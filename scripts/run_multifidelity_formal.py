#!/usr/bin/env python3
"""Fail-closed runner for the preregistered v0.3 multi-fidelity experiment.

Each invocation executes exactly one phase.  In particular, ``train`` has no
code path that opens a sealed label archive, while ``evaluate`` opens each of
the four archives once, after a frozen :class:`CheckpointRegistry` exists.

The ``tiny-mock`` backend exists only for state-machine/integrity tests.  Its
artifacts are stamped ``effect_claim_allowed=false`` and cannot be analyzed as
formal evidence.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import os
import platform
import sys
import time
import tracemalloc
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from tunnelgeopt.multifidelity import CheckpointRegistry
from tunnelgeopt.multifidelity_learning import (
    LearningBatch,
    LearningContractError,
    TrainingContract,
    aggregate_case_errors_by_parent,
    build_training_contract,
    case_weighted_stress_error,
    checkpoint_payload,
    hierarchical_paired_bootstrap,
    load_formal_model_from_checkpoint,
    make_model,
    method_arrays,
    mismatched_coarse_indices,
    nested_geometry_subsets,
    predict,
    reconstruct_fine_prediction,
    save_formal_checkpoint_atomic,
    train_formal_with_dev_selection,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "multifidelity_formal.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "experiment" / "mf-residual-formal-v0.3.0"
PHASES = ("prepare", "generate", "train", "evaluate", "analyze")
SECTION_NAMES = ("circle", "horseshoe", "straight_wall_arch")
LOCKED_PARTITIONS = (
    "locked_iid",
    "locked_geometry_ood",
    "locked_load_ood",
    "locked_joint_ood",
)
SEALED_FILENAMES = {
    "locked_iid": "sealed_locked_iid_fine_labels.npz",
    "locked_geometry_ood": "sealed_locked_geometry_ood_fine_labels.npz",
    "locked_load_ood": "sealed_locked_load_ood_fine_labels.npz",
    "locked_joint_ood": "sealed_locked_joint_ood_fine_labels.npz",
}
PUBLIC_FILENAME = "public_inputs_and_coarse_fields.npz"
TRAIN_DEV_FILENAME = "train_dev_fine_labels.npz"
DATASET_MANIFEST_FILENAME = "formal_dataset_manifest.json"
STATE_FILENAME = "formal_run_state.json"
ACCESS_LOG_FILENAME = "access_audit.jsonl"
REGISTRY_FILENAME = "checkpoint_registry.json"
CHECKPOINT_MANIFEST_FILENAME = "checkpoint_manifest.json"
EXACT_METHOD_FRACTIONS = {
    "scratch": (1.0,),
    "direct_coarse": (1.0,),
    "residual_coarse": (0.25, 0.5, 0.75, 1.0),
    "mismatched_coarse": (0.5,),
}


class FormalRunError(RuntimeError):
    """Raised when a formal execution contract is violated."""


class SealedAccessError(FormalRunError):
    """Raised before any forbidden sealed-label path is opened."""


class FormalAbstain(FormalRunError):
    """Raised for an integrity failure whose only valid result is ABSTAIN."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_value(value: Any) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, name: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise FormalRunError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _atomic_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_bytes(_canonical_bytes(value) + b"\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return _file_sha256(path)


def _atomic_npz(path: Path, **arrays: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return _file_sha256(path)


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FormalRunError(f"could not read {description}: {exc}") from exc
    if not isinstance(value, dict):
        raise FormalRunError(f"{description} must be a JSON object")
    return value


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_bytes(dict(value)) + b"\n"
    with path.open("ab") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _load_npz_bytes(payload: bytes, description: str) -> dict[str, np.ndarray]:
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            return {name: np.asarray(archive[name]) for name in archive.files}
    except (OSError, ValueError, KeyError) as exc:
        raise FormalRunError(f"could not parse {description}: {exc}") from exc


def _load_npz(path: Path, description: str) -> dict[str, np.ndarray]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise FormalRunError(f"could not open {description}: {exc}") from exc
    return _load_npz_bytes(payload, description)


def _format_fraction(value: float) -> str:
    return f"{round(float(value) * 100):03d}"


def _checkpoint_key(method: str, fraction: float, seed: int) -> str:
    return f"{method}__f{_format_fraction(fraction)}__seed{int(seed)}"


def _distribution_summary(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise FormalAbstain("cannot summarize empty or non-finite metrics")
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "maximum": float(np.max(array)),
    }


def _wall_offset_discrepancy(
    prediction: np.ndarray,
    fine: np.ndarray,
    arc_weights: np.ndarray,
    normals_yz: np.ndarray,
    stress_scales: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-case D_t and D_r relative to fine at the frozen offset."""

    prediction = np.asarray(prediction, dtype=np.float64)
    fine = np.asarray(fine, dtype=np.float64)
    weights = np.asarray(arc_weights, dtype=np.float64)
    normals = np.asarray(normals_yz, dtype=np.float64)
    scales = np.asarray(stress_scales, dtype=np.float64)
    if (
        prediction.shape != fine.shape
        or prediction.ndim != 3
        or prediction.shape[-1] != 3
        or weights.shape != prediction.shape[:2]
        or normals.shape != (*prediction.shape[:2], 2)
        or scales.shape != (prediction.shape[0],)
    ):
        raise FormalRunError("wall-offset diagnostic arrays do not align")
    if (
        not all(np.isfinite(value).all() for value in (prediction, fine, weights, normals, scales))
        or np.any(weights < 0.0)
        or np.any(scales <= 0.0)
        or not np.allclose(weights.sum(axis=1), 1.0)
    ):
        raise FormalAbstain("wall-offset diagnostic inputs are invalid")
    difference = (prediction - fine) * scales[:, None, None]
    tensor = np.empty((*difference.shape[:2], 2, 2), dtype=np.float64)
    tensor[..., 0, 0] = difference[..., 0]
    tensor[..., 1, 1] = difference[..., 1]
    tensor[..., 0, 1] = difference[..., 2]
    tensor[..., 1, 0] = difference[..., 2]
    traction = np.einsum("cpij,cpj->cpi", tensor, normals)
    traction_squared = np.sum(traction**2, axis=-1)
    d_t = np.sqrt(np.sum(weights * traction_squared, axis=1)) / scales
    resultant = np.sum(weights[..., None] * traction, axis=1)
    d_r = np.linalg.norm(resultant, axis=1) / scales
    if not np.isfinite(d_t).all() or not np.isfinite(d_r).all():
        raise FormalAbstain("wall-offset diagnostic produced a non-finite value")
    return d_t, d_r


def _expected_checkpoint_specs(config: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    learning = config["learning"]
    matrix = {
        str(method): tuple(float(value) for value in fractions)
        for method, fractions in learning["method_fraction_matrix"].items()
    }
    if matrix != EXACT_METHOD_FRACTIONS:
        raise FormalRunError("formal method/fraction matrix is not the frozen 7-per-seed design")
    seeds = tuple(int(value) for value in learning["training_seeds"])
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise FormalRunError("formal training requires exactly five unique seeds")
    specifications = tuple(
        {
            "method": method,
            "fraction": fraction,
            "seed": seed,
            "checkpoint_key": _checkpoint_key(method, fraction, seed),
        }
        for seed in seeds
        for method, fractions in EXACT_METHOD_FRACTIONS.items()
        for fraction in fractions
    )
    if len(specifications) != 35 or int(learning["expected_checkpoint_count"]) != 35:
        raise FormalRunError("formal checkpoint count must be exactly 35")
    return specifications


def load_formal_config(path: Path, *, tiny_mock: bool = False) -> tuple[dict[str, Any], str]:
    """Load the complete config and validate invariants used by every phase."""

    config = _read_json(path, "formal config")
    digest = _sha256_value(config)
    required = {
        "schema_version",
        "preregistration_id",
        "config_name",
        "run_id",
        "status",
        "scope",
        "claim_exclusions",
        "hashing",
        "preformal_development_evidence",
        "identity_and_split",
        "geometry",
        "material_and_loads",
        "dataset",
        "mesh",
        "query",
        "quality_control",
        "learning",
        "sealed_evaluation",
        "evaluation",
        "scientific_decision",
    }
    if set(config) != required:
        raise FormalRunError("formal config top-level key set changed")
    if config["schema_version"] != "tunnelgeopt.multifidelity.formal.v1":
        raise FormalRunError("formal config schema is unsupported")
    if tuple(config["geometry"]["section_families"]) != SECTION_NAMES:
        raise FormalRunError("formal section families changed")
    if config["identity_and_split"]["split_unit"] != "geometry_group_id":
        raise FormalRunError("formal split unit must remain the parent geometry")
    if tuple(map(float, config["learning"]["fine_train_fractions"])) != (
        0.25,
        0.5,
        0.75,
        1.0,
    ):
        raise FormalRunError("formal nested fractions changed")
    _expected_checkpoint_specs(config)
    stores = config["sealed_evaluation"]["locked_label_stores"]
    if tuple(stores) != LOCKED_PARTITIONS:
        raise FormalRunError("formal locked partition set or order changed")
    if int(config["sealed_evaluation"]["max_evaluation_calls_per_locked_partition"]) != 1:
        raise FormalRunError("each locked partition must be evaluated exactly once")
    if int(config["sealed_evaluation"]["checkpoint_count_required_before_authorization"]) != 35:
        raise FormalRunError("sealed authorization must require all 35 checkpoints")
    if tiny_mock:
        if not str(config["run_id"]).startswith("tiny-mock-"):
            raise FormalRunError("tiny-mock config run_id must start with 'tiny-mock-'")
        if config.get("scope") != "tiny_mock_state_machine_only_no_scientific_claim":
            raise FormalRunError("tiny-mock scope cannot be formal evidence")
    return config, digest


def _validate_approval(
    approval_path: Path,
    *,
    config: Mapping[str, Any],
    config_sha256: str,
    tiny_mock: bool,
) -> dict[str, Any]:
    approval = _read_json(approval_path, "execution approval")
    required_true = (
        "config_frozen",
        "development_convergence_audit_passed",
        "ultrafine_development_audit_passed",
        "formal_execution_authorized",
    )
    if approval.get("run_id") != config["run_id"]:
        raise FormalRunError("approval run_id does not match the config")
    if approval.get("config_sha256") != config_sha256:
        raise FormalRunError("approval is not bound to the complete config hash")
    if any(approval.get(key) is not True for key in required_true):
        raise FormalRunError("formal approval gates are not all true")
    if tiny_mock:
        if approval.get("schema") != "tunnelgeopt.tiny_mock_approval.v1":
            raise FormalRunError("tiny-mock approval schema is invalid")
        if approval.get("test_only") is not True:
            raise FormalRunError("tiny-mock approval must be marked test_only")
        return approval
    if approval.get("schema") != "tunnelgeopt.formal_execution_approval.v1":
        raise FormalRunError("formal approval schema is invalid")
    status = config["status"]
    if (
        status.get("eligible_to_generate_formal_data") is not True
        or status.get("requires_dev_convergence_pass") is not False
        or status.get("state") != "frozen_preregistered_pre_generation"
        or status.get("formal_data_generated") is not False
        or status.get("locked_labels_opened") is not False
    ):
        raise FormalRunError("candidate config is not eligible for formal generation")
    return approval


def make_tiny_mock_config(base: Mapping[str, Any], *, run_id: str) -> dict[str, Any]:
    """Return a small, explicitly non-scientific config for runner tests."""

    config = json.loads(json.dumps(base))
    if not run_id.startswith("tiny-mock-"):
        raise ValueError("tiny mock run_id must start with 'tiny-mock-'")
    config["run_id"] = run_id
    config["scope"] = "tiny_mock_state_machine_only_no_scientific_claim"
    config["status"].update(
        {
            "state": "tiny_mock_test_only",
            "eligible_to_generate_formal_data": False,
            "formal_data_generated": False,
            "formal_effect_computation_started": False,
            "locked_labels_opened": False,
        }
    )
    counts = {
        "train_id": (12, 4, 2),
        "dev_id": (3, 1, 2),
        "locked_iid": (3, 1, 2),
        "locked_geometry_ood": (3, 1, 1),
        "locked_load_ood": (3, 1, 1),
        "locked_joint_ood": (3, 1, 1),
    }
    for name, (parents, per_section, loads) in counts.items():
        partition = config["dataset"]["partitions"][name]
        partition.update(
            {
                "parent_geometries": parents,
                "parents_per_section": per_section,
                "loads_per_parent": loads,
                "cases": parents * loads,
            }
        )
    config["dataset"]["total_parent_geometries"] = sum(value[0] for value in counts.values())
    config["dataset"]["total_cases"] = sum(value[0] * value[2] for value in counts.values())
    config["query"].update(
        {"points_per_case": 8, "nearfield_volume": 4, "wall_offset": 2, "farfield": 2}
    )
    config["learning"]["nested_parent_counts"] = {
        "0.25": {"total": 3, "per_section": 1},
        "0.5": {"total": 6, "per_section": 2},
        "0.75": {"total": 9, "per_section": 3},
        "1.0": {"total": 12, "per_section": 4},
    }
    config["evaluation"]["bootstrap"].update({"replicates": 50})
    config["quality_control"]["fine_ultrafine"].update(
        {"expected_formal_audit_cases": 0, "locked_audit_runs_inside_sealed_generator": True}
    )
    return config


def write_tiny_mock_approval(path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    """Write a hash-bound approval usable only with ``tiny-mock``."""

    approval = {
        "schema": "tunnelgeopt.tiny_mock_approval.v1",
        "test_only": True,
        "run_id": config["run_id"],
        "config_sha256": _sha256_value(config),
        "config_frozen": True,
        "development_convergence_audit_passed": True,
        "ultrafine_development_audit_passed": True,
        "formal_execution_authorized": True,
        "approved_at_utc": _now(),
    }
    _atomic_json(path, approval)
    return approval


@dataclass(frozen=True)
class RunnerPaths:
    root: Path

    @property
    def state(self) -> Path:
        return self.root / STATE_FILENAME

    @property
    def access_log(self) -> Path:
        return self.root / ACCESS_LOG_FILENAME

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def evaluation(self) -> Path:
        return self.root / "evaluation"

    @property
    def analysis(self) -> Path:
        return self.root / "analysis"


class FormalExperimentRunner:
    """Five-phase resumable runner with a fail-closed sealed access boundary."""

    def __init__(
        self,
        *,
        config_path: Path,
        approval_path: Path,
        output_dir: Path,
        backend: str = "formal",
        device: str = "cuda",
    ) -> None:
        if backend not in {"formal", "tiny-mock"}:
            raise ValueError("backend must be 'formal' or 'tiny-mock'")
        self.config_path = Path(config_path).resolve()
        self.approval_path = Path(approval_path).resolve()
        self.paths = RunnerPaths(Path(output_dir).resolve())
        self.backend = backend
        self.tiny_mock = backend == "tiny-mock"
        self.device = str(device)
        self.config, self.config_sha256 = load_formal_config(
            self.config_path, tiny_mock=self.tiny_mock
        )
        self.approval = _validate_approval(
            self.approval_path,
            config=self.config,
            config_sha256=self.config_sha256,
            tiny_mock=self.tiny_mock,
        )

    def _dataset_paths(self) -> tuple[Path, Path, Path]:
        """Return manifest/public/train paths without exposing a sealed path."""

        if self.tiny_mock:
            return (
                self.paths.data / DATASET_MANIFEST_FILENAME,
                self.paths.data / PUBLIC_FILENAME,
                self.paths.data / TRAIN_DEV_FILENAME,
            )
        try:
            from tunnelgeopt.formal_generation import training_data_paths
        except ImportError as exc:  # pragma: no cover - integration guard
            raise FormalRunError("formal generation module is unavailable") from exc
        values = training_data_paths(self.paths.data)
        public = Path(values.public_inputs_path)
        train = Path(values.train_dev_fine_labels_path)
        manifest_candidates = (
            self.paths.data / DATASET_MANIFEST_FILENAME,
            self.paths.data / "manifest.json",
            self.paths.data / "formal_manifest.json",
        )
        manifest = next((path for path in manifest_candidates if path.is_file()), None)
        if manifest is None:
            raise FormalRunError("formal dataset manifest is missing")
        return manifest, public, train

    def _training_paths(self) -> tuple[Path, Path]:
        """Trainer-facing API: deliberately returns no manifest/sealed path."""

        _, public, train = self._dataset_paths()
        return public, train

    def _trusted_sealed_path(self, partition: str) -> Path:
        if self.tiny_mock:
            return self.paths.data / SEALED_FILENAMES[partition]
        try:
            from tunnelgeopt.formal_generation import trusted_locked_label_path
        except ImportError as exc:  # pragma: no cover - integration guard
            raise FormalRunError("formal generation module is unavailable") from exc
        return Path(trusted_locked_label_path(self.paths.data, partition))

    def _manifest_file_digest(
        self, manifest: Mapping[str, Any], path: Path, *, required: bool = True
    ) -> str | None:
        files = manifest.get("files", manifest.get("file_hashes", {}))
        if not isinstance(files, Mapping):
            raise FormalRunError("dataset manifest file hash map is invalid")
        candidates = {
            str(path),
            path.name,
            path.as_posix(),
        }
        try:
            candidates.add(path.relative_to(self.paths.data).as_posix())
        except ValueError:
            pass
        for key, value in files.items():
            if str(key) in candidates or Path(str(key)).name == path.name:
                if isinstance(value, Mapping):
                    value = value.get("sha256")
                return _require_sha256(value, f"manifest digest for {path.name}")
        if required:
            raise FormalRunError(f"dataset manifest omits file digest for {path.name}")
        return None

    def _initial_state(self) -> dict[str, Any]:
        return {
            "schema": "tunnelgeopt.multifidelity.formal_run_state.v1",
            "run_id": self.config["run_id"],
            "config_sha256": self.config_sha256,
            "backend": self.backend,
            "effect_claim_allowed": not self.tiny_mock,
            "created_at_utc": _now(),
            "updated_at_utc": _now(),
            "phases": {
                phase: {"status": "pending", "attempts": 0, "artifacts": {}} for phase in PHASES
            },
            "denied_premature_sealed_accesses": 0,
            "sealed_partition_open_counts": {name: 0 for name in LOCKED_PARTITIONS},
            "checkpoint_evaluation_counts": {},
            "abstain_reasons": [],
        }

    def _load_state(self, *, create: bool = False) -> dict[str, Any]:
        if not self.paths.state.exists():
            if not create:
                raise FormalRunError("prepare phase has not created run state")
            state = self._initial_state()
            self.paths.root.mkdir(parents=True, exist_ok=True)
            _atomic_json(self.paths.state, state)
            return state
        state = _read_json(self.paths.state, "formal run state")
        if (
            state.get("run_id") != self.config["run_id"]
            or state.get("config_sha256") != self.config_sha256
            or state.get("backend") != self.backend
        ):
            raise FormalRunError("run state belongs to a different config/backend")
        if set(state.get("phases", {})) != set(PHASES):
            raise FormalRunError("run state phase order changed")
        return state

    def _save_state(self, state: dict[str, Any]) -> None:
        state["updated_at_utc"] = _now()
        _atomic_json(self.paths.state, state)

    def _event(self, state: dict[str, Any], event: str, **details: Any) -> None:
        _append_jsonl(
            self.paths.access_log,
            {
                "at_utc": _now(),
                "run_id": self.config["run_id"],
                "config_sha256": self.config_sha256,
                "event": event,
                **details,
            },
        )
        self._save_state(state)

    def _verify_artifacts(self, artifacts: Mapping[str, str]) -> None:
        for relative, expected in artifacts.items():
            path = self.paths.root / relative
            if not path.is_file() or _file_sha256(path) != _require_sha256(expected, relative):
                raise FormalAbstain(f"completed phase artifact drifted: {relative}")

    def _begin_phase(self, phase: str) -> tuple[dict[str, Any], bool]:
        if phase not in PHASES:
            raise ValueError(f"unknown phase {phase!r}")
        state = self._load_state(create=phase == "prepare")
        record = state["phases"][phase]
        if record["status"] == "completed":
            self._verify_artifacts(record.get("artifacts", {}))
            return state, False
        phase_index = PHASES.index(phase)
        for predecessor in PHASES[:phase_index]:
            if state["phases"][predecessor]["status"] != "completed":
                raise FormalRunError(f"phase {phase!r} requires completed {predecessor!r}")
        if phase == "evaluate" and record["status"] == "in_progress":
            reason = "evaluation was interrupted after a sealed partition may have been opened"
            if reason not in state["abstain_reasons"]:
                state["abstain_reasons"].append(reason)
            record["status"] = "abstained"
            self._event(state, "evaluation_resume_refused", reason=reason)
            raise FormalAbstain(reason)
        if record["status"] == "abstained":
            raise FormalAbstain("run is already ABSTAIN and cannot resume in place")
        record["status"] = "in_progress"
        record["attempts"] = int(record.get("attempts", 0)) + 1
        record["started_at_utc"] = _now()
        self._event(state, "phase_started", phase=phase, attempt=record["attempts"])
        return state, True

    def _complete_phase(
        self, state: dict[str, Any], phase: str, artifacts: Mapping[str, str]
    ) -> dict[str, Any]:
        record = state["phases"][phase]
        record.update(
            {
                "status": "completed",
                "completed_at_utc": _now(),
                "artifacts": dict(sorted(artifacts.items())),
            }
        )
        self._event(state, "phase_completed", phase=phase, artifacts=dict(artifacts))
        return {"phase": phase, "status": "completed", "artifacts": dict(artifacts)}

    def run_phase(self, phase: str) -> dict[str, Any]:
        state, execute = self._begin_phase(phase)
        if not execute:
            return {"phase": phase, "status": "already_completed"}
        handlers = {
            "prepare": self._run_prepare,
            "generate": self._run_generate,
            "train": self._run_train,
            "evaluate": self._run_evaluate,
            "analyze": self._run_analyze,
        }
        try:
            artifacts = handlers[phase](state)
        except Exception as exc:
            state = self._load_state()
            if phase != "evaluate":
                state["phases"][phase]["status"] = "failed"
            state["phases"][phase]["last_error"] = f"{type(exc).__name__}: {exc}"
            self._event(state, "phase_failed", phase=phase, error=str(exc))
            raise
        # Handlers may persist monotonic audit counters through a separately
        # loaded state object.  Never overwrite those counters with the stale
        # object created at phase entry.
        return self._complete_phase(self._load_state(), phase, artifacts)

    def open_sealed_partition(self, partition: str) -> dict[str, np.ndarray]:
        """The sole sealed-reader; it is authorization- and count-gated."""

        if partition not in LOCKED_PARTITIONS:
            raise ValueError("unknown locked partition")
        state = self._load_state()
        registry_path = self.paths.checkpoints / REGISTRY_FILENAME
        authorized = (
            state["phases"]["train"]["status"] == "completed"
            and state["phases"]["evaluate"]["status"] == "in_progress"
            and registry_path.is_file()
        )
        if not authorized:
            state["denied_premature_sealed_accesses"] += 1
            self._event(state, "sealed_access_denied", partition=partition)
            raise SealedAccessError("sealed labels cannot be opened before frozen evaluation")
        if int(state["sealed_partition_open_counts"][partition]) != 0:
            reason = f"sealed partition {partition} would be opened more than once"
            state["abstain_reasons"].append(reason)
            self._event(state, "sealed_reopen_denied", partition=partition)
            raise FormalAbstain(reason)
        path = self._trusted_sealed_path(partition)
        try:
            payload = path.read_bytes()  # exactly one filesystem open
        except OSError as exc:
            raise FormalAbstain(f"could not open sealed partition {partition}: {exc}") from exc
        state["sealed_partition_open_counts"][partition] = 1
        self._event(
            state,
            "sealed_partition_opened",
            partition=partition,
            bytes=len(payload),
            sha256=_sha256_bytes(payload),
        )
        return _load_npz_bytes(payload, f"sealed partition {partition}")

    def _run_prepare(self, state: dict[str, Any]) -> dict[str, str]:
        self.paths.root.mkdir(parents=True, exist_ok=True)
        snapshot = {
            "schema": "tunnelgeopt.multifidelity.formal_prepare.v1",
            "run_id": self.config["run_id"],
            "backend": self.backend,
            "effect_claim_allowed": not self.tiny_mock,
            "config_sha256": self.config_sha256,
            "approval_sha256": _file_sha256(self.approval_path),
            "prepared_at_utc": _now(),
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "device_requested": self.device,
            },
            "formal_data_generated": False,
            "locked_labels_opened": False,
        }
        config_snapshot = self.paths.root / "frozen_config_snapshot.json"
        approval_snapshot = self.paths.root / "execution_approval_snapshot.json"
        prepare_manifest = self.paths.root / "prepare_manifest.json"
        hashes = {
            "frozen_config_snapshot.json": _atomic_json(config_snapshot, self.config),
            "execution_approval_snapshot.json": _atomic_json(approval_snapshot, self.approval),
        }
        snapshot["snapshot_hashes"] = dict(hashes)
        hashes["prepare_manifest.json"] = _atomic_json(prepare_manifest, snapshot)
        return hashes

    def _run_generate(self, state: dict[str, Any]) -> dict[str, str]:
        if not self.tiny_mock:
            try:
                from tunnelgeopt.formal_generation import (
                    build_formal_generation_plan,
                    generate_formal_dataset,
                )
            except ImportError as exc:  # pragma: no cover - integration guard
                raise FormalRunError("formal generation module is unavailable") from exc
            plan = build_formal_generation_plan(self.config)
            if plan.formal_eligible is not True:
                raise FormalRunError("formal generation plan is not eligible")

            def progress(event: Mapping[str, Any]) -> None:
                _append_jsonl(
                    self.paths.root / "generation_progress.jsonl",
                    {"at_utc": _now(), **dict(event)},
                )

            result = generate_formal_dataset(
                self.config,
                self.paths.data,
                resume=True,
                progress_callback=progress,
            )
            if not result.manifest_path.is_file():
                raise FormalRunError("trusted generator did not write its manifest")
            artifacts: dict[str, str] = {}
            allowed_paths = {
                Path(result.manifest_path).resolve(),
                Path(result.public_inputs_path).resolve(),
                Path(result.train_dev_labels_path).resolve(),
            }
            for path_value, expected_digest in result.file_hashes.items():
                path = Path(path_value)
                if not path.is_absolute():
                    path = self.paths.data / path
                if path.resolve() not in allowed_paths:
                    continue
                digest = _file_sha256(path)
                if digest != _require_sha256(expected_digest, str(path_value)):
                    raise FormalRunError("trusted generator returned a stale file digest")
                artifacts[path.relative_to(self.paths.root).as_posix()] = digest
            manifest_path = Path(result.manifest_path)
            artifacts[manifest_path.relative_to(self.paths.root).as_posix()] = _file_sha256(
                manifest_path
            )
            self._event(
                state,
                "trusted_generator_completed",
                test_only=False,
                solved_cases=int(result.solved_cases),
                resumed_cases=int(result.resumed_cases),
                audit_summary=dict(result.audit_summary),
            )
            return artifacts
        manifest = self._generate_tiny_mock_dataset()
        self._event(
            state,
            "trusted_generator_completed",
            test_only=True,
            sealed_files_written=len(LOCKED_PARTITIONS),
        )
        artifacts = {
            f"data/{DATASET_MANIFEST_FILENAME}": _file_sha256(
                self.paths.data / DATASET_MANIFEST_FILENAME
            ),
            f"data/{PUBLIC_FILENAME}": manifest["files"][PUBLIC_FILENAME],
            f"data/{TRAIN_DEV_FILENAME}": manifest["files"][TRAIN_DEV_FILENAME],
        }
        # Sealed identities remain opaque inside the trusted manifest; neither
        # phase state nor the trainer-facing path API receives their paths.
        return artifacts

    def _generate_tiny_mock_dataset(self) -> dict[str, Any]:
        rng = np.random.default_rng(90125)
        partitions = self.config["dataset"]["partitions"]
        points = int(self.config["query"]["points_per_case"])
        metadata: list[dict[str, str]] = []
        base_rows: list[np.ndarray] = []
        coarse_rows: list[np.ndarray] = []
        fine_rows: list[np.ndarray] = []
        training_weight_rows: list[np.ndarray] = []
        metric_weight_rows: list[np.ndarray] = []
        arc_weight_rows: list[np.ndarray] = []
        wall_normal_rows: list[np.ndarray] = []
        region_rows: list[np.ndarray] = []
        partition_order = ("train_id", "dev_id", *LOCKED_PARTITIONS)
        split_map = {"train_id": "train", "dev_id": "dev"}
        for partition_index, partition in enumerate(partition_order):
            spec = partitions[partition]
            parents_per_section = int(spec["parents_per_section"])
            loads_per_parent = int(spec["loads_per_parent"])
            for section_index, section in enumerate(SECTION_NAMES):
                for parent_index in range(parents_per_section):
                    geometry_id = f"{partition}__{section}__g{parent_index:03d}"
                    boundary_hash = _sha256_bytes(geometry_id.encode("utf-8"))
                    for load_index in range(loads_per_parent):
                        case_id = f"{geometry_id}__case{load_index:02d}"
                        load_id = _sha256_bytes(f"{partition}:{geometry_id}:{load_index}".encode())
                        base = rng.normal(size=(points, 11)).astype(np.float32)
                        base[:, 0] = np.linspace(-1.0, 1.0, points, dtype=np.float32)
                        base[:, 1] = float(section_index)
                        coarse = rng.normal(size=(points, 3)).astype(
                            np.float32
                        ) * 0.15 + np.asarray([-1.0, -0.55, 0.05], dtype=np.float32)
                        correction = (
                            0.04 * np.tanh(base[:, :3])
                            + 0.005 * float(partition_index + load_index)
                        ).astype(np.float32)
                        fine = coarse + correction
                        regions = np.concatenate(
                            [
                                np.full(int(self.config["query"]["nearfield_volume"]), 0),
                                np.full(int(self.config["query"]["wall_offset"]), 1),
                                np.full(int(self.config["query"]["farfield"]), 2),
                            ]
                        ).astype(np.int8)
                        region_masses = np.asarray([0.8, 0.15, 0.05], dtype=np.float32)
                        training_weights = np.zeros(points, dtype=np.float32)
                        for region, mass in enumerate(region_masses):
                            mask = regions == region
                            training_weights[mask] = mass / int(mask.sum())
                        metric_weights = np.zeros(points, dtype=np.float32)
                        metric_weights[regions == 0] = 1.0 / int(np.sum(regions == 0))
                        arc_weights = np.zeros(points, dtype=np.float32)
                        arc_weights[regions == 1] = 1.0 / int(np.sum(regions == 1))
                        wall_normals = np.zeros((points, 2), dtype=np.float32)
                        wall_normals[regions == 1, 0] = 1.0
                        metadata.append(
                            {
                                "case_group_id": case_id,
                                "geometry_group_id": geometry_id,
                                "section_family": section,
                                "partition": partition,
                                "split": split_map.get(partition, partition),
                                "load_group_id": load_id,
                                "load_subtype": (
                                    "id"
                                    if partition in {"train_id", "dev_id", "locked_iid"}
                                    else (
                                        "geometry_ood"
                                        if partition == "locked_geometry_ood"
                                        else "ood"
                                    )
                                ),
                                "boundary_float64_sha256": boundary_hash,
                            }
                        )
                        base_rows.append(base)
                        coarse_rows.append(coarse)
                        fine_rows.append(fine)
                        training_weight_rows.append(training_weights)
                        metric_weight_rows.append(metric_weights)
                        arc_weight_rows.append(arc_weights)
                        wall_normal_rows.append(wall_normals)
                        region_rows.append(regions)
        base_array = np.stack(base_rows)
        coarse_array = np.stack(coarse_rows)
        fine_array = np.stack(fine_rows)
        training_weights_array = np.stack(training_weight_rows)
        metric_weights_array = np.stack(metric_weight_rows)
        arc_weights_array = np.stack(arc_weight_rows)
        wall_normals_array = np.stack(wall_normal_rows)
        regions_array = np.stack(region_rows)
        public = {
            "base_features": base_array,
            "coarse_stress": coarse_array,
            "training_weights": training_weights_array,
            "metric_weights": metric_weights_array,
            "arc_weights": arc_weights_array,
            "wall_rock_outward_normals_yz": wall_normals_array,
            "query_regions": regions_array,
            "nearfield_mask": regions_array == 0,
            "wall_offset_mask": regions_array == 1,
            "farfield_mask": regions_array == 2,
            "stress_scales": np.ones(len(metadata), dtype=np.float64),
            "query_hashes": np.asarray(
                [_sha256_bytes(row["geometry_group_id"].encode()) for row in metadata],
                dtype="U64",
            ),
            "case_group_ids": np.asarray([row["case_group_id"] for row in metadata], dtype="U160"),
            "geometry_group_ids": np.asarray(
                [row["geometry_group_id"] for row in metadata], dtype="U128"
            ),
            "section_families": np.asarray(
                [row["section_family"] for row in metadata], dtype="U32"
            ),
            "partitions": np.asarray([row["partition"] for row in metadata], dtype="U32"),
            "splits": np.asarray([row["split"] for row in metadata], dtype="U32"),
            "load_group_ids": np.asarray([row["load_group_id"] for row in metadata], dtype="U64"),
            "load_subtypes": np.asarray([row["load_subtype"] for row in metadata], dtype="U32"),
            "boundary_float64_sha256": np.asarray(
                [row["boundary_float64_sha256"] for row in metadata], dtype="U64"
            ),
        }
        self.paths.data.mkdir(parents=True, exist_ok=True)
        file_hashes = {PUBLIC_FILENAME: _atomic_npz(self.paths.data / PUBLIC_FILENAME, **public)}
        train_dev_indices = np.flatnonzero(
            np.isin(public["partitions"], np.asarray(["train_id", "dev_id"]))
        )
        file_hashes[TRAIN_DEV_FILENAME] = _atomic_npz(
            self.paths.data / TRAIN_DEV_FILENAME,
            indices=train_dev_indices,
            case_group_ids=public["case_group_ids"][train_dev_indices],
            fine_stress=fine_array[train_dev_indices],
        )
        for partition, filename in SEALED_FILENAMES.items():
            indices = np.flatnonzero(public["partitions"] == partition)
            file_hashes[filename] = _atomic_npz(
                self.paths.data / filename,
                indices=indices,
                case_group_ids=public["case_group_ids"][indices],
                fine_stress=fine_array[indices],
                ultrafine_audit_passed=np.asarray([True]),
            )
        manifest = self._validate_generated_dataset(public, metadata, file_hashes)
        _atomic_json(self.paths.data / DATASET_MANIFEST_FILENAME, manifest)
        return manifest

    def _validate_generated_dataset(
        self,
        public: Mapping[str, np.ndarray],
        records: Sequence[Mapping[str, str]],
        file_hashes: Mapping[str, str],
    ) -> dict[str, Any]:
        case_ids = tuple(str(value) for value in public["case_group_ids"])
        if len(case_ids) != len(set(case_ids)):
            raise FormalRunError("case_group_id collision in generated data")
        if len(records) != len(case_ids):
            raise FormalRunError("case manifest and public tensors do not align")
        partitions = tuple(str(value) for value in public["partitions"])
        geometry_ids = tuple(str(value) for value in public["geometry_group_ids"])
        load_ids = tuple(str(value) for value in public["load_group_ids"])
        boundary_hashes = tuple(str(value) for value in public["boundary_float64_sha256"])
        geometry_partition: dict[str, str] = {}
        boundary_partition: dict[str, str] = {}
        load_partition: dict[str, str] = {}
        for geometry_id, boundary_hash, load_id, partition in zip(
            geometry_ids, boundary_hashes, load_ids, partitions, strict=True
        ):
            for identity, mapping, name in (
                (geometry_id, geometry_partition, "geometry_group_id"),
                (boundary_hash, boundary_partition, "boundary_float64_sha256"),
                (load_id, load_partition, "load_group_id"),
            ):
                previous = mapping.setdefault(identity, partition)
                if previous != partition:
                    raise FormalRunError(f"{name} crosses formal partitions")
        partition_records: dict[str, Any] = {}
        for partition, spec in self.config["dataset"]["partitions"].items():
            indices = np.flatnonzero(np.asarray(partitions) == partition)
            parent_count = len(set(np.asarray(geometry_ids)[indices].tolist()))
            section_counts = {
                section: len(
                    set(
                        np.asarray(geometry_ids)[
                            indices[np.asarray(public["section_families"])[indices] == section]
                        ].tolist()
                    )
                )
                for section in SECTION_NAMES
            }
            if (
                indices.size != int(spec["cases"])
                or parent_count != int(spec["parent_geometries"])
                or any(
                    value != int(spec["parents_per_section"]) for value in section_counts.values()
                )
            ):
                raise FormalRunError(f"generated partition count mismatch for {partition}")
            partition_records[partition] = {
                "case_count": int(indices.size),
                "parent_geometry_count": parent_count,
                "parents_per_section": section_counts,
                "case_group_ids_sha256": _sha256_value(
                    sorted(np.asarray(case_ids)[indices].tolist())
                ),
                "geometry_group_ids_sha256": _sha256_value(
                    sorted(set(np.asarray(geometry_ids)[indices].tolist()))
                ),
                "load_group_ids_sha256": _sha256_value(
                    sorted(np.asarray(load_ids)[indices].tolist())
                ),
            }
        manifest = {
            "schema": "tunnelgeopt.multifidelity.formal_dataset_manifest.v1",
            "run_id": self.config["run_id"],
            "config_sha256": self.config_sha256,
            "backend": self.backend,
            "effect_claim_allowed": not self.tiny_mock,
            "generated_at_utc": _now(),
            "public_case_count": len(case_ids),
            "files": dict(sorted(file_hashes.items())),
            "partitions": partition_records,
            "identity_checks": {
                "unique_case_group_ids": True,
                "zero_cross_partition_geometry_group_id": True,
                "zero_cross_partition_boundary_float64_sha256": True,
                "zero_cross_partition_case_group_id": True,
                "zero_cross_partition_load_group_id": True,
                "section_stratified_counts_exact": True,
            },
            "quality_control": {
                "minimum_valid_case_fraction_per_partition_section": 1.0,
                "fine_ultrafine_independent_gate_passed": True,
                "nonfinite_count": 0,
            },
            "case_manifest_sha256": _sha256_value([dict(record) for record in records]),
        }
        return manifest

    def _load_training_inputs(
        self,
    ) -> tuple[dict[str, np.ndarray], LearningBatch, dict[str, Any]]:
        """Load only public data and train/dev labels (never a sealed path)."""

        manifest_path, public_path, labels_path = self._dataset_paths()
        manifest = _read_json(manifest_path, "formal dataset manifest")
        # The trusted generator may omit the runner backend stamp, but it may
        # never disagree with it or the complete config hash.
        if manifest.get("config_sha256") != self.config_sha256 or manifest.get("backend") not in (
            None,
            self.backend,
        ):
            raise FormalRunError("dataset manifest belongs to a different run")
        for path in (public_path, labels_path):
            if _file_sha256(path) != self._manifest_file_digest(manifest, path):
                raise FormalAbstain(f"training input hash mismatch: {path.name}")
        public = _load_npz(public_path, "public inputs and coarse fields")
        labels = _load_npz(labels_path, "train/dev fine labels")
        required_public = {
            "base_features",
            "coarse_stress",
            "training_weights",
            "metric_weights",
            "arc_weights",
            "wall_rock_outward_normals_yz",
            "stress_scales",
            "nearfield_mask",
            "wall_offset_mask",
            "farfield_mask",
            "query_hashes",
            "case_group_ids",
            "geometry_group_ids",
            "section_families",
            "partitions",
            "load_group_ids",
            "load_subtypes",
        }
        if not required_public.issubset(public):
            missing = sorted(required_public - set(public))
            raise FormalRunError(f"public archive omits required fields: {missing}")
        if not {"case_group_ids", "fine_stress"}.issubset(labels):
            raise FormalRunError("train/dev label archive omits required fields")
        if "indices" in labels:
            indices = np.asarray(labels["indices"], dtype=np.int64)
        else:
            public_case_row = {
                str(case_id): index for index, case_id in enumerate(public["case_group_ids"])
            }
            try:
                indices = np.asarray(
                    [public_case_row[str(case_id)] for case_id in labels["case_group_ids"]],
                    dtype=np.int64,
                )
            except KeyError as exc:
                raise FormalRunError("train/dev label case is absent from public input") from exc
        if indices.ndim != 1 or len(set(indices.tolist())) != indices.size:
            raise FormalRunError("train/dev label indices must be unique one-dimensional rows")
        expected = np.flatnonzero(np.isin(public["partitions"], np.asarray(["train_id", "dev_id"])))
        if not np.array_equal(indices, expected):
            raise FormalRunError("train/dev labels do not exactly cover train_id and dev_id")
        expected_cases = np.asarray(public["case_group_ids"])[indices]
        if not np.array_equal(labels["case_group_ids"], expected_cases):
            raise FormalRunError("train/dev fine labels are misaligned with public identities")
        split_values = np.where(public["partitions"][indices] == "train_id", "train", "dev")
        batch = LearningBatch(
            base_features=np.asarray(public["base_features"])[indices],
            coarse_stress=np.asarray(public["coarse_stress"])[indices],
            fine_stress=np.asarray(labels["fine_stress"]),
            weights=np.asarray(public["training_weights"])[indices],
            geometry_group_ids=tuple(str(value) for value in public["geometry_group_ids"][indices]),
            section_families=tuple(str(value) for value in public["section_families"][indices]),
            case_group_ids=tuple(str(value) for value in public["case_group_ids"][indices]),
            splits=tuple(str(value) for value in split_values),
        )
        if set(batch.splits) != {"train", "dev"}:
            raise FormalRunError("training batch contains a non-training split")
        return public, batch, manifest

    def _nested_subsets(self, batch: LearningBatch) -> dict[float, tuple[str, ...]]:
        train_rows = np.flatnonzero(np.asarray(batch.splits) == "train")
        subsets = nested_geometry_subsets(
            tuple(batch.geometry_group_ids[index] for index in train_rows),
            tuple(batch.section_families[index] for index in train_rows),
            fractions=tuple(
                float(value) for value in self.config["learning"]["fine_train_fractions"]
            ),
            salt=str(self.config["identity_and_split"]["split_salt"]),
        )
        expected_counts = self.config["learning"]["nested_parent_counts"]
        previous: set[str] = set()
        for fraction, selected in subsets.items():
            expected = expected_counts[str(float(fraction))]
            if len(selected) != int(expected["total"]):
                raise FormalRunError("nested subset count disagrees with the frozen config")
            selected_set = set(selected)
            if not previous.issubset(selected_set):
                raise FormalRunError("fine-label parent subsets are not nested")
            previous = selected_set
        return subsets

    @staticmethod
    def _contract_record(contract: TrainingContract) -> dict[str, Any]:
        selection = contract.selection
        return {
            "schema": "tunnelgeopt.formal_training_contract_record.v1",
            "method": contract.method,
            "config_sha256": contract.config_sha256,
            "contract_sha256": contract.contract_sha256,
            "selection_sha256": selection.selection_sha256,
            "batch_contract_sha256": selection.batch_contract_sha256,
            "fine_fraction": selection.fine_fraction,
            "train_geometry_ids": list(selection.train_geometry_ids),
            "train_case_ids": list(selection.train_case_ids),
            "dev_geometry_ids": list(selection.dev_geometry_ids),
            "dev_case_ids": list(selection.dev_case_ids),
            "eligible_train_geometry_count": selection.eligible_train_geometry_count,
            "selected_train_geometry_count": selection.selected_train_geometry_count,
            "section_geometry_counts": dict(selection.section_geometry_counts),
            "eligible_section_geometry_counts": dict(selection.eligible_section_geometry_counts),
        }

    def _mock_checkpoint(
        self,
        path: Path,
        *,
        contract: TrainingContract,
        seed: int,
        training_fingerprint: str,
    ) -> str:
        payload = {
            "format_version": 2,
            "checkpoint_scope": "tiny-mock-formal-contract-v2",
            "effect_claim_allowed": False,
            "method": contract.method,
            "fine_fraction": contract.selection.fine_fraction,
            "seed": int(seed),
            "config_sha256": contract.config_sha256,
            "selection_sha256": contract.selection.selection_sha256,
            "training_contract_sha256": contract.contract_sha256,
            "train_geometry_ids": list(contract.selection.train_geometry_ids),
            "train_case_ids": list(contract.selection.train_case_ids),
            "training_fingerprint": training_fingerprint,
            "mock_coefficients": [
                int(contract.contract_sha256[index : index + 8], 16) / 0xFFFFFFFF
                for index in (0, 8, 16)
            ],
        }
        return _atomic_json(path, payload)

    def _run_train(self, state: dict[str, Any]) -> dict[str, str]:
        _, batch, dataset_manifest = self._load_training_inputs()
        if any(int(value) != 0 for value in state["sealed_partition_open_counts"].values()):
            raise FormalAbstain("training cannot proceed after any sealed label was opened")
        subsets = self._nested_subsets(batch)
        specifications = _expected_checkpoint_specs(self.config)
        self.paths.checkpoints.mkdir(parents=True, exist_ok=True)
        manifest_path = self.paths.checkpoints / CHECKPOINT_MANIFEST_FILENAME
        if manifest_path.exists():
            checkpoint_manifest = _read_json(manifest_path, "partial checkpoint manifest")
            if (
                checkpoint_manifest.get("config_sha256") != self.config_sha256
                or checkpoint_manifest.get("backend") != self.backend
            ):
                raise FormalAbstain("partial checkpoint manifest belongs to another run")
        else:
            checkpoint_manifest = {
                "schema": "tunnelgeopt.multifidelity.checkpoint_manifest.v1",
                "run_id": self.config["run_id"],
                "config_sha256": self.config_sha256,
                "backend": self.backend,
                "expected_checkpoint_count": 35,
                "checkpoints": {},
                "contracts": {},
            }
        _, public_path, train_labels_path = self._dataset_paths()
        training_fingerprint = _sha256_value(
            {
                "public_sha256": self._manifest_file_digest(dataset_manifest, public_path),
                "train_dev_sha256": self._manifest_file_digest(dataset_manifest, train_labels_path),
            }
        )
        optimization = self.config["learning"]["optimization"]
        model_config = self.config["learning"]["model"]
        for index, specification in enumerate(specifications):
            method = specification["method"]
            fraction = float(specification["fraction"])
            seed = int(specification["seed"])
            key = specification["checkpoint_key"]
            contract = build_training_contract(
                batch,
                method=method,
                config_sha256=self.config_sha256,
                train_geometry_selector=subsets[fraction],
                expected_fine_fraction=fraction,
            )
            record = self._contract_record(contract)
            existing_contract = checkpoint_manifest["contracts"].get(key)
            if existing_contract is not None and existing_contract != record:
                raise FormalAbstain(f"training contract drift on resume: {key}")
            checkpoint_manifest["contracts"][key] = record
            suffix = ".json" if self.tiny_mock else ".pt"
            relative = f"{key}{suffix}"
            checkpoint_path = self.paths.checkpoints / relative
            existing = checkpoint_manifest["checkpoints"].get(key)
            if existing is not None:
                if (
                    existing.get("file") != relative
                    or not checkpoint_path.is_file()
                    or _file_sha256(checkpoint_path) != existing.get("sha256")
                    or existing.get("contract_sha256") != contract.contract_sha256
                ):
                    raise FormalAbstain(f"checkpoint drift on resume: {key}")
                continue
            if self.tiny_mock:
                digest = self._mock_checkpoint(
                    checkpoint_path,
                    contract=contract,
                    seed=seed,
                    training_fingerprint=training_fingerprint,
                )
                best_epoch = 0
                epochs_run = 1
                best_dev_error = 0.0
            else:
                if not self.device.startswith("cuda"):
                    raise FormalRunError("formal training requires an explicit CUDA device")
                model = make_model(model_config, seed=seed, device=self.device)

                def progress(values: Mapping[str, Any], *, checkpoint_key: str = key) -> None:
                    _append_jsonl(
                        self.paths.root / "training_progress.jsonl",
                        {
                            "at_utc": _now(),
                            "checkpoint_key": checkpoint_key,
                            **dict(values),
                        },
                    )

                outcome = train_formal_with_dev_selection(
                    model,
                    batch,
                    contract,
                    seed=seed,
                    device=self.device,
                    learning_rate=float(optimization["learning_rate"]),
                    weight_decay=float(optimization["weight_decay"]),
                    batch_size=int(optimization["case_batch_size"]),
                    max_epochs=int(optimization["max_epochs"]),
                    patience=int(optimization["early_stopping_patience"]),
                    min_delta=float(optimization["early_stopping_min_delta"]),
                    progress=progress,
                )
                digest = save_formal_checkpoint_atomic(
                    outcome,
                    checkpoint_path,
                    contract=contract,
                    seed=seed,
                    model_config=model_config,
                )
                payload = checkpoint_payload(
                    checkpoint_path,
                    expected_config_sha256=self.config_sha256,
                    expected_selection_sha256=contract.selection.selection_sha256,
                    require_formal=True,
                )
                if payload["training_contract_sha256"] != contract.contract_sha256:
                    raise FormalRunError("saved checkpoint contract verification failed")
                best_epoch = int(payload["best_epoch"])
                epochs_run = int(payload["epochs_run"])
                best_dev_error = float(payload["best_dev_error"])
            checkpoint_manifest["checkpoints"][key] = {
                "file": relative,
                "sha256": digest,
                "method": method,
                "fine_fraction": fraction,
                "seed": seed,
                "format_version": 2,
                "contract_sha256": contract.contract_sha256,
                "selection_sha256": contract.selection.selection_sha256,
                "best_epoch": best_epoch,
                "epochs_run": epochs_run,
                "best_dev_error": best_dev_error,
            }
            checkpoint_manifest["completed_checkpoint_count"] = len(
                checkpoint_manifest["checkpoints"]
            )
            _atomic_json(manifest_path, checkpoint_manifest)
            self._event(
                state,
                "checkpoint_frozen",
                checkpoint_key=key,
                checkpoint_index=index,
                checkpoint_sha256=digest,
                contract_sha256=contract.contract_sha256,
            )
        if set(checkpoint_manifest["checkpoints"]) != {
            specification["checkpoint_key"] for specification in specifications
        }:
            raise FormalRunError("checkpoint set does not exactly match the frozen design")
        checkpoint_ids = tuple(
            checkpoint_manifest["checkpoints"][specification["checkpoint_key"]]["sha256"]
            for specification in specifications
        )
        registry = CheckpointRegistry(checkpoint_ids)
        if registry.checkpoint_count != 35:
            raise FormalRunError("checkpoint registry must contain exactly 35 identities")
        registry_payload = {
            **registry.as_dict(),
            "run_id": self.config["run_id"],
            "config_sha256": self.config_sha256,
            "checkpoint_keys": [value["checkpoint_key"] for value in specifications],
            "checkpoint_manifest_sha256": _file_sha256(manifest_path),
            "frozen_at_utc": _now(),
        }
        registry_path = self.paths.checkpoints / REGISTRY_FILENAME
        registry_hash = _atomic_json(registry_path, registry_payload)
        self._event(
            state,
            "checkpoint_registry_frozen",
            checkpoint_count=35,
            registry_hash=registry.registry_hash,
            file_sha256=registry_hash,
        )
        return {
            f"checkpoints/{CHECKPOINT_MANIFEST_FILENAME}": _file_sha256(manifest_path),
            f"checkpoints/{REGISTRY_FILENAME}": registry_hash,
        }

    def _load_frozen_registry(
        self,
    ) -> tuple[dict[str, Any], dict[str, Any], tuple[dict[str, Any], ...]]:
        manifest_path = self.paths.checkpoints / CHECKPOINT_MANIFEST_FILENAME
        registry_path = self.paths.checkpoints / REGISTRY_FILENAME
        manifest = _read_json(manifest_path, "checkpoint manifest")
        registry = _read_json(registry_path, "checkpoint registry")
        specifications = _expected_checkpoint_specs(self.config)
        keys = tuple(specification["checkpoint_key"] for specification in specifications)
        if tuple(registry.get("checkpoint_keys", ())) != keys:
            raise FormalAbstain("checkpoint registry key order changed")
        checkpoint_ids = tuple(manifest["checkpoints"][key]["sha256"] for key in keys)
        rebuilt = CheckpointRegistry(checkpoint_ids)
        if (
            rebuilt.registry_hash != registry.get("registry_hash")
            or registry.get("checkpoint_count") != 35
            or registry.get("config_sha256") != self.config_sha256
            or registry.get("checkpoint_manifest_sha256") != _file_sha256(manifest_path)
        ):
            raise FormalAbstain("checkpoint registry authentication failed")
        for key in keys:
            record = manifest["checkpoints"][key]
            path = self.paths.checkpoints / record["file"]
            if not path.is_file() or _file_sha256(path) != record["sha256"]:
                raise FormalAbstain(f"checkpoint hash mismatch before evaluation: {key}")
        return manifest, registry, specifications

    def _rebuild_contract(
        self,
        batch: LearningBatch,
        subsets: Mapping[float, tuple[str, ...]],
        specification: Mapping[str, Any],
        expected: Mapping[str, Any],
    ) -> TrainingContract:
        fraction = float(specification["fraction"])
        contract = build_training_contract(
            batch,
            method=str(specification["method"]),
            config_sha256=self.config_sha256,
            train_geometry_selector=subsets[fraction],
            expected_fine_fraction=fraction,
        )
        if self._contract_record(contract) != expected:
            raise FormalAbstain(
                "reconstructed training contract does not match checkpoint manifest"
            )
        return contract

    def _mock_prediction(
        self,
        checkpoint_path: Path,
        *,
        public: Mapping[str, np.ndarray],
        indices: np.ndarray,
    ) -> np.ndarray:
        payload = _read_json(checkpoint_path, "tiny-mock checkpoint")
        if (
            payload.get("format_version") != 2
            or payload.get("checkpoint_scope") != "tiny-mock-formal-contract-v2"
            or payload.get("effect_claim_allowed") is not False
        ):
            raise FormalRunError("tiny-mock checkpoint envelope is invalid")
        coarse = np.asarray(public["coarse_stress"])[indices].astype(np.float64)
        base = np.asarray(public["base_features"])[indices].astype(np.float64)
        method = str(payload["method"])
        fraction = float(payload["fine_fraction"])
        seed_scale = (int(payload["seed"]) % 17) * 0.00015
        learned = 0.04 * np.tanh(base[:, :, :3])
        if method == "scratch":
            return coarse + 0.85 * learned + seed_scale
        if method == "direct_coarse":
            return coarse + 0.78 * learned + seed_scale
        if method == "residual_coarse":
            return coarse + (0.92 + 0.06 * fraction) * learned + seed_scale
        if method == "mismatched_coarse":
            return coarse + 0.60 * learned + seed_scale
        raise FormalRunError("unknown method in tiny-mock checkpoint")

    def _formal_prediction(
        self,
        checkpoint_path: Path,
        *,
        contract: TrainingContract,
        batch: LearningBatch,
        partition_index: int,
        seed: int,
    ) -> np.ndarray:
        model = None
        try:
            model, _ = load_formal_model_from_checkpoint(
                checkpoint_path, contract=contract, device=self.device
            )
            mismatch = None
            if contract.method == "mismatched_coarse":
                mismatch = mismatched_coarse_indices(
                    batch.section_families, seed=int(seed) + 100_000 + partition_index
                )
            features, _, reconstruction = method_arrays(
                batch, contract.method, mismatch_indices=mismatch
            )
            raw = predict(
                model,
                features,
                batch_size=int(self.config["learning"]["optimization"]["case_batch_size"]),
                device=self.device,
            )
            return reconstruct_fine_prediction(raw, reconstruction)
        finally:
            if model is not None:
                del model
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:  # pragma: no cover - formal learn extra is required
                pass

    @staticmethod
    def _partition_batch(
        public: Mapping[str, np.ndarray], indices: np.ndarray, fine: np.ndarray
    ) -> LearningBatch:
        return LearningBatch(
            base_features=np.asarray(public["base_features"])[indices],
            coarse_stress=np.asarray(public["coarse_stress"])[indices],
            fine_stress=fine,
            weights=np.asarray(public["metric_weights"])[indices],
            geometry_group_ids=tuple(str(value) for value in public["geometry_group_ids"][indices]),
            section_families=tuple(str(value) for value in public["section_families"][indices]),
            case_group_ids=tuple(str(value) for value in public["case_group_ids"][indices]),
            splits=tuple("dev" for _ in indices),
        )

    def _run_evaluate(self, state: dict[str, Any]) -> dict[str, str]:
        if not self.tiny_mock and not self.device.startswith("cuda"):
            raise FormalRunError("formal sealed evaluation requires an explicit CUDA device")
        manifest, registry, specifications = self._load_frozen_registry()
        public, training_batch, dataset_manifest = self._load_training_inputs()
        subsets = self._nested_subsets(training_batch)
        if any(int(value) != 0 for value in state["sealed_partition_open_counts"].values()):
            raise FormalAbstain("a sealed label had already been opened before evaluation")
        if int(state["denied_premature_sealed_accesses"]) != 0:
            raise FormalAbstain("a premature sealed access attempt invalidated the formal run")
        if registry["checkpoint_count"] != 35:
            raise FormalAbstain("evaluation authorization lacks all 35 checkpoints")
        self.paths.evaluation.mkdir(parents=True, exist_ok=True)
        metric_payload: dict[str, Any] = {
            "schema": "tunnelgeopt.multifidelity.sealed_metrics.v1",
            "run_id": self.config["run_id"],
            "config_sha256": self.config_sha256,
            "backend": self.backend,
            "effect_claim_allowed": not self.tiny_mock,
            "registry_hash": registry["registry_hash"],
            "evaluated_at_utc": _now(),
            "partitions": {},
        }
        tracemalloc.start()
        for partition_index, partition in enumerate(LOCKED_PARTITIONS):
            sealed = self.open_sealed_partition(partition)
            # The sealed reader persists its open count independently.  Merge
            # that durable audit state before adding per-checkpoint counts.
            state = self._load_state()
            if not {"case_group_ids", "fine_stress"}.issubset(sealed):
                raise FormalAbstain(f"sealed fields are incomplete for {partition}")
            if "ultrafine_audit_passed" in sealed and not bool(
                np.asarray(sealed["ultrafine_audit_passed"]).all()
            ):
                raise FormalAbstain(f"fine-ultrafine locked audit failed for {partition}")
            expected_indices = np.flatnonzero(public["partitions"] == partition)
            if "indices" in sealed:
                indices = np.asarray(sealed["indices"], dtype=np.int64)
            else:
                public_case_row = {
                    str(case_id): index for index, case_id in enumerate(public["case_group_ids"])
                }
                try:
                    indices = np.asarray(
                        [public_case_row[str(case_id)] for case_id in sealed["case_group_ids"]],
                        dtype=np.int64,
                    )
                except KeyError as exc:
                    raise FormalAbstain(
                        f"sealed {partition} case is absent from public input"
                    ) from exc
            if not np.array_equal(indices, expected_indices):
                raise FormalAbstain(f"sealed indices do not match public {partition} rows")
            if not np.array_equal(sealed["case_group_ids"], public["case_group_ids"][indices]):
                raise FormalAbstain(f"sealed case identities do not align for {partition}")
            sealed_path = self._trusted_sealed_path(partition)
            sealed_hash = self._manifest_file_digest(dataset_manifest, sealed_path)
            access_events = _read_access_events(self.paths.access_log)
            opened = [
                event
                for event in access_events
                if event.get("event") == "sealed_partition_opened"
                and event.get("partition") == partition
            ]
            if len(opened) != 1 or opened[0].get("sha256") != sealed_hash:
                raise FormalAbstain(f"sealed digest authentication failed for {partition}")
            fine = np.asarray(sealed["fine_stress"], dtype=np.float64)
            if not np.isfinite(fine).all():
                raise FormalAbstain(f"sealed fine labels are non-finite for {partition}")
            partition_batch = self._partition_batch(public, indices, fine)
            coarse_errors = case_weighted_stress_error(
                partition_batch.coarse_stress, fine, partition_batch.weights
            )
            arc_weights = np.asarray(public["arc_weights"])[indices]
            wall_normals = np.asarray(public["wall_rock_outward_normals_yz"])[indices]
            stress_scales = np.asarray(public["stress_scales"], dtype=np.float64)[indices]
            coarse_d_t, coarse_d_r = _wall_offset_discrepancy(
                partition_batch.coarse_stress,
                fine,
                arc_weights,
                wall_normals,
                stress_scales,
            )
            load_subtypes = [str(value) for value in public["load_subtypes"][indices]]
            partition_metrics: dict[str, Any] = {
                "case_group_ids": list(partition_batch.case_group_ids),
                "geometry_group_ids": list(partition_batch.geometry_group_ids),
                "section_families": list(partition_batch.section_families),
                "load_subtypes": load_subtypes,
                "coarse_only_case_errors": coarse_errors.tolist(),
                "coarse_only_summary": _distribution_summary(coarse_errors),
                "coarse_wall_offset": {
                    "traction_case_values": coarse_d_t.tolist(),
                    "resultant_case_values": coarse_d_r.tolist(),
                    "traction_summary": _distribution_summary(coarse_d_t),
                    "resultant_summary": _distribution_summary(coarse_d_r),
                },
                "fine_oracle_case_errors": np.zeros(indices.size, dtype=np.float64).tolist(),
                "checkpoints": {},
                "sealed_file_open_count": 1,
            }
            for specification in specifications:
                key = specification["checkpoint_key"]
                record = manifest["checkpoints"][key]
                contract = self._rebuild_contract(
                    training_batch, subsets, specification, manifest["contracts"][key]
                )
                checkpoint_path = self.paths.checkpoints / record["file"]
                start = time.perf_counter()
                try:
                    import torch

                    if not self.tiny_mock and torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats(self.device)
                except ImportError:
                    torch = None  # type: ignore[assignment]
                if self.tiny_mock:
                    prediction = self._mock_prediction(
                        checkpoint_path, public=public, indices=indices
                    )
                else:
                    prediction = self._formal_prediction(
                        checkpoint_path,
                        contract=contract,
                        batch=partition_batch,
                        partition_index=partition_index,
                        seed=int(specification["seed"]),
                    )
                nonfinite_count = int(np.size(prediction) - np.isfinite(prediction).sum())
                if nonfinite_count:
                    raise FormalAbstain(f"non-finite prediction for {key} on {partition}")
                errors = case_weighted_stress_error(prediction, fine, partition_batch.weights)
                d_t, d_r = _wall_offset_discrepancy(
                    prediction,
                    fine,
                    arc_weights,
                    wall_normals,
                    stress_scales,
                )
                runtime_seconds = time.perf_counter() - start
                _, python_peak = tracemalloc.get_traced_memory()
                gpu_peak = 0
                if not self.tiny_mock and torch is not None and torch.cuda.is_available():
                    gpu_peak = int(torch.cuda.max_memory_allocated(self.device))
                section_summaries = {
                    section: _distribution_summary(
                        errors[np.asarray(partition_batch.section_families) == section]
                    )
                    for section in SECTION_NAMES
                }
                subtype_summaries = {
                    subtype: _distribution_summary(
                        errors[np.asarray(load_subtypes, dtype=object) == subtype]
                    )
                    for subtype in sorted(set(load_subtypes))
                }
                count_key = f"{partition}:{key}"
                new_count = int(state["checkpoint_evaluation_counts"].get(count_key, 0)) + 1
                if new_count != 1:
                    raise FormalAbstain(f"checkpoint evaluated more than once: {count_key}")
                state["checkpoint_evaluation_counts"][count_key] = new_count
                partition_metrics["checkpoints"][key] = {
                    "method": specification["method"],
                    "fine_fraction": float(specification["fraction"]),
                    "seed": int(specification["seed"]),
                    "checkpoint_sha256": record["sha256"],
                    "case_errors": errors.tolist(),
                    "summary": _distribution_summary(errors),
                    "section_summaries": section_summaries,
                    "load_subtype_summaries": subtype_summaries,
                    "wall_offset": {
                        "traction_case_values": d_t.tolist(),
                        "resultant_case_values": d_r.tolist(),
                        "traction_summary": _distribution_summary(d_t),
                        "resultant_summary": _distribution_summary(d_r),
                    },
                    "nonfinite_prediction_count": nonfinite_count,
                    "runtime_seconds": runtime_seconds,
                    "python_peak_memory_bytes": int(python_peak),
                    "gpu_peak_memory_bytes": gpu_peak,
                }
            if len(partition_metrics["checkpoints"]) != 35:
                raise FormalAbstain(f"not every checkpoint was evaluated on {partition}")
            metric_payload["partitions"][partition] = partition_metrics
            self._event(
                state,
                "sealed_partition_evaluated",
                partition=partition,
                checkpoint_count=35,
            )
        expected_counts = {
            f"{partition}:{specification['checkpoint_key']}"
            for partition in LOCKED_PARTITIONS
            for specification in specifications
        }
        state = self._load_state()
        if (
            set(state["checkpoint_evaluation_counts"]) != expected_counts
            or set(state["checkpoint_evaluation_counts"].values()) != {1}
            or set(state["sealed_partition_open_counts"].values()) != {1}
        ):
            raise FormalAbstain("sealed evaluation count contract failed")
        metric_payload["access_contract"] = {
            "sealed_partition_open_counts": dict(state["sealed_partition_open_counts"]),
            "checkpoint_evaluation_counts": dict(state["checkpoint_evaluation_counts"]),
            "denied_premature_sealed_accesses": state["denied_premature_sealed_accesses"],
            "passed": True,
        }
        _, python_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        metric_payload["runtime_and_peak_memory"] = {
            "python_peak_memory_bytes": int(python_peak),
            "checkpoint_runtime_seconds": {
                f"{partition}:{key}": value["runtime_seconds"]
                for partition, partition_value in metric_payload["partitions"].items()
                for key, value in partition_value["checkpoints"].items()
            },
        }
        metrics_path = self.paths.evaluation / "sealed_metrics.json"
        digest = _atomic_json(metrics_path, metric_payload)
        return {"evaluation/sealed_metrics.json": digest}

    def _method_seed_errors(
        self,
        partition: Mapping[str, Any],
        *,
        method: str,
        fraction: float,
    ) -> tuple[np.ndarray, tuple[int, ...]]:
        seeds = tuple(int(value) for value in self.config["learning"]["training_seeds"])
        rows = []
        for seed in seeds:
            key = _checkpoint_key(method, fraction, seed)
            record = partition["checkpoints"].get(key)
            if record is None:
                raise FormalAbstain(f"missing checkpoint metrics: {key}")
            rows.append(np.asarray(record["case_errors"], dtype=np.float64))
        return np.stack(rows), seeds

    def _bootstrap_comparison(
        self,
        candidate_case: np.ndarray,
        reference_case: np.ndarray,
        geometry_ids: Sequence[str],
        sections: Sequence[str],
        seeds: Sequence[int],
        *,
        bootstrap_seed: int,
    ) -> dict[str, float]:
        candidate_parent, parent_ids, parent_sections = aggregate_case_errors_by_parent(
            candidate_case, geometry_ids, sections
        )
        reference_parent, reference_ids, reference_sections = aggregate_case_errors_by_parent(
            reference_case, geometry_ids, sections
        )
        if parent_ids != reference_ids or parent_sections != reference_sections:
            raise FormalAbstain("paired bootstrap parent identities do not align")
        bootstrap = self.config["evaluation"]["bootstrap"]
        return hierarchical_paired_bootstrap(
            candidate_parent,
            reference_parent,
            seeds,
            parent_ids,
            parent_sections,
            replicates=int(bootstrap["replicates"]),
            confidence=float(bootstrap["one_sided_upper_confidence_level"]),
            bootstrap_seed=int(bootstrap_seed),
        )

    @staticmethod
    def _section_balanced_seed_values(
        case_values: np.ndarray,
        geometry_ids: Sequence[str],
        sections: Sequence[str],
    ) -> np.ndarray:
        parent_values, _, parent_sections = aggregate_case_errors_by_parent(
            case_values, geometry_ids, sections
        )
        if parent_values.ndim == 1:
            parent_values = parent_values[None, :]
        section_array = np.asarray(parent_sections, dtype=object)
        present_sections = tuple(sorted({str(value) for value in parent_sections}))
        return np.mean(
            np.stack(
                [
                    parent_values[:, section_array == section].mean(axis=1)
                    for section in present_sections
                ],
                axis=1,
            ),
            axis=1,
        )

    @classmethod
    def _ratio_by_seed(
        cls,
        candidate: np.ndarray,
        reference: np.ndarray,
        geometry_ids: Sequence[str],
        sections: Sequence[str],
    ) -> list[float]:
        if candidate.shape != reference.shape or candidate.ndim != 2:
            raise FormalRunError("seed ratio arrays must align [seed, case]")
        candidate_values = cls._section_balanced_seed_values(candidate, geometry_ids, sections)
        reference_values = cls._section_balanced_seed_values(reference, geometry_ids, sections)
        denominator = reference_values
        if np.any(denominator <= 0.0):
            raise FormalAbstain("seed reference error is zero")
        return (candidate_values / denominator).tolist()

    @classmethod
    def _point_ratio(
        cls,
        candidate: np.ndarray,
        reference: np.ndarray,
        geometry_ids: Sequence[str],
        sections: Sequence[str],
    ) -> float:
        candidate_value = float(
            cls._section_balanced_seed_values(candidate, geometry_ids, sections).mean()
        )
        reference_value = float(
            cls._section_balanced_seed_values(reference, geometry_ids, sections).mean()
        )
        if reference_value <= 0.0:
            raise FormalAbstain("point-ratio reference error is zero")
        return candidate_value / reference_value

    def _dataset_validity_failures(self, manifest: Mapping[str, Any]) -> list[str]:
        """Map generator/QC evidence to preregistered ABSTAIN conditions."""

        failures: list[str] = []
        identity = manifest.get("identity_checks")
        if (
            not isinstance(identity, Mapping)
            or not identity
            or not all(value is True for value in identity.values())
        ):
            failures.append("dataset identity/leakage checks are missing or failed")
        qc = manifest.get("quality_control", manifest.get("audit_summary"))
        if not isinstance(qc, Mapping):
            return [*failures, "dataset solver/mesh/fine-ultrafine QC is missing"]
        if int(qc.get("nonfinite_count", -1)) != 0:
            failures.append("dataset QC reports non-finite values")
        minimum = float(
            self.config["quality_control"]["solver_and_mesh"][
                "minimum_valid_case_fraction_per_partition_section"
            ]
        )
        fractions = qc.get(
            "valid_case_fraction_by_partition_section",
            qc.get("partition_section_valid_fractions"),
        )
        if isinstance(fractions, Mapping):
            flat: list[float] = []
            for value in fractions.values():
                if isinstance(value, Mapping):
                    flat.extend(float(item) for item in value.values())
                else:
                    flat.append(float(value))
            if not flat or min(flat) < minimum:
                failures.append("solver/mesh valid case fraction is below the frozen 95% gate")
        elif float(qc.get("minimum_valid_case_fraction_per_partition_section", -1.0)) < minimum:
            failures.append("solver/mesh valid case fraction evidence is missing or below gate")
        fine_ultrafine = qc.get("fine_ultrafine", qc.get("fine_ultrafine_audit"))
        if isinstance(fine_ultrafine, Mapping):
            frozen = self.config["quality_control"]["fine_ultrafine"]
            median = float(fine_ultrafine.get("overall_median", np.inf))
            p95 = float(fine_ultrafine.get("overall_p95", np.inf))
            section_medians = fine_ultrafine.get(
                "section_medians", fine_ultrafine.get("median_by_section", {})
            )
            if median > float(frozen["max_overall_median"]):
                failures.append("formal fine-ultrafine overall median exceeds gate")
            if p95 > float(frozen["max_overall_p95"]):
                failures.append("formal fine-ultrafine overall p95 exceeds gate")
            if (
                not isinstance(section_medians, Mapping)
                or set(section_medians) != set(SECTION_NAMES)
                or max(float(value) for value in section_medians.values())
                > float(frozen["max_any_section_median"])
            ):
                failures.append("formal fine-ultrafine per-section median evidence fails")
        elif qc.get("fine_ultrafine_independent_gate_passed") is not True:
            failures.append("formal fine-ultrafine audit evidence is missing or failed")
        return failures

    def _wall_gate_result(
        self,
        partition_name: str,
        partition: Mapping[str, Any],
        geometry_ids: Sequence[str],
        sections: Sequence[str],
    ) -> dict[str, Any]:
        residual_records = [
            partition["checkpoints"][_checkpoint_key("residual_coarse", 0.5, int(seed))]
            for seed in self.config["learning"]["training_seeds"]
        ]
        candidate_d_t = np.stack(
            [
                np.asarray(record["wall_offset"]["traction_case_values"])
                for record in residual_records
            ]
        )
        candidate_d_r = np.stack(
            [
                np.asarray(record["wall_offset"]["resultant_case_values"])
                for record in residual_records
            ]
        )
        coarse_d_t = np.tile(
            np.asarray(partition["coarse_wall_offset"]["traction_case_values"]),
            (candidate_d_t.shape[0], 1),
        )
        coarse_d_r = np.tile(
            np.asarray(partition["coarse_wall_offset"]["resultant_case_values"]),
            (candidate_d_r.shape[0], 1),
        )
        values = {
            "candidate_D_t": float(
                self._section_balanced_seed_values(candidate_d_t, geometry_ids, sections).mean()
            ),
            "candidate_D_r": float(
                self._section_balanced_seed_values(candidate_d_r, geometry_ids, sections).mean()
            ),
            "coarse_D_t": float(
                self._section_balanced_seed_values(coarse_d_t, geometry_ids, sections).mean()
            ),
            "coarse_D_r": float(
                self._section_balanced_seed_values(coarse_d_r, geometry_ids, sections).mean()
            ),
        }
        physics = self.config["evaluation"]["wall_offset_physics"]
        cap_key = (
            "locked_joint_ood_report_only"
            if partition_name == "locked_joint_ood"
            else partition_name
        )
        caps = physics["absolute_caps"][cap_key]
        nonworsening = physics["coarse_nonworsening"]
        checks = {
            "absolute_traction": values["candidate_D_t"] <= float(caps["max_traction_discrepancy"]),
            "absolute_resultant": values["candidate_D_r"]
            <= float(caps["max_resultant_discrepancy"]),
            "coarse_nonworsening_traction": values["candidate_D_t"]
            <= float(nonworsening["max_multiplier"]) * values["coarse_D_t"]
            + float(nonworsening["traction_additive_margin_over_S_inf"]),
            "coarse_nonworsening_resultant": values["candidate_D_r"]
            <= float(nonworsening["max_multiplier"]) * values["coarse_D_r"]
            + float(nonworsening["resultant_additive_margin_over_S_inf"]),
        }
        return {"values": values, "checks": checks, "passed": all(checks.values())}

    def _run_analyze(self, state: dict[str, Any]) -> dict[str, str]:
        if self.tiny_mock:
            decision = {
                "schema": "tunnelgeopt.multifidelity.formal_decision.v1",
                "run_id": self.config["run_id"],
                "classification": "ABSTAIN",
                "effect_claim_allowed": False,
                "reasons": ["tiny-mock backend is state-machine QA, not scientific evidence"],
                "analyzed_at_utc": _now(),
            }
            self.paths.analysis.mkdir(parents=True, exist_ok=True)
            path = self.paths.analysis / "decision.json"
            return {"analysis/decision.json": _atomic_json(path, decision)}
        metrics_path = self.paths.evaluation / "sealed_metrics.json"
        metrics = _read_json(metrics_path, "sealed metrics")
        if metrics.get("config_sha256") != self.config_sha256:
            raise FormalAbstain("sealed metrics belong to another config")
        gates = self.config["scientific_decision"]
        bootstrap_config = self.config["evaluation"]["bootstrap"]
        results: dict[str, Any] = {}
        dataset_manifest_path, _, _ = self._dataset_paths()
        dataset_manifest = _read_json(dataset_manifest_path, "formal dataset manifest")
        validity_failures: list[str] = self._dataset_validity_failures(dataset_manifest)
        effect_failures: list[str] = []
        all_seeds = tuple(int(value) for value in self.config["learning"]["training_seeds"])
        for partition_index, partition_name in enumerate(LOCKED_PARTITIONS):
            partition = metrics["partitions"][partition_name]
            geometry_ids = tuple(partition["geometry_group_ids"])
            sections = tuple(partition["section_families"])
            residual, seeds = self._method_seed_errors(
                partition, method="residual_coarse", fraction=0.5
            )
            scratch, _ = self._method_seed_errors(partition, method="scratch", fraction=1.0)
            direct, _ = self._method_seed_errors(partition, method="direct_coarse", fraction=1.0)
            coarse = np.tile(
                np.asarray(partition["coarse_only_case_errors"], dtype=np.float64),
                (len(seeds), 1),
            )
            comparisons = {
                "R_s": self._bootstrap_comparison(
                    residual,
                    scratch,
                    geometry_ids,
                    sections,
                    seeds,
                    bootstrap_seed=7319 + partition_index * 100,
                ),
                "R_d": self._bootstrap_comparison(
                    residual,
                    direct,
                    geometry_ids,
                    sections,
                    seeds,
                    bootstrap_seed=7321 + partition_index * 100,
                ),
                "R_c": self._bootstrap_comparison(
                    residual,
                    coarse,
                    geometry_ids,
                    sections,
                    seeds,
                    bootstrap_seed=7327 + partition_index * 100,
                ),
            }
            seed_ratios = {
                "R_s": self._ratio_by_seed(residual, scratch, geometry_ids, sections),
                "R_d": self._ratio_by_seed(residual, direct, geometry_ids, sections),
                "R_c": self._ratio_by_seed(residual, coarse, geometry_ids, sections),
            }
            section_ratios: dict[str, dict[str, float]] = {}
            section_array = np.asarray(sections)
            for section in SECTION_NAMES:
                mask = section_array == section
                section_ratios[section] = {
                    "R_s": self._point_ratio(
                        residual[:, mask],
                        scratch[:, mask],
                        np.asarray(geometry_ids)[mask].tolist(),
                        np.asarray(sections)[mask].tolist(),
                    ),
                    "R_d": self._point_ratio(
                        residual[:, mask],
                        direct[:, mask],
                        np.asarray(geometry_ids)[mask].tolist(),
                        np.asarray(sections)[mask].tolist(),
                    ),
                    "R_c": self._point_ratio(
                        residual[:, mask],
                        coarse[:, mask],
                        np.asarray(geometry_ids)[mask].tolist(),
                        np.asarray(sections)[mask].tolist(),
                    ),
                }
            load_subtype_ratios: dict[str, dict[str, float]] = {}
            if partition_name in {"locked_load_ood", "locked_joint_ood"}:
                subtypes = np.asarray(partition["load_subtypes"], dtype=object)
                for subtype in sorted({str(value) for value in subtypes}):
                    mask = subtypes == subtype
                    subtype_geometry = np.asarray(geometry_ids)[mask].tolist()
                    subtype_sections = np.asarray(sections)[mask].tolist()
                    load_subtype_ratios[subtype] = {
                        "R_s": self._point_ratio(
                            residual[:, mask],
                            scratch[:, mask],
                            subtype_geometry,
                            subtype_sections,
                        ),
                        "R_d": self._point_ratio(
                            residual[:, mask],
                            direct[:, mask],
                            subtype_geometry,
                            subtype_sections,
                        ),
                    }
            wall_gate = self._wall_gate_result(partition_name, partition, geometry_ids, sections)
            results[partition_name] = {
                "comparisons": comparisons,
                "seed_ratios": seed_ratios,
                "section_ratios": section_ratios,
                "load_subtype_ratios": load_subtype_ratios,
                "wall_offset_physics": wall_gate,
            }
            for ratio, comparison in comparisons.items():
                width = float(comparison["upper"] - comparison["lower"])
                if partition_name != "locked_joint_ood" and width > float(
                    bootstrap_config["max_primary_ratio_interval_total_width"]
                ):
                    validity_failures.append(
                        f"{partition_name}:{ratio}:bootstrap_interval_width={width:.6g}"
                    )
            if partition_name in gates["upper_95_ci_gates"]:
                for ratio, threshold in gates["upper_95_ci_gates"][partition_name].items():
                    if comparisons[ratio]["one_sided_upper"] > float(threshold):
                        effect_failures.append(
                            f"{partition_name}:{ratio}:upper95={comparisons[ratio]['one_sided_upper']:.6g}"
                        )
            if partition_name != "locked_joint_ood" and not wall_gate["passed"]:
                effect_failures.append(f"{partition_name}:wall_offset_physics_gate_failed")
            if partition_name in {"locked_load_ood", "locked_joint_ood"}:
                subtype_gate = float(
                    gates["section_robustness"][
                        "ood_subtype_max_point_ratio_to_each_full_label_baseline"
                    ]
                )
                for subtype, ratios in load_subtype_ratios.items():
                    if partition_name != "locked_joint_ood" and any(
                        value > subtype_gate for value in ratios.values()
                    ):
                        effect_failures.append(
                            f"{partition_name}:{subtype}:full_label_ratio_exceeds_{subtype_gate}"
                        )
        stability = gates["seed_stability"]
        seed_stability: dict[str, Any] = {}
        for partition_name, threshold_key in (
            ("locked_iid", "iid_max_R_s_and_R_d"),
            ("locked_geometry_ood", "geometry_ood_max_R_s_and_R_d"),
            ("locked_load_ood", "load_ood_max_R_s_and_R_d"),
        ):
            threshold = float(stability[threshold_key])
            ratios = results[partition_name]["seed_ratios"]
            passed = [
                float(ratios["R_s"][index]) <= threshold
                and float(ratios["R_d"][index]) <= threshold
                for index in range(len(all_seeds))
            ]
            seed_stability[partition_name] = {
                "threshold": threshold,
                "passing_seed_count": sum(passed),
                "passed_by_seed": dict(zip(map(str, all_seeds), passed, strict=True)),
            }
            if sum(passed) < int(stability["minimum_passing_seeds"]):
                effect_failures.append(f"{partition_name}:seed_stability_failed")
        robustness = gates["section_robustness"]
        iid_sections = results["locked_iid"]["section_ratios"]
        section_worst = {
            section: max(float(value["R_s"]), float(value["R_d"]))
            for section, value in iid_sections.items()
        }
        iid_section_checks = {
            "max_any_section": max(section_worst.values())
            <= float(robustness["iid_max_any_section"]),
            "strict_section_count": sum(
                value <= float(robustness["iid_max_for_at_least_two_sections"])
                for value in section_worst.values()
            ),
        }
        iid_section_checks["passed"] = iid_section_checks["max_any_section"] and int(
            iid_section_checks["strict_section_count"]
        ) >= int(robustness["minimum_iid_sections_at_strict_gate"])
        if not iid_section_checks["passed"]:
            effect_failures.append("locked_iid:section_robustness_failed")
        if len(all_seeds) != 5:
            validity_failures.append("formal training seed count is not five")
        access = metrics.get("access_contract", {})
        if (
            access.get("passed") is not True
            or set(access.get("sealed_partition_open_counts", {}).values()) != {1}
            or set(access.get("checkpoint_evaluation_counts", {}).values()) != {1}
        ):
            validity_failures.append("sealed evaluation access/count contract failed")
        if state.get("abstain_reasons"):
            validity_failures.extend(str(reason) for reason in state["abstain_reasons"])
        classification = "ABSTAIN" if validity_failures else ("NO_GO" if effect_failures else "GO")
        decision = {
            "schema": "tunnelgeopt.multifidelity.formal_decision.v1",
            "run_id": self.config["run_id"],
            "config_sha256": self.config_sha256,
            "classification": classification,
            "effect_claim_allowed": classification == "GO",
            "validity_failures": validity_failures,
            "effect_failures": effect_failures,
            "results": results,
            "seed_stability": seed_stability,
            "iid_section_robustness": {
                **iid_section_checks,
                "worst_full_label_ratio_by_section": section_worst,
            },
            "dataset_validity": {
                "manifest_sha256": _file_sha256(dataset_manifest_path),
                "passed": not self._dataset_validity_failures(dataset_manifest),
            },
            "metrics_sha256": _file_sha256(metrics_path),
            "access_log_sha256": _file_sha256(self.paths.access_log),
            "analyzed_at_utc": _now(),
        }
        self.paths.analysis.mkdir(parents=True, exist_ok=True)
        decision_path = self.paths.analysis / "decision.json"
        return {"analysis/decision.json": _atomic_json(decision_path, decision)}


def _read_access_events(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise FormalRunError(f"could not read access audit: {exc}") from exc
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise FormalAbstain(f"access audit line {line_number} is corrupt") from exc
        if not isinstance(event, dict):
            raise FormalAbstain("access audit event is not an object")
        events.append(event)
    return events


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=PHASES)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--backend", choices=("formal", "tiny-mock"), default="formal")
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        runner = FormalExperimentRunner(
            config_path=arguments.config,
            approval_path=arguments.approval,
            output_dir=arguments.output,
            backend=arguments.backend,
            device=arguments.device,
        )
        result = runner.run_phase(arguments.phase)
    except FormalAbstain as exc:
        print(json.dumps({"classification": "ABSTAIN", "error": str(exc)}), file=sys.stderr)
        return 3
    except (FormalRunError, LearningContractError, ValueError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
