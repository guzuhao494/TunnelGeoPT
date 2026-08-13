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
import re
import subprocess
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
    build_training_contract,
    case_weighted_stress_error,
    checkpoint_payload,
    load_formal_model_from_checkpoint,
    method_arrays,
    mismatched_coarse_indices,
    nested_geometry_subsets,
    predict,
    reconstruct_fine_prediction,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "multifidelity_formal.json"
DEFAULT_EXCLUSIONS = ROOT / "configs" / "multifidelity_seen_identity_exclusions.json"
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
IMPLEMENTATION_MANIFEST_FILENAME = "implementation_manifest.json"
TRAINING_WORKER_CONTRACT_FILENAME = "training_worker_contract.json"
TRAINING_WORKER_AUDIT_FILENAME = "training_worker_audit.json"
TRAINING_WORKER_PROGRESS_FILENAME = "training_progress.jsonl"
TRAINING_WORKER_SCHEMA = "tunnelgeopt.formal_training_worker_contract.v1"
TRAINING_WORKER_ALLOWED_KEYS = frozenset(
    {
        "schema",
        "run_id",
        "config_sha256",
        "backend",
        "device",
        "public_inputs",
        "train_dev_labels",
        "checkpoint_output_dir",
        "split_salt",
        "fine_train_fractions",
        "nested_parent_counts",
        "model",
        "optimization",
        "checkpoint_specifications",
    }
)
TRAINING_WORKER_ALLOWED_IMPORTS = frozenset(
    {
        "tunnelgeopt",
        "tunnelgeopt.cases",
        "tunnelgeopt.elastic_schema",
        "tunnelgeopt.elastic_validation",
        "tunnelgeopt.elasticity",
        "tunnelgeopt.geometry",
        "tunnelgeopt.kirsch",
        "tunnelgeopt.lift",
        "tunnelgeopt.mesh",
        "tunnelgeopt.multifidelity_learning",
        "tunnelgeopt.schema",
    }
)
IMPLEMENTATION_SOURCE_PATHS = (
    "scripts/run_multifidelity_formal.py",
    "scripts/run_multifidelity_train_worker.py",
    "src/tunnelgeopt/formal_generation.py",
    "src/tunnelgeopt/formal_analysis.py",
    "src/tunnelgeopt/__init__.py",
    "src/tunnelgeopt/cases.py",
    "src/tunnelgeopt/elastic_schema.py",
    "src/tunnelgeopt/elastic_validation.py",
    "src/tunnelgeopt/multifidelity.py",
    "src/tunnelgeopt/multifidelity_learning.py",
    "src/tunnelgeopt/geometry.py",
    "src/tunnelgeopt/mesh.py",
    "src/tunnelgeopt/elasticity.py",
    "src/tunnelgeopt/field_sampling.py",
    "src/tunnelgeopt/kirsch.py",
    "src/tunnelgeopt/lift.py",
    "src/tunnelgeopt/schema.py",
    "configs/multifidelity_formal.json",
    "configs/multifidelity_formal_approval.json",
    "configs/multifidelity_seen_identity_exclusions.json",
)
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


def _run_readonly_command(arguments: Sequence[str], *, description: str) -> str:
    completed = subprocess.run(
        list(arguments),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise FormalRunError(f"could not inspect {description}: {detail}")
    return completed.stdout.strip()


def _git_output(*arguments: str, description: str) -> str:
    return _run_readonly_command(("git", *arguments), description=description)


def _sanitize_remote_url(value: str) -> str:
    remote = value.strip()
    # Strip userinfo from HTTP(S) remotes and user names from SCP-style SSH.
    remote = re.sub(r"^(https?://)[^/@]+@", r"\1", remote, flags=re.IGNORECASE)
    remote = re.sub(r"^(ssh://)[^/@]+@", r"\1", remote, flags=re.IGNORECASE)
    remote = re.sub(r"^[^/@:]+@([^:]+):", r"ssh://\1/", remote)
    return remote


def _module_version(name: str) -> str | None:
    try:
        module = __import__(name)
    except ImportError:
        return None
    value = getattr(module, "__version__", None)
    return None if value is None else str(value)


def _cuda_environment() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "torch": None,
        "cuda_runtime": None,
        "cuda_available": False,
        "device_name": None,
        "device_total_memory_bytes": None,
        "driver_version": None,
    }
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None:
        payload["torch"] = str(torch.__version__)
        payload["cuda_runtime"] = None if torch.version.cuda is None else str(torch.version.cuda)
        payload["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            payload["device_name"] = str(properties.name)
            payload["device_total_memory_bytes"] = int(properties.total_memory)
    try:
        driver = _run_readonly_command(
            ("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"),
            description="NVIDIA driver version",
        )
    except FormalRunError:
        driver = ""
    if driver:
        payload["driver_version"] = driver.splitlines()[0].strip()
    return payload


def _environment_manifest(device_requested: str) -> dict[str, Any]:
    cuda = _cuda_environment()
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": _module_version("scipy"),
        "skfem": _module_version("skfem"),
        "gmsh": _module_version("gmsh"),
        "torch": cuda["torch"],
        "cuda_runtime": cuda["cuda_runtime"],
        "cuda_available": cuda["cuda_available"],
        "device_requested": str(device_requested),
        "device_name": cuda["device_name"],
        "device_total_memory_bytes": cuda["device_total_memory_bytes"],
        "driver_version": cuda["driver_version"],
    }


def _require_sha256(value: Any, name: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise FormalRunError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _require_git_commit(value: Any, name: str) -> str:
    commit = str(value)
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise FormalRunError(f"{name} is not a lowercase 40-hex Git commit")
    return commit


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
        exclusions_path: Path | None = None,
        backend: str = "formal",
        device: str = "cuda",
    ) -> None:
        if backend not in {"formal", "tiny-mock"}:
            raise ValueError("backend must be 'formal' or 'tiny-mock'")
        self.config_path = Path(config_path).resolve()
        self.approval_path = Path(approval_path).resolve()
        self.exclusions_path = None if exclusions_path is None else Path(exclusions_path).resolve()
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
        self.forbidden_identities = self._load_identity_exclusions()
        self._prepared_implementation: dict[str, Any] | None = None

    def _source_hashes(self) -> dict[str, str]:
        hashes: dict[str, str] = {}
        for relative in IMPLEMENTATION_SOURCE_PATHS:
            path = ROOT / relative
            if not path.is_file():
                raise FormalRunError(f"critical implementation source is missing: {relative}")
            hashes[relative] = _file_sha256(path)
        return hashes

    def _implementation_manifest(self) -> dict[str, Any]:
        if self.tiny_mock:
            source_hashes = self._source_hashes()
            return {
                "schema": "tunnelgeopt.formal_implementation_manifest.v1",
                "run_id": self.config["run_id"],
                "config_sha256": self.config_sha256,
                "effect_claim_allowed": False,
                "recorded_at_utc": _now(),
                "source_provenance": {
                    "git_head": "0" * 40,
                    "upstream_ref": "tiny-mock/no-upstream-required",
                    "upstream_head": "0" * 40,
                    "head_matches_upstream": False,
                    "worktree_clean_before_prepare": False,
                    "remote_url_sanitized": "tiny-mock://not-applicable",
                    "all_sources_tracked": False,
                    "source_sha256": source_hashes,
                },
                "environment": _environment_manifest(self.device),
            }
        git_head = _git_output("rev-parse", "HEAD", description="git HEAD")
        upstream_ref = _git_output(
            "rev-parse",
            "--abbrev-ref",
            "--symbolic-full-name",
            "@{upstream}",
            description="git upstream ref",
        )
        upstream_head = _git_output("rev-parse", "@{upstream}", description="git upstream HEAD")
        status = _git_output(
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            description="git worktree status",
        )
        worktree_clean = not bool(status)
        head_matches_upstream = git_head == upstream_head
        tracked = set(
            _git_output(
                "ls-files", "--", *IMPLEMENTATION_SOURCE_PATHS, description="tracked sources"
            ).splitlines()
        )
        all_sources_tracked = tracked == set(IMPLEMENTATION_SOURCE_PATHS)
        remote_name = upstream_ref.split("/", 1)[0]
        remote_url = _git_output(
            "remote", "get-url", remote_name, description="upstream remote URL"
        )
        if not self.tiny_mock and not worktree_clean:
            raise FormalRunError("formal prepare requires a clean git worktree")
        if not self.tiny_mock and not head_matches_upstream:
            raise FormalRunError("formal prepare requires HEAD equal to its configured upstream")
        if not self.tiny_mock and not all_sources_tracked:
            raise FormalRunError("formal prepare requires every critical source to be tracked")
        source_hashes = self._source_hashes()
        runtime_inputs = {
            "configs/multifidelity_formal.json": self.config_path,
            "configs/multifidelity_formal_approval.json": self.approval_path,
            "configs/multifidelity_seen_identity_exclusions.json": self.exclusions_path,
        }
        if any(
            path is None or _file_sha256(path) != source_hashes[relative]
            for relative, path in runtime_inputs.items()
        ):
            raise FormalRunError("formal runtime inputs differ from the tracked frozen sources")
        environment = _environment_manifest(self.device)
        required_strings = set(environment) - {"cuda_available", "device_total_memory_bytes"}
        if (
            environment["cuda_available"] is not True
            or not str(environment["device_requested"]).startswith("cuda")
            or not isinstance(environment["device_total_memory_bytes"], int)
            or environment["device_total_memory_bytes"] <= 0
            or any(
                not isinstance(environment[name], str) or not environment[name].strip()
                for name in required_strings
            )
        ):
            raise FormalRunError("formal prepare requires complete CUDA/package provenance")
        return {
            "schema": "tunnelgeopt.formal_implementation_manifest.v1",
            "run_id": self.config["run_id"],
            "config_sha256": self.config_sha256,
            "effect_claim_allowed": not self.tiny_mock,
            "recorded_at_utc": _now(),
            "source_provenance": {
                "git_head": _require_git_commit(git_head, "git HEAD"),
                "upstream_ref": upstream_ref,
                "upstream_head": _require_git_commit(upstream_head, "git upstream HEAD"),
                "head_matches_upstream": head_matches_upstream,
                "worktree_clean_before_prepare": worktree_clean,
                "remote_url_sanitized": _sanitize_remote_url(remote_url),
                "all_sources_tracked": all_sources_tracked,
                "source_sha256": source_hashes,
            },
            "environment": environment,
        }

    def _verify_implementation_unchanged(self) -> None:
        path = self.paths.root / IMPLEMENTATION_MANIFEST_FILENAME
        if not path.is_file():
            raise FormalAbstain("implementation provenance manifest is missing")
        implementation = _read_json(path, "implementation provenance manifest")
        provenance = implementation.get("source_provenance")
        if not isinstance(provenance, Mapping):
            raise FormalAbstain("implementation source provenance is missing")
        expected = provenance.get("source_sha256")
        if not isinstance(expected, Mapping) or set(expected) != set(IMPLEMENTATION_SOURCE_PATHS):
            raise FormalAbstain("implementation source set changed after prepare")
        if self._source_hashes() != dict(expected):
            raise FormalAbstain("critical source hash changed after prepare")
        if _environment_manifest(self.device) != implementation.get("environment"):
            raise FormalAbstain("implementation environment changed after prepare")
        if self.tiny_mock:
            return
        runtime_inputs = {
            "configs/multifidelity_formal.json": self.config_path,
            "configs/multifidelity_formal_approval.json": self.approval_path,
            "configs/multifidelity_seen_identity_exclusions.json": self.exclusions_path,
        }
        if any(
            path is None or _file_sha256(path) != expected[relative]
            for relative, path in runtime_inputs.items()
        ):
            raise FormalAbstain("runtime frozen input changed after prepare")
        head = _git_output("rev-parse", "HEAD", description="git HEAD revalidation")
        if head != provenance.get("git_head"):
            raise FormalAbstain("git HEAD changed after prepare")
        upstream = _git_output(
            "rev-parse", "@{upstream}", description="git upstream HEAD revalidation"
        )
        if upstream != head or upstream != provenance.get("upstream_head"):
            raise FormalAbstain("git upstream no longer matches the prepared HEAD")
        dirty_tracked = subprocess.run(
            ("git", "diff", "--quiet", "--", *IMPLEMENTATION_SOURCE_PATHS),
            cwd=ROOT,
            check=False,
        ).returncode
        staged_tracked = subprocess.run(
            ("git", "diff", "--cached", "--quiet", "--", *IMPLEMENTATION_SOURCE_PATHS),
            cwd=ROOT,
            check=False,
        ).returncode
        if dirty_tracked != 0 or staged_tracked != 0:
            raise FormalAbstain("critical tracked source changed after prepare")

    def _load_identity_exclusions(self) -> Any:
        from tunnelgeopt.formal_generation import FrozenIdentityExclusions

        if self.tiny_mock and self.exclusions_path is None:
            return FrozenIdentityExclusions()
        if self.exclusions_path is None or not self.exclusions_path.is_file():
            raise FormalRunError("formal seen-identity exclusions artifact is required")
        payload = _read_json(self.exclusions_path, "seen-identity exclusions")
        required = {
            "schema",
            "geometry_group_ids",
            "boundary_float64_sha256",
            "case_group_ids",
            "load_group_ids",
            "source_artifact_sha256",
            "source_record_count",
        }
        if not required.issubset(payload):
            raise FormalRunError("seen-identity exclusions field set is incomplete")
        identities = FrozenIdentityExclusions(
            geometry_group_ids=frozenset(map(str, payload["geometry_group_ids"])),
            boundary_float64_sha256=frozenset(map(str, payload["boundary_float64_sha256"])),
            case_group_ids=frozenset(map(str, payload["case_group_ids"])),
            load_group_ids=frozenset(map(str, payload["load_group_ids"])),
            source_artifact_sha256=str(payload["source_artifact_sha256"]),
            source_record_count=int(payload["source_record_count"]),
        )
        if not self.tiny_mock and not any(
            (
                identities.geometry_group_ids,
                identities.boundary_float64_sha256,
                identities.case_group_ids,
                identities.load_group_ids,
            )
        ):
            raise FormalRunError("formal seen-identity exclusions may not be empty")
        if not self.tiny_mock and identities.source_record_count <= 0:
            raise FormalRunError("formal seen-identity exclusions source count must be positive")
        canonical_digest = _sha256_value(payload)
        expected_canonical = self.approval.get("seen_identity_exclusions_canonical_sha256")
        expected_file = self.approval.get("seen_identity_exclusions_file_sha256")
        if not self.tiny_mock and (
            expected_canonical != canonical_digest
            or expected_file != _file_sha256(self.exclusions_path)
        ):
            raise FormalRunError("seen-identity exclusions are not bound to formal approval")
        self.exclusions_canonical_sha256 = canonical_digest
        self.exclusions_file_sha256 = _file_sha256(self.exclusions_path)
        return identities

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
        manifest = Path(values.dataset_manifest_path)
        if not manifest.is_file():
            raise FormalRunError("formal dataset manifest is missing")
        return manifest, public, train

    def _training_paths(self) -> tuple[Path, Path]:
        """Trainer-facing API: deliberately returns no manifest/sealed path."""

        _, public, train = self._dataset_paths()
        return public, train

    def _trusted_sealed_path(self, partition: str) -> Path:
        state = self._load_state()
        phase = next(
            (name for name in PHASES if state["phases"][name]["status"] == "in_progress"),
            "none",
        )
        self._event(
            state,
            "trusted_sealed_path_resolved",
            partition=partition,
            phase=phase,
        )
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

    def _opaque_sealed_digest(self, manifest: Mapping[str, Any], partition: str) -> str:
        stores = manifest.get("opaque_sealed_stores")
        if not isinstance(stores, Mapping):
            raise FormalRunError("dataset manifest omits opaque sealed store identities")
        opaque_id = _sha256_value({"run_id": self.config["run_id"], "partition": partition})
        if opaque_id not in stores:
            raise FormalRunError(f"opaque sealed store identity is missing for {partition}")
        return _require_sha256(stores[opaque_id], f"opaque sealed digest for {partition}")

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
            predecessor_record = state["phases"][predecessor]
            if predecessor_record["status"] != "completed":
                raise FormalRunError(f"phase {phase!r} requires completed {predecessor!r}")
            self._verify_artifacts(predecessor_record.get("artifacts", {}))
        if phase != "prepare":
            self._verify_implementation_unchanged()
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
        # Formal provenance must observe the caller's clean checkout before
        # state/access artifacts are created inside the repository output.
        if phase == "prepare" and not self.paths.state.exists():
            self._prepared_implementation = self._implementation_manifest()
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
        except FormalAbstain as exc:
            state = self._load_state()
            record = state["phases"][phase]
            record["status"] = "abstained"
            record["last_error"] = f"{type(exc).__name__}: {exc}"
            if str(exc) not in state["abstain_reasons"]:
                state["abstain_reasons"].append(str(exc))
            self._event(state, "phase_abstained", phase=phase, reason=str(exc))
            raise
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
            active_phase = next(
                (name for name in PHASES if state["phases"][name]["status"] == "in_progress"),
                "none",
            )
            self._event(
                state,
                "sealed_access_denied",
                partition=partition,
                phase=active_phase,
            )
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
        active_phase = next(
            (name for name in PHASES if state["phases"][name]["status"] == "in_progress"),
            "none",
        )
        self._event(
            state,
            "sealed_partition_opened",
            partition=partition,
            phase=active_phase,
            bytes=len(payload),
            sha256=_sha256_bytes(payload),
        )
        return _load_npz_bytes(payload, f"sealed partition {partition}")

    def _run_prepare(self, state: dict[str, Any]) -> dict[str, str]:
        self.paths.root.mkdir(parents=True, exist_ok=True)
        implementation = self._prepared_implementation
        if implementation is None:
            raise FormalRunError("implementation provenance was not captured before output writes")
        self._prepared_implementation = None
        implementation_path = self.paths.root / IMPLEMENTATION_MANIFEST_FILENAME
        implementation_sha256 = _atomic_json(implementation_path, implementation)
        snapshot = {
            "schema": "tunnelgeopt.multifidelity.formal_prepare.v1",
            "run_id": self.config["run_id"],
            "backend": self.backend,
            "effect_claim_allowed": not self.tiny_mock,
            "config_sha256": self.config_sha256,
            "approval_sha256": _file_sha256(self.approval_path),
            "seen_identity_exclusions": (
                None
                if self.exclusions_path is None
                else {
                    "canonical_sha256": self.exclusions_canonical_sha256,
                    "file_sha256": self.exclusions_file_sha256,
                }
            ),
            "prepared_at_utc": _now(),
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "device_requested": self.device,
            },
            "formal_data_generated": False,
            "locked_labels_opened": False,
            "implementation_manifest_file_sha256": implementation_sha256,
        }
        config_snapshot = self.paths.root / "frozen_config_snapshot.json"
        approval_snapshot = self.paths.root / "execution_approval_snapshot.json"
        prepare_manifest = self.paths.root / "prepare_manifest.json"
        hashes = {
            IMPLEMENTATION_MANIFEST_FILENAME: implementation_sha256,
            "frozen_config_snapshot.json": _atomic_json(config_snapshot, self.config),
            "execution_approval_snapshot.json": _atomic_json(approval_snapshot, self.approval),
        }
        if self.exclusions_path is not None:
            exclusions_snapshot = self.paths.root / "seen_identity_exclusions_snapshot.json"
            hashes["seen_identity_exclusions_snapshot.json"] = _atomic_json(
                exclusions_snapshot,
                _read_json(self.exclusions_path, "seen-identity exclusions"),
            )
        snapshot["snapshot_hashes"] = dict(hashes)
        hashes["prepare_manifest.json"] = _atomic_json(prepare_manifest, snapshot)
        return hashes

    def _run_generate(self, state: dict[str, Any]) -> dict[str, str]:
        if not self.tiny_mock:
            try:
                from tunnelgeopt.formal_generation import (
                    FormalGenerationError,
                    build_formal_generation_plan,
                    generate_formal_dataset,
                )
            except ImportError as exc:  # pragma: no cover - integration guard
                raise FormalRunError("formal generation module is unavailable") from exc
            try:
                plan = build_formal_generation_plan(
                    self.config, forbidden_identities=self.forbidden_identities
                )
            except FormalGenerationError as exc:
                raise FormalRunError(f"formal generation planning failed: {exc}") from exc
            if plan.formal_eligible is not True:
                raise FormalRunError("formal generation plan is not eligible")

            def progress(event: Mapping[str, Any]) -> None:
                _append_jsonl(
                    self.paths.root / "generation_progress.jsonl",
                    {"at_utc": _now(), **dict(event)},
                )

            try:
                result = generate_formal_dataset(
                    self.config,
                    self.paths.data,
                    forbidden_identities=self.forbidden_identities,
                    resume=True,
                    progress_callback=progress,
                )
            except FormalGenerationError as exc:
                manifest_path = self.paths.data / DATASET_MANIFEST_FILENAME
                if manifest_path.is_file():
                    manifest = _read_json(manifest_path, "generation ABSTAIN manifest")
                    if (
                        manifest.get("generation_status") == "ABSTAIN"
                        and manifest.get("config_sha256") == self.config_sha256
                        and manifest.get("run_id") == self.config["run_id"]
                    ):
                        digest = _file_sha256(manifest_path)
                        persisted = self._load_state()
                        reason = f"trusted generation validity gate ABSTAIN: {exc}"
                        persisted["phases"]["generate"].update(
                            {
                                "status": "abstained",
                                "artifacts": {
                                    f"data/{DATASET_MANIFEST_FILENAME}": digest,
                                },
                                "last_error": f"FormalGenerationError: {exc}",
                            }
                        )
                        if reason not in persisted["abstain_reasons"]:
                            persisted["abstain_reasons"].append(reason)
                        self._event(
                            persisted,
                            "generation_abstain_evidence_frozen",
                            reason=reason,
                            manifest_sha256=digest,
                        )
                        raise FormalAbstain(reason) from exc
                raise FormalRunError(
                    f"formal generation failed without ABSTAIN evidence: {exc}"
                ) from exc
            result_manifest_path = Path(result.manifest_path)
            if not result_manifest_path.is_file():
                raise FormalRunError("trusted generator did not write its manifest")
            artifacts: dict[str, str] = {}
            allowed_paths = {
                result_manifest_path.resolve(),
                Path(result.public_inputs_path).resolve(),
                Path(result.train_dev_labels_path).resolve(),
            }
            for path_value, expected_digest in result.public_file_hashes.items():
                path = Path(path_value)
                if not path.is_absolute():
                    path = self.paths.data / path
                if path.resolve() not in allowed_paths:
                    continue
                digest = _file_sha256(path)
                if digest != _require_sha256(expected_digest, str(path_value)):
                    raise FormalRunError("trusted generator returned a stale file digest")
                artifacts[path.relative_to(self.paths.root).as_posix()] = digest
            manifest_path = result_manifest_path
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
                audit_case_group_ids=np.asarray([], dtype="U160"),
                audit_section_families=np.asarray([], dtype="U32"),
                audit_partitions=np.asarray([], dtype="U32"),
                audit_relative_errors=np.asarray([], dtype=np.float64),
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
            "files": {
                filename: file_hashes[filename]
                for filename in (PUBLIC_FILENAME, TRAIN_DEV_FILENAME)
            },
            "opaque_sealed_stores": {
                _sha256_value(
                    {"run_id": self.config["run_id"], "partition": partition}
                ): file_hashes[filename]
                for partition, filename in SEALED_FILENAMES.items()
            },
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

    def _training_worker_payload(self) -> dict[str, Any]:
        manifest_path, public_path, labels_path = self._dataset_paths()
        manifest = _read_json(manifest_path, "formal dataset manifest")
        if manifest.get("config_sha256") != self.config_sha256 or manifest.get("backend") not in (
            None,
            self.backend,
        ):
            raise FormalRunError("dataset manifest belongs to a different run")
        input_hashes: dict[str, str] = {}
        for role, path in (("public_inputs", public_path), ("train_dev_labels", labels_path)):
            expected = self._manifest_file_digest(manifest, path)
            if not path.is_file() or _file_sha256(path) != expected:
                raise FormalAbstain(f"training input hash mismatch: {path.name}")
            input_hashes[role] = expected
        learning = self.config["learning"]
        return {
            "schema": TRAINING_WORKER_SCHEMA,
            "run_id": self.config["run_id"],
            "config_sha256": self.config_sha256,
            "backend": self.backend,
            "device": self.device,
            "public_inputs": {"path": str(public_path), "sha256": input_hashes["public_inputs"]},
            "train_dev_labels": {
                "path": str(labels_path),
                "sha256": input_hashes["train_dev_labels"],
            },
            "checkpoint_output_dir": str(self.paths.checkpoints),
            "split_salt": self.config["identity_and_split"]["split_salt"],
            "fine_train_fractions": list(learning["fine_train_fractions"]),
            "nested_parent_counts": dict(learning["nested_parent_counts"]),
            "model": dict(learning["model"]),
            "optimization": dict(learning["optimization"]),
            "checkpoint_specifications": [
                dict(value) for value in _expected_checkpoint_specs(self.config)
            ],
        }

    def _verify_training_worker(
        self,
        *,
        contract_path: Path,
        contract_payload: Mapping[str, Any],
        command: Sequence[str],
    ) -> tuple[dict[str, Any], dict[str, Any], tuple[dict[str, Any], ...]]:
        audit_path = self.paths.checkpoints / TRAINING_WORKER_AUDIT_FILENAME
        manifest_path = self.paths.checkpoints / CHECKPOINT_MANIFEST_FILENAME
        audit = _read_json(audit_path, "training worker audit")
        manifest = _read_json(manifest_path, "worker checkpoint manifest")
        expected_paths = {
            "worker_contract": str(contract_path.resolve()),
            "public_inputs": str(Path(contract_payload["public_inputs"]["path"]).resolve()),
            "train_dev_labels": str(Path(contract_payload["train_dev_labels"]["path"]).resolve()),
            "checkpoint_output_dir": str(self.paths.checkpoints.resolve()),
        }
        worker_argv = audit.get("argv")
        argv_matches = (
            isinstance(worker_argv, list)
            and len(worker_argv) == 4
            and Path(str(worker_argv[0])).name.lower() == Path(command[0]).name.lower()
            and Path(str(worker_argv[1])).name == Path(command[1]).name
            and worker_argv[2:] == list(command[2:])
        )
        if (
            audit.get("passed") is not True
            or audit.get("entrypoint") != "isolated_training_worker"
            or audit.get("contract_sha256") != _file_sha256(contract_path)
            or audit.get("received_contract_keys") != sorted(TRAINING_WORKER_ALLOWED_KEYS)
            or audit.get("received_paths") != expected_paths
            or not argv_matches
            or set(audit.get("imported_tunnelgeopt_modules", ())) != TRAINING_WORKER_ALLOWED_IMPORTS
            or audit.get("unexpected_tunnelgeopt_modules") != []
        ):
            raise FormalAbstain("training worker isolation audit failed")
        specifications = _expected_checkpoint_specs(self.config)
        if [dict(value) for value in specifications] != list(
            contract_payload["checkpoint_specifications"]
        ):
            raise FormalAbstain("training worker checkpoint specification contract drifted")
        expected_keys = {str(value["checkpoint_key"]) for value in specifications}
        if (
            manifest.get("config_sha256") != self.config_sha256
            or manifest.get("backend") != self.backend
            or manifest.get("training_worker_contract_sha256") != _file_sha256(contract_path)
            or manifest.get("expected_checkpoint_count") != 35
            or set(manifest.get("checkpoints", {})) != expected_keys
            or set(manifest.get("contracts", {})) != expected_keys
            or manifest.get("training_worker_audit_sha256") != _file_sha256(audit_path)
        ):
            raise FormalAbstain("training worker checkpoint manifest failed authentication")
        by_key = {str(value["checkpoint_key"]): value for value in specifications}
        for key in sorted(expected_keys):
            specification = by_key[key]
            checkpoint = manifest["checkpoints"][key]
            contract = manifest["contracts"][key]
            path = (self.paths.checkpoints / str(checkpoint.get("file"))).resolve()
            if path.parent != self.paths.checkpoints.resolve():
                raise FormalAbstain(f"worker checkpoint escaped output directory: {key}")
            if (
                not path.is_file()
                or _file_sha256(path) != checkpoint.get("sha256")
                or checkpoint.get("method") != specification["method"]
                or float(checkpoint.get("fine_fraction", -1.0)) != float(specification["fraction"])
                or int(checkpoint.get("seed", -1)) != int(specification["seed"])
                or checkpoint.get("format_version") != 2
                or checkpoint.get("contract_sha256") != contract.get("contract_sha256")
                or checkpoint.get("selection_sha256") != contract.get("selection_sha256")
                or contract.get("config_sha256") != self.config_sha256
                or contract.get("method") != specification["method"]
                or float(contract.get("fine_fraction", -1.0)) != float(specification["fraction"])
            ):
                raise FormalAbstain(f"worker checkpoint record failed verification: {key}")
            if self.tiny_mock:
                saved = _read_json(path, "tiny worker checkpoint")
                if (
                    saved.get("training_contract_sha256") != contract["contract_sha256"]
                    or saved.get("selection_sha256") != contract["selection_sha256"]
                    or saved.get("config_sha256") != self.config_sha256
                ):
                    raise FormalAbstain(f"tiny worker checkpoint envelope failed: {key}")
            else:
                saved = checkpoint_payload(
                    path,
                    expected_config_sha256=self.config_sha256,
                    expected_selection_sha256=str(contract["selection_sha256"]),
                    require_formal=True,
                )
                if saved.get("training_contract_sha256") != contract["contract_sha256"]:
                    raise FormalAbstain(f"formal worker checkpoint envelope failed: {key}")
        return manifest, audit, specifications

    def _run_train(self, state: dict[str, Any]) -> dict[str, str]:
        if any(int(value) != 0 for value in state["sealed_partition_open_counts"].values()):
            raise FormalAbstain("training cannot proceed after any sealed label was opened")
        self.paths.checkpoints.mkdir(parents=True, exist_ok=True)
        worker_payload = self._training_worker_payload()
        serialized = _canonical_bytes(worker_payload).decode("utf-8").lower()
        if any(
            token in serialized
            for token in (
                "formal_dataset_manifest",
                "execution_approval",
                "locked_iid",
                "locked_geometry_ood",
                "locked_load_ood",
                "locked_joint_ood",
                "sealed_locked",
            )
        ):
            raise FormalRunError("orchestrator attempted to disclose evaluator data to trainer")
        contract_path = self.paths.checkpoints / TRAINING_WORKER_CONTRACT_FILENAME
        contract_sha256 = _atomic_json(contract_path, worker_payload)
        command = [
            sys.executable,
            str((ROOT / "scripts" / "run_multifidelity_train_worker.py").resolve()),
            "--contract",
            str(contract_path.resolve()),
        ]
        self._event(
            state,
            "training_worker_started",
            worker_argv=command,
            worker_contract_sha256=contract_sha256,
            received_path_roles=[
                "worker_contract",
                "public_inputs",
                "train_dev_labels",
                "checkpoint_output_dir",
            ],
        )
        completed = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise FormalRunError(
                f"isolated training worker failed with exit {completed.returncode}: {detail}"
            )
        manifest, worker_audit, specifications = self._verify_training_worker(
            contract_path=contract_path,
            contract_payload=worker_payload,
            command=command,
        )
        access_events = _read_access_events(self.paths.access_log)
        path_resolutions = sum(
            event.get("event") == "trusted_sealed_path_resolved" and event.get("phase") == "train"
            for event in access_events
        )
        sealed_opens = sum(
            event.get("event") == "sealed_partition_opened" and event.get("phase") == "train"
            for event in access_events
        )
        denied_accesses = sum(
            event.get("event") == "sealed_access_denied" and event.get("phase") == "train"
            for event in access_events
        )
        training_access_audit = {
            "schema": "tunnelgeopt.formal_training_access_audit.v2",
            "process_isolated": True,
            "worker_argv": list(worker_audit["argv"]),
            "worker_contract_sha256": contract_sha256,
            "worker_received_contract_keys": list(worker_audit["received_contract_keys"]),
            "worker_received_paths": dict(worker_audit["received_paths"]),
            "worker_imported_tunnelgeopt_modules": list(
                worker_audit["imported_tunnelgeopt_modules"]
            ),
            "worker_unexpected_tunnelgeopt_modules": list(
                worker_audit["unexpected_tunnelgeopt_modules"]
            ),
            "sealed_path_resolution_calls": path_resolutions,
            "sealed_open_calls": sealed_opens,
            "denied_sealed_access_calls": denied_accesses,
            "trainer_input_api": "redacted_contract_public_train_dev_subprocess",
            "passed": worker_audit["passed"] is True
            and path_resolutions == sealed_opens == denied_accesses == 0,
        }
        if training_access_audit["passed"] is not True:
            raise FormalAbstain("isolated trainer touched or received evaluator-only data")
        manifest["training_access_audit"] = training_access_audit
        manifest_path = self.paths.checkpoints / CHECKPOINT_MANIFEST_FILENAME
        _atomic_json(manifest_path, manifest)
        checkpoint_ids: list[str] = []
        for index, specification in enumerate(specifications):
            key = str(specification["checkpoint_key"])
            record = manifest["checkpoints"][key]
            checkpoint_ids.append(str(record["sha256"]))
            self._event(
                state,
                "checkpoint_frozen",
                checkpoint_key=key,
                checkpoint_index=index,
                checkpoint_sha256=record["sha256"],
                contract_sha256=record["contract_sha256"],
            )
        registry = CheckpointRegistry(tuple(checkpoint_ids))
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
            "training_worker_completed",
            worker_pid=worker_audit["worker_pid"],
            checkpoint_count=35,
            import_audit=worker_audit["unexpected_tunnelgeopt_modules"],
            received_paths=worker_audit["received_paths"],
        )
        self._event(
            state,
            "checkpoint_registry_frozen",
            checkpoint_count=35,
            registry_hash=registry.registry_hash,
            file_sha256=registry_hash,
        )
        artifacts = {
            f"checkpoints/{TRAINING_WORKER_CONTRACT_FILENAME}": contract_sha256,
            f"checkpoints/{TRAINING_WORKER_AUDIT_FILENAME}": _file_sha256(
                self.paths.checkpoints / TRAINING_WORKER_AUDIT_FILENAME
            ),
            f"checkpoints/{CHECKPOINT_MANIFEST_FILENAME}": _file_sha256(manifest_path),
            f"checkpoints/{REGISTRY_FILENAME}": registry_hash,
        }
        progress_path = self.paths.checkpoints / TRAINING_WORKER_PROGRESS_FILENAME
        if progress_path.is_file():
            artifacts[f"checkpoints/{TRAINING_WORKER_PROGRESS_FILENAME}"] = _file_sha256(
                progress_path
            )
        return artifacts

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
        fine_ultrafine_audit = {
            "case_group_ids": [],
            "section_families": [],
            "partitions": [],
            "relative_errors": [],
        }
        _, train_dev_path = self._training_paths()
        train_dev_archive = _load_npz(train_dev_path, "train/dev fine labels")
        audit_fields = {
            "audit_case_group_ids",
            "audit_section_families",
            "audit_partitions",
            "audit_relative_errors",
        }
        if not self.tiny_mock and not audit_fields.issubset(train_dev_archive):
            raise FormalAbstain("train/dev fine-ultrafine audit fields are missing")
        if audit_fields.issubset(train_dev_archive):
            audit_lengths = {np.asarray(train_dev_archive[name]).size for name in audit_fields}
            if len(audit_lengths) != 1:
                raise FormalAbstain("train/dev fine-ultrafine audit arrays misalign")
            train_dev_audit_errors = np.asarray(
                train_dev_archive["audit_relative_errors"], dtype=np.float64
            )
            train_dev_audit_partitions = [
                str(value) for value in train_dev_archive["audit_partitions"]
            ]
            if (
                not np.isfinite(train_dev_audit_errors).all()
                or np.any(train_dev_audit_errors < 0.0)
                or any(value not in {"train_id", "dev_id"} for value in train_dev_audit_partitions)
            ):
                raise FormalAbstain("train/dev fine-ultrafine audit values are invalid")
            fine_ultrafine_audit["case_group_ids"].extend(
                str(value) for value in train_dev_archive["audit_case_group_ids"]
            )
            fine_ultrafine_audit["section_families"].extend(
                str(value) for value in train_dev_archive["audit_section_families"]
            )
            fine_ultrafine_audit["partitions"].extend(train_dev_audit_partitions)
            fine_ultrafine_audit["relative_errors"].extend(train_dev_audit_errors.tolist())
        tracemalloc.start()
        for partition_index, partition in enumerate(LOCKED_PARTITIONS):
            partition_started = time.perf_counter()
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
            if not self.tiny_mock and not audit_fields.issubset(sealed):
                raise FormalAbstain(
                    f"sealed fine-ultrafine audit fields are missing for {partition}"
                )
            if audit_fields.issubset(sealed):
                audit_lengths = {np.asarray(sealed[name]).size for name in audit_fields}
                if len(audit_lengths) != 1:
                    raise FormalAbstain(
                        f"sealed fine-ultrafine audit arrays misalign for {partition}"
                    )
                audit_errors = np.asarray(sealed["audit_relative_errors"], dtype=np.float64)
                if not np.isfinite(audit_errors).all() or np.any(audit_errors < 0.0):
                    raise FormalAbstain(
                        f"sealed fine-ultrafine audit errors are invalid for {partition}"
                    )
                audit_partitions = [str(value) for value in sealed["audit_partitions"]]
                if any(value != partition for value in audit_partitions):
                    raise FormalAbstain(
                        f"sealed fine-ultrafine audit partition mismatch for {partition}"
                    )
                fine_ultrafine_audit["case_group_ids"].extend(
                    str(value) for value in sealed["audit_case_group_ids"]
                )
                fine_ultrafine_audit["section_families"].extend(
                    str(value) for value in sealed["audit_section_families"]
                )
                fine_ultrafine_audit["partitions"].extend(audit_partitions)
                fine_ultrafine_audit["relative_errors"].extend(audit_errors.tolist())
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
            sealed_hash = self._opaque_sealed_digest(dataset_manifest, partition)
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
                "checkpoint_evaluation_counts": {},
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
                partition_metrics["checkpoint_evaluation_counts"][key] = new_count
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
            residual_by_seed = {
                str(seed): {
                    "traction_discrepancy_by_case": partition_metrics["checkpoints"][
                        _checkpoint_key("residual_coarse", 0.5, int(seed))
                    ]["wall_offset"]["traction_case_values"],
                    "resultant_discrepancy_by_case": partition_metrics["checkpoints"][
                        _checkpoint_key("residual_coarse", 0.5, int(seed))
                    ]["wall_offset"]["resultant_case_values"],
                }
                for seed in self.config["learning"]["training_seeds"]
            }
            partition_metrics["wall_offset_physics"] = {
                "coarse_only": {
                    "traction_discrepancy_by_case": coarse_d_t.tolist(),
                    "resultant_discrepancy_by_case": coarse_d_r.tolist(),
                },
                "residual_coarse_0.5": {"by_seed": residual_by_seed},
            }
            _, current_peak = tracemalloc.get_traced_memory()
            partition_metrics["resource_usage"] = {
                "runtime_seconds": time.perf_counter() - partition_started,
                "peak_memory_bytes": int(current_peak),
            }
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
        metric_payload["fine_ultrafine_audit"] = fine_ultrafine_audit
        selected_audit_ids = dataset_manifest.get("fine_ultrafine_selection", {}).get(
            "selected_case_ids"
        )
        if not self.tiny_mock and list(fine_ultrafine_audit["case_group_ids"]) != list(
            selected_audit_ids or ()
        ):
            raise FormalAbstain(
                "unsealed fine-ultrafine values do not match the preselected manifest order"
            )
        checkpoint_manifest = _read_json(
            self.paths.checkpoints / CHECKPOINT_MANIFEST_FILENAME,
            "checkpoint manifest",
        )
        generation_resource = dataset_manifest.get(
            "resource_usage", dataset_manifest.get("generation_resource_usage", {})
        )
        if isinstance(generation_resource, Mapping) and "generation" in generation_resource:
            generation_resource = generation_resource["generation"]
        metric_payload["resource_usage"] = {
            "generation": dict(generation_resource)
            if isinstance(generation_resource, Mapping)
            else {},
            "training": {
                key: {
                    "runtime_seconds": value.get("runtime_seconds"),
                    "peak_memory_bytes": value.get("peak_memory_bytes"),
                }
                for key, value in checkpoint_manifest["checkpoints"].items()
            },
            "evaluation": {
                partition: value["resource_usage"]
                for partition, value in metric_payload["partitions"].items()
            },
        }
        metrics_path = self.paths.evaluation / "sealed_metrics.json"
        digest = _atomic_json(metrics_path, metric_payload)
        return {"evaluation/sealed_metrics.json": digest}

    def _formal_analysis_access_state(
        self,
        *,
        metrics: Mapping[str, Any],
        dataset_manifest: Mapping[str, Any],
        state: Mapping[str, Any],
    ) -> dict[str, Any]:
        checkpoint_manifest_path = self.paths.checkpoints / CHECKPOINT_MANIFEST_FILENAME
        registry_path = self.paths.checkpoints / REGISTRY_FILENAME
        checkpoint_manifest = _read_json(checkpoint_manifest_path, "checkpoint manifest")
        registry = _read_json(registry_path, "checkpoint registry")
        events = _read_access_events(self.paths.access_log)
        training_audit = checkpoint_manifest.get("training_access_audit", {})
        training_path_resolutions = sum(
            event.get("event") == "trusted_sealed_path_resolved" and event.get("phase") == "train"
            for event in events
        )
        training_sealed_opens = sum(
            event.get("event") == "sealed_partition_opened" and event.get("phase") == "train"
            for event in events
        )
        worker_paths = training_audit.get("worker_received_paths", {})
        worker_argv = training_audit.get("worker_argv", [])
        worker_isolation_passed = (
            isinstance(training_audit, Mapping)
            and training_audit.get("schema") == "tunnelgeopt.formal_training_access_audit.v2"
            and training_audit.get("passed") is True
            and training_audit.get("process_isolated") is True
            and training_audit.get("worker_received_contract_keys")
            == sorted(TRAINING_WORKER_ALLOWED_KEYS)
            and isinstance(worker_paths, Mapping)
            and set(worker_paths)
            == {
                "worker_contract",
                "public_inputs",
                "train_dev_labels",
                "checkpoint_output_dir",
            }
            and isinstance(worker_argv, list)
            and len(worker_argv) == 4
            and Path(str(worker_argv[1])).name == "run_multifidelity_train_worker.py"
            and worker_argv[2] == "--contract"
            and training_audit.get("worker_unexpected_tunnelgeopt_modules") == []
        )
        trainer_received_locked_path = not (
            worker_isolation_passed
            and int(training_audit.get("sealed_path_resolution_calls", -1)) == 0
            and training_path_resolutions == 0
        )
        registry_event_indices = [
            index
            for index, event in enumerate(events)
            if event.get("event") == "checkpoint_registry_frozen"
        ]
        sealed_open_indices = [
            index
            for index, event in enumerate(events)
            if event.get("event") == "sealed_partition_opened"
        ]
        opened_before_registry = not registry_event_indices or any(
            index < registry_event_indices[-1] for index in sealed_open_indices
        )
        prepare_complete_indices = [
            index
            for index, event in enumerate(events)
            if event.get("event") == "phase_completed" and event.get("phase") == "prepare"
        ]
        generate_start_indices = [
            index
            for index, event in enumerate(events)
            if event.get("event") == "phase_started" and event.get("phase") == "generate"
        ]
        config_frozen_before_generation = (
            self.approval.get("config_frozen") is True
            and bool(prepare_complete_indices)
            and bool(generate_start_indices)
            and prepare_complete_indices[0] < generate_start_indices[0]
        )
        records = {
            key: {
                "sha256": value["sha256"],
                "training_contract_sha256": value["contract_sha256"],
                "config_sha256": self.config_sha256,
            }
            for key, value in checkpoint_manifest["checkpoints"].items()
        }
        dataset_manifest_path, _, _ = self._dataset_paths()
        metrics_path = self.paths.evaluation / "sealed_metrics.json"
        implementation_path = self.paths.root / IMPLEMENTATION_MANIFEST_FILENAME
        prepare_manifest_path = self.paths.root / "prepare_manifest.json"
        implementation = _read_json(implementation_path, "implementation provenance manifest")
        return {
            "run_id": self.config["run_id"],
            "config_sha256": self.config_sha256,
            "hashes": {
                "config_canonical_sha256": self.config_sha256,
                "dataset_manifest_canonical_sha256": _sha256_value(dataset_manifest),
                "sealed_metrics_canonical_sha256": _sha256_value(metrics),
                "config_file_sha256": _file_sha256(ROOT / "configs/multifidelity_formal.json"),
                "dataset_manifest_file_sha256": _file_sha256(dataset_manifest_path),
                "sealed_metrics_file_sha256": _file_sha256(metrics_path),
                "access_log_file_sha256": _file_sha256(self.paths.access_log),
                "checkpoint_manifest_file_sha256": _file_sha256(checkpoint_manifest_path),
                "checkpoint_registry_file_sha256": _file_sha256(registry_path),
                "implementation_manifest_file_sha256": _file_sha256(implementation_path),
                "prepare_manifest_file_sha256": _file_sha256(prepare_manifest_path),
            },
            "checkpoint_registry": {
                "frozen": True,
                "config_sha256": self.config_sha256,
                "checkpoint_count": registry["checkpoint_count"],
                "registry_hash": registry["registry_hash"],
                "checkpoints": records,
            },
            "config_frozen_before_generation": config_frozen_before_generation,
            "implementation_manifest": implementation,
            "locked_labels_opened_before_checkpoint_freeze": opened_before_registry,
            "locked_labels_used_for_tuning": training_sealed_opens > 0,
            "trainer_received_locked_label_path": trainer_received_locked_path,
            "access_log_append_only": True,
            "denied_premature_sealed_accesses": state["denied_premature_sealed_accesses"],
            "sealed_partition_open_counts": dict(state["sealed_partition_open_counts"]),
            "checkpoint_evaluation_counts": dict(state["checkpoint_evaluation_counts"]),
            "abstain_reasons": list(state["abstain_reasons"]),
            "training_access_evidence": {
                "checkpoint_manifest_audit": dict(training_audit),
                "worker_process_contract_passed": worker_isolation_passed,
                "worker_argv": list(worker_argv) if isinstance(worker_argv, list) else [],
                "worker_received_paths": dict(worker_paths)
                if isinstance(worker_paths, Mapping)
                else {},
                "worker_import_audit": training_audit.get(
                    "worker_imported_tunnelgeopt_modules", []
                ),
                "observed_train_sealed_path_resolution_calls": training_path_resolutions,
                "observed_train_sealed_open_calls": training_sealed_opens,
            },
        }

    def _run_independent_formal_analysis(self, state: Mapping[str, Any]) -> dict[str, str]:
        from tunnelgeopt.formal_analysis import evaluate_formal_decision

        metrics_path = self.paths.evaluation / "sealed_metrics.json"
        metrics = _read_json(metrics_path, "sealed metrics")
        dataset_manifest_path, _, _ = self._dataset_paths()
        dataset_manifest = _read_json(dataset_manifest_path, "formal dataset manifest")
        access_state = self._formal_analysis_access_state(
            metrics=metrics, dataset_manifest=dataset_manifest, state=state
        )
        decision = evaluate_formal_decision(self.config, metrics, dataset_manifest, access_state)
        if decision.get("classification") not in {"GO", "NO_GO", "ABSTAIN"}:
            raise FormalRunError("independent analysis returned an invalid classification")
        self.paths.analysis.mkdir(parents=True, exist_ok=True)
        evidence_path = self.paths.analysis / "analysis_access_state.json"
        decision_path = self.paths.analysis / "decision.json"
        return {
            "analysis/analysis_access_state.json": _atomic_json(evidence_path, access_state),
            "analysis/decision.json": _atomic_json(decision_path, decision),
        }

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
        return self._run_independent_formal_analysis(state)


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
    parser.add_argument("--exclusions", type=Path, default=DEFAULT_EXCLUSIONS)
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
            exclusions_path=arguments.exclusions,
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
