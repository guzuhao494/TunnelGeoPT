#!/usr/bin/env python3
"""Minimal subprocess entry point for audited train/dev-only fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import tracemalloc
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from tunnelgeopt.multifidelity_learning import (
    LearningBatch,
    LearningContractError,
    TrainingContract,
    build_training_contract,
    checkpoint_payload,
    make_model,
    nested_geometry_subsets,
    save_formal_checkpoint_atomic,
    train_formal_with_dev_selection,
)

CONTRACT_SCHEMA = "tunnelgeopt.formal_training_worker_contract.v1"
CONTRACT_FILENAME = "training_worker_contract.json"
AUDIT_FILENAME = "training_worker_audit.json"
PROGRESS_FILENAME = "training_progress.jsonl"
CHECKPOINT_MANIFEST_FILENAME = "checkpoint_manifest.json"
ALLOWED_CONTRACT_KEYS = frozenset(
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
METHOD_FRACTIONS = {
    "scratch": (1.0,),
    "direct_coarse": (1.0,),
    "residual_coarse": (0.25, 0.5, 0.75, 1.0),
    "mismatched_coarse": (0.5,),
}


class WorkerError(RuntimeError):
    """Raised when the subprocess input or output contract is invalid."""


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


def _sha256_value(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = _canonical_bytes(payload) + b"\n"
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return hashlib.sha256(encoded).hexdigest()


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        stream.write(_canonical_bytes(payload) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise WorkerError(f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkerError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise WorkerError(f"{label} must be an object")
    return value


def _load_npz(path: Path, label: str) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {name: archive[name] for name in archive.files}
    except (OSError, ValueError) as exc:
        raise WorkerError(f"invalid {label}: {path}") from exc


def _load_batch(public_path: Path, labels_path: Path) -> LearningBatch:
    public = _load_npz(public_path, "public input archive")
    labels = _load_npz(labels_path, "train/dev label archive")
    required_public = {
        "base_features",
        "coarse_stress",
        "training_weights",
        "case_group_ids",
        "geometry_group_ids",
        "section_families",
        "partitions",
    }
    if not required_public.issubset(public):
        raise WorkerError(
            f"public input archive omits fields: {sorted(required_public - set(public))}"
        )
    if not {"case_group_ids", "fine_stress"}.issubset(labels):
        raise WorkerError("train/dev label archive omits required fields")
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
            raise WorkerError("train/dev label identity is absent from public input") from exc
    if indices.ndim != 1 or len(set(indices.tolist())) != indices.size:
        raise WorkerError("train/dev indices must be unique one-dimensional rows")
    partitions = np.asarray(public["partitions"])
    expected = np.flatnonzero(np.isin(partitions, np.asarray(["train_id", "dev_id"])))
    if not np.array_equal(indices, expected):
        raise WorkerError("labels do not exactly cover the two fitting partitions")
    expected_cases = np.asarray(public["case_group_ids"])[indices]
    if not np.array_equal(np.asarray(labels["case_group_ids"]), expected_cases):
        raise WorkerError("fine labels are misaligned with public identities")
    split_values = np.where(partitions[indices] == "train_id", "train", "dev")
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
        raise WorkerError("fitting batch contains an unexpected split")
    return batch


def _subsets(
    batch: LearningBatch,
    payload: Mapping[str, Any],
) -> dict[float, tuple[str, ...]]:
    train_rows = np.flatnonzero(np.asarray(batch.splits) == "train")
    subsets = nested_geometry_subsets(
        tuple(batch.geometry_group_ids[index] for index in train_rows),
        tuple(batch.section_families[index] for index in train_rows),
        fractions=tuple(float(value) for value in payload["fine_train_fractions"]),
        salt=str(payload["split_salt"]),
    )
    expected_counts = payload["nested_parent_counts"]
    previous: set[str] = set()
    for fraction, selected in subsets.items():
        expected = expected_counts[str(float(fraction))]
        if len(selected) != int(expected["total"]):
            raise WorkerError("nested subset count disagrees with the worker contract")
        selected_set = set(selected)
        if not previous.issubset(selected_set):
            raise WorkerError("fine-label parent subsets are not nested")
        previous = selected_set
    return subsets


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
    path: Path,
    *,
    contract: TrainingContract,
    seed: int,
    training_fingerprint: str,
) -> str:
    return _atomic_json(
        path,
        {
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
        },
    )


def _specifications(values: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    specifications = tuple(dict(value) for value in values)
    if len(specifications) != 35:
        raise WorkerError("worker contract must contain exactly 35 checkpoint specifications")
    keys = [str(value.get("checkpoint_key")) for value in specifications]
    if len(set(keys)) != 35:
        raise WorkerError("checkpoint keys must be unique")
    expected = {
        (method, float(fraction))
        for method, fractions in METHOD_FRACTIONS.items()
        for fraction in fractions
    }
    by_seed: dict[int, set[tuple[str, float]]] = {}
    for value in specifications:
        if set(value) != {"checkpoint_key", "method", "fraction", "seed"}:
            raise WorkerError("checkpoint specification field set changed")
        by_seed.setdefault(int(value["seed"]), set()).add(
            (str(value["method"]), float(value["fraction"]))
        )
    if len(by_seed) != 5 or any(values != expected for values in by_seed.values()):
        raise WorkerError("checkpoint matrix changed")
    return specifications


def execute(contract_path: Path) -> dict[str, Any]:
    contract_path = Path(contract_path).resolve()
    payload = _read_json(contract_path, "worker contract")
    if set(payload) != ALLOWED_CONTRACT_KEYS or payload.get("schema") != CONTRACT_SCHEMA:
        raise WorkerError("worker contract field set changed")
    backend = str(payload["backend"])
    if backend not in {"formal", "tiny-mock"}:
        raise WorkerError("worker backend is invalid")
    records = {
        "public_inputs": payload["public_inputs"],
        "train_dev_labels": payload["train_dev_labels"],
    }
    for role, record in records.items():
        if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
            raise WorkerError(f"{role} input record is invalid")
    public_path = Path(records["public_inputs"]["path"]).resolve()
    labels_path = Path(records["train_dev_labels"]["path"]).resolve()
    checkpoint_dir = Path(payload["checkpoint_output_dir"]).resolve()
    if contract_path.parent != checkpoint_dir or contract_path.name != CONTRACT_FILENAME:
        raise WorkerError("worker contract location is invalid")
    for role, path in (("public_inputs", public_path), ("train_dev_labels", labels_path)):
        if not path.is_file() or _file_sha256(path) != str(records[role]["sha256"]):
            raise WorkerError(f"{role} input hash mismatch")
    specifications = _specifications(payload["checkpoint_specifications"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    batch = _load_batch(public_path, labels_path)
    subsets = _subsets(batch, payload)
    manifest_path = checkpoint_dir / CHECKPOINT_MANIFEST_FILENAME
    worker_contract_sha256 = _file_sha256(contract_path)
    if manifest_path.exists():
        manifest = _read_json(manifest_path, "partial checkpoint manifest")
        if (
            manifest.get("config_sha256") != payload["config_sha256"]
            or manifest.get("backend") != backend
            or manifest.get("training_worker_contract_sha256") != worker_contract_sha256
        ):
            raise WorkerError("partial checkpoint manifest is not from this worker contract")
    else:
        manifest = {
            "schema": "tunnelgeopt.multifidelity.checkpoint_manifest.v1",
            "run_id": payload["run_id"],
            "config_sha256": payload["config_sha256"],
            "backend": backend,
            "training_worker_contract_sha256": worker_contract_sha256,
            "expected_checkpoint_count": 35,
            "checkpoints": {},
            "contracts": {},
        }
    training_fingerprint = _sha256_value(
        {
            "public_sha256": records["public_inputs"]["sha256"],
            "train_dev_sha256": records["train_dev_labels"]["sha256"],
        }
    )
    optimization = payload["optimization"]
    model_config = payload["model"]
    progress_path = checkpoint_dir / PROGRESS_FILENAME
    for specification in specifications:
        method = str(specification["method"])
        fraction = float(specification["fraction"])
        seed = int(specification["seed"])
        key = str(specification["checkpoint_key"])
        contract = build_training_contract(
            batch,
            method=method,
            config_sha256=str(payload["config_sha256"]),
            train_geometry_selector=subsets[fraction],
            expected_fine_fraction=fraction,
        )
        record = _contract_record(contract)
        existing_contract = manifest["contracts"].get(key)
        if existing_contract is not None and existing_contract != record:
            raise WorkerError(f"training contract drift on resume: {key}")
        manifest["contracts"][key] = record
        suffix = ".json" if backend == "tiny-mock" else ".pt"
        relative = f"{key}{suffix}"
        checkpoint_path = checkpoint_dir / relative
        existing = manifest["checkpoints"].get(key)
        if existing is not None:
            if (
                existing.get("file") != relative
                or not checkpoint_path.is_file()
                or _file_sha256(checkpoint_path) != existing.get("sha256")
                or existing.get("contract_sha256") != contract.contract_sha256
            ):
                raise WorkerError(f"checkpoint drift on resume: {key}")
            continue
        checkpoint_started = time.perf_counter()
        tracemalloc.start()
        try:
            if backend == "tiny-mock":
                digest = _mock_checkpoint(
                    checkpoint_path,
                    contract=contract,
                    seed=seed,
                    training_fingerprint=training_fingerprint,
                )
                best_epoch = 0
                epochs_run = 1
                best_dev_error = 0.0
            else:
                device = str(payload["device"])
                if not device.startswith("cuda"):
                    raise WorkerError("production fitting requires an explicit CUDA device")
                model = make_model(model_config, seed=seed, device=device)

                def progress(values: Mapping[str, Any], *, checkpoint_key: str = key) -> None:
                    _append_jsonl(
                        progress_path,
                        {"at_utc": _now(), "checkpoint_key": checkpoint_key, **dict(values)},
                    )

                outcome = train_formal_with_dev_selection(
                    model,
                    batch,
                    contract,
                    seed=seed,
                    device=device,
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
                saved = checkpoint_payload(
                    checkpoint_path,
                    expected_config_sha256=str(payload["config_sha256"]),
                    expected_selection_sha256=contract.selection.selection_sha256,
                    require_formal=True,
                )
                if saved["training_contract_sha256"] != contract.contract_sha256:
                    raise WorkerError("saved checkpoint contract verification failed")
                best_epoch = int(saved["best_epoch"])
                epochs_run = int(saved["epochs_run"])
                best_dev_error = float(saved["best_dev_error"])
            _, python_peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        gpu_peak = 0
        if backend == "formal":
            try:
                import torch

                if torch.cuda.is_available():
                    gpu_peak = int(torch.cuda.max_memory_allocated(str(payload["device"])))
            except ImportError:  # pragma: no cover
                pass
        manifest["checkpoints"][key] = {
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
            "runtime_seconds": time.perf_counter() - checkpoint_started,
            "peak_memory_bytes": max(int(python_peak), gpu_peak),
        }
        manifest["completed_checkpoint_count"] = len(manifest["checkpoints"])
        _atomic_json(manifest_path, manifest)
    expected_keys = {str(value["checkpoint_key"]) for value in specifications}
    if set(manifest["checkpoints"]) != expected_keys:
        raise WorkerError("checkpoint set does not exactly match the worker contract")
    imported_modules = sorted(
        name for name in sys.modules if name == "tunnelgeopt" or name.startswith("tunnelgeopt.")
    )
    # Importing a package submodule executes the package initializer.  These
    # are its current dependency-only imports; the parent process separately
    # rejects any module outside this frozen list.
    allowed_modules = {
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
    unexpected_modules = sorted(set(imported_modules) - allowed_modules)
    audit = {
        "schema": "tunnelgeopt.formal_training_worker_audit.v1",
        "worker_pid": os.getpid(),
        "argv": [sys.executable, *sys.argv],
        "contract_sha256": worker_contract_sha256,
        "received_contract_keys": sorted(payload),
        "received_paths": {
            "worker_contract": str(contract_path),
            "public_inputs": str(public_path),
            "train_dev_labels": str(labels_path),
            "checkpoint_output_dir": str(checkpoint_dir),
        },
        "imported_tunnelgeopt_modules": imported_modules,
        "unexpected_tunnelgeopt_modules": unexpected_modules,
        "entrypoint": "isolated_training_worker",
        "passed": not unexpected_modules,
    }
    audit_path = checkpoint_dir / AUDIT_FILENAME
    audit_sha256 = _atomic_json(audit_path, audit)
    manifest["training_worker_audit_sha256"] = audit_sha256
    _atomic_json(manifest_path, manifest)
    return {
        "status": "completed",
        "checkpoint_count": 35,
        "checkpoint_manifest_sha256": _file_sha256(manifest_path),
        "training_worker_audit_sha256": audit_sha256,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        result = execute(arguments.contract)
    except (WorkerError, LearningContractError, ValueError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
