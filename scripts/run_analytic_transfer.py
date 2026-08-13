"""Run the preregistered circular Kirsch analytic-transfer smoke experiment."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from tunnelgeopt.transfer import (
    ALL_METHODS,
    build_analytic_dataset,
    checkpoint_identity,
    config_sha256,
    dry_run_contract,
    evaluate_locked_test,
    load_checkpoint_payload,
    load_model_checkpoint,
    load_transfer_config,
    resolve_device,
    save_cpu_checkpoint_atomic,
    summarize_full_gate,
    train_method,
)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _release_model(model: Any, device: str) -> None:
    """Release one model before continuing to the next method/seed."""

    import torch

    # Moving the object mutates the caller-held module too, so the GPU storage
    # is released even before that caller local is rebound on the next loop.
    model.to("cpu")
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


def _checkpoint_path(output: Path, method: str, seed: int) -> Path:
    return output / "checkpoints" / f"{method}__seed-{seed}.pt"


def _environment(device: str) -> dict[str, Any]:
    import scipy
    import torch

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": device,
        "cuda_device_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the circle-only Kirsch analytic-transfer pipeline. A passing result does not "
            "establish non-circular, fracture, rockburst, or field validity."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "analytic_transfer_smoke.json",
    )
    parser.add_argument("--mode", choices=("dry-run", "full"), default="dry-run")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts" / "experiment" / "analytic-transfer-v0.2.0",
    )
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    started = time.perf_counter()
    config = load_transfer_config(args.config)
    device = resolve_device(args.device)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    dataset = build_analytic_dataset(config)
    manifest: dict[str, Any] = {
        "run_id": output.name,
        "mode": args.mode,
        "status": "running",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "command": " ".join(sys.argv),
        "config_path": str(args.config.resolve()),
        "config_sha256": config_sha256(args.config),
        "config_name": config["config_name"],
        "environment": _environment(device),
        "claim_scope": "analytic_circle_pipeline_only",
        "claim_exclusions": config["claim_exclusions"],
        "test_used_for_model_selection": False,
    }
    _atomic_json(output / "manifest.json", manifest)

    if args.mode == "dry-run":
        report = dry_run_contract(dataset, config, device=device)
        _atomic_json(output / "dry_run.json", report)
        manifest.update(
            {
                "status": "dry_run_passed",
                "elapsed_seconds": time.perf_counter() - started,
                "locked_test_inference_count": 0,
                "result_path": str((output / "dry_run.json").resolve()),
            }
        )
        _atomic_json(output / "manifest.json", manifest)
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        return 0

    seeds = [int(seed) for seed in config["optimization"]["training_seeds"]]
    results: dict[str, dict[int, dict[str, Any]]] = {method: {} for method in ALL_METHODS}
    audits: dict[str, dict[int, dict[str, Any]]] = {method: {} for method in ALL_METHODS}
    expected_checkpoint_count = len(ALL_METHODS) * len(seeds)
    checkpoint_records: dict[str, dict[str, Any]] = {}

    def progress(event: dict[str, Any]) -> None:
        event = {**event, "at_utc": datetime.now(UTC).isoformat()}
        with (output / "progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")

    # Train and freeze every method before any locked-test label is read.
    for method in ALL_METHODS:
        for seed in seeds:
            checkpoint_key = f"{method}__seed-{seed}"
            checkpoint_path = _checkpoint_path(output, method, seed)
            expected_metadata = {
                "method": method,
                "seed": seed,
                "config_sha256": manifest["config_sha256"],
            }
            # Crash recovery is evidence-preserving: reuse only a structurally
            # valid CPU checkpoint whose method/seed/config metadata matches.
            if checkpoint_path.is_file():
                try:
                    payload = load_checkpoint_payload(checkpoint_path)
                    if all(
                        payload["metadata"].get(key) == expected
                        for key, expected in expected_metadata.items()
                    ):
                        identity = checkpoint_identity(checkpoint_path)
                        audit = dict(payload["metadata"].get("training_audit", {}))
                        audits[method][seed] = audit
                        checkpoint_records[checkpoint_key] = {
                            "path": str(checkpoint_path.resolve()),
                            "sha256": identity,
                            "status": "reused_frozen_cpu_checkpoint",
                        }
                        progress(
                            {
                                "event": "checkpoint_reused",
                                "method": method,
                                "seed": seed,
                                "sha256": identity,
                            }
                        )
                        _atomic_json(output / "checkpoint_index.json", checkpoint_records)
                        _atomic_json(output / "training_audit.json", audits)
                        continue
                except (OSError, ValueError, RuntimeError):
                    progress(
                        {
                            "event": "checkpoint_rejected_for_resume",
                            "method": method,
                            "seed": seed,
                        }
                    )
            progress({"event": "method_start", "method": method, "seed": seed})
            model, audit = train_method(
                dataset,
                config,
                method,
                seed,
                device=device,
                progress=lambda event, m=method, s=seed: progress(
                    {**event, "method": m, "seed": s}
                ),
            )
            audits[method][seed] = audit
            identity = save_cpu_checkpoint_atomic(
                model,
                checkpoint_path,
                config,
                seed=seed,
                metadata={**expected_metadata, "training_audit": audit},
            )
            checkpoint_records[checkpoint_key] = {
                "path": str(checkpoint_path.resolve()),
                "sha256": identity,
                "status": "trained_and_frozen_cpu_checkpoint",
            }
            progress(
                {
                    "event": "checkpoint_frozen",
                    "method": method,
                    "seed": seed,
                    "sha256": identity,
                }
            )
            _atomic_json(output / "checkpoint_index.json", checkpoint_records)
            _atomic_json(output / "training_audit.json", audits)
            _release_model(model, device)

    if len(checkpoint_records) != expected_checkpoint_count:
        raise RuntimeError("not all expected checkpoints were durably frozen")
    before_unlock = dataset.access_snapshot()
    if before_unlock["materialized_cases"]["locked_test"] != 0:
        raise RuntimeError("critical leakage: locked_test labels materialized before freeze")
    if before_unlock["label_case_reads"]["locked_test"] != 0:
        raise RuntimeError("critical leakage: locked_test labels read before freeze")
    if before_unlock["denied_locked_test_accesses"] != 0:
        raise RuntimeError("critical leakage attempt: locked_test gate was triggered before freeze")
    dataset.authorize_locked_test(
        [record["sha256"] for record in checkpoint_records.values()],
        expected_checkpoint_count=expected_checkpoint_count,
    )
    dataset.materialize_split("locked_test", purpose="post_freeze_locked_evaluation")
    _atomic_json(output / "dataset_access_audit.json", dataset.access_snapshot())
    progress(
        {
            "event": "locked_test_materialized_after_all_checkpoints",
            "frozen_checkpoint_count": expected_checkpoint_count,
        }
    )

    # Exactly one evaluation call per frozen method/seed checkpoint.  Each call
    # performs three complete test-set forward passes: primary prediction plus
    # original/rotated predictions for the equivariance diagnostic.
    evaluation_calls = 0
    locked_test_forward_passes = 0
    locked_test_forward_batches = 0
    locked_test_label_case_reads = 0
    for method in ALL_METHODS:
        for seed in seeds:
            checkpoint_path = _checkpoint_path(output, method, seed)
            model, _ = load_model_checkpoint(
                checkpoint_path,
                config,
                device=device,
                expected_metadata={
                    "method": method,
                    "seed": seed,
                    "config_sha256": manifest["config_sha256"],
                },
            )
            results[method][seed] = evaluate_locked_test(model, dataset, config, device=device)
            counts = results[method][seed]["access_counts"]
            evaluation_calls += int(counts["evaluation_calls"])
            locked_test_forward_passes += int(counts["locked_test_model_forward_passes"])
            locked_test_forward_batches += int(counts["locked_test_model_forward_batches"])
            locked_test_label_case_reads += int(counts["locked_test_label_case_reads"])
            progress(
                {
                    "event": "locked_test_evaluation_call_completed",
                    "method": method,
                    "seed": seed,
                    **counts,
                }
            )
            _atomic_json(output / "locked_test_results.json", results)
            _atomic_json(output / "dataset_access_audit.json", dataset.access_snapshot())
            _release_model(model, device)
    gate = summarize_full_gate(results, config)
    _atomic_json(output / "gate.json", gate)
    manifest.update(
        {
            "status": "completed",
            "elapsed_seconds": time.perf_counter() - started,
            "locked_test_evaluation_calls": evaluation_calls,
            "actual_locked_test_forward_passes": locked_test_forward_passes,
            "actual_locked_test_forward_batches": locked_test_forward_batches,
            "locked_test_label_case_reads": locked_test_label_case_reads,
            "dataset_access_audit": dataset.access_snapshot(),
            "gate_status": gate["status"],
            "formal_gate_status": "not_evaluated_by_this_smoke",
            "result_paths": {
                "training_audit": str((output / "training_audit.json").resolve()),
                "checkpoint_index": str((output / "checkpoint_index.json").resolve()),
                "dataset_access_audit": str((output / "dataset_access_audit.json").resolve()),
                "locked_test_results": str((output / "locked_test_results.json").resolve()),
                "gate": str((output / "gate.json").resolve()),
                "progress": str((output / "progress.jsonl").resolve()),
            },
        }
    )
    _atomic_json(output / "manifest.json", manifest)
    print(json.dumps(gate, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
