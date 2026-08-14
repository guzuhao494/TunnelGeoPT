#!/usr/bin/env python3
"""Run one immutable paired SENT+SENS coarse intact timing/QC campaign.

This development runner is deliberately narrower than a fracture trajectory:
damage is fixed at zero and exactly three formal-grid states are solved for
each benchmark.  Its output may be used only for paired timing/QC triage.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import re
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = Path(__file__).resolve()

CAMPAIGN_SCHEMA = "tunnelgeopt.fracture.sent_sens.paired_intact_campaign.v1"
CAMPAIGN_RESULT_SCHEMA = "tunnelgeopt.fracture.sent_sens.paired_intact_campaign_result.v1"
CASE_RESULT_SCHEMA = "tunnelgeopt.fracture.sent_sens.paired_intact_case_result.v1"
IMPLEMENTATION_MANIFEST_SCHEMA = (
    "tunnelgeopt.fracture.sent_sens.paired_intact_implementation_manifest.v1"
)
ARTIFACT_MANIFEST_SCHEMA = "tunnelgeopt.fracture.sent_sens.paired_intact_artifact_manifest.v1"
CASE_ORDER = ("sent", "sens")
TIER = "coarse"
EXPECTED_THREE_STATE_GRID_MM = (0.0, 1.0e-5, 2.0e-5)
CAMPAIGN_ARTIFACT_RELATIVE_PATHS = (
    "cases/sent/result.json",
    "cases/sens/result.json",
    "implementation_manifest.json",
    "campaign_result.json",
    "artifact_manifest.json",
)
_PROBE_SCHEMA = "tunnelgeopt.fracture.sent_sens.intact_probe.v1"
_PROBE_STATUS = "DEVELOPMENT_INTACT_FIXED_DAMAGE_PROBE_ONLY"
_DAMAGE_STATUS = "NOT_APPLICABLE_INTACT_D0_PROBE"
_ZERO_STATE_REACTION_TOLERANCE_KN = 1.0e-12
_THREAD_ENVIRONMENT_NAMES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


class FractureBenchmarkCampaignError(RuntimeError):
    """Raised when the paired campaign identity or QC contract is violated."""


@dataclass(frozen=True)
class CampaignArtifactBundle:
    """Content hashes for the four files in a completed paired campaign."""

    sent_result_sha256: str
    sens_result_sha256: str
    implementation_manifest_sha256: str
    campaign_result_sha256: str
    artifact_manifest_sha256: str


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _apply_cpu_single_thread_policy() -> dict[str, Any]:
    """Set thread controls before the CLI lazily imports numerical packages."""

    for name in _THREAD_ENVIRONMENT_NAMES:
        os.environ[name] = "1"
    return {
        "execution_device": "CPU",
        "requested_single_thread": True,
        "environment": {name: os.environ[name] for name in _THREAD_ENVIRONMENT_NAMES},
        "application_point": "before_lazy_tunnelgeopt_numpy_scipy_imports_in_this_runner",
        "runtime_library_thread_count_directly_verified": False,
        "verification_boundary": "environment_controls_recorded_not_OS_scheduler_guarantee",
    }


def _peak_rss_measurement() -> dict[str, Any]:
    """Read process-lifetime peak RSS without adding a package dependency."""

    try:
        if os.name == "nt":
            size_type = ctypes.c_size_t

            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_ulong),
                    ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", size_type),
                    ("WorkingSetSize", size_type),
                    ("QuotaPeakPagedPoolUsage", size_type),
                    ("QuotaPagedPoolUsage", size_type),
                    ("QuotaPeakNonPagedPoolUsage", size_type),
                    ("QuotaNonPagedPoolUsage", size_type),
                    ("PagefileUsage", size_type),
                    ("PeakPagefileUsage", size_type),
                ]

            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            get_current_process = kernel32.GetCurrentProcess
            get_current_process.restype = ctypes.c_void_p
            get_process_memory_info = psapi.GetProcessMemoryInfo
            get_process_memory_info.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ProcessMemoryCounters),
                ctypes.c_ulong,
            ]
            if not get_process_memory_info(
                get_current_process(), ctypes.byref(counters), counters.cb
            ):
                raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
            peak = int(counters.PeakWorkingSetSize)
            method = "windows_GetProcessMemoryInfo_PeakWorkingSetSize"
        else:
            import resource

            raw_peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            if os.uname().sysname == "Darwin":
                peak = raw_peak
                method = "resource_getrusage_ru_maxrss_bytes"
            else:
                peak = raw_peak * 1024
                method = "resource_getrusage_ru_maxrss_kib"
        if peak <= 0:
            raise ValueError("non-positive peak RSS")
        return {
            "status": "AVAILABLE",
            "peak_rss_bytes": peak,
            "method": method,
            "scope": "whole_campaign_process_lifetime_upper_bound_not_incremental",
        }
    except (AttributeError, ImportError, OSError, ValueError):
        return {
            "status": "UNAVAILABLE",
            "peak_rss_bytes": None,
            "method": None,
            "reason_code": "platform_process_peak_rss_api_unavailable",
            "scope": "whole_campaign_process_lifetime_upper_bound_not_incremental",
        }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, help="new, unique outer campaign leaf")
    parser.add_argument(
        "--expected-project-head",
        required=True,
        help="full pushed Git SHA shared by both cases",
    )
    parser.add_argument("--run-paired-intact-probe", action="store_true")
    parser.add_argument("--approved-development-probe", action="store_true")
    return parser


def _validate_cli_arguments(args: argparse.Namespace) -> None:
    if not args.run_paired_intact_probe:
        raise SystemExit("campaign requires --run-paired-intact-probe")
    if not args.approved_development_probe:
        raise SystemExit("campaign requires --approved-development-probe")
    if args.output is None:
        raise SystemExit("campaign requires --output for a unique outer leaf")
    if re.fullmatch(r"[0-9a-fA-F]{40}", args.expected_project_head.strip()) is None:
        raise SystemExit("--expected-project-head must be a full 40-character SHA-1")


def _load_campaign_contract(
    config_path: Path,
) -> tuple[Mapping[str, Any], dict[str, tuple[float, ...]]]:
    from tunnelgeopt.fracture_benchmark_validation import (
        load_fracture_sent_sens_config,
        prescribed_displacements,
    )

    config = load_fracture_sent_sens_config(config_path)
    displacements: dict[str, tuple[float, ...]] = {}
    for benchmark_id in CASE_ORDER:
        formal_grid = prescribed_displacements(config, benchmark_id)
        if len(formal_grid) < 3:
            raise FractureBenchmarkCampaignError(
                f"{benchmark_id} formal grid does not contain three states"
            )
        first_three = tuple(float(value) for value in formal_grid[:3])
        if first_three != EXPECTED_THREE_STATE_GRID_MM:
            raise FractureBenchmarkCampaignError(
                f"{benchmark_id} first two formal increments are not both 1e-5 mm"
            )
        displacements[benchmark_id] = first_three
    return config, displacements


def _validate_snapshot_source_closure(snapshot: Any) -> None:
    try:
        expected_runner = RUNNER_PATH.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError as exc:  # pragma: no cover - fixed repository layout
        raise FractureBenchmarkCampaignError("campaign runner is outside project root") from exc
    if snapshot.runner_path != expected_runner:
        raise FractureBenchmarkCampaignError("snapshot is not bound to the paired campaign runner")
    source_paths = {source.path for source in snapshot.source_files}
    if expected_runner not in source_paths:
        raise FractureBenchmarkCampaignError("campaign runner is absent from source closure")
    if not isinstance(snapshot.source_inventory_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", snapshot.source_inventory_sha256
    ):
        raise FractureBenchmarkCampaignError("snapshot source inventory identity is invalid")


def _require_finite_json(value: Any, path: str = "artifact") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise FractureBenchmarkCampaignError(f"{path} contains a non-finite float")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise FractureBenchmarkCampaignError(f"{path} contains a non-string key")
            _require_finite_json(child, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _require_finite_json(child, f"{path}[{index}]")
        return
    raise FractureBenchmarkCampaignError(
        f"{path} contains unsupported JSON value {type(value).__name__}"
    )


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    _require_finite_json(value)
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _reject_host_path_strings(value: Any, project_root: Path, path: str = "artifact") -> None:
    forbidden = str(project_root.resolve()).replace("\\", "/").casefold()
    if isinstance(value, str):
        if forbidden and forbidden in value.replace("\\", "/").casefold():
            raise FractureBenchmarkCampaignError(f"{path} contains the local project path")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            _reject_host_path_strings(key, project_root, f"{path}.<key>")
            _reject_host_path_strings(child, project_root, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_host_path_strings(child, project_root, f"{path}[{index}]")


def _write_exclusive_file(path: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary_name, path)
        os.unlink(temporary_name)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def reserve_campaign_output_directory(
    project_root: str | Path, output_directory: str | Path
) -> tuple[Path, str]:
    """Reject every fixed campaign artifact if ignored, then reserve the leaf."""

    from tunnelgeopt.fracture_benchmark import (
        _git_path_is_ignored,
        reserve_probe_output_directory,
    )

    root = Path(project_root).resolve(strict=True)
    candidate = Path(output_directory)
    if not candidate.is_absolute():
        candidate = root / candidate
    target = Path(os.path.abspath(candidate))
    try:
        relative = target.relative_to(root).as_posix()
    except ValueError as exc:
        raise FractureBenchmarkCampaignError("campaign output is outside project root") from exc
    for artifact_relative in CAMPAIGN_ARTIFACT_RELATIVE_PATHS:
        path = f"{relative}/{artifact_relative}"
        if _git_path_is_ignored(root, path):
            raise FractureBenchmarkCampaignError(
                f"campaign artifact path must not be ignored by Git: {path}"
            )
    return reserve_probe_output_directory(root, target)


def _hash_record(path: str, payload: bytes) -> dict[str, Any]:
    return {
        "path": path,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _validate_case_probe(
    probe: Any,
    *,
    benchmark_id: str,
    displacements_mm: tuple[float, ...],
    formal_increment_count: int,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = probe.as_dict()
    if not isinstance(raw, dict):
        raise FractureBenchmarkCampaignError(f"{benchmark_id} probe is not a mapping")
    expected_identity = {
        "schema": _PROBE_SCHEMA,
        "status": _PROBE_STATUS,
        "benchmark_id": benchmark_id,
        "tier": TIER,
        "authorizes_medium_fine_or_formal_run": False,
    }
    for key, expected in expected_identity.items():
        if raw.get(key) != expected:
            raise FractureBenchmarkCampaignError(
                f"{benchmark_id} probe identity field {key!r} differs"
            )
    for key in (
        "protocol_sha256",
        "mesh_plan_sha256",
        "mesh_topology_sha256",
        "bvp_mesh_sha256",
    ):
        if not isinstance(raw.get(key), str) or re.fullmatch(r"[0-9a-f]{64}", raw[key]) is None:
            raise FractureBenchmarkCampaignError(f"{benchmark_id} {key} is invalid")
    if tuple(raw.get("prescribed_U_mm", ())) != displacements_mm:
        raise FractureBenchmarkCampaignError(f"{benchmark_id} prescribed states differ")
    if raw.get("projected_formal_increment_count") != formal_increment_count:
        raise FractureBenchmarkCampaignError(
            f"{benchmark_id} formal increment projection identity differs"
        )
    mesh_counts = raw.get("mesh_counts")
    if not isinstance(mesh_counts, Mapping) or any(
        not isinstance(mesh_counts.get(name), int) or mesh_counts[name] <= 0
        for name in ("node_count", "element_count", "top_node_count", "bottom_node_count")
    ):
        raise FractureBenchmarkCampaignError(f"{benchmark_id} mesh counts are invalid")
    steps = raw.get("steps")
    if not isinstance(steps, list) or len(steps) != len(displacements_mm):
        raise FractureBenchmarkCampaignError(f"{benchmark_id} must contain exactly three steps")
    thresholds = config["per_tier_qc"]
    maxima = {
        "equilibrium_relative_residual": 0.0,
        "global_force_relative_imbalance": 0.0,
        "global_moment_relative_imbalance": 0.0,
        "path_energy_relative_imbalance": 0.0,
    }
    threshold_keys = {
        "equilibrium_relative_residual": "max_equilibrium_relative_residual",
        "global_force_relative_imbalance": "max_global_force_relative_imbalance",
        "global_moment_relative_imbalance": "max_global_moment_relative_imbalance",
        "path_energy_relative_imbalance": "max_path_energy_relative_imbalance",
    }
    positive_reaction_checks: list[bool] = []
    for index, (step, displacement) in enumerate(zip(steps, displacements_mm, strict=True)):
        if not isinstance(step, Mapping):
            raise FractureBenchmarkCampaignError(f"{benchmark_id} step {index} is not a mapping")
        if step.get("sequence_index") != index or step.get("prescribed_U_mm") != displacement:
            raise FractureBenchmarkCampaignError(f"{benchmark_id} step order/state differs")
        if step.get("converged") is not True:
            raise FractureBenchmarkCampaignError(f"{benchmark_id} step {index} did not converge")
        if step.get("damage_component_status") != _DAMAGE_STATUS:
            raise FractureBenchmarkCampaignError(
                f"{benchmark_id} step {index} is not an intact fixed-d probe"
            )
        generalized_load = step.get("generalized_load_kN")
        load_is_finite_number = (
            isinstance(generalized_load, (int, float))
            and not isinstance(generalized_load, bool)
            and math.isfinite(float(generalized_load))
        )
        if not load_is_finite_number:
            raise FractureBenchmarkCampaignError(
                f"{benchmark_id} step {index} has an invalid reaction magnitude"
            )
        if displacement > 0.0:
            positive = float(generalized_load) > 0.0
            positive_reaction_checks.append(positive)
            if not positive:
                raise FractureBenchmarkCampaignError(
                    f"{benchmark_id} step {index} reaction magnitude must be strictly positive"
                )
        elif abs(float(generalized_load)) > _ZERO_STATE_REACTION_TOLERANCE_KN:
            raise FractureBenchmarkCampaignError(
                f"{benchmark_id} zero state reaction magnitude exceeds numerical zero tolerance"
            )
        for metric, threshold_key in threshold_keys.items():
            value = step.get(metric)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise FractureBenchmarkCampaignError(
                    f"{benchmark_id} step {index} {metric} is invalid"
                )
            value = float(value)
            maxima[metric] = max(maxima[metric], value)
            if value < 0.0 or value > float(thresholds[threshold_key]):
                raise FractureBenchmarkCampaignError(
                    f"{benchmark_id} step {index} exceeds {threshold_key}"
                )
    _require_finite_json(raw, f"cases.{benchmark_id}.probe")
    positive_reaction_gate_passed = len(positive_reaction_checks) == sum(
        value > 0.0 for value in displacements_mm
    ) and all(positive_reaction_checks)
    return raw, {
        "all_three_states_converged": True,
        "all_intact_damage_statuses_verified": True,
        "positive_reaction_magnitude_gate_passed": positive_reaction_gate_passed,
        "maxima": maxima,
        "thresholds": {metric: float(thresholds[key]) for metric, key in threshold_keys.items()},
        "per_tier_qc_passed_for_intact_applicable_fields": True,
        "coupled_damage_qc_applicability": "NOT_APPLICABLE_FIXED_D0_PROBE",
    }


def _sanitized_command(
    *, expected_project_head: str, config_path: str, output_relative: str
) -> list[str]:
    return [
        "python",
        "scripts/run_fracture_benchmark_campaign.py",
        "--config",
        config_path,
        "--output",
        output_relative,
        "--expected-project-head",
        expected_project_head.lower(),
        "--run-paired-intact-probe",
        "--approved-development-probe",
    ]


def write_paired_campaign_artifact_atomic(
    output_directory: str | Path,
    *,
    probes: Mapping[str, Any],
    config: Mapping[str, Any],
    displacements: Mapping[str, tuple[float, ...]],
    formal_increment_counts: Mapping[str, int],
    project_snapshot: Any,
    started_utc: str,
    completed_utc: str,
    postflight_verified_utc: str,
    sanitized_command: Sequence[str],
    solver_options: Any,
    runtime_environment: Mapping[str, Any],
    resource_measurement: Mapping[str, Any],
    paired_wall_seconds: float,
) -> CampaignArtifactBundle:
    """Publish both cases and manifests; the final manifest is the completion marker."""

    target = Path(output_directory).resolve(strict=False)
    root = project_snapshot._project_root.resolve()
    try:
        output_relative = target.relative_to(root).as_posix()
    except ValueError as exc:
        raise FractureBenchmarkCampaignError("campaign output is outside project root") from exc
    if not target.is_dir() or target.is_symlink() or any(target.iterdir()):
        raise FractureBenchmarkCampaignError(
            "campaign output leaf must remain exclusively reserved and empty until publication"
        )
    if tuple(probes) != CASE_ORDER or set(displacements) != set(CASE_ORDER):
        raise FractureBenchmarkCampaignError("campaign cases must be ordered SENT then SENS")
    if not sanitized_command or not all(
        isinstance(argument, str) and argument for argument in sanitized_command
    ):
        raise FractureBenchmarkCampaignError("sanitized command contains an empty argument")

    raw_probes: dict[str, dict[str, Any]] = {}
    qc_summaries: dict[str, dict[str, Any]] = {}
    protocol_sha256: str | None = None
    for benchmark_id in CASE_ORDER:
        raw, qc = _validate_case_probe(
            probes[benchmark_id],
            benchmark_id=benchmark_id,
            displacements_mm=displacements[benchmark_id],
            formal_increment_count=formal_increment_counts[benchmark_id],
            config=config,
        )
        if protocol_sha256 is None:
            protocol_sha256 = raw["protocol_sha256"]
        elif raw["protocol_sha256"] != protocol_sha256:
            raise FractureBenchmarkCampaignError("SENT/SENS protocol identities differ")
        raw_probes[benchmark_id] = raw
        qc_summaries[benchmark_id] = qc

    evidence_scope = {
        "real_paired_probe_completed": True,
        "timing_and_qc_triage_only": True,
        "paper_effect_evidence": False,
        "formal_fracture_trajectory": False,
        "coupled_damage_evolution": False,
        "authorizes_medium_fine_or_formal_run": False,
    }
    case_payloads: dict[str, bytes] = {}
    case_records: list[dict[str, Any]] = []
    for index, benchmark_id in enumerate(CASE_ORDER):
        case_result = {
            "schema": CASE_RESULT_SCHEMA,
            "status": "COMPLETED_PAIRED_MEMBER_INTACT_FIXED_D0",
            "campaign_schema": CAMPAIGN_SCHEMA,
            "case_order_index": index,
            "benchmark_id": benchmark_id,
            "tier": TIER,
            "project_head": project_snapshot.project_head,
            "source_inventory_sha256": project_snapshot.source_inventory_sha256,
            "evidence_scope": evidence_scope,
            "qc": qc_summaries[benchmark_id],
            "probe": raw_probes[benchmark_id],
        }
        _reject_host_path_strings(case_result, root)
        payload = _canonical_json_bytes(case_result)
        relative = f"cases/{benchmark_id}/result.json"
        case_payloads[benchmark_id] = payload
        case_records.append(_hash_record(relative, payload))

    implementation_manifest = {
        "schema": IMPLEMENTATION_MANIFEST_SCHEMA,
        "status": "PAIRED_RESULTS_WRITTEN_INCOMPLETE_UNTIL_FINAL_MANIFEST_PRESENT",
        "campaign_schema": CAMPAIGN_SCHEMA,
        "output_directory": output_relative,
        "timing": {
            "started_utc": started_utc,
            "completed_utc": completed_utc,
            "postflight_verified_utc": postflight_verified_utc,
        },
        "execution": {
            "sanitized_command": list(sanitized_command),
            "solver_options": asdict(solver_options),
            "runtime_environment": dict(runtime_environment),
            "resource_measurement": dict(resource_measurement),
            "case_execution_order": list(CASE_ORDER),
            "case_execution_mode": "strictly_serial_no_parallel_case_or_state_solves",
            "states_per_case": 3,
            "total_fixed_damage_equilibrium_solves": 6,
        },
        "project_provenance": project_snapshot.as_dict(),
        "postflight": {
            "verification_call_count": 1,
            "head_equals_upstream_equals_expected": True,
            "source_inventory_unchanged": True,
            "outer_leaf_empty_at_verification": True,
        },
        "protocol_sha256": protocol_sha256,
        "case_artifacts": case_records,
        "case_qc": qc_summaries,
        "evidence_scope": evidence_scope,
    }
    _reject_host_path_strings(implementation_manifest, root)
    implementation_payload = _canonical_json_bytes(implementation_manifest)
    implementation_record = _hash_record("implementation_manifest.json", implementation_payload)
    case_timing = {
        benchmark_id: {
            "raw_step_wall_seconds": [
                float(step["wall_seconds"]) for step in raw_probes[benchmark_id]["steps"]
            ],
            "median_step_wall_seconds": float(raw_probes[benchmark_id]["median_step_wall_seconds"]),
            "projected_formal_case_wall_hours": float(
                raw_probes[benchmark_id]["projected_formal_case_wall_hours"]
            ),
            "projection_interpretation": raw_probes[benchmark_id]["projection_interpretation"],
        }
        for benchmark_id in CASE_ORDER
    }
    if not math.isfinite(paired_wall_seconds) or paired_wall_seconds < 0.0:
        raise FractureBenchmarkCampaignError("paired campaign wall time is invalid")
    campaign_result = {
        "schema": CAMPAIGN_RESULT_SCHEMA,
        "status": "COMPLETED_PAIRED_INTACT_FIXED_D0_TIMING_QC_TRIAGE",
        "classification": "REAL_PAIRED_PROBE_NOT_COUPLED_NOT_FORMAL_NOT_PAPER_EFFECT",
        "campaign_schema": CAMPAIGN_SCHEMA,
        "case_order": list(CASE_ORDER),
        "shared_identity": {
            "project_head": project_snapshot.project_head,
            "upstream_head": project_snapshot.upstream_head,
            "protocol_sha256": protocol_sha256,
            "config_path": project_snapshot.config_path,
            "source_inventory_sha256": project_snapshot.source_inventory_sha256,
            "runtime_environment": dict(runtime_environment),
            "solver_options": asdict(solver_options),
        },
        "case_qc": qc_summaries,
        "case_timing": case_timing,
        "paired_resources": {
            "wall_seconds": paired_wall_seconds,
            "peak_rss": dict(resource_measurement),
        },
        "implementation_manifest": implementation_record,
        "evidence_scope": evidence_scope,
        "authorizes_coupled_fracture_run": False,
        "paper_effect_evidence": False,
    }
    _reject_host_path_strings(campaign_result, root)
    campaign_result_payload = _canonical_json_bytes(campaign_result)
    campaign_result_record = _hash_record("campaign_result.json", campaign_result_payload)
    artifact_records = [
        *case_records,
        implementation_record,
        campaign_result_record,
    ]
    artifact_manifest = {
        "schema": ARTIFACT_MANIFEST_SCHEMA,
        "status": "COMPLETED_IMMUTABLE_PAIRED_INTACT_CAMPAIGN",
        "campaign_schema": CAMPAIGN_SCHEMA,
        "output_directory": output_relative,
        "project_head": project_snapshot.project_head,
        "source_inventory_sha256": project_snapshot.source_inventory_sha256,
        "artifacts": artifact_records,
        "artifact_count": len(artifact_records),
        "artifact_set_files": [
            *(record["path"] for record in artifact_records),
            "artifact_manifest.json",
        ],
        "artifact_set_file_count_including_this_manifest": len(artifact_records) + 1,
        "completion_marker_rule": "this_final_manifest_written_exclusively_after_all_hash_checks",
        "manifest_sha256_reporting": "returned_by_writer_and_printed_by_cli_not_self_embedded",
        "evidence_scope": evidence_scope,
    }
    _reject_host_path_strings(artifact_manifest, root)
    artifact_manifest_payload = _canonical_json_bytes(artifact_manifest)

    cases_directory = target / "cases"
    cases_directory.mkdir()
    for benchmark_id in CASE_ORDER:
        case_directory = cases_directory / benchmark_id
        case_directory.mkdir()
        _write_exclusive_file(case_directory / "result.json", case_payloads[benchmark_id])
    _write_exclusive_file(target / "implementation_manifest.json", implementation_payload)
    _write_exclusive_file(target / "campaign_result.json", campaign_result_payload)
    for record in artifact_records:
        artifact_path = target / record["path"]
        if hashlib.sha256(artifact_path.read_bytes()).hexdigest() != record["sha256"]:
            raise FractureBenchmarkCampaignError(f"published hash mismatch for {record['path']}")
    # This is the only completion marker and is always linked last.
    _write_exclusive_file(target / "artifact_manifest.json", artifact_manifest_payload)
    artifact_manifest_sha256 = hashlib.sha256(artifact_manifest_payload).hexdigest()
    if hashlib.sha256((target / "artifact_manifest.json").read_bytes()).hexdigest() != (
        artifact_manifest_sha256
    ):
        raise FractureBenchmarkCampaignError("published final manifest hash mismatch")
    return CampaignArtifactBundle(
        sent_result_sha256=case_records[0]["sha256"],
        sens_result_sha256=case_records[1]["sha256"],
        implementation_manifest_sha256=implementation_record["sha256"],
        campaign_result_sha256=campaign_result_record["sha256"],
        artifact_manifest_sha256=artifact_manifest_sha256,
    )


def main(argv: list[str] | None = None) -> int:
    single_thread_policy = _apply_cpu_single_thread_policy()
    args = _parser().parse_args(argv)
    _validate_cli_arguments(args)

    # Numerical/project imports are intentionally lazy so the thread policy is
    # installed first in a fresh campaign process.
    from tunnelgeopt.fracture import FractureSolverOptions
    from tunnelgeopt.fracture_benchmark import (
        capture_probe_project_preflight,
        probe_runtime_environment,
        run_intact_fracture_benchmark_probe,
        verify_probe_project_postflight,
    )
    from tunnelgeopt.fracture_benchmark_mesh import generate_fracture_benchmark_mesh
    from tunnelgeopt.fracture_benchmark_validation import (
        default_fracture_sent_sens_config_path,
        prescribed_displacements,
    )

    config_path = (
        default_fracture_sent_sens_config_path()
        if args.config is None
        else args.config.resolve(strict=True)
    )
    config, displacements = _load_campaign_contract(config_path)
    formal_increment_counts = {
        benchmark_id: len(prescribed_displacements(config, benchmark_id)) - 1
        for benchmark_id in CASE_ORDER
    }

    # Exactly one clean/pushed snapshot is shared by both case solves.
    snapshot = capture_probe_project_preflight(
        PROJECT_ROOT,
        expected_project_head=args.expected_project_head,
        config_path=config_path,
        runner_path=RUNNER_PATH,
    )
    _validate_snapshot_source_closure(snapshot)
    assert args.output is not None  # guarded by _validate_cli_arguments
    output, output_relative = reserve_campaign_output_directory(PROJECT_ROOT, args.output)
    controls = FractureSolverOptions()
    runtime_environment = probe_runtime_environment()
    runtime_environment["campaign_cpu_policy"] = single_thread_policy
    paired_start = time.perf_counter()
    started_utc = _utc_now()
    probes: dict[str, Any] = {}
    for benchmark_id in CASE_ORDER:
        benchmark_mesh = generate_fracture_benchmark_mesh(
            loading=benchmark_id,
            tier=TIER,
        )
        probes[benchmark_id] = run_intact_fracture_benchmark_probe(
            config,
            benchmark_mesh,
            benchmark_id=benchmark_id,
            tier=TIER,
            displacements_mm=displacements[benchmark_id],
            options=controls,
        )
    completed_utc = _utc_now()
    paired_wall_seconds = float(time.perf_counter() - paired_start)
    resource_measurement = _peak_rss_measurement()

    # Results remain only in memory until one postflight confirms that the
    # shared source snapshot is unchanged and the reserved leaf is still empty.
    if any(output.iterdir()):
        raise FractureBenchmarkCampaignError("campaign leaf changed before postflight")
    postflight_verified_utc = verify_probe_project_postflight(snapshot)
    if any(output.iterdir()):
        raise FractureBenchmarkCampaignError("campaign leaf changed during postflight")
    bundle = write_paired_campaign_artifact_atomic(
        output,
        probes=probes,
        config=config,
        displacements=displacements,
        formal_increment_counts=formal_increment_counts,
        project_snapshot=snapshot,
        started_utc=started_utc,
        completed_utc=completed_utc,
        postflight_verified_utc=postflight_verified_utc,
        sanitized_command=_sanitized_command(
            expected_project_head=snapshot.expected_project_head,
            config_path=snapshot.config_path,
            output_relative=output_relative,
        ),
        solver_options=controls,
        runtime_environment=runtime_environment,
        resource_measurement=resource_measurement,
        paired_wall_seconds=paired_wall_seconds,
    )
    print(
        json.dumps(
            {
                "status": "COMPLETED_IMMUTABLE_PAIRED_INTACT_CAMPAIGN",
                "artifact_manifest_sha256": bundle.artifact_manifest_sha256,
                "real_paired_probe_completed": True,
                "timing_and_qc_triage_only": True,
                "paper_effect_evidence": False,
                "formal_fracture_trajectory": False,
                "authorizes_medium_fine_or_formal_run": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
