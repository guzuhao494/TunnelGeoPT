"""Frozen validate-only contract for the Miehe-type SENT/SENS benchmarks.

The module deliberately has no mesh or solver imports.  It validates the
development protocol, enumerates the six benchmark/tier identities, and
constructs the prescribed displacement grids.  The separate real-Gmsh mesh
contract has been implemented and audited, but this module does not execute a
fracture solve or turn the current ``ABSTAIN_NOT_RUN`` state into benchmark
result evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "tunnelgeopt.fracture.sent_sens.development_protocol.v1"
PROTOCOL_ID = "miehe-sent-sens-three-grid-development-v1"
BENCHMARK_IDS = ("sent", "sens")
MESH_TIERS = ("coarse", "medium", "fine")
CASE_COUNT = 6

# SHA-256 of canonical JSON (sorted keys, compact separators, UTF-8).  The
# value is intentionally pinned: changing any scientific or decision field
# requires a new protocol version instead of silently relaxing this validator.
FROZEN_CANONICAL_SHA256 = "59975e7832faedaca11210b452b80ca981335a9a9d3cf850b55d7ae2430445e1"

_EXPECTED_TOP_LEVEL = {
    "schema_version",
    "protocol_id",
    "status",
    "scope",
    "evidence_basis",
    "coordinate_system",
    "geometry",
    "material",
    "fracture_model",
    "loading",
    "mesh",
    "topology_qc",
    "solver",
    "per_tier_qc",
    "three_grid_convergence",
    "digitized_sanity_windows",
    "compute_preflight",
    "decision",
    "identity",
}


class FractureBenchmarkContractError(ValueError):
    """Raised when the frozen SENT/SENS development contract is changed."""


@dataclass(frozen=True)
class FractureBenchmarkCase:
    """One frozen benchmark and mesh-tier identity."""

    case_id: str
    benchmark_id: str
    benchmark_name: str
    benchmark_index: int
    mesh_tier: str
    mesh_tier_index: int
    h_target_over_ell: float
    h_target_mm: float
    bulk_h_target_mm: float
    final_displacement_mm: float
    required_state_count: int
    reaction_component: str

    def as_dict(self) -> dict[str, str | int | float]:
        return {
            "case_id": self.case_id,
            "benchmark_id": self.benchmark_id,
            "benchmark_name": self.benchmark_name,
            "benchmark_index": self.benchmark_index,
            "mesh_tier": self.mesh_tier,
            "mesh_tier_index": self.mesh_tier_index,
            "h_target_over_ell": self.h_target_over_ell,
            "h_target_mm": self.h_target_mm,
            "bulk_h_target_mm": self.bulk_h_target_mm,
            "final_displacement_mm": self.final_displacement_mm,
            "required_state_count": self.required_state_count,
            "reaction_component": self.reaction_component,
        }


def default_fracture_sent_sens_config_path() -> Path:
    """Return the repository's frozen SENT/SENS protocol path."""

    return Path(__file__).resolve().parents[2] / "configs" / "fracture_sent_sens_v1.json"


def _walk_json(value: Any, path: str = "config") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str) or not key:
                raise FractureBenchmarkContractError(f"{path} contains a non-string or empty key")
            _walk_json(child, f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _walk_json(child, f"{path}[{index}]")
        return
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise FractureBenchmarkContractError(f"{path} must be finite")
        return
    raise FractureBenchmarkContractError(f"{path} is not a JSON value")


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise FractureBenchmarkContractError(f"config is not canonical JSON: {exc}") from exc
    return hashlib.sha256(payload).hexdigest()


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FractureBenchmarkContractError(f"{path} must be an object")
    return value


def _require_sequence(value: Any, path: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise FractureBenchmarkContractError(f"{path} must be an array")
    return value


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FractureBenchmarkContractError(f"{path} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise FractureBenchmarkContractError(f"{path} must be finite")
    return result


def _integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FractureBenchmarkContractError(f"{path} must be an integer")
    return value


def _require_close(actual: Any, expected: float, path: str) -> None:
    value = _number(actual, path)
    if not math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-15):
        raise FractureBenchmarkContractError(
            f"{path} must equal frozen value {expected!r}, got {value!r}"
        )


def _segment_states(segments: Sequence[Any], path: str) -> tuple[float, ...]:
    states: list[float] = []
    previous_end: float | None = None
    for segment_index, raw_segment in enumerate(segments):
        segment_path = f"{path}[{segment_index}]"
        segment = _require_mapping(raw_segment, segment_path)
        expected_keys = {
            "start_U_mm",
            "end_U_mm",
            "increment_mm",
            "accepted_increment_count",
            "include_start",
            "include_end",
        }
        if set(segment) != expected_keys:
            raise FractureBenchmarkContractError(
                f"{segment_path} keys differ; missing={sorted(expected_keys - set(segment))}, "
                f"extra={sorted(set(segment) - expected_keys)}"
            )
        start = _number(segment["start_U_mm"], f"{segment_path}.start_U_mm")
        end = _number(segment["end_U_mm"], f"{segment_path}.end_U_mm")
        increment = _number(segment["increment_mm"], f"{segment_path}.increment_mm")
        count = _integer(
            segment["accepted_increment_count"],
            f"{segment_path}.accepted_increment_count",
        )
        include_start = segment["include_start"]
        include_end = segment["include_end"]
        if not isinstance(include_start, bool) or not isinstance(include_end, bool):
            raise FractureBenchmarkContractError(
                f"{segment_path}.include_start and include_end must be booleans"
            )
        if increment <= 0.0 or count <= 0 or end <= start:
            raise FractureBenchmarkContractError(f"{segment_path} is not a monotone segment")
        _require_close(
            start + count * increment,
            end,
            f"{segment_path}.end_from_increment_count",
        )
        if previous_end is not None:
            _require_close(start, previous_end, f"{segment_path}.contiguous_start")
        first_index = 0 if include_start else 1
        last_index = count if include_end else count - 1
        if first_index > last_index:
            raise FractureBenchmarkContractError(f"{segment_path} emits no states")
        states.extend(
            round(start + index * increment, 12) for index in range(first_index, last_index + 1)
        )
        previous_end = end
    if not states or any(right <= left for left, right in pairwise(states)):
        raise FractureBenchmarkContractError(f"{path} must emit a strictly increasing grid")
    return tuple(states)


def _validate_semantics(config: Mapping[str, Any]) -> None:
    if set(config) != _EXPECTED_TOP_LEVEL:
        raise FractureBenchmarkContractError(
            "config top-level keys differ from the frozen SENT/SENS contract"
        )
    if config["schema_version"] != SCHEMA_VERSION or config["protocol_id"] != PROTOCOL_ID:
        raise FractureBenchmarkContractError("schema_version or protocol_id is not frozen v1")

    status = _require_mapping(config["status"], "status")
    expected_status = {
        "state": "protocol_frozen_mesh_contract_verified_solver_not_run",
        "development_only": True,
        "mesh_contract_implemented_and_audited": True,
        "mesh_contract_verification_scope": (
            "six_Gmsh_meshes_SENT_SENS_x_coarse_medium_fine_topology_labels_and_size_contract_only"
        ),
        "fracture_solver_runs_completed": False,
        "benchmark_results_available": False,
        "current_decision": "ABSTAIN_NOT_RUN",
        "phase1_pilot_authorized": False,
        "formal_or_paper_claim_allowed": False,
    }
    if dict(status) != expected_status:
        raise FractureBenchmarkContractError(
            "status must preserve verified mesh-only scope and unrun fracture benchmarks"
        )

    coordinates = _require_mapping(config["coordinate_system"], "coordinate_system")
    if coordinates.get("order") != "y_vertical_then_z_horizontal":
        raise FractureBenchmarkContractError(
            "coordinate order must be y vertical then z horizontal"
        )
    bounds = _require_mapping(coordinates.get("domain_bounds_mm"), "coordinate_system.bounds")
    if bounds != {"y": [0.0, 1.0], "z": [0.0, 1.0]}:
        raise FractureBenchmarkContractError("domain must be [0,1] x [0,1] mm in (y,z)")

    notch = _require_mapping(
        _require_mapping(config["geometry"], "geometry").get("notch"), "geometry.notch"
    )
    line = _require_mapping(notch.get("line_mm"), "geometry.notch.line_mm")
    _require_close(line.get("y"), 0.5, "geometry.notch.line_mm.y")
    if line.get("z") != [0.0, 0.5]:
        raise FractureBenchmarkContractError("horizontal slit must span z=0 to 0.5 at y=0.5")
    if notch.get("type") != "explicit_zero_width_double_face_slit":
        raise FractureBenchmarkContractError("notch must be an explicit double-face slit")
    if notch.get("initial_damage_d0") != 0.0 or notch.get("initial_history_H0_kN_per_mm2") != 0.0:
        raise FractureBenchmarkContractError("the explicit slit requires d0=H0=0 in rock")

    material = _require_mapping(config["material"], "material")
    for key, expected in {
        "lame_lambda_kN_per_mm2": 121.15,
        "shear_modulus_kN_per_mm2": 80.77,
        "critical_fracture_energy_kN_per_mm": 0.0027,
        "regularization_length_ell_mm": 0.015,
        "viscosity_eta": 0.0,
    }.items():
        _require_close(material.get(key), expected, f"material.{key}")
    model = _require_mapping(config["fracture_model"], "fracture_model")
    _require_close(model.get("residual_stiffness_k"), 1e-8, "fracture_model.residual_stiffness_k")

    loading = _require_mapping(config["loading"], "loading")
    benchmarks = _require_sequence(loading.get("benchmarks"), "loading.benchmarks")
    if len(benchmarks) != 2:
        raise FractureBenchmarkContractError("loading.benchmarks must contain SENT and SENS")
    expected_benchmarks = {
        "sent": (2001, 0.0065, "sum_top_reaction_y_kN", {"u_y_mm": "U", "u_z_mm": 0.0}),
        "sens": (1501, 0.015, "sum_top_reaction_z_kN", {"u_y_mm": 0.0, "u_z_mm": "U"}),
    }
    seen: list[str] = []
    for index, raw in enumerate(benchmarks):
        benchmark = _require_mapping(raw, f"loading.benchmarks[{index}]")
        benchmark_id = benchmark.get("id")
        if benchmark_id not in expected_benchmarks:
            raise FractureBenchmarkContractError(f"unknown benchmark id {benchmark_id!r}")
        seen.append(str(benchmark_id))
        count, final_u, reaction, top = expected_benchmarks[str(benchmark_id)]
        states = _segment_states(
            _require_sequence(benchmark.get("segments"), f"loading.{benchmark_id}.segments"),
            f"loading.{benchmark_id}.segments",
        )
        if len(states) != count or benchmark.get("required_state_count") != count:
            raise FractureBenchmarkContractError(f"{benchmark_id} prescribed state count changed")
        _require_close(states[-1], final_u, f"loading.{benchmark_id}.final_state")
        _require_close(benchmark.get("final_U_mm"), final_u, f"loading.{benchmark_id}.final_U_mm")
        if benchmark.get("reaction_component") != reaction or benchmark.get("top_y1") != top:
            raise FractureBenchmarkContractError(
                f"{benchmark_id} boundary/reaction convention changed"
            )
    if tuple(seen) != BENCHMARK_IDS:
        raise FractureBenchmarkContractError("benchmark order must be SENT then SENS")

    ell = _number(material["regularization_length_ell_mm"], "material.ell")
    mesh = _require_mapping(config["mesh"], "mesh")
    tiers = _require_sequence(mesh.get("tiers"), "mesh.tiers")
    expected_tiers = (("coarse", 0.5), ("medium", 0.25), ("fine", 0.125))
    if len(tiers) != 3:
        raise FractureBenchmarkContractError("mesh.tiers must contain three entries")
    for index, (raw, (tier_id, ratio)) in enumerate(zip(tiers, expected_tiers, strict=True)):
        tier = _require_mapping(raw, f"mesh.tiers[{index}]")
        if tier.get("id") != tier_id:
            raise FractureBenchmarkContractError("mesh tier order must be coarse, medium, fine")
        _require_close(
            tier.get("h_target_over_ell_in_refined_corridor"), ratio, f"mesh.{tier_id}.ratio"
        )
        h_target = ell * ratio
        _require_close(tier.get("h_target_mm"), h_target, f"mesh.{tier_id}.h_target_mm")
        _require_close(
            tier.get("bulk_h_target_mm"), min(4.0 * h_target, 0.04), f"mesh.{tier_id}.bulk"
        )

    compute = _require_mapping(config["compute_preflight"], "compute_preflight")
    if compute.get("unmeasured_DOF_scaling_exponent_allowed") is not False:
        raise FractureBenchmarkContractError("unmeasured DOF scaling exponents are forbidden")
    _require_close(
        compute.get("max_projected_single_medium_case_wall_hours"),
        12.0,
        "compute_preflight.single_medium_hours",
    )
    _require_close(
        compute.get("max_projected_all_six_cases_wall_hours"),
        72.0,
        "compute_preflight.all_six_hours",
    )
    preflight = _require_mapping(
        compute.get("coarse_subsampled_loading"), "compute_preflight.coarse_subsampled_loading"
    )
    for benchmark_id, expected_count in (("sent", 201), ("sens", 151)):
        entry = _require_mapping(preflight.get(benchmark_id), f"compute_preflight.{benchmark_id}")
        states = _segment_states(
            _require_sequence(entry.get("segments"), f"compute_preflight.{benchmark_id}.segments"),
            f"compute_preflight.{benchmark_id}.segments",
        )
        if len(states) != expected_count or entry.get("required_state_count") != expected_count:
            raise FractureBenchmarkContractError(
                f"compute preflight {benchmark_id} state count changed"
            )

    decision = _require_mapping(config["decision"], "decision")
    if decision.get("go_family_code") != "READY_FOR_PHASE1_PILOT":
        raise FractureBenchmarkContractError("passing route is only READY_FOR_PHASE1_PILOT")
    if decision.get("go_is_not_paper_or_field_GO") is not True:
        raise FractureBenchmarkContractError("benchmark readiness must not become a paper GO")


def validate_fracture_sent_sens_config(config: Mapping[str, Any]) -> None:
    """Fail closed unless *config* is the exact frozen validate-only protocol."""

    root = _require_mapping(config, "config")
    _walk_json(root)
    _validate_semantics(root)
    digest = _canonical_sha256(root)
    if digest != FROZEN_CANONICAL_SHA256:
        raise FractureBenchmarkContractError(
            "config differs from frozen canonical SENT/SENS v1; create a new protocol version "
            f"instead (sha256={digest})"
        )


def load_fracture_sent_sens_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load UTF-8 JSON and validate the frozen SENT/SENS development protocol."""

    source = default_fracture_sent_sens_config_path() if path is None else Path(path)
    try:
        with source.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise FractureBenchmarkContractError(
            f"could not load SENT/SENS config {source}: {exc}"
        ) from exc
    config = _require_mapping(value, "config")
    validate_fracture_sent_sens_config(config)
    return dict(config)


def prescribed_displacements(
    config: Mapping[str, Any], benchmark_id: str, *, compute_preflight: bool = False
) -> tuple[float, ...]:
    """Enumerate one frozen displacement grid without running a solver."""

    validate_fracture_sent_sens_config(config)
    if benchmark_id not in BENCHMARK_IDS:
        raise FractureBenchmarkContractError(
            f"benchmark_id must be one of {BENCHMARK_IDS}, got {benchmark_id!r}"
        )
    if compute_preflight:
        source = _require_mapping(
            _require_mapping(config["compute_preflight"], "compute_preflight")[
                "coarse_subsampled_loading"
            ],
            "compute_preflight.coarse_subsampled_loading",
        )[benchmark_id]
    else:
        source = next(
            entry
            for entry in _require_sequence(config["loading"]["benchmarks"], "loading.benchmarks")
            if entry["id"] == benchmark_id
        )
    return _segment_states(source["segments"], f"{benchmark_id}.segments")


def enumerate_fracture_benchmark_cases(
    config: Mapping[str, Any],
) -> tuple[FractureBenchmarkCase, ...]:
    """Enumerate SENT/SENS x coarse/medium/fine in frozen canonical order."""

    validate_fracture_sent_sens_config(config)
    benchmarks = _require_sequence(config["loading"]["benchmarks"], "loading.benchmarks")
    tiers = _require_sequence(config["mesh"]["tiers"], "mesh.tiers")
    cases = tuple(
        FractureBenchmarkCase(
            case_id=f"fss1-{benchmark['id']}-{tier['id']}",
            benchmark_id=str(benchmark["id"]),
            benchmark_name=str(benchmark["name"]),
            benchmark_index=benchmark_index,
            mesh_tier=str(tier["id"]),
            mesh_tier_index=tier_index,
            h_target_over_ell=float(tier["h_target_over_ell_in_refined_corridor"]),
            h_target_mm=float(tier["h_target_mm"]),
            bulk_h_target_mm=float(tier["bulk_h_target_mm"]),
            final_displacement_mm=float(benchmark["final_U_mm"]),
            required_state_count=int(benchmark["required_state_count"]),
            reaction_component=str(benchmark["reaction_component"]),
        )
        for benchmark_index, benchmark in enumerate(benchmarks)
        for tier_index, tier in enumerate(tiers)
    )
    if len(cases) != CASE_COUNT or len({case.case_id for case in cases}) != CASE_COUNT:
        raise FractureBenchmarkContractError("case enumeration did not produce six identities")
    return cases


__all__ = [
    "BENCHMARK_IDS",
    "CASE_COUNT",
    "FROZEN_CANONICAL_SHA256",
    "MESH_TIERS",
    "PROTOCOL_ID",
    "SCHEMA_VERSION",
    "FractureBenchmarkCase",
    "FractureBenchmarkContractError",
    "default_fracture_sent_sens_config_path",
    "enumerate_fracture_benchmark_cases",
    "load_fracture_sent_sens_config",
    "prescribed_displacements",
    "validate_fracture_sent_sens_config",
]
