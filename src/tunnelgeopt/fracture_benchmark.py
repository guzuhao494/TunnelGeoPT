"""Development adapter and bounded timing probe for SENT/SENS coupons.

This module connects the frozen protocol, an audited benchmark mesh, and the
generic prescribed-displacement BVP.  It deliberately does not implement a
formal SENT/SENS trajectory or claim a reproduction of Miehe et al.  The
built-in probe keeps damage fixed at zero; its timing is therefore a labelled,
non-authorizing lower bound for the coupled fracture calculation.

Coordinates and displacement DOFs are frozen as ``[y, z]`` and node-major
``[u_y, u_z]``.  Top/bottom are selected from physical labels, never inferred
from the second coordinate column.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .fracture import AT2Material, FractureSolverOptions
from .fracture_benchmark_validation import (
    FROZEN_CANONICAL_SHA256,
    prescribed_displacements,
    validate_fracture_sent_sens_config,
)
from .fracture_bvp import (
    FixedDamageDisplacementBVPResult,
    PrescribedDisplacementState,
    prescribed_displacement_mesh_identity,
    solve_fixed_damage_displacement_bvp,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

PROBE_SCHEMA = "tunnelgeopt.fracture.sent_sens.intact_probe.v1"
_COORDINATE_TOLERANCE_MM = 2.0e-12
_FORCE_FLOOR_KN = 1.0e-15
_MOMENT_FLOOR_KN_MM = 1.0e-15
_ENERGY_FLOOR_KN_MM = 1.0e-18


class FractureBenchmarkPreflightError(ValueError):
    """Raised before solving when protocol, mesh, or loading identities differ."""


@dataclass(frozen=True)
class FractureBenchmarkPreflight:
    benchmark_id: str
    tier: str
    protocol_sha256: str
    mesh_plan_sha256: str
    mesh_topology_sha256: str
    bvp_mesh_sha256: str
    node_count: int
    element_count: int
    top_node_count: int
    bottom_node_count: int


@dataclass(frozen=True)
class ProbeStep:
    sequence_index: int
    prescribed_U_mm: float
    wall_seconds: float
    converged: bool
    generalized_load_kN: float
    elastic_energy_kN_mm: float
    equilibrium_relative_residual: float
    global_force_relative_imbalance: float
    global_moment_relative_imbalance: float
    path_energy_relative_imbalance: float
    damage_component_status: str


@dataclass(frozen=True)
class FractureBenchmarkProbe:
    schema: str
    status: str
    claim_boundary: str
    benchmark_id: str
    tier: str
    protocol_sha256: str
    mesh_plan_sha256: str
    mesh_topology_sha256: str
    bvp_mesh_sha256: str
    material: Mapping[str, float]
    prescribed_U_mm: tuple[float, ...]
    steps: tuple[ProbeStep, ...]
    median_step_wall_seconds: float
    projected_formal_increment_count: int
    projected_formal_case_wall_hours: float
    projection_interpretation: str
    authorizes_medium_fine_or_formal_run: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "status": self.status,
            "claim_boundary": self.claim_boundary,
            "benchmark_id": self.benchmark_id,
            "tier": self.tier,
            "protocol_sha256": self.protocol_sha256,
            "mesh_plan_sha256": self.mesh_plan_sha256,
            "mesh_topology_sha256": self.mesh_topology_sha256,
            "bvp_mesh_sha256": self.bvp_mesh_sha256,
            "material": dict(self.material),
            "prescribed_U_mm": list(self.prescribed_U_mm),
            "steps": [asdict(step) for step in self.steps],
            "median_step_wall_seconds": self.median_step_wall_seconds,
            "projected_formal_increment_count": self.projected_formal_increment_count,
            "projected_formal_case_wall_hours": self.projected_formal_case_wall_hours,
            "projection_interpretation": self.projection_interpretation,
            "authorizes_medium_fine_or_formal_run": (self.authorizes_medium_fine_or_formal_run),
        }


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FractureBenchmarkPreflightError(f"{name} must be a mapping")
    return value


def _benchmark_entry(config: Mapping[str, Any], benchmark_id: str) -> Mapping[str, Any]:
    entries = config["loading"]["benchmarks"]
    try:
        return next(entry for entry in entries if entry["id"] == benchmark_id)
    except StopIteration as exc:
        raise FractureBenchmarkPreflightError(f"unknown benchmark {benchmark_id!r}") from exc


def _tier_entry(config: Mapping[str, Any], tier: str) -> Mapping[str, Any]:
    try:
        return next(entry for entry in config["mesh"]["tiers"] if entry["id"] == tier)
    except StopIteration as exc:
        raise FractureBenchmarkPreflightError(f"unknown mesh tier {tier!r}") from exc


def lame_to_young_poisson(lame_lambda: float, shear_modulus: float) -> tuple[float, float]:
    """Convert 3-D isotropic Lame parameters to ``(E, nu)`` and regress back."""

    lam = float(lame_lambda)
    mu = float(shear_modulus)
    if not math.isfinite(lam) or not math.isfinite(mu) or mu <= 0.0 or 3.0 * lam + 2.0 * mu <= 0.0:
        raise ValueError("Lame lambda and shear modulus must define a stable isotropic material")
    denominator = lam + mu
    if denominator == 0.0:
        raise ValueError("lambda + mu must be nonzero")
    young = mu * (3.0 * lam + 2.0 * mu) / denominator
    poisson = lam / (2.0 * denominator)
    material = AT2Material(young, poisson, 1.0, 1.0)
    if not math.isclose(material.lame_lambda, lam, rel_tol=2.0e-14, abs_tol=1.0e-14):
        raise RuntimeError("lambda -> (E, nu) -> lambda regression failed")
    if not math.isclose(material.shear_modulus, mu, rel_tol=2.0e-14, abs_tol=1.0e-14):
        raise RuntimeError("mu -> (E, nu) -> mu regression failed")
    return young, poisson


def benchmark_material(config: Mapping[str, Any]) -> AT2Material:
    """Build the exact AT2 material encoded by the frozen protocol."""

    validate_fracture_sent_sens_config(config)
    values = config["material"]
    young, poisson = lame_to_young_poisson(
        values["lame_lambda_kN_per_mm2"], values["shear_modulus_kN_per_mm2"]
    )
    return AT2Material(
        young_modulus=young,
        poisson_ratio=poisson,
        fracture_toughness=float(values["critical_fracture_energy_kN_per_mm"]),
        length_scale=float(values["regularization_length_ell_mm"]),
        residual_stiffness=float(config["fracture_model"]["residual_stiffness_k"]),
    )


def _facet_nodes(benchmark_mesh: Any, label: str) -> IntArray:
    facets_by_label = _require_mapping(benchmark_mesh.boundary_facets, "mesh.boundary_facets")
    if label not in facets_by_label:
        raise FractureBenchmarkPreflightError(f"mesh is missing facet label {label!r}")
    facets = np.asarray(facets_by_label[label], dtype=np.int64)
    mesh_facets = np.asarray(benchmark_mesh.mesh.facets, dtype=np.int64)
    if (
        facets.ndim != 1
        or facets.size == 0
        or np.any(facets < 0)
        or np.any(facets >= mesh_facets.shape[1])
    ):
        raise FractureBenchmarkPreflightError(f"facet label {label!r} is empty or invalid")
    return np.unique(mesh_facets[:, facets])


def _close(actual: Any, expected: Any, name: str) -> None:
    if not np.allclose(
        np.asarray(actual, dtype=float), np.asarray(expected, dtype=float), rtol=0.0, atol=1.0e-14
    ):
        raise FractureBenchmarkPreflightError(f"{name} differs from frozen config")


def preflight_fracture_benchmark(
    config: Mapping[str, Any], benchmark_mesh: Any, *, benchmark_id: str, tier: str
) -> FractureBenchmarkPreflight:
    """Validate config/mesh/loading identity without invoking either BVP solver."""

    validate_fracture_sent_sens_config(config)
    benchmark = _benchmark_entry(config, benchmark_id)
    mesh_tier = _tier_entry(config, tier)
    plan = benchmark_mesh.plan
    if plan.loading != benchmark_id or plan.tier != tier:
        raise FractureBenchmarkPreflightError("mesh loading/tier differs from requested case")
    _close(plan.target_h_mm, mesh_tier["h_target_mm"], "mesh target_h_mm")
    _close(plan.farfield_h_mm, mesh_tier["bulk_h_target_mm"], "mesh farfield_h_mm")
    corridor = benchmark["refined_corridor"]
    notch_line = config["geometry"]["notch"]["line_mm"]
    expected_notch = (
        (float(notch_line["y"]), float(notch_line["z"][0])),
        (float(notch_line["y"]), float(notch_line["z"][1])),
    )
    _close(plan.notch_polyline_yz_mm, expected_notch, "mesh notch polyline")
    _close(
        plan.notch_band_half_width_mm,
        corridor["notch_face_and_tip_refinement_distance_mm"],
        "mesh notch refinement distance",
    )
    _close(
        plan.propagation_corridor_polyline_yz_mm,
        corridor["centerline_yz_mm"],
        "mesh propagation corridor",
    )
    _close(
        plan.propagation_corridor_half_width_mm,
        corridor["half_width_mm"],
        "mesh propagation corridor half width",
    )

    identity = _require_mapping(benchmark_mesh.identity, "mesh.identity")
    metadata = _require_mapping(benchmark_mesh.metadata, "mesh.metadata")
    if tuple(identity.get("coordinate_order", ())) != ("y", "z"):
        raise FractureBenchmarkPreflightError("mesh coordinate order must be [y,z]")
    if identity.get("plan_sha256") != plan.plan_sha256:
        raise FractureBenchmarkPreflightError("mesh contains stale plan identity")
    topology_sha = identity.get("topology_sha256")
    if not isinstance(topology_sha, str) or len(topology_sha) != 64:
        raise FractureBenchmarkPreflightError("mesh topology identity is missing")
    recompute_topology = getattr(benchmark_mesh, "recompute_topology_sha256", None)
    if not callable(recompute_topology):
        raise FractureBenchmarkPreflightError("mesh must expose live topology-hash recomputation")
    if recompute_topology() != topology_sha:
        raise FractureBenchmarkPreflightError("mesh topology differs from stored identity")
    for audit in (
        "topology_audit_passed",
        "boundary_coverage_audit_passed",
        "zero_width_double_face_slit_audit_passed",
        "corridor_hmax_audit_passed",
    ):
        if metadata.get(audit) is not True:
            raise FractureBenchmarkPreflightError(f"mesh audit {audit!r} did not pass")

    expected_labels = tuple(config["geometry"]["boundary_labels"])
    point_labels = {"notch_tip"}
    facet_labels = set(expected_labels) - point_labels
    if set(benchmark_mesh.boundary_facets) != facet_labels:
        raise FractureBenchmarkPreflightError("mesh facet labels differ from frozen config")
    boundary_nodes = _require_mapping(
        getattr(benchmark_mesh, "boundary_nodes", None), "mesh.boundary_nodes"
    )
    if set(boundary_nodes) != point_labels:
        raise FractureBenchmarkPreflightError("mesh point labels differ from frozen config")
    tip_nodes = np.asarray(boundary_nodes["notch_tip"], dtype=np.int64)
    if tip_nodes.shape != (1,):
        raise FractureBenchmarkPreflightError("notch_tip must identify exactly one node")

    nodes = np.asarray(benchmark_mesh.nodes, dtype=np.float64)
    if nodes.ndim != 2 or nodes.shape[1] != 2 or not np.isfinite(nodes).all():
        raise FractureBenchmarkPreflightError("mesh nodes must be finite [N,2] [y,z]")
    _close(nodes[tip_nodes[0]], expected_notch[-1], "notch tip coordinate")
    top_nodes = _facet_nodes(benchmark_mesh, "top")
    bottom_nodes = _facet_nodes(benchmark_mesh, "bottom")
    if not np.allclose(nodes[top_nodes, 0], 1.0, rtol=0.0, atol=_COORDINATE_TOLERANCE_MM):
        raise FractureBenchmarkPreflightError("top label is not y=1; possible [z,y] swap")
    if not np.allclose(nodes[bottom_nodes, 0], 0.0, rtol=0.0, atol=_COORDINATE_TOLERANCE_MM):
        raise FractureBenchmarkPreflightError("bottom label is not y=0; possible [z,y] swap")
    if np.intersect1d(top_nodes, bottom_nodes).size:
        raise FractureBenchmarkPreflightError("top and bottom node sets overlap")

    current_bvp_sha = prescribed_displacement_mesh_identity(benchmark_mesh.mesh)
    return FractureBenchmarkPreflight(
        benchmark_id=benchmark_id,
        tier=tier,
        protocol_sha256=FROZEN_CANONICAL_SHA256,
        mesh_plan_sha256=plan.plan_sha256,
        mesh_topology_sha256=topology_sha,
        bvp_mesh_sha256=current_bvp_sha,
        node_count=int(nodes.shape[0]),
        element_count=int(np.asarray(benchmark_mesh.elements).shape[0]),
        top_node_count=int(top_nodes.size),
        bottom_node_count=int(bottom_nodes.size),
    )


def build_prescribed_displacement_states(
    config: Mapping[str, Any],
    benchmark_mesh: Any,
    *,
    benchmark_id: str,
    tier: str,
    displacements_mm: Sequence[float],
) -> tuple[PrescribedDisplacementState, ...]:
    """Create an exact, immutable node-major path after strict preflight."""

    preflight = preflight_fracture_benchmark(
        config, benchmark_mesh, benchmark_id=benchmark_id, tier=tier
    )
    values = np.asarray(displacements_mm, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("displacements_mm must be a finite non-empty vector")
    if values[0] != 0.0 or (values.size > 1 and np.any(np.diff(values) <= 0.0)):
        raise ValueError("probe path must start at U=0 and then be strictly increasing")
    formal_grid = prescribed_displacements(config, benchmark_id)
    if values[-1] > formal_grid[-1]:
        raise ValueError("probe displacement exceeds the frozen formal endpoint")

    nodes = np.asarray(benchmark_mesh.nodes)
    top_nodes = _facet_nodes(benchmark_mesh, "top")
    bottom_nodes = _facet_nodes(benchmark_mesh, "bottom")
    bottom_y = np.sort(2 * bottom_nodes)
    bottom_z = np.sort(2 * bottom_nodes + 1)
    top_y = np.sort(2 * top_nodes)
    top_z = np.sort(2 * top_nodes + 1)
    constrained = np.unique(np.concatenate((bottom_y, bottom_z, top_y, top_z)))
    groups = {"bottom_u_y": bottom_y, "bottom_u_z": bottom_z, "top_u_y": top_y, "top_u_z": top_z}
    driven = "top_u_y" if benchmark_id == "sent" else "top_u_z"
    states: list[PrescribedDisplacementState] = []
    for index, displacement in enumerate(values):
        prescribed = np.zeros(constrained.size, dtype=np.float64)
        prescribed[np.isin(constrained, groups[driven])] = displacement
        payload = {
            "schema": "tunnelgeopt.fracture.benchmark.state.v1",
            "protocol_sha256": preflight.protocol_sha256,
            "mesh_sha256": preflight.bvp_mesh_sha256,
            "benchmark_id": benchmark_id,
            "tier": tier,
            "sequence_index": index,
            "U_mm": float(displacement),
            "dirichlet_dofs": constrained.tolist(),
            "dirichlet_values": prescribed.tolist(),
            "driven_group": driven,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        states.append(
            PrescribedDisplacementState(
                identity=f"fss1-{benchmark_id}-{tier}-{index:04d}-{digest[:16]}",
                mesh_identity=preflight.bvp_mesh_sha256,
                sequence_index=index,
                path_parameter=float(displacement),
                dirichlet_dofs=constrained,
                dirichlet_values=prescribed,
                external_force=np.zeros(2 * nodes.shape[0]),
                reaction_groups=groups,
                driven_group=driven,
            )
        )
    return tuple(states)


def _relative_global_balances(
    result: FixedDamageDisplacementBVPResult, nodes: FloatArray
) -> tuple[float, float]:
    # Only constrained residual entries are physical support reactions.  Free
    # residuals belong to the independent equilibrium-residual gate.
    support = np.zeros(2 * nodes.shape[0], dtype=np.float64)
    support[result.dirichlet_dofs] = np.asarray(result.reaction)[result.dirichlet_dofs]
    support = support.reshape((-1, 2))
    applied = np.asarray(result.external_force, dtype=np.float64).reshape((-1, 2))
    resultant = support + applied
    force_numerator = float(np.linalg.norm(resultant.sum(axis=0)))
    force_denominator = max(
        float(np.linalg.norm(support, axis=1).sum() + np.linalg.norm(applied, axis=1).sum()),
        _FORCE_FLOOR_KN,
    )
    # Out-of-plane moment about frozen origin (y,z)=(0,0): M_x = y*F_z-z*F_y.
    support_moment = nodes[:, 0] * support[:, 1] - nodes[:, 1] * support[:, 0]
    applied_moment = nodes[:, 0] * applied[:, 1] - nodes[:, 1] * applied[:, 0]
    moment_numerator = abs(float((support_moment + applied_moment).sum()))
    moment_denominator = max(
        float(np.abs(support_moment).sum() + np.abs(applied_moment).sum()),
        _MOMENT_FLOOR_KN_MM,
    )
    return force_numerator / force_denominator, moment_numerator / moment_denominator


def run_intact_fracture_benchmark_probe(
    config: Mapping[str, Any],
    benchmark_mesh: Any,
    *,
    benchmark_id: str,
    tier: str,
    displacements_mm: Sequence[float],
    options: FractureSolverOptions | None = None,
    step_solver: Callable[
        ..., FixedDamageDisplacementBVPResult
    ] = solve_fixed_damage_displacement_bvp,
    clock: Callable[[], float] = time.perf_counter,
) -> FractureBenchmarkProbe:
    """Run a bounded fixed-``d=0`` probe; never authorize a formal computation."""

    if len(displacements_mm) > 12:
        raise ValueError("development probe is capped at 12 explicit states")
    before = preflight_fracture_benchmark(
        config, benchmark_mesh, benchmark_id=benchmark_id, tier=tier
    )
    states = build_prescribed_displacement_states(
        config,
        benchmark_mesh,
        benchmark_id=benchmark_id,
        tier=tier,
        displacements_mm=displacements_mm,
    )
    material = benchmark_material(config)
    nodes = np.asarray(benchmark_mesh.nodes, dtype=np.float64)
    damage = np.zeros(nodes.shape[0], dtype=np.float64)
    controls = options or FractureSolverOptions()
    steps: list[ProbeStep] = []
    prior_reaction: FloatArray | None = None
    prior_u: FloatArray | None = None
    initial_energy: float | None = None
    path_work = 0.0
    initial_displacement: FloatArray | None = None
    for state in states:
        start = clock()
        result = step_solver(
            benchmark_mesh.mesh,
            material,
            state,
            damage=damage,
            initial_displacement=initial_displacement,
            options=controls,
        )
        elapsed = float(clock() - start)
        if not math.isfinite(elapsed) or elapsed < 0.0:
            raise RuntimeError("probe clock produced an invalid duration")
        if (
            result.state.identity != state.identity
            or result.mesh_identity != before.bvp_mesh_sha256
        ):
            raise RuntimeError("probe solver returned a mismatched state or mesh identity")
        if initial_energy is None:
            initial_energy = float(result.elastic_energy)
        current_u = np.asarray(result.displacement).ravel()[state.dirichlet_dofs]
        current_reaction = np.asarray(result.reaction)[state.dirichlet_dofs]
        if prior_reaction is not None and prior_u is not None:
            path_work += float(0.5 * (prior_reaction + current_reaction) @ (current_u - prior_u))
        energy_change = float(result.elastic_energy) - initial_energy
        energy_numerator = abs(energy_change - path_work)
        energy_denominator = max(abs(energy_change), abs(path_work), _ENERGY_FLOOR_KN_MM)
        force_balance, moment_balance = _relative_global_balances(result, nodes)
        steps.append(
            ProbeStep(
                sequence_index=state.sequence_index,
                prescribed_U_mm=state.path_parameter,
                wall_seconds=elapsed,
                converged=bool(result.converged),
                generalized_load_kN=float(result.generalized_load),
                elastic_energy_kN_mm=float(result.elastic_energy),
                equilibrium_relative_residual=float(result.equilibrium_residual),
                global_force_relative_imbalance=force_balance,
                global_moment_relative_imbalance=moment_balance,
                path_energy_relative_imbalance=energy_numerator / energy_denominator,
                damage_component_status="NOT_APPLICABLE_INTACT_D0_PROBE",
            )
        )
        prior_reaction = current_reaction.copy()
        prior_u = current_u.copy()
        initial_displacement = np.asarray(result.displacement).copy()

    after = preflight_fracture_benchmark(
        config, benchmark_mesh, benchmark_id=benchmark_id, tier=tier
    )
    if after != before:
        raise RuntimeError("mesh identity or metadata changed during the probe")
    durations = np.asarray([step.wall_seconds for step in steps], dtype=np.float64)
    median = float(np.median(durations))
    formal_increment_count = len(prescribed_displacements(config, benchmark_id)) - 1
    projection = formal_increment_count * median / 3600.0
    return FractureBenchmarkProbe(
        schema=PROBE_SCHEMA,
        status="DEVELOPMENT_INTACT_FIXED_DAMAGE_PROBE_ONLY",
        claim_boundary="not_Miehe_reproduction_not_coupled_timing_not_Phase1_ready",
        benchmark_id=benchmark_id,
        tier=tier,
        protocol_sha256=before.protocol_sha256,
        mesh_plan_sha256=before.mesh_plan_sha256,
        mesh_topology_sha256=before.mesh_topology_sha256,
        bvp_mesh_sha256=before.bvp_mesh_sha256,
        material=MappingProxyType(
            {
                "young_modulus_kN_per_mm2": material.young_modulus,
                "poisson_ratio": material.poisson_ratio,
                "fracture_toughness_kN_per_mm": material.fracture_toughness,
                "length_scale_mm": material.length_scale,
                "residual_stiffness": material.residual_stiffness,
            }
        ),
        prescribed_U_mm=tuple(float(value) for value in displacements_mm),
        steps=tuple(steps),
        median_step_wall_seconds=median,
        projected_formal_increment_count=formal_increment_count,
        projected_formal_case_wall_hours=projection,
        projection_interpretation="intact_fixed_damage_lower_bound_non_authorizing",
        authorizes_medium_fine_or_formal_run=False,
    )


def write_probe_artifact_atomic(probe: FractureBenchmarkProbe, destination: str | Path) -> str:
    """Atomically write canonical JSON and return its SHA-256 (no host paths stored)."""

    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            probe.as_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    digest = hashlib.sha256(payload).hexdigest()
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, target)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return digest


__all__ = [
    "PROBE_SCHEMA",
    "FractureBenchmarkPreflight",
    "FractureBenchmarkPreflightError",
    "FractureBenchmarkProbe",
    "ProbeStep",
    "benchmark_material",
    "build_prescribed_displacement_states",
    "lame_to_young_poisson",
    "preflight_fracture_benchmark",
    "run_intact_fracture_benchmark_probe",
    "write_probe_artifact_atomic",
]
