"""Frozen validation contract for the development-only fracture Phase-1 pilot.

This module validates and enumerates a protocol.  It does not solve the AT2
problem, serialize a trajectory, or make a fracture/rockburst result claim.
The contract deliberately rejects fields inherited from the older Stress-Lift
and dynamic pilot so those concepts cannot silently enter the 2-D quasi-static
experiment.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

SCHEMA_VERSION = "tunnelgeopt.fracture.phase1.v2"
PROTOCOL_ID = "fracture-phase1-development-pilot-v2"
SECTION_FAMILIES = ("circle", "horseshoe", "straight_wall_arch")
MATERIAL_LEVEL_IDS = ("m1", "m2", "m3")
LOAD_PATH_IDS = ("p1", "p2", "p3", "p4")
REQUIRED_OUTPUT_STEP_COUNT = 41

_EXPECTED_TOP_LEVEL = {
    "schema_version",
    "protocol_id",
    "status",
    "scope",
    "exclusions",
    "design",
    "geometry",
    "materials",
    "load_paths",
    "time_discretization",
    "fracture_model",
    "mesh",
    "solver",
    "quality_control",
    "identity",
}

# These are forbidden *field names*, not words in the human-readable exclusion
# list.  Normalization catches case, spaces and hyphens.  The list is purposely
# broader than the old pilot's exact spelling so a renamed legacy field fails
# closed instead of being ignored.
_FORBIDDEN_KEY_FRAGMENTS = (
    "stress_lift",
    "stresslift",
    "lifted_dynamic",
    "synthetic_dynamic",
    "reference_velocity",
    "reference_time",
    "compressional_wave_speed",
    "unloading_time_cp",
    "advance_length",
    "intermediate_stress",
    "velocity_over",
    "max_cfl",
    "joint",
    "roughness",
    "heterogeneity",
    "random_field",
    "microseismic",
    "acoustic_emission",
    "fragment_contact",
    "inertia",
    "three_dimensional",
)

_EXPECTED_GEOMETRY_PARAMETERS: Mapping[str, Mapping[str, float]] = MappingProxyType(
    {
        "circle": MappingProxyType({"axis_ratio": 1.0, "superellipse_exponent": 2.0}),
        "horseshoe": MappingProxyType(
            {
                "span_height_ratio": 0.82,
                "sidewall_height_ratio": 1.0,
                "crown_shape": 2.0,
            }
        ),
        "straight_wall_arch": MappingProxyType(
            {
                "span_height_ratio": 1.0,
                "springline_height_ratio": 0.2,
                "crown_rise_span": 0.8,
            }
        ),
    }
)

_EXPECTED_MATERIAL_LEVELS = (
    ("m1", 0.04, 0.000008),
    ("m2", 0.06, 0.000012),
    ("m3", 0.08, 0.000016),
)

_EXPECTED_PATH_NAMES = (
    "proportional_uniform_release",
    "stress_ratio_change_then_uniform_release",
    "principal_rotation_with_uniform_release",
    "crown_sidewalls_invert_staged_release",
)

# (s, sigma1/UCS, sigma3/sigma1, principal angle, release values).  The first
# three paths use one uniform value; P4 uses crown/right/invert/left.
_EXPECTED_PATH_KNOTS: tuple[tuple[tuple[Any, ...], ...], ...] = (
    (
        (0.0, 0.45, 0.70, 0.0, (0.0,)),
        (0.25, 0.45, 0.70, 0.0, (0.25,)),
        (0.50, 0.45, 0.70, 0.0, (0.50,)),
        (0.75, 0.45, 0.70, 0.0, (0.75,)),
        (1.0, 0.45, 0.70, 0.0, (1.0,)),
    ),
    (
        (0.0, 0.45, 0.80, 0.0, (0.0,)),
        (0.25, 0.55, 0.80, 0.0, (0.15,)),
        (0.50, 0.65, 0.55, 0.0, (0.40,)),
        (0.75, 0.65, 0.35, 0.0, (0.70,)),
        (1.0, 0.65, 0.35, 0.0, (1.0,)),
    ),
    (
        (0.0, 0.55, 0.55, -30.0, (0.0,)),
        (0.25, 0.55, 0.55, -30.0, (0.20,)),
        (0.50, 0.55, 0.55, 0.0, (0.45,)),
        (0.75, 0.55, 0.55, 30.0, (0.70,)),
        (1.0, 0.55, 0.55, 30.0, (1.0,)),
    ),
    (
        (0.0, 0.55, 0.45, 15.0, (0.0, 0.0, 0.0, 0.0)),
        (0.25, 0.55, 0.45, 15.0, (0.75, 0.0, 0.0, 0.0)),
        (0.50, 0.55, 0.45, 15.0, (1.0, 0.5, 0.0, 0.5)),
        (0.75, 0.55, 0.45, 15.0, (1.0, 1.0, 0.5, 1.0)),
        (1.0, 0.55, 0.45, 15.0, (1.0, 1.0, 1.0, 1.0)),
    ),
)

_UNIFORM_RELEASE_KEYS = ("all",)
_STAGED_RELEASE_KEYS = ("crown", "right_sidewall", "invert", "left_sidewall")


class FracturePhase1ContractError(ValueError):
    """Raised when configuration, identity, or trajectory QC breaks Phase 1."""


@dataclass(frozen=True)
class FracturePhase1Case:
    """One member of the frozen 3 x 3 x 4 development cross-product."""

    case_id: str
    section_family: str
    material_level_id: str
    load_path_id: str
    section_index: int
    material_index: int
    load_path_index: int
    ultrafine_audit: bool
    primary_mesh_tier: str = "fine"

    def as_dict(self) -> dict[str, str | int | bool]:
        return {
            "case_id": self.case_id,
            "section_family": self.section_family,
            "material_level_id": self.material_level_id,
            "load_path_id": self.load_path_id,
            "section_index": self.section_index,
            "material_index": self.material_index,
            "load_path_index": self.load_path_index,
            "ultrafine_audit": self.ultrafine_audit,
            "primary_mesh_tier": self.primary_mesh_tier,
        }


@dataclass(frozen=True)
class TrajectoryQCResult:
    """Outcome of the frozen per-trajectory gates; failure never changes identity."""

    case_id: str
    passed: bool
    checks: Mapping[str, bool]
    failed_checks: tuple[str, ...]
    ultrafine_audit: bool
    replacement_allowed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "checks", MappingProxyType(dict(self.checks)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "passed": self.passed,
            "checks": dict(self.checks),
            "failed_checks": list(self.failed_checks),
            "ultrafine_audit": self.ultrafine_audit,
            "replacement_allowed": self.replacement_allowed,
        }


def default_fracture_phase1_config_path() -> Path:
    """Return the repository's versioned Phase-1 protocol path."""

    return Path(__file__).resolve().parents[2] / "configs" / "fracture_phase1_pilot.json"


def _normalise_key(value: object) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _walk_and_validate_json(value: Any, path: str = "config") -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            if not isinstance(raw_key, str) or not raw_key:
                raise FracturePhase1ContractError(f"{path} contains a non-string or empty key")
            key = _normalise_key(raw_key)
            if (
                key == "cfl"
                or key == "dynamic"
                or key == "3d"
                or any(fragment in key for fragment in _FORBIDDEN_KEY_FRAGMENTS)
            ):
                raise FracturePhase1ContractError(
                    f"forbidden legacy, 3-D, or dynamic field at {path}.{raw_key}"
                )
            _walk_and_validate_json(child, f"{path}.{raw_key}")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _walk_and_validate_json(child, f"{path}[{index}]")
        return
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise FracturePhase1ContractError(f"{path} must be finite")
        return
    raise FracturePhase1ContractError(f"{path} is not a JSON value")


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FracturePhase1ContractError(f"{path} must be an object")
    return value


def _sequence(value: Any, path: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise FracturePhase1ContractError(f"{path} must be an array")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise FracturePhase1ContractError(
            f"{path} keys differ from the frozen contract; missing={missing}, extra={extra}"
        )


def _require_equal(actual: Any, expected: Any, path: str) -> None:
    if actual != expected:
        raise FracturePhase1ContractError(
            f"{path} must equal frozen value {expected!r}, got {actual!r}"
        )


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FracturePhase1ContractError(f"{path} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise FracturePhase1ContractError(f"{path} must be finite")
    return result


def _require_close(actual: Any, expected: float, path: str, *, atol: float = 1e-15) -> None:
    value = _number(actual, path)
    if not math.isclose(value, expected, rel_tol=1e-12, abs_tol=atol):
        raise FracturePhase1ContractError(
            f"{path} must equal frozen value {expected!r}, got {value!r}"
        )


def _require_bool(value: Any, expected: bool, path: str) -> None:
    if not isinstance(value, bool) or value is not expected:
        raise FracturePhase1ContractError(f"{path} must be {expected}")


def _validate_status_and_scope(config: Mapping[str, Any]) -> None:
    status = _mapping(config["status"], "status")
    _require_exact_keys(
        status,
        {
            "state",
            "development_only",
            "solver_implemented",
            "labels_generated",
            "formal_or_locked_use_allowed",
        },
        "status",
    )
    _require_equal(status["state"], "frozen_development_only_not_run", "status.state")
    _require_bool(status["development_only"], True, "status.development_only")
    _require_bool(status["solver_implemented"], False, "status.solver_implemented")
    _require_bool(status["labels_generated"], False, "status.labels_generated")
    _require_bool(
        status["formal_or_locked_use_allowed"], False, "status.formal_or_locked_use_allowed"
    )

    scope = _mapping(config["scope"], "scope")
    _require_exact_keys(
        scope,
        {
            "spatial_dimension",
            "kinematics",
            "regime",
            "material_class",
            "fracture_model",
            "claim_boundary",
        },
        "scope",
    )
    _require_equal(scope["spatial_dimension"], 2, "scope.spatial_dimension")
    _require_equal(scope["kinematics"], "plane_strain", "scope.kinematics")
    _require_equal(scope["regime"], "quasi_static", "scope.regime")
    _require_equal(scope["material_class"], "homogeneous_isotropic_brittle", "scope.material_class")
    _require_equal(scope["fracture_model"], "AT2", "scope.fracture_model")
    _require_equal(
        scope["claim_boundary"],
        "synthetic_two_dimensional_quasi_static_brittle_fracture_development_only",
        "scope.claim_boundary",
    )

    exclusions = _sequence(config["exclusions"], "exclusions")
    required_exclusion_terms = (
        "stress-lift",
        "three-dimensional",
        "dynamic",
        "joints",
        "roughness",
        "random",
        "acoustic-emission",
        "contact",
        "plasticity",
        "engineering rockburst",
    )
    text = "\n".join(str(item).lower() for item in exclusions)
    missing = [term for term in required_exclusion_terms if term not in text]
    if missing:
        raise FracturePhase1ContractError(f"exclusions omit required Phase-1 boundaries: {missing}")


def _validate_design(config: Mapping[str, Any]) -> None:
    design = _mapping(config["design"], "design")
    _require_exact_keys(
        design,
        {
            "factorization",
            "section_families",
            "material_level_ids",
            "load_path_ids",
            "replicates_per_cell",
            "trajectory_count",
            "random_sampling",
            "result_conditioned_replacement_allowed",
            "ultrafine_audit",
        },
        "design",
    )
    _require_equal(design["factorization"], "full_cross_product", "design.factorization")
    _require_equal(tuple(design["section_families"]), SECTION_FAMILIES, "design.section_families")
    _require_equal(
        tuple(design["material_level_ids"]), MATERIAL_LEVEL_IDS, "design.material_level_ids"
    )
    _require_equal(tuple(design["load_path_ids"]), LOAD_PATH_IDS, "design.load_path_ids")
    _require_equal(design["replicates_per_cell"], 1, "design.replicates_per_cell")
    expected_count = len(SECTION_FAMILIES) * len(MATERIAL_LEVEL_IDS) * len(LOAD_PATH_IDS)
    _require_equal(design["trajectory_count"], expected_count, "design.trajectory_count")
    _require_bool(design["random_sampling"], False, "design.random_sampling")
    _require_bool(
        design["result_conditioned_replacement_allowed"],
        False,
        "design.result_conditioned_replacement_allowed",
    )

    audit = _mapping(design["ultrafine_audit"], "design.ultrafine_audit")
    _require_exact_keys(
        audit,
        {
            "count",
            "coverage",
            "material_selection_rule",
            "selection_before_solver_results",
            "replacement_allowed",
        },
        "design.ultrafine_audit",
    )
    _require_equal(audit["count"], 12, "design.ultrafine_audit.count")
    _require_equal(
        audit["coverage"],
        "one_case_for_every_section_family_x_load_path_cell",
        "design.ultrafine_audit.coverage",
    )
    _require_equal(
        audit["material_selection_rule"],
        "material_index_zero_based=(section_index_zero_based+path_index_zero_based)%3",
        "design.ultrafine_audit.material_selection_rule",
    )
    _require_bool(
        audit["selection_before_solver_results"],
        True,
        "design.ultrafine_audit.selection_before_solver_results",
    )
    _require_bool(audit["replacement_allowed"], False, "design.ultrafine_audit.replacement_allowed")


def _validate_geometry(config: Mapping[str, Any]) -> None:
    geometry = _mapping(config["geometry"], "geometry")
    _require_exact_keys(
        geometry,
        {
            "characteristic_radius_R",
            "boundary_points",
            "boundary_perturbation",
            "sections",
            "outer_domain",
        },
        "geometry",
    )
    _require_close(geometry["characteristic_radius_R"], 1.0, "geometry.characteristic_radius_R")
    _require_equal(geometry["boundary_points"], 256, "geometry.boundary_points")
    _require_equal(geometry["boundary_perturbation"], "none", "geometry.boundary_perturbation")
    sections = _sequence(geometry["sections"], "geometry.sections")
    if len(sections) != 3:
        raise FracturePhase1ContractError("geometry.sections must contain exactly three entries")
    for index, (entry, section_id) in enumerate(zip(sections, SECTION_FAMILIES, strict=True)):
        section = _mapping(entry, f"geometry.sections[{index}]")
        _require_exact_keys(section, {"id", "parameters"}, f"geometry.sections[{index}]")
        _require_equal(section["id"], section_id, f"geometry.sections[{index}].id")
        parameters = _mapping(section["parameters"], f"geometry.sections[{index}].parameters")
        expected = _EXPECTED_GEOMETRY_PARAMETERS[section_id]
        _require_exact_keys(parameters, set(expected), f"geometry.sections[{index}].parameters")
        for name, expected_value in expected.items():
            _require_close(
                parameters[name], expected_value, f"geometry.sections[{index}].parameters.{name}"
            )

    outer = _mapping(geometry["outer_domain"], "geometry.outer_domain")
    _require_exact_keys(
        outer,
        {"type", "bounds_over_R", "fixed_across_sections_materials_paths_and_meshes"},
        "geometry.outer_domain",
    )
    _require_equal(outer["type"], "centered_axis_aligned_square", "geometry.outer_domain.type")
    bounds = _mapping(outer["bounds_over_R"], "geometry.outer_domain.bounds_over_R")
    _require_exact_keys(bounds, {"y", "z"}, "geometry.outer_domain.bounds_over_R")
    for axis in ("y", "z"):
        values = _sequence(bounds[axis], f"geometry.outer_domain.bounds_over_R.{axis}")
        if len(values) != 2:
            raise FracturePhase1ContractError(
                f"geometry.outer_domain.bounds_over_R.{axis} must have two values"
            )
        _require_close(values[0], -8.0, f"geometry.outer_domain.bounds_over_R.{axis}[0]")
        _require_close(values[1], 8.0, f"geometry.outer_domain.bounds_over_R.{axis}[1]")
    _require_bool(
        outer["fixed_across_sections_materials_paths_and_meshes"],
        True,
        "geometry.outer_domain.fixed_across_sections_materials_paths_and_meshes",
    )


def _validate_materials(config: Mapping[str, Any]) -> None:
    materials = _mapping(config["materials"], "materials")
    _require_exact_keys(
        materials,
        {"stress_scale", "length_scale", "spatial_variation", "fixed", "coupling", "levels"},
        "materials",
    )
    _require_equal(materials["stress_scale"], "UCS", "materials.stress_scale")
    _require_equal(materials["length_scale"], "R", "materials.length_scale")
    _require_equal(materials["spatial_variation"], "none", "materials.spatial_variation")
    fixed = _mapping(materials["fixed"], "materials.fixed")
    _require_exact_keys(fixed, {"youngs_modulus_over_UCS", "poisson_ratio"}, "materials.fixed")
    e_over_ucs = _number(
        fixed["youngs_modulus_over_UCS"], "materials.fixed.youngs_modulus_over_UCS"
    )
    poisson = _number(fixed["poisson_ratio"], "materials.fixed.poisson_ratio")
    _require_close(e_over_ucs, 500.0, "materials.fixed.youngs_modulus_over_UCS")
    _require_close(poisson, 0.25, "materials.fixed.poisson_ratio")
    if not -1.0 < poisson < 0.5:
        raise FracturePhase1ContractError("plane-strain poisson_ratio must lie in (-1, 0.5)")

    coupling = _mapping(materials["coupling"], "materials.coupling")
    _require_exact_keys(
        coupling,
        {
            "rule",
            "Gc_over_UCS_ell",
            "AT2_homogeneous_peak_stress_formula",
            "implied_sigma_peak_over_UCS",
            "interpretation",
        },
        "materials.coupling",
    )
    _require_equal(
        coupling["rule"],
        "Gc_over_UCS_R=ell_over_R_times_Gc_over_UCS_ell",
        "materials.coupling.rule",
    )
    ratio = _number(coupling["Gc_over_UCS_ell"], "materials.coupling.Gc_over_UCS_ell")
    _require_close(ratio, 0.0002, "materials.coupling.Gc_over_UCS_ell")
    _require_equal(
        coupling["AT2_homogeneous_peak_stress_formula"],
        "sigma_peak_over_UCS=sqrt((27/256)*(E_over_UCS)*(Gc_over_UCS_ell))",
        "materials.coupling.AT2_homogeneous_peak_stress_formula",
    )
    implied = math.sqrt((27.0 / 256.0) * e_over_ucs * ratio)
    _require_close(
        coupling["implied_sigma_peak_over_UCS"],
        implied,
        "materials.coupling.implied_sigma_peak_over_UCS",
    )
    _require_equal(
        coupling["interpretation"],
        "regularization_sensitivity_at_fixed_dimensionless_AT2_strength_scale",
        "materials.coupling.interpretation",
    )

    levels = _sequence(materials["levels"], "materials.levels")
    if len(levels) != 3:
        raise FracturePhase1ContractError("materials.levels must contain exactly three entries")
    for index, (entry, expected) in enumerate(zip(levels, _EXPECTED_MATERIAL_LEVELS, strict=True)):
        level = _mapping(entry, f"materials.levels[{index}]")
        _require_exact_keys(
            level, {"id", "ell_over_R", "Gc_over_UCS_R"}, f"materials.levels[{index}]"
        )
        level_id, expected_ell, expected_gc = expected
        _require_equal(level["id"], level_id, f"materials.levels[{index}].id")
        ell = _number(level["ell_over_R"], f"materials.levels[{index}].ell_over_R")
        gc = _number(level["Gc_over_UCS_R"], f"materials.levels[{index}].Gc_over_UCS_R")
        _require_close(ell, expected_ell, f"materials.levels[{index}].ell_over_R")
        _require_close(gc, expected_gc, f"materials.levels[{index}].Gc_over_UCS_R")
        if ell <= 0.0 or gc <= 0.0:
            raise FracturePhase1ContractError("ell/R and Gc/(UCS R) must be positive")
        if not math.isclose(gc / ell, ratio, rel_tol=1e-12, abs_tol=1e-15):
            raise FracturePhase1ContractError(
                f"materials.levels[{index}] violates the frozen Gc-ell coupling"
            )


def _validate_load_paths(config: Mapping[str, Any]) -> None:
    load_paths = _mapping(config["load_paths"], "load_paths")
    _require_exact_keys(
        load_paths,
        {
            "stress_components",
            "path_parameter",
            "interpolation",
            "wall_release_definition",
            "wall_zones_for_p4",
            "paths",
        },
        "load_paths",
    )
    _require_equal(
        load_paths["stress_components"],
        "principal_compression_magnitudes_reported_positive_then_converted_to_solver_tension_positive_tensor",
        "load_paths.stress_components",
    )
    _require_equal(load_paths["path_parameter"], "s_in_[0,1]", "load_paths.path_parameter")
    _require_equal(
        load_paths["interpolation"],
        "piecewise_linear_between_control_knots",
        "load_paths.interpolation",
    )
    _require_equal(
        load_paths["wall_release_definition"],
        "lambda=0_supported_in_situ_wall_traction_lambda=1_fully_released",
        "load_paths.wall_release_definition",
    )
    zones = _mapping(load_paths["wall_zones_for_p4"], "load_paths.wall_zones_for_p4")
    _require_exact_keys(
        zones,
        {
            "coordinate_rule",
            "crown",
            "right_sidewall",
            "invert",
            "left_sidewall",
            "transition_blend_deg",
        },
        "load_paths.wall_zones_for_p4",
    )
    _require_equal(
        zones["coordinate_rule"],
        "polar_angle_about_boundary_centroid_measured_from_positive_z_toward_positive_y",
        "load_paths.wall_zones_for_p4.coordinate_rule",
    )
    for key, expected in {
        "crown": "[-45deg,45deg]",
        "right_sidewall": "(45deg,135deg)",
        "invert": "[135deg,225deg]",
        "left_sidewall": "(225deg,315deg)",
    }.items():
        _require_equal(zones[key], expected, f"load_paths.wall_zones_for_p4.{key}")
    _require_close(
        zones["transition_blend_deg"], 5.0, "load_paths.wall_zones_for_p4.transition_blend_deg"
    )

    paths = _sequence(load_paths["paths"], "load_paths.paths")
    if len(paths) != 4:
        raise FracturePhase1ContractError("load_paths.paths must contain exactly four histories")
    seen_histories: set[tuple[tuple[Any, ...], ...]] = set()
    for path_index, (entry, expected_id, expected_name, expected_knots) in enumerate(
        zip(paths, LOAD_PATH_IDS, _EXPECTED_PATH_NAMES, _EXPECTED_PATH_KNOTS, strict=True)
    ):
        path = _mapping(entry, f"load_paths.paths[{path_index}]")
        _require_exact_keys(
            path, {"id", "name", "control_knots"}, f"load_paths.paths[{path_index}]"
        )
        _require_equal(path["id"], expected_id, f"load_paths.paths[{path_index}].id")
        _require_equal(path["name"], expected_name, f"load_paths.paths[{path_index}].name")
        knots = _sequence(path["control_knots"], f"load_paths.paths[{path_index}].control_knots")
        if len(knots) != len(expected_knots):
            raise FracturePhase1ContractError(
                f"load_paths.paths[{path_index}] must contain five frozen control knots"
            )
        actual_history: list[tuple[Any, ...]] = []
        release_keys = _STAGED_RELEASE_KEYS if expected_id == "p4" else _UNIFORM_RELEASE_KEYS
        previous_release = [-math.inf] * len(release_keys)
        for knot_index, (entry_knot, expected) in enumerate(
            zip(knots, expected_knots, strict=True)
        ):
            knot_path = f"load_paths.paths[{path_index}].control_knots[{knot_index}]"
            knot = _mapping(entry_knot, knot_path)
            _require_exact_keys(
                knot,
                {
                    "s",
                    "sigma1_over_UCS",
                    "sigma3_over_sigma1",
                    "principal_angle_deg",
                    "wall_release",
                },
                knot_path,
            )
            values = [
                _number(knot["s"], f"{knot_path}.s"),
                _number(knot["sigma1_over_UCS"], f"{knot_path}.sigma1_over_UCS"),
                _number(knot["sigma3_over_sigma1"], f"{knot_path}.sigma3_over_sigma1"),
                _number(knot["principal_angle_deg"], f"{knot_path}.principal_angle_deg"),
            ]
            release = _mapping(knot["wall_release"], f"{knot_path}.wall_release")
            _require_exact_keys(release, set(release_keys), f"{knot_path}.wall_release")
            releases = [
                _number(release[key], f"{knot_path}.wall_release.{key}") for key in release_keys
            ]
            actual = (*values, tuple(releases))
            for item_index, (actual_value, expected_value) in enumerate(
                zip(values, expected[:4], strict=True)
            ):
                _require_close(actual_value, expected_value, f"{knot_path}[value={item_index}]")
            for release_index, (actual_value, expected_value) in enumerate(
                zip(releases, expected[4], strict=True)
            ):
                _require_close(
                    actual_value,
                    expected_value,
                    f"{knot_path}.wall_release.{release_keys[release_index]}",
                )
                if not 0.0 <= actual_value <= 1.0:
                    raise FracturePhase1ContractError("wall-release fractions must lie in [0, 1]")
                if actual_value < previous_release[release_index]:
                    raise FracturePhase1ContractError("every wall-zone release must be monotone")
                previous_release[release_index] = actual_value
            if values[1] <= 0.0 or not 0.0 < values[2] <= 1.0:
                raise FracturePhase1ContractError("principal compression and ratio are invalid")
            actual_history.append(actual)
        history_tuple = tuple(actual_history)
        if history_tuple in seen_histories:
            raise FracturePhase1ContractError(
                "the four load paths must be distinct actual histories"
            )
        seen_histories.add(history_tuple)


def _validate_discretization_model_mesh_solver(config: Mapping[str, Any]) -> None:
    time = _mapping(config["time_discretization"], "time_discretization")
    _require_exact_keys(
        time,
        {
            "required_output_s",
            "required_output_step_count",
            "adaptive_substeps_allowed_between_required_outputs",
            "store_every_accepted_internal_step",
        },
        "time_discretization",
    )
    output = _sequence(time["required_output_s"], "time_discretization.required_output_s")
    expected = tuple(index / 40.0 for index in range(REQUIRED_OUTPUT_STEP_COUNT))
    if len(output) != REQUIRED_OUTPUT_STEP_COUNT:
        raise FracturePhase1ContractError("required_output_s must contain exactly 41 states")
    for index, (actual, expected_value) in enumerate(zip(output, expected, strict=True)):
        _require_close(actual, expected_value, f"time_discretization.required_output_s[{index}]")
    _require_equal(
        time["required_output_step_count"],
        REQUIRED_OUTPUT_STEP_COUNT,
        "time_discretization.required_output_step_count",
    )
    _require_bool(
        time["adaptive_substeps_allowed_between_required_outputs"],
        True,
        "time_discretization.adaptive_substeps_allowed_between_required_outputs",
    )
    _require_bool(
        time["store_every_accepted_internal_step"],
        True,
        "time_discretization.store_every_accepted_internal_step",
    )

    model = _mapping(config["fracture_model"], "fracture_model")
    _require_exact_keys(
        model,
        {
            "damage_convention",
            "degradation",
            "residual_stiffness_k",
            "split",
            "history",
            "irreversibility",
            "displacement_element",
            "damage_element",
            "energy_and_history_integration",
            "unconstrained_solve_then_clipping_allowed",
        },
        "fracture_model",
    )
    frozen_model_values = {
        "damage_convention": "d=0_intact_d=1_fully_damaged",
        "degradation": "g(d)=(1-d)^2+k",
        "split": "three_dimensional_spectral_strain_split_evaluated_under_plane_strain_epsilon_xx_zero",
        "history": "integration_point_H_n=max(H_previous,psi_plus_current)",
        "irreversibility": "bound_constrained_d_n>=d_previous_and_0<=d_n<=1",
        "displacement_element": "P1",
        "damage_element": "P1",
        "energy_and_history_integration": "explicit_quadrature",
    }
    for key, value in frozen_model_values.items():
        _require_equal(model[key], value, f"fracture_model.{key}")
    _require_close(model["residual_stiffness_k"], 1e-8, "fracture_model.residual_stiffness_k")
    _require_bool(
        model["unconstrained_solve_then_clipping_allowed"],
        False,
        "fracture_model.unconstrained_solve_then_clipping_allowed",
    )

    mesh = _mapping(config["mesh"], "mesh")
    _require_exact_keys(
        mesh,
        {
            "backend",
            "element",
            "same_boundary_and_outer_domain_across_tiers",
            "potential_fracture_region",
            "tiers",
        },
        "mesh",
    )
    _require_equal(mesh["backend"], "gmsh", "mesh.backend")
    _require_equal(mesh["element"], "first_order_triangle", "mesh.element")
    _require_bool(
        mesh["same_boundary_and_outer_domain_across_tiers"],
        True,
        "mesh.same_boundary_and_outer_domain_across_tiers",
    )
    region = _mapping(mesh["potential_fracture_region"], "mesh.potential_fracture_region")
    _require_exact_keys(
        region,
        {
            "definition",
            "max_wall_distance_over_R",
            "trajectory_invalid_if_damage_reaches_outer_edge",
        },
        "mesh.potential_fracture_region",
    )
    _require_equal(
        region["definition"],
        "rock_points_with_wall_distance_at_most_2R",
        "mesh.potential_fracture_region.definition",
    )
    _require_close(
        region["max_wall_distance_over_R"],
        2.0,
        "mesh.potential_fracture_region.max_wall_distance_over_R",
    )
    _require_bool(
        region["trajectory_invalid_if_damage_reaches_outer_edge"],
        True,
        "mesh.potential_fracture_region.trajectory_invalid_if_damage_reaches_outer_edge",
    )
    tiers = _mapping(mesh["tiers"], "mesh.tiers")
    _require_exact_keys(tiers, {"fine", "ultrafine_audit"}, "mesh.tiers")
    for tier_name, h_over_ell, farfield_h, role in (
        ("fine", 0.25, 0.4, "all_36_primary_development_trajectories"),
        ("ultrafine_audit", 0.125, 0.25, "12_preselected_fine_to_ultrafine_audits"),
    ):
        tier = _mapping(tiers[tier_name], f"mesh.tiers.{tier_name}")
        _require_exact_keys(
            tier,
            {"role", "max_h_over_ell_in_potential_fracture_region", "farfield_h_over_R"},
            f"mesh.tiers.{tier_name}",
        )
        _require_equal(tier["role"], role, f"mesh.tiers.{tier_name}.role")
        _require_close(
            tier["max_h_over_ell_in_potential_fracture_region"],
            h_over_ell,
            f"mesh.tiers.{tier_name}.max_h_over_ell_in_potential_fracture_region",
        )
        _require_close(
            tier["farfield_h_over_R"], farfield_h, f"mesh.tiers.{tier_name}.farfield_h_over_R"
        )
    if (
        _number(
            tiers["fine"]["max_h_over_ell_in_potential_fracture_region"],
            "mesh.tiers.fine.max_h_over_ell",
        )
        > 0.25
    ):
        raise FracturePhase1ContractError("fine mesh must retain at least four elements per ell")

    solver = _mapping(config["solver"], "solver")
    expected_solver: dict[str, Any] = {
        "formulation": "total_field_affine_farfield_plus_correction",
        "nonlinear_scheme": "alternate_minimization_displacement_history_bound_constrained_damage",
        "damage_constraint_method": "primal_dual_active_set",
        "max_staggered_iterations": 100,
        "max_active_set_iterations": 100,
        "relative_displacement_increment_tolerance": 1e-8,
        "relative_damage_increment_tolerance": 1e-8,
        "relative_energy_increment_tolerance": 1e-8,
        "equilibrium_relative_residual_tolerance": 1e-6,
        "kkt_complementarity_relative_residual_tolerance": 1e-6,
        "step_retry_factor": 0.5,
        "max_step_retries_per_required_output_interval": 6,
        "minimum_s_increment": 0.000390625,
        "retry_exhaustion_action": "invalidate_trajectory_without_replacement",
        "final_unconverged_iterate_may_be_stored_as_label": False,
    }
    _require_exact_keys(solver, set(expected_solver), "solver")
    for key, expected_value in expected_solver.items():
        if isinstance(expected_value, float):
            _require_close(solver[key], expected_value, f"solver.{key}")
        elif isinstance(expected_value, bool):
            _require_bool(solver[key], expected_value, f"solver.{key}")
        else:
            _require_equal(solver[key], expected_value, f"solver.{key}")


def _validate_quality_and_identity(config: Mapping[str, Any]) -> None:
    quality = _mapping(config["quality_control"], "quality_control")
    _require_exact_keys(
        quality,
        {
            "per_trajectory",
            "fine_to_ultrafine",
            "solver_prerequisites_before_pilot",
            "failed_trajectory_policy",
            "minimum_pass_fraction",
        },
        "quality_control",
    )
    per = _mapping(quality["per_trajectory"], "quality_control.per_trajectory")
    expected_per: dict[str, Any] = {
        "required_output_step_count": 41,
        "max_nonfinite_fraction": 0.0,
        "max_equilibrium_relative_residual": 1e-6,
        "max_kkt_complementarity_relative_residual": 1e-6,
        "max_damage_irreversibility_violation": 1e-10,
        "max_damage_range_violation": 1e-10,
        "max_history_monotonicity_violation": 1e-10,
        "max_relative_energy_imbalance": 0.05,
        "min_incremental_dissipation_over_UCS_R2": -1e-10,
        "max_damage_on_refined_region_outer_edge": 0.0001,
        "require_all_steps_converged": True,
        "require_complete_failure_ledger": True,
        "require_zero_replacement_attempts": True,
    }
    _require_exact_keys(per, set(expected_per), "quality_control.per_trajectory")
    for key, expected_value in expected_per.items():
        if isinstance(expected_value, bool):
            _require_bool(per[key], expected_value, f"quality_control.per_trajectory.{key}")
        elif isinstance(expected_value, float):
            _require_close(per[key], expected_value, f"quality_control.per_trajectory.{key}")
        else:
            _require_equal(per[key], expected_value, f"quality_control.per_trajectory.{key}")

    audit = _mapping(quality["fine_to_ultrafine"], "quality_control.fine_to_ultrafine")
    _require_exact_keys(
        audit,
        {
            "max_peak_reaction_relative_change",
            "max_total_fracture_energy_relative_change",
            "both_metrics_required_for_every_preselected_audit",
        },
        "quality_control.fine_to_ultrafine",
    )
    _require_close(
        audit["max_peak_reaction_relative_change"],
        0.05,
        "quality_control.fine_to_ultrafine.max_peak_reaction_relative_change",
    )
    _require_close(
        audit["max_total_fracture_energy_relative_change"],
        0.05,
        "quality_control.fine_to_ultrafine.max_total_fracture_energy_relative_change",
    )
    _require_bool(
        audit["both_metrics_required_for_every_preselected_audit"],
        True,
        "quality_control.fine_to_ultrafine.both_metrics_required_for_every_preselected_audit",
    )
    prerequisites = _mapping(
        quality["solver_prerequisites_before_pilot"],
        "quality_control.solver_prerequisites_before_pilot",
    )
    expected_prerequisites: dict[str, Any] = {
        "intact_regression_max_relative_error": 1e-6,
        "require_rigid_body_zero_energy_test": True,
        "require_spectral_reconstruction_and_compression_tests": True,
        "require_tangent_directional_derivative_test": True,
        "require_constrained_damage_kkt_test": True,
        "require_pinned_MOOSE_crack2d_iso_reference_self_test": True,
        "require_local_vs_MOOSE_same_problem_cross_check": True,
        "require_single_edge_notch_tension_and_shear_three_grid_benchmarks": True,
    }
    _require_exact_keys(
        prerequisites,
        set(expected_prerequisites),
        "quality_control.solver_prerequisites_before_pilot",
    )
    for key, expected_value in expected_prerequisites.items():
        if isinstance(expected_value, bool):
            _require_bool(
                prerequisites[key],
                expected_value,
                f"quality_control.solver_prerequisites_before_pilot.{key}",
            )
        else:
            _require_close(
                prerequisites[key],
                expected_value,
                f"quality_control.solver_prerequisites_before_pilot.{key}",
            )
    _require_equal(
        quality["failed_trajectory_policy"],
        "retain_identity_and_failure_ledger_exclude_from_successful_labels_no_replacement",
        "quality_control.failed_trajectory_policy",
    )
    _require_close(quality["minimum_pass_fraction"], 1.0, "quality_control.minimum_pass_fraction")

    identity = _mapping(config["identity"], "identity")
    _require_exact_keys(
        identity,
        {
            "case_id_template",
            "canonical_order",
            "mesh_tier_excluded_from_case_identity",
            "attempt_or_restart_excluded_from_case_identity",
        },
        "identity",
    )
    _require_equal(
        identity["case_id_template"],
        "fp1-{section_family}-{material_level_id}-{load_path_id}",
        "identity.case_id_template",
    )
    _require_equal(
        identity["canonical_order"],
        "section_then_material_then_path_in_config_order",
        "identity.canonical_order",
    )
    _require_bool(
        identity["mesh_tier_excluded_from_case_identity"],
        True,
        "identity.mesh_tier_excluded_from_case_identity",
    )
    _require_bool(
        identity["attempt_or_restart_excluded_from_case_identity"],
        True,
        "identity.attempt_or_restart_excluded_from_case_identity",
    )


def validate_fracture_phase1_config(config: Mapping[str, Any]) -> None:
    """Fail closed unless ``config`` is the frozen 36-trajectory Phase-1 contract."""

    root = _mapping(config, "config")
    _walk_and_validate_json(root)
    _require_exact_keys(root, _EXPECTED_TOP_LEVEL, "config")
    _require_equal(root["schema_version"], SCHEMA_VERSION, "schema_version")
    _require_equal(root["protocol_id"], PROTOCOL_ID, "protocol_id")
    _validate_status_and_scope(root)
    _validate_design(root)
    _validate_geometry(root)
    _validate_materials(root)
    _validate_load_paths(root)
    _validate_discretization_model_mesh_solver(root)
    _validate_quality_and_identity(root)


def load_fracture_phase1_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load UTF-8 JSON and semantically validate the frozen Phase-1 protocol."""

    source = default_fracture_phase1_config_path() if path is None else Path(path)
    try:
        with source.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise FracturePhase1ContractError(f"could not load Phase-1 config {source}: {exc}") from exc
    validate_fracture_phase1_config(config)
    return config


def _case_id(section: str, material: str, path: str) -> str:
    return f"fp1-{section}-{material}-{path}"


def _is_ultrafine_selection(section_index: int, material_index: int, path_index: int) -> bool:
    return material_index == (section_index + path_index) % len(MATERIAL_LEVEL_IDS)


def enumerate_fracture_phase1_cases(
    config: Mapping[str, Any],
) -> tuple[FracturePhase1Case, ...]:
    """Enumerate all 36 identities in frozen section/material/path order."""

    validate_fracture_phase1_config(config)
    cases = tuple(
        FracturePhase1Case(
            case_id=_case_id(section, material, path),
            section_family=section,
            material_level_id=material,
            load_path_id=path,
            section_index=section_index,
            material_index=material_index,
            load_path_index=path_index,
            ultrafine_audit=_is_ultrafine_selection(section_index, material_index, path_index),
        )
        for section_index, section in enumerate(SECTION_FAMILIES)
        for material_index, material in enumerate(MATERIAL_LEVEL_IDS)
        for path_index, path in enumerate(LOAD_PATH_IDS)
    )
    if len(cases) != 36 or len({case.case_id for case in cases}) != 36:
        raise FracturePhase1ContractError("internal case enumeration did not produce 36 identities")
    return cases


def enumerate_ultrafine_audits(
    config: Mapping[str, Any],
) -> tuple[FracturePhase1Case, ...]:
    """Return the 12 pre-outcome Latin-cycled section-by-path audit cases."""

    cases = enumerate_fracture_phase1_cases(config)
    by_cell = {
        (case.section_family, case.load_path_id): case for case in cases if case.ultrafine_audit
    }
    audits = tuple(
        by_cell[(section, path)] for section in SECTION_FAMILIES for path in LOAD_PATH_IDS
    )
    if len(audits) != 12 or len({case.case_id for case in audits}) != 12:
        raise FracturePhase1ContractError("internal ultrafine selection did not produce 12 cases")
    if {case.material_level_id for case in audits} != set(MATERIAL_LEVEL_IDS):
        raise FracturePhase1ContractError("ultrafine Latin cycle omitted a material level")
    return audits


_REQUIRED_QC_METRICS = {
    "completed",
    "required_output_state_count",
    "stored_accepted_step_count",
    "required_output_s_complete",
    "nonfinite_fraction",
    "max_equilibrium_relative_residual",
    "max_kkt_complementarity_relative_residual",
    "max_damage_irreversibility_violation",
    "max_damage_range_violation",
    "max_history_monotonicity_violation",
    "max_relative_energy_imbalance",
    "min_incremental_dissipation_over_UCS_R2",
    "max_damage_on_refined_region_outer_edge",
    "all_steps_converged",
    "failure_ledger_complete",
    "retry_budget_exhausted",
    "replacement_attempts",
}


def _metric_bool(metrics: Mapping[str, Any], key: str) -> bool:
    value = metrics[key]
    if not isinstance(value, bool):
        raise FracturePhase1ContractError(f"trajectory metric {key!r} must be boolean")
    return value


def _metric_nonnegative(metrics: Mapping[str, Any], key: str) -> float:
    value = _number(metrics[key], f"trajectory_metrics.{key}")
    if value < 0.0:
        raise FracturePhase1ContractError(f"trajectory metric {key!r} must be non-negative")
    return value


def evaluate_trajectory_qc(
    config: Mapping[str, Any],
    case_id: str,
    metrics: Mapping[str, Any],
) -> TrajectoryQCResult:
    """Evaluate one frozen identity; a failed case remains failed without replacement.

    Audit cases additionally require fine-to-ultrafine peak-reaction and total
    fracture-energy changes.  The function only evaluates supplied diagnostics;
    it does not infer convergence from stored fields or rerun a solver.
    """

    validate_fracture_phase1_config(config)
    if not isinstance(case_id, str):
        raise FracturePhase1ContractError("case_id must be a string")
    cases = {case.case_id: case for case in enumerate_fracture_phase1_cases(config)}
    if case_id not in cases:
        raise FracturePhase1ContractError(f"unknown or replacement Phase-1 case_id {case_id!r}")
    values = _mapping(metrics, "trajectory_metrics")
    missing = sorted(_REQUIRED_QC_METRICS - set(values))
    if missing:
        raise FracturePhase1ContractError(
            f"trajectory metrics are missing required keys: {missing}"
        )

    required_state_count = values["required_output_state_count"]
    stored_accepted_count = values["stored_accepted_step_count"]
    replacement_attempts = values["replacement_attempts"]
    for key, count in (
        ("required_output_state_count", required_state_count),
        ("stored_accepted_step_count", stored_accepted_count),
    ):
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise FracturePhase1ContractError(
                f"trajectory metric {key!r} must be a non-negative integer"
            )
    if (
        isinstance(replacement_attempts, bool)
        or not isinstance(replacement_attempts, int)
        or replacement_attempts < 0
    ):
        raise FracturePhase1ContractError(
            "trajectory metric 'replacement_attempts' must be a non-negative integer"
        )

    thresholds = config["quality_control"]["per_trajectory"]
    checks: dict[str, bool] = {
        "completed": _metric_bool(values, "completed"),
        "required_output_step_count": required_state_count
        == thresholds["required_output_step_count"],
        "stored_all_required_and_adaptive_steps": stored_accepted_count >= required_state_count,
        "required_output_s_complete": _metric_bool(values, "required_output_s_complete"),
        "finite": _metric_nonnegative(values, "nonfinite_fraction")
        <= thresholds["max_nonfinite_fraction"],
        "equilibrium": _metric_nonnegative(values, "max_equilibrium_relative_residual")
        <= thresholds["max_equilibrium_relative_residual"],
        "kkt_complementarity": _metric_nonnegative(
            values, "max_kkt_complementarity_relative_residual"
        )
        <= thresholds["max_kkt_complementarity_relative_residual"],
        "damage_irreversibility": _metric_nonnegative(
            values, "max_damage_irreversibility_violation"
        )
        <= thresholds["max_damage_irreversibility_violation"],
        "damage_range": _metric_nonnegative(values, "max_damage_range_violation")
        <= thresholds["max_damage_range_violation"],
        "history_monotonicity": _metric_nonnegative(values, "max_history_monotonicity_violation")
        <= thresholds["max_history_monotonicity_violation"],
        "energy_balance": _metric_nonnegative(values, "max_relative_energy_imbalance")
        <= thresholds["max_relative_energy_imbalance"],
        "nonnegative_dissipation": _number(
            values["min_incremental_dissipation_over_UCS_R2"],
            "trajectory_metrics.min_incremental_dissipation_over_UCS_R2",
        )
        >= thresholds["min_incremental_dissipation_over_UCS_R2"],
        "damage_contained_in_refined_region": _metric_nonnegative(
            values, "max_damage_on_refined_region_outer_edge"
        )
        <= thresholds["max_damage_on_refined_region_outer_edge"],
        "all_steps_converged": _metric_bool(values, "all_steps_converged"),
        "failure_ledger_complete": _metric_bool(values, "failure_ledger_complete"),
        "retry_budget_not_exhausted": not _metric_bool(values, "retry_budget_exhausted"),
        "zero_replacement_attempts": replacement_attempts == 0,
    }

    case = cases[case_id]
    if case.ultrafine_audit:
        audit_metric_names = {
            "fine_to_ultrafine_peak_reaction_relative_change",
            "fine_to_ultrafine_total_fracture_energy_relative_change",
        }
        missing_audit = sorted(audit_metric_names - set(values))
        if missing_audit:
            raise FracturePhase1ContractError(
                f"preselected ultrafine audit metrics are missing: {missing_audit}"
            )
        audit_thresholds = config["quality_control"]["fine_to_ultrafine"]
        checks["fine_to_ultrafine_peak_reaction"] = (
            _metric_nonnegative(values, "fine_to_ultrafine_peak_reaction_relative_change")
            <= audit_thresholds["max_peak_reaction_relative_change"]
        )
        checks["fine_to_ultrafine_fracture_energy"] = (
            _metric_nonnegative(values, "fine_to_ultrafine_total_fracture_energy_relative_change")
            <= audit_thresholds["max_total_fracture_energy_relative_change"]
        )

    failed = tuple(name for name, passed in checks.items() if not passed)
    return TrajectoryQCResult(
        case_id=case_id,
        passed=not failed,
        checks=checks,
        failed_checks=failed,
        ultrafine_audit=case.ultrafine_audit,
        replacement_allowed=False,
    )


__all__ = [
    "LOAD_PATH_IDS",
    "MATERIAL_LEVEL_IDS",
    "PROTOCOL_ID",
    "REQUIRED_OUTPUT_STEP_COUNT",
    "SCHEMA_VERSION",
    "SECTION_FAMILIES",
    "FracturePhase1Case",
    "FracturePhase1ContractError",
    "TrajectoryQCResult",
    "default_fracture_phase1_config_path",
    "enumerate_fracture_phase1_cases",
    "enumerate_ultrafine_audits",
    "evaluate_trajectory_qc",
    "load_fracture_phase1_config",
    "validate_fracture_phase1_config",
]
