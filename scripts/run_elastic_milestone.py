"""Execute the frozen TunnelGeoPT B-elastic milestone campaign.

The runner deliberately separates the preregistered run contract from the
computed evidence.  It writes the case/derived-record manifest before the
first solver call, attempts every planned case exactly once, and retains every
failure.  A passing decision authorizes only linear-elastic solver-emulation
data generation; it is not evidence of fracture, damage, or rockburst physics.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from importlib import metadata
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from tunnelgeopt.cases import (
    build_case_manifest,
    load_case_manifest,
    sha256_canonical,
    write_case_manifest,
)
from tunnelgeopt.elastic_schema import (
    elastic_record_from_result,
    load_elastic_record,
    save_elastic_record,
)
from tunnelgeopt.elastic_validation import (
    kirsch_metrics,
    run_affine_patch_test,
    validate_elastic_result,
)
from tunnelgeopt.elasticity import solve_plane_strain_excavation
from tunnelgeopt.geometry import make_tunnel_boundary, points_inside_polygon
from tunnelgeopt.mesh import FARFIELD, ROCK, WALL, generate_tunnel_mesh

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "elastic_milestone.json"
CASE_DESIGN_VERSION = "balanced-cyclic-six-v1"
SOLVER_ID = "tunnelgeopt-skfem-p1-plane-strain-v1"
REQUIRED_SECTIONS = ("circle", "horseshoe", "straight_wall_arch")
REQUIRED_TIERS = ("coarse", "medium", "fine")
REQUIRED_KIRSCH_LOADS = ("uniaxial", "equal_biaxial", "pure_shear")


class MilestoneConfigError(ValueError):
    """Raised when the frozen configuration is incomplete or weakened."""


@dataclass(frozen=True)
class PreparedRun:
    """Frozen inputs and paths written before numerical execution begins."""

    config: dict[str, Any]
    config_hash: str
    output_dir: Path
    config_snapshot_path: Path
    environment_path: Path
    case_manifest_path: Path
    environment: dict[str, Any]
    manifest: dict[str, Any]


class RunLogger:
    """Append-only JSON-lines logger flushed after every event."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("x", encoding="utf-8", newline="\n")

    def write(self, event: str, **payload: Any) -> None:
        record = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "event": event,
            **_json_native(payload),
        }
        self._stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        self._stream.flush()
        os.fsync(self._stream.fileno())

    def close(self) -> None:
        self._stream.close()


def _json_native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(
            _json_native(value), ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required(mapping: dict[str, Any], key: str, path: str) -> Any:
    if key not in mapping:
        raise MilestoneConfigError(f"{path}.{key} is required")
    return mapping[key]


def _finite_sequence(mapping: dict[str, Any], key: str, *, positive: bool = False) -> list[float]:
    raw = _required(mapping, key, "$.cases")
    if not isinstance(raw, list) or len(raw) != 3:
        raise MilestoneConfigError(f"$.cases.{key} must contain exactly three levels")
    values = [float(item) for item in raw]
    if not all(math.isfinite(item) for item in values):
        raise MilestoneConfigError(f"$.cases.{key} contains a non-finite value")
    if positive and not all(item > 0.0 for item in values):
        raise MilestoneConfigError(f"$.cases.{key} must contain positive values")
    return values


def validate_frozen_config(config: dict[str, Any]) -> None:
    """Reject incomplete contracts and any relaxation of the frozen gates."""

    if not isinstance(config, dict):
        raise MilestoneConfigError("configuration root must be an object")
    if _required(config, "schema_version", "$") != "0.2.0":
        raise MilestoneConfigError("only the frozen 0.2.0 milestone contract is supported")
    if _required(config, "status", "$") != "preregistered_before_run":
        raise MilestoneConfigError("configuration must remain preregistered_before_run")
    cases = _required(config, "cases", "$")
    if not isinstance(cases, dict):
        raise MilestoneConfigError("$.cases must be an object")
    if tuple(_required(cases, "section_families", "$.cases")) != REQUIRED_SECTIONS:
        raise MilestoneConfigError(f"section families must remain {REQUIRED_SECTIONS}")
    per_section = int(_required(cases, "cases_per_section", "$.cases"))
    total = int(_required(cases, "total_cases", "$.cases"))
    if per_section != 6 or total != len(REQUIRED_SECTIONS) * per_section:
        raise MilestoneConfigError("the frozen design requires three sections times six cases")
    _finite_sequence(cases, "roughness_amplitude_over_radius")
    _finite_sequence(cases, "youngs_modulus_over_reference_stress", positive=True)
    poisson = _finite_sequence(cases, "poisson_ratio")
    if not all(-1.0 < value < 0.5 for value in poisson):
        raise MilestoneConfigError("Poisson ratios must lie in the elastic domain (-1, 0.5)")
    _finite_sequence(cases, "sigma1_over_reference_stress_compression_positive", positive=True)
    ratios = _finite_sequence(cases, "sigma3_over_sigma1", positive=True)
    if not all(value <= 1.0 for value in ratios):
        raise MilestoneConfigError("sigma3/sigma1 must lie in (0, 1]")
    _finite_sequence(cases, "sigma1_azimuth_deg")

    physics = _required(config, "physics", "$")
    if physics.get("internal_stress_sign") != "tension_positive":
        raise MilestoneConfigError("internal solver sign must remain tension_positive")
    if physics.get("input_rock_stress_sign") != "compression_positive_converted_at_boundary":
        raise MilestoneConfigError("compression-positive input conversion must remain explicit")

    split = _required(config, "split", "$")
    if split.get("expected_counts_per_section") != {
        "train": 4,
        "dev": 1,
        "locked_test": 1,
    }:
        raise MilestoneConfigError("the 4/1/1 per-section split is frozen")

    mesh = _required(config, "mesh", "$")
    if tuple(mesh.get("tiers", {}).keys()) != REQUIRED_TIERS:
        raise MilestoneConfigError(f"mesh tiers must remain ordered as {REQUIRED_TIERS}")
    if float(mesh.get("farfield_half_width_over_radius", 0.0)) <= 1.0:
        raise MilestoneConfigError("farfield half-width must exceed one radius")
    for tier in REQUIRED_TIERS:
        spec = mesh["tiers"][tier]
        if (
            float(spec.get("wall_size_over_radius", 0.0)) <= 0.0
            or float(spec.get("farfield_size_over_radius", 0.0)) <= 0.0
        ):
            raise MilestoneConfigError(f"mesh tier {tier!r} has a non-positive size")

    kirsch = _required(config, "kirsch_validation", "$")
    if tuple(kirsch.get("load_cases", ())) != REQUIRED_KIRSCH_LOADS:
        raise MilestoneConfigError(f"Kirsch loads must remain {REQUIRED_KIRSCH_LOADS}")
    if float(kirsch.get("max_fine_area_weighted_stress_relative_l2", math.inf)) > 0.08:
        raise MilestoneConfigError("the frozen fine Kirsch error gate may not exceed 0.08")
    if float(kirsch.get("max_uniaxial_stress_concentration_relative_error", math.inf)) > 0.10:
        raise MilestoneConfigError("the frozen uniaxial SCF error gate may not exceed 0.10")
    if kirsch.get("require_monotonic_error_improvement") is not True:
        raise MilestoneConfigError("monotonic Kirsch improvement must remain required")

    quality = _required(config, "quality_control", "$")
    frozen_upper_bounds = {
        "max_nonfinite_fraction": 0.0,
        "max_matrix_symmetry_relative_error": 1e-12,
        "max_free_dof_algebraic_residual": 1e-9,
        "max_clapeyron_relative_error": 1e-9,
        "max_patch_stress_relative_error": 1e-9,
    }
    for key, frozen_maximum in frozen_upper_bounds.items():
        if float(quality.get(key, math.inf)) > frozen_maximum:
            raise MilestoneConfigError(f"the frozen QC gate {key} may not be relaxed")
    if float(quality.get("min_triangle_signed_area", 0.0)) < 1e-12:
        raise MilestoneConfigError("the minimum signed-area gate may not be relaxed")
    for key in (
        "require_explicit_wall_and_farfield_tags",
        "require_no_element_centroid_inside_cavity",
    ):
        if quality.get(key) is not True:
            raise MilestoneConfigError(f"{key} must remain required")


def load_config(path: str | Path = DEFAULT_CONFIG) -> tuple[dict[str, Any], str]:
    source = Path(path)
    try:
        config = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MilestoneConfigError(f"could not read frozen config {source}: {exc}") from exc
    validate_frozen_config(config)
    return config, sha256_canonical(config)


def _design_indices(case_index: int) -> dict[str, int]:
    """Balanced deterministic six-run design; every three-level factor occurs twice."""

    if not 0 <= case_index < 6:
        raise ValueError("case_index must lie in [0, 6)")
    level = case_index % 3
    block = case_index // 3
    return {
        "roughness": level,
        "youngs_modulus": (level + 2 * block) % 3,
        "poisson_ratio": (2 * level + block) % 3,
        "sigma1": (level + block) % 3,
        "sigma3_ratio": (2 * level + 2 * block) % 3,
        "azimuth": level if block == 0 else 2 - level,
    }


def compression_positive_inplane_tensor(
    sigma1: float, sigma3: float, azimuth_deg: float
) -> np.ndarray:
    """Return the compression-positive transverse tensor in ``(y,z)`` order."""

    if not all(math.isfinite(value) for value in (sigma1, sigma3, azimuth_deg)):
        raise ValueError("stress parameters must be finite")
    if sigma1 <= 0.0 or sigma3 <= 0.0 or sigma3 > sigma1:
        raise ValueError("principal compressive stresses require 0 < sigma3 <= sigma1")
    angle = math.radians(azimuth_deg)
    principal_1 = np.asarray([math.cos(angle), math.sin(angle)], dtype=np.float64)
    principal_3 = np.asarray([-math.sin(angle), math.cos(angle)], dtype=np.float64)
    return sigma1 * np.outer(principal_1, principal_1) + sigma3 * np.outer(principal_3, principal_3)


def solver_tension_positive_stress(compression_positive: np.ndarray) -> np.ndarray:
    """Perform the single explicit sign conversion at the solver boundary."""

    stress = np.asarray(compression_positive, dtype=np.float64)
    if stress.shape != (2, 2) or not np.isfinite(stress).all():
        raise ValueError("compression-positive stress must be a finite 2x2 tensor")
    if not np.allclose(stress, stress.T, rtol=0.0, atol=1e-15):
        raise ValueError("compression-positive stress must be symmetric")
    return -stress


def build_physical_cases(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand the frozen three-section, six-case-per-section design."""

    validate_frozen_config(config)
    spec = config["cases"]
    levels = {
        "roughness": [float(x) for x in spec["roughness_amplitude_over_radius"]],
        "youngs_modulus": [float(x) for x in spec["youngs_modulus_over_reference_stress"]],
        "poisson_ratio": [float(x) for x in spec["poisson_ratio"]],
        "sigma1": [float(x) for x in spec["sigma1_over_reference_stress_compression_positive"]],
        "sigma3_ratio": [float(x) for x in spec["sigma3_over_sigma1"]],
        "azimuth": [float(x) for x in spec["sigma1_azimuth_deg"]],
    }
    records: list[dict[str, Any]] = []
    for family_index, family in enumerate(spec["section_families"]):
        for case_index in range(int(spec["cases_per_section"])):
            indices = _design_indices(case_index)
            selected = {name: values[indices[name]] for name, values in levels.items()}
            sigma1 = selected["sigma1"]
            sigma3 = sigma1 * selected["sigma3_ratio"]
            transverse = compression_positive_inplane_tensor(sigma1, sigma3, selected["azimuth"])
            axial = sigma3
            stress_3d = [
                [axial, 0.0, 0.0],
                [0.0, float(transverse[0, 0]), float(transverse[0, 1])],
                [0.0, float(transverse[1, 0]), float(transverse[1, 1])],
            ]
            seed = family_index * 10_000 + case_index
            records.append(
                {
                    "section_family": family,
                    "section_parameters": {
                        "radius": 1.0,
                        "roughness_amplitude_over_radius": selected["roughness"],
                    },
                    "material_field_seed": seed,
                    "joint_network_seed": 1_000_000 + seed,
                    "dimensionless_material_parameters": {
                        "young_modulus_over_reference_stress": selected["youngs_modulus"],
                        "poisson_ratio": selected["poisson_ratio"],
                    },
                    "initial_stress_tensor": stress_3d,
                    "stress_orientation": selected["azimuth"],
                    "excavation_schedule": [[0.0, 0.0], [1.0, 1.0]],
                    "unloading_schedule": [[0.0, 1.0], [1.0, 0.0]],
                }
            )
    if len(records) != int(spec["total_cases"]):
        raise RuntimeError("case expansion did not produce the frozen total")
    return records


def _planned_derived_records(
    frozen_parents: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    medium = config["mesh"]["tiers"]["medium"]
    return [
        {
            "case_group_id": parent["case_group_id"],
            "run_id": config["run_id"],
            "fidelity": "b_elastic",
            "solver": SOLVER_ID,
            "mesh_tier": "medium",
            "mesh_spec": medium,
            "planned_attempt_count": 1,
        }
        for parent in frozen_parents
    ]


def _environment_snapshot() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in ("numpy", "scipy", "scikit-fem", "gmsh", "tunnelgeopt"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    git: dict[str, Any] = {"commit": "unknown", "dirty": None}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
        git = {"commit": commit, "dirty": bool(dirty.strip())}
    except (OSError, subprocess.SubprocessError):
        pass
    return {
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": packages,
        "git": git,
    }


def prepare_run_contract(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path | None = None,
) -> PreparedRun:
    """Freeze config, environment, parent cases, and child plans before solving."""

    source = Path(config_path).resolve()
    config, config_hash = load_config(source)
    destination = (
        Path(output_dir).resolve()
        if output_dir is not None
        else REPO_ROOT / "artifacts" / "experiment" / str(config["run_id"])
    )
    if destination.exists():
        raise FileExistsError(
            f"immutable run directory already exists: {destination}; choose a new output directory"
        )
    destination.mkdir(parents=True)
    snapshot_path = destination / "config.snapshot.json"
    snapshot_path.write_bytes(source.read_bytes())
    environment = _environment_snapshot()
    environment_path = destination / "environment.json"
    _atomic_json(environment_path, environment)

    cases = build_physical_cases(config)
    # Build once to obtain frozen parent ids, then rebuild with all planned
    # medium-grid children.  No solver is called before the final manifest is
    # verified on disk.
    parents_only = build_case_manifest(
        cases,
        ratios=config["split"]["fractions"],
        metadata={"config_hash": config_hash},
    )
    derived = _planned_derived_records(parents_only["cases"], config)
    manifest = build_case_manifest(
        cases,
        derived_records=derived,
        ratios=config["split"]["fractions"],
        metadata={
            "run_id": config["run_id"],
            "config_hash": config_hash,
            "case_design_version": CASE_DESIGN_VERSION,
            "solver_id": SOLVER_ID,
            "written_before_any_solver_call": True,
        },
    )
    manifest_path = destination / "case_manifest.json"
    write_case_manifest(manifest_path, manifest)
    verified = load_case_manifest(manifest_path)
    return PreparedRun(
        config=config,
        config_hash=config_hash,
        output_dir=destination,
        config_snapshot_path=snapshot_path,
        environment_path=environment_path,
        case_manifest_path=manifest_path,
        environment=environment,
        manifest=verified,
    )


def _boundary_point_count(wall_size: float) -> int:
    return max(32, math.ceil(2.0 * math.pi / wall_size))


def _make_mesh(
    config: dict[str, Any],
    *,
    section: str,
    roughness: float,
    seed: int,
    tier: str,
) -> tuple[Any, Any]:
    radius = 1.0
    mesh_spec = config["mesh"]["tiers"][tier]
    wall_size = float(mesh_spec["wall_size_over_radius"]) * radius
    farfield_size = float(mesh_spec["farfield_size_over_radius"]) * radius
    half_width = float(config["mesh"]["farfield_half_width_over_radius"]) * radius
    geometry = make_tunnel_boundary(
        section,
        n_points=_boundary_point_count(wall_size),
        radius=radius,
        roughness_amplitude=roughness,
        seed=seed,
    )
    tunnel_mesh = generate_tunnel_mesh(
        geometry,
        outer_bounds=(-half_width, half_width, -half_width, half_width),
        mesh_size=farfield_size,
        wall_mesh_size=wall_size,
        farfield_mesh_size=farfield_size,
    )
    return geometry, tunnel_mesh


def _kirsch_input(load_case: str) -> np.ndarray:
    if load_case == "uniaxial":
        return np.asarray([[1.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    if load_case == "equal_biaxial":
        return np.eye(2, dtype=np.float64)
    if load_case == "pure_shear":
        return np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    raise ValueError(f"unknown Kirsch load {load_case!r}")


def _exception_record(stage: str, exc: BaseException, **identity: Any) -> dict[str, Any]:
    return {
        "stage": stage,
        **identity,
        "exception_type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }


def run_kirsch_campaign(prepared: PreparedRun, logger: RunLogger) -> list[dict[str, Any]]:
    config = prepared.config
    records: list[dict[str, Any]] = []
    annulus = tuple(
        float(x) for x in config["kirsch_validation"]["comparison_annulus_radius_over_radius"]
    )
    for load_case in config["kirsch_validation"]["load_cases"]:
        for tier in config["mesh"]["tiers"]:
            started = perf_counter()
            identity = {"load_case": load_case, "mesh_tier": tier}
            try:
                _, tunnel_mesh = _make_mesh(
                    config,
                    section="circle",
                    roughness=float(config["kirsch_validation"]["roughness_amplitude_over_radius"]),
                    seed=0,
                    tier=tier,
                )
                result = solve_plane_strain_excavation(
                    tunnel_mesh,
                    young_modulus=500.0,
                    poisson_ratio=0.25,
                    sigma_inf=solver_tension_positive_stress(_kirsch_input(load_case)),
                )
                metric = kirsch_metrics(result, tunnel_mesh, radius=1.0, annulus=annulus)
                generic = _generic_qc(
                    result,
                    tunnel_mesh,
                    make_tunnel_boundary(
                        "circle",
                        n_points=_boundary_point_count(
                            float(config["mesh"]["tiers"][tier]["wall_size_over_radius"])
                        ),
                        radius=1.0,
                        roughness_amplitude=0.0,
                        seed=0,
                    ),
                    config,
                )
                record = {
                    **identity,
                    "status": "completed",
                    "wall_seconds": perf_counter() - started,
                    "node_count": int(result.nodes.shape[0]),
                    "element_count": int(result.elements.shape[0]),
                    "metrics": metric.as_dict(),
                    "generic_qc": generic,
                }
                logger.write("kirsch_completed", **identity, metrics=record["metrics"])
            except Exception as exc:  # noqa: BLE001 - retain this preregistered outcome
                record = {
                    **identity,
                    "status": "failed",
                    "wall_seconds": perf_counter() - started,
                    "failure": _exception_record("kirsch", exc, **identity),
                }
                logger.write("kirsch_failed", **identity, failure=record["failure"])
            records.append(record)
    return records


def _generic_qc(
    result: Any, tunnel_mesh: Any, geometry: Any, config: dict[str, Any]
) -> dict[str, Any]:
    quality = config["quality_control"]
    solver = validate_elastic_result(
        result,
        max_symmetry_error=float(quality["max_matrix_symmetry_relative_error"]),
        max_algebraic_residual=float(quality["max_free_dof_algebraic_residual"]),
        max_energy_closure=float(quality["max_clapeyron_relative_error"]),
    )
    triangles = np.asarray(result.nodes)[np.asarray(result.elements)]
    signed_area = 0.5 * (
        (triangles[:, 1, 0] - triangles[:, 0, 0]) * (triangles[:, 2, 1] - triangles[:, 0, 1])
        - (triangles[:, 2, 0] - triangles[:, 0, 0]) * (triangles[:, 1, 1] - triangles[:, 0, 1])
    )
    minimum_signed_area = float(np.min(signed_area))
    centroid_inside_count = int(
        np.count_nonzero(points_inside_polygon(result.element_centers, geometry.boundary_yz))
    )
    tags_explicit = (
        set(tunnel_mesh.boundary_facets) == {WALL, FARFIELD}
        and set(tunnel_mesh.physical_tags) == {ROCK, WALL, FARFIELD}
        and len(tunnel_mesh.boundary_facets[WALL]) > 0
        and len(tunnel_mesh.boundary_facets[FARFIELD]) > 0
    )
    extra_checks = {
        "nonfinite_fraction_within_gate": solver["metrics"]["nonfinite_fraction"]
        <= float(quality["max_nonfinite_fraction"]),
        "minimum_triangle_signed_area": minimum_signed_area
        >= float(quality["min_triangle_signed_area"]),
        "explicit_wall_and_farfield_tags": bool(tags_explicit),
        "no_element_centroid_inside_cavity": centroid_inside_count == 0,
    }
    return {
        "passed": bool(solver["passed"] and all(extra_checks.values())),
        "solver": solver,
        "checks": extra_checks,
        "metrics": {
            "minimum_triangle_signed_area": minimum_signed_area,
            "element_centroid_inside_cavity_count": centroid_inside_count,
        },
    }


def _case_stresses(parent: dict[str, Any]) -> tuple[np.ndarray, float]:
    compression = np.asarray(parent["initial_stress_tensor"], dtype=np.float64)
    transverse = compression[1:, 1:]
    solver_stress = solver_tension_positive_stress(transverse)
    solver_sigma_xx = -float(compression[0, 0])
    return solver_stress, solver_sigma_xx


def run_physical_cases(prepared: PreparedRun, logger: RunLogger) -> list[dict[str, Any]]:
    config = prepared.config
    parents = {item["case_group_id"]: item for item in prepared.manifest["cases"]}
    records: list[dict[str, Any]] = []
    for child in prepared.manifest["derived_records"]:
        started = perf_counter()
        parent = parents[child["case_group_id"]]
        identity = {
            "case_group_id": child["case_group_id"],
            "derived_record_id": child["derived_record_id"],
            "section_family": parent["section_family"],
            "split": child["split"],
            "mesh_tier": child["mesh_tier"],
        }
        try:
            roughness = float(parent["section_parameters"]["roughness_amplitude_over_radius"])
            geometry, tunnel_mesh = _make_mesh(
                config,
                section=parent["section_family"],
                roughness=roughness,
                seed=int(parent["material_field_seed"]),
                tier=child["mesh_tier"],
            )
            solver_stress, solver_sigma_xx = _case_stresses(parent)
            material = parent["dimensionless_material_parameters"]
            result = solve_plane_strain_excavation(
                tunnel_mesh,
                young_modulus=float(material["young_modulus_over_reference_stress"]),
                poisson_ratio=float(material["poisson_ratio"]),
                sigma_inf=solver_stress,
                sigma_xx_inf=solver_sigma_xx,
            )
            qc = _generic_qc(result, tunnel_mesh, geometry, config)
            record_dir = prepared.output_dir / "cases" / child["derived_record_id"]
            elastic_record = elastic_record_from_result(
                result,
                case_group_id=child["case_group_id"],
                config_hash=prepared.config_hash,
                env=prepared.environment,
                meta={
                    "derived_record_id": child["derived_record_id"],
                    "run_id": config["run_id"],
                    "mesh_tier": child["mesh_tier"],
                    "section_family": parent["section_family"],
                    "split": child["split"],
                    "input_stress_sign": "compression_positive",
                    "solver_stress_sign": "tension_positive",
                    "reference_length_m": 1.0,
                    "reference_stress_pa": 1.0,
                },
            )
            paths = save_elastic_record(record_dir, elastic_record)
            loaded = load_elastic_record(record_dir)
            if loaded.case_group_id != child["case_group_id"]:
                raise RuntimeError("saved schema case_group_id failed round-trip")
            record = {
                **identity,
                "status": "passed" if qc["passed"] else "failed_qc",
                "wall_seconds": perf_counter() - started,
                "node_count": loaded.num_nodes,
                "element_count": loaded.num_elements,
                "mesh_id": loaded.mesh_id,
                "solver_sigma_inf_tension_positive": loaded.sigma_inf.tolist(),
                "input_sigma_inf_compression_positive": (-loaded.sigma_inf).tolist(),
                "qc": qc,
                "record_paths": {
                    "arrays": paths.arrays.relative_to(prepared.output_dir).as_posix(),
                    "meta": paths.meta.relative_to(prepared.output_dir).as_posix(),
                },
                "record_round_trip_verified": True,
            }
            logger.write("physical_case_completed", **identity, status=record["status"])
        except Exception as exc:  # noqa: BLE001 - retain one outcome, never replace it
            failure = _exception_record("physical_case", exc, **identity)
            failure_path = prepared.output_dir / "failures" / f"{child['derived_record_id']}.json"
            _atomic_json(failure_path, failure)
            record = {
                **identity,
                "status": "failed",
                "wall_seconds": perf_counter() - started,
                "failure": failure,
                "failure_path": failure_path.relative_to(prepared.output_dir).as_posix(),
            }
            logger.write("physical_case_failed", **identity, failure=failure)
        records.append(record)
    return records


def evaluate_kirsch_gate(records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    spec = config["kirsch_validation"]
    by_load = {
        load: [record for record in records if record["load_case"] == load]
        for load in spec["load_cases"]
    }
    details: dict[str, Any] = {}
    all_pass = True
    for load_case, load_records in by_load.items():
        ordered = []
        missing = []
        for tier in REQUIRED_TIERS:
            matches = [item for item in load_records if item["mesh_tier"] == tier]
            if len(matches) != 1 or matches[0].get("status") != "completed":
                missing.append(tier)
            else:
                ordered.append(matches[0])
        if missing:
            details[load_case] = {"passed": False, "missing_or_failed_tiers": missing}
            all_pass = False
            continue
        errors = [float(item["metrics"]["annulus_stress_relative_l2"]) for item in ordered]
        monotonic = errors[0] >= errors[1] >= errors[2]
        fine_pass = errors[2] < float(spec["max_fine_area_weighted_stress_relative_l2"])
        generic_pass = all(item["generic_qc"]["passed"] for item in ordered)
        scf_pass: bool | None = None
        if load_case == "uniaxial":
            scf_pass = float(ordered[-1]["metrics"]["peak_hoop_relative_error"]) < float(
                spec["max_uniaxial_stress_concentration_relative_error"]
            )
        passed = monotonic and fine_pass and generic_pass and (scf_pass is not False)
        details[load_case] = {
            "passed": passed,
            "annulus_stress_relative_l2_by_tier": dict(zip(REQUIRED_TIERS, errors, strict=True)),
            "monotonic_coarse_to_fine": monotonic,
            "fine_below_threshold": fine_pass,
            "all_generic_qc_passed": generic_pass,
            "uniaxial_scf_below_threshold": scf_pass,
        }
        all_pass = all_pass and passed
    return {"passed": all_pass, "loads": details}


def evaluate_decision(raw: dict[str, Any], prepared: PreparedRun) -> dict[str, Any]:
    quality = prepared.config["quality_control"]
    patch = raw["patch"]
    patch_pass = (
        patch.get("status") == "completed"
        and float(patch["metrics"]["stress_relative_l2"])
        <= float(quality["max_patch_stress_relative_error"])
        and bool(patch["metrics"]["passed"])
    )
    kirsch_gate = evaluate_kirsch_gate(raw["kirsch"], prepared.config)
    physical = raw["physical_cases"]
    planned_ids = [item["derived_record_id"] for item in prepared.manifest["derived_records"]]
    actual_ids = [item["derived_record_id"] for item in physical]
    exactly_once = len(actual_ids) == len(set(actual_ids)) == len(planned_ids) and set(
        actual_ids
    ) == set(planned_ids)
    physical_pass = len(physical) == int(prepared.config["cases"]["total_cases"]) and all(
        item.get("status") == "passed"
        and item.get("qc", {}).get("passed") is True
        and item.get("record_round_trip_verified") is True
        for item in physical
    )
    checks = {
        "affine_patch": patch_pass,
        "kirsch_three_load_three_mesh": kirsch_gate["passed"],
        "all_18_medium_cases_generic_qc_and_schema": physical_pass,
        "planned_cases_attempted_exactly_once_no_replacements": exactly_once,
        "frozen_manifest_written_before_solver": prepared.manifest["metadata"].get(
            "written_before_any_solver_call"
        )
        is True,
    }
    passed = all(checks.values())
    return {
        "status": "go" if passed else "no_go",
        "passed": passed,
        "checks": checks,
        "kirsch_gate": kirsch_gate,
        "failed_checks": [key for key, value in checks.items() if not value],
        "passing_scope": prepared.config["milestone_decision"]["passing_scope"],
        "claim_boundary": (
            "validated synthetic homogeneous isotropic small-strain plane-strain elasticity only"
        ),
        "excluded_claims": list(prepared.config["claim_exclusions"]),
        "next_action": (
            "generate a larger split-safe B-elastic corpus and train the first surrogate baseline"
            if passed
            else "repair the recorded failed gate without changing cases, meshes, or thresholds"
        ),
    }


def _run_manifest(
    prepared: PreparedRun, decision: dict[str, Any], command: list[str]
) -> dict[str, Any]:
    files = []
    for path in sorted(prepared.output_dir.rglob("*")):
        if path.is_file() and path.name != "run_manifest.json":
            files.append(
                {
                    "path": path.relative_to(prepared.output_dir).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
    return {
        "run_id": prepared.config["run_id"],
        "schema_version": prepared.config["schema_version"],
        "status": decision["status"],
        "config_hash": prepared.config_hash,
        "case_manifest_hash": prepared.manifest["manifest_hash"],
        "exact_command": command,
        "case_design_version": CASE_DESIGN_VERSION,
        "solver_id": SOLVER_ID,
        "planned_physical_cases": len(prepared.manifest["derived_records"]),
        "files": files,
    }


def run_milestone(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path | None = None,
    *,
    command: list[str] | None = None,
) -> dict[str, Any]:
    prepared = prepare_run_contract(config_path, output_dir)
    logger = RunLogger(prepared.output_dir / "run.log")
    logger.write(
        "run_contract_frozen",
        config_hash=prepared.config_hash,
        manifest_hash=prepared.manifest["manifest_hash"],
        parent_count=len(prepared.manifest["cases"]),
        derived_count=len(prepared.manifest["derived_records"]),
    )
    raw: dict[str, Any] = {
        "run_id": prepared.config["run_id"],
        "config_hash": prepared.config_hash,
        "case_manifest_hash": prepared.manifest["manifest_hash"],
    }
    patch_started = perf_counter()
    try:
        patch_metrics = run_affine_patch_test()
        raw["patch"] = {
            "status": "completed",
            "wall_seconds": perf_counter() - patch_started,
            "metrics": patch_metrics,
        }
        logger.write("patch_completed", metrics=patch_metrics)
    except Exception as exc:  # noqa: BLE001 - a failed patch is milestone evidence
        failure = _exception_record("affine_patch", exc)
        raw["patch"] = {
            "status": "failed",
            "wall_seconds": perf_counter() - patch_started,
            "failure": failure,
        }
        logger.write("patch_failed", failure=failure)

    raw["kirsch"] = run_kirsch_campaign(prepared, logger)
    raw["physical_cases"] = run_physical_cases(prepared, logger)
    decision = evaluate_decision(raw, prepared)
    raw_path = prepared.output_dir / "raw_metrics.json"
    decision_path = prepared.output_dir / "decision.json"
    _atomic_json(raw_path, raw)
    _atomic_json(decision_path, decision)
    logger.write("decision_recorded", status=decision["status"], checks=decision["checks"])
    logger.close()
    exact_command = command or [
        sys.executable,
        str(Path(__file__).resolve()),
        "--config",
        str(config_path),
    ]
    manifest = _run_manifest(prepared, decision, exact_command)
    _atomic_json(prepared.output_dir / "run_manifest.json", manifest)
    return {"decision": decision, "raw_metrics": raw, "run_manifest": manifest}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Immutable output directory; default is artifacts/experiment/<run_id>",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = [sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])]
    result = run_milestone(args.config, args.output_dir, command=command)
    print(json.dumps(result["decision"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["decision"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
