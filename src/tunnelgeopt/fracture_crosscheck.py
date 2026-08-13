"""Fail-closed local-versus-MOOSE same-problem fracture cross-check.

This module validates a deliberately small cross-solver problem before the
internal AT2 kernel is allowed to generate development labels.  Both solvers
read the same immutable Gmsh 2.2 TRI3 mesh.  MOOSE ``(x, y, z)`` maps to local
``(y, z, tunnel-axis)`` and both use tension-positive stress.

The first gate is intact plane-strain elasticity under three independent
far-field stress bases.  A fixed, nonuniform P1 damage gate is conditional on
the intact gate passing.  Neither gate validates coupled crack evolution,
adaptive loading, field transfer, or rockburst dynamics.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .fracture import (
    AT2Material,
    FractureSolverOptions,
    miehe_spectral_response,
    solve_fixed_damage_displacement,
)
from .mesh import FARFIELD, ROCK, WALL, TunnelMesh

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

SCHEMA_VERSION = 1
EXPECTED_PHYSICAL_NAMES = {(1, 2): WALL, (1, 3): FARFIELD, (2, 1): ROCK}
UNRESOLVED_TOKEN = re.compile(r"@[A-Z][A-Z0-9_]*@")
PROJECT_NONPACKAGE_INPUTS = (
    "configs/fracture_crosscheck_v1.json",
    "moose/fracture_crosscheck/fixed_damage_same_mesh.i",
    "moose/fracture_crosscheck/intact_same_mesh.i",
    "pyproject.toml",
    "scripts/run_fracture_crosscheck.py",
)


class CrosscheckError(RuntimeError):
    """Base class for a cross-check that cannot produce valid evidence."""


class CrosscheckValidationError(CrosscheckError):
    """Raised when configuration, identity, or solver output is invalid."""


class MooseUnavailableError(CrosscheckError):
    """Raised when the pinned MOOSE executable cannot be proven available."""


@dataclass(frozen=True)
class CanonicalMesh:
    """Strictly parsed Gmsh 2.2 ASCII TRI3 mesh and boundary identity."""

    path: Path
    file_sha256: str
    structure_sha256: str
    node_tags: IntArray
    nodes_xy: FloatArray
    triangle_tags: IntArray
    triangles: IntArray
    line_tags: IntArray
    line_physical_tags: IntArray
    lines: IntArray

    @property
    def element_centroids_xy(self) -> FloatArray:
        return np.asarray(self.nodes_xy[self.triangles].mean(axis=1), dtype=np.float64)

    @property
    def triangle_areas(self) -> FloatArray:
        tri = self.nodes_xy[self.triangles]
        determinant = (tri[:, 1, 0] - tri[:, 0, 0]) * (tri[:, 2, 1] - tri[:, 0, 1])
        determinant -= (tri[:, 1, 1] - tri[:, 0, 1]) * (tri[:, 2, 0] - tri[:, 0, 0])
        return np.asarray(0.5 * np.abs(determinant), dtype=np.float64)

    def boundary_edges(self, physical_tag: int) -> IntArray:
        return np.asarray(self.lines[self.line_physical_tags == int(physical_tag)], dtype=np.int64)


@dataclass(frozen=True)
class MooseEnvironment:
    """Verified, current MOOSE executable provenance."""

    distribution: str
    linux_home: str
    executable_linux: str
    application_version: str
    executable_sha256: str
    source_commit: str
    upstream_commit: str
    source_tree_clean: bool


@dataclass(frozen=True)
class LocalCaseResult:
    """Local fixed-damage equilibrium fields in the frozen coordinate contract."""

    case_id: str
    gate: str
    applied_stress_yz: FloatArray
    nodes_xy: FloatArray
    displacement_xy: FloatArray
    internal_force_xy: FloatArray
    farfield_node_rows: IntArray
    element_centroids_xy: FloatArray
    strain_engineering: FloatArray
    stress_inplane: FloatArray
    stress_axis: FloatArray
    energy_density: FloatArray
    elastic_energy: float
    reaction_xy: FloatArray
    damage: FloatArray

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "applied_stress_yz": self.applied_stress_yz,
            "nodes_xy": self.nodes_xy,
            "displacement_xy": self.displacement_xy,
            "internal_force_xy": self.internal_force_xy,
            "farfield_node_rows": self.farfield_node_rows,
            "element_centroids_xy": self.element_centroids_xy,
            "strain_engineering": self.strain_engineering,
            "stress_inplane": self.stress_inplane,
            "stress_axis": self.stress_axis,
            "energy_density": self.energy_density,
            "elastic_energy": np.asarray(self.elastic_energy, dtype=np.float64),
            "reaction_xy": self.reaction_xy,
            "damage": self.damage,
        }


@dataclass(frozen=True)
class MooseCaseResult:
    """Strictly parsed MOOSE nodal and element fields."""

    node_ids: IntArray
    nodes_xy: FloatArray
    displacement_xy: FloatArray
    residual_xy: FloatArray
    damage: FloatArray | None
    element_ids: IntArray
    element_centroids_xy: FloatArray
    strain_engineering: FloatArray
    stress_inplane: FloatArray
    stress_axis: FloatArray
    energy_density: FloatArray


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise CrosscheckValidationError(f"{label} keys mismatch; missing={missing}, extra={extra}")


def _finite_float(value: Any, label: str, *, positive: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CrosscheckValidationError(f"{label} must be numeric") from exc
    if not np.isfinite(result) or (positive and result <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise CrosscheckValidationError(f"{label} must be {qualifier}")
    return result


def load_crosscheck_config(path: str | Path) -> dict[str, Any]:
    """Load the v1 config with strict keys and scientific-domain checks."""

    config_path = Path(path).resolve()
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CrosscheckValidationError(f"cannot load cross-check config: {config_path}") from exc
    if not isinstance(raw, dict):
        raise CrosscheckValidationError("cross-check config must be a JSON object")
    _require_exact_keys(
        raw,
        {
            "schema_version",
            "gate_id",
            "claim_boundary",
            "coordinate_contract",
            "mesh",
            "material",
            "intact_gate",
            "fixed_damage_gate",
            "comparison",
            "execution",
        },
        "config",
    )
    if raw["schema_version"] != SCHEMA_VERSION:
        raise CrosscheckValidationError(
            f"schema_version must be exactly {SCHEMA_VERSION}; migration is not implicit"
        )
    if raw["gate_id"] != "moose-local-same-problem-v1":
        raise CrosscheckValidationError("gate_id is not the frozen v1 identifier")
    if not isinstance(raw["claim_boundary"], str) or not raw["claim_boundary"].strip():
        raise CrosscheckValidationError("claim_boundary must be a non-empty string")

    coordinate = raw["coordinate_contract"]
    _require_exact_keys(
        coordinate,
        {
            "local_coordinates",
            "moose_coordinates",
            "mapping",
            "stress_sign",
            "strain_order",
            "stress_order",
        },
        "coordinate_contract",
    )
    if coordinate != {
        "local_coordinates": ["y", "z"],
        "moose_coordinates": ["x", "y", "z"],
        "mapping": "moose_x=local_y; moose_y=local_z; moose_z=tunnel_axis",
        "stress_sign": "tension_positive",
        "strain_order": ["yy", "zz", "gamma_yz"],
        "stress_order": ["yy", "zz", "yz"],
    }:
        raise CrosscheckValidationError("coordinate_contract differs from the frozen v1 mapping")

    mesh = raw["mesh"]
    _require_exact_keys(
        mesh,
        {"path", "format", "element_type", "physical_names", "allow_renumbering"},
        "mesh",
    )
    if (
        mesh["format"] != "gmsh_2.2_ascii"
        or mesh["element_type"] != "TRI3"
        or mesh["physical_names"] != {"rock": 1, "wall": 2, "farfield": 3}
        or mesh["allow_renumbering"] is not False
    ):
        raise CrosscheckValidationError("mesh contract differs from frozen Gmsh v2.2 TRI3 identity")

    material = raw["material"]
    _require_exact_keys(
        material,
        {
            "young_modulus_pa",
            "poisson_ratio",
            "fracture_toughness_j_per_m2",
            "length_scale_m",
            "residual_stiffness",
        },
        "material",
    )
    _finite_float(material["young_modulus_pa"], "young_modulus_pa", positive=True)
    poisson = _finite_float(material["poisson_ratio"], "poisson_ratio")
    if not -1.0 < poisson < 0.5:
        raise CrosscheckValidationError("poisson_ratio must lie in (-1, 0.5)")
    _finite_float(material["fracture_toughness_j_per_m2"], "fracture_toughness", positive=True)
    _finite_float(material["length_scale_m"], "length_scale", positive=True)
    residual = _finite_float(material["residual_stiffness"], "residual_stiffness")
    if not 0.0 <= residual < 1.0:
        raise CrosscheckValidationError("residual_stiffness must lie in [0, 1)")

    intact = raw["intact_gate"]
    _require_exact_keys(
        intact, {"enabled", "damage", "stress_basis_tension_positive_pa"}, "intact_gate"
    )
    if intact["enabled"] is not True or intact["damage"] != "zero":
        raise CrosscheckValidationError("the intact v1 gate must be enabled with zero damage")
    _validate_stress_basis(intact["stress_basis_tension_positive_pa"], "intact_gate")

    fixed = raw["fixed_damage_gate"]
    _require_exact_keys(
        fixed,
        {
            "enabled",
            "status",
            "damage_expression_moose_xy",
            "damage_coefficients_local_yz",
            "stress_basis_tension_positive_pa",
        },
        "fixed_damage_gate",
    )
    if fixed["enabled"] is not True or fixed["status"] != "conditional_after_intact_gate":
        raise CrosscheckValidationError("fixed-damage gate must remain conditional after intact")
    coefficients = np.asarray(fixed["damage_coefficients_local_yz"], dtype=np.float64)
    if coefficients.shape != (3,) or not np.isfinite(coefficients).all():
        raise CrosscheckValidationError("damage coefficients must be finite [intercept, dy, dz]")
    expression = fixed["damage_expression_moose_xy"]
    if not isinstance(expression, str) or not expression.strip():
        raise CrosscheckValidationError("damage_expression_moose_xy must be non-empty")
    _validate_stress_basis(fixed["stress_basis_tension_positive_pa"], "fixed_damage_gate")

    comparison = raw["comparison"]
    _require_exact_keys(
        comparison,
        {
            "primary_relative_l2_tolerance",
            "absolute_tolerances",
            "normalization_floors",
            "require_finite",
            "require_bijective_coordinate_mapping",
            "require_identical_mesh_file_sha256",
        },
        "comparison",
    )
    tolerance = _finite_float(
        comparison["primary_relative_l2_tolerance"], "primary relative tolerance", positive=True
    )
    if tolerance > 1.0e-6:
        raise CrosscheckValidationError("primary tolerance may not exceed the frozen 1e-6 gate")
    absolute_keys = {
        "coordinate_m",
        "displacement_m",
        "strain",
        "stress_pa",
        "energy_j_per_m",
        "reaction_n_per_m",
    }
    floor_keys = absolute_keys - {"coordinate_m"}
    _require_exact_keys(comparison["absolute_tolerances"], absolute_keys, "absolute_tolerances")
    _require_exact_keys(comparison["normalization_floors"], floor_keys, "normalization_floors")
    for label, value in comparison["absolute_tolerances"].items():
        _finite_float(value, f"absolute_tolerances.{label}", positive=True)
    for label, value in comparison["normalization_floors"].items():
        _finite_float(value, f"normalization_floors.{label}", positive=True)
    for flag in (
        "require_finite",
        "require_bijective_coordinate_mapping",
        "require_identical_mesh_file_sha256",
    ):
        if comparison[flag] is not True:
            raise CrosscheckValidationError(f"comparison.{flag} must remain true")

    execution = raw["execution"]
    _require_exact_keys(
        execution,
        {
            "wsl_distribution",
            "moose_executable_linux",
            "intact_template",
            "fixed_damage_template",
            "petsc_options",
            "threads",
        },
        "execution",
    )
    if execution["petsc_options"] != "direct_lu" or execution["threads"] != 1:
        raise CrosscheckValidationError("v1 execution must use one thread and direct LU")
    for label in (
        "wsl_distribution",
        "moose_executable_linux",
        "intact_template",
        "fixed_damage_template",
    ):
        if not isinstance(execution[label], str) or not execution[label].strip():
            raise CrosscheckValidationError(f"execution.{label} must be a non-empty string")
    raw["_config_path"] = str(config_path)
    raw["_config_sha256"] = _sha256_file(config_path)
    return raw


def _validate_stress_basis(raw: Any, label: str) -> None:
    basis = np.asarray(raw, dtype=np.float64)
    if basis.shape != (3, 3) or not np.isfinite(basis).all():
        raise CrosscheckValidationError(f"{label} stress basis must be finite [3, 3]")
    if np.linalg.matrix_rank(basis) != 3:
        raise CrosscheckValidationError(f"{label} stress basis must have rank three")


def _parse_section(lines: list[str], name: str) -> list[str]:
    start_token = f"${name}"
    end_token = f"$End{name}"
    starts = [index for index, line in enumerate(lines) if line == start_token]
    ends = [index for index, line in enumerate(lines) if line == end_token]
    if len(starts) != 1 or len(ends) != 1 or ends[0] <= starts[0]:
        raise CrosscheckValidationError(f"Gmsh section {name} must occur exactly once")
    return lines[starts[0] + 1 : ends[0]]


def parse_gmsh_v22_ascii(path: str | Path) -> CanonicalMesh:
    """Parse only the frozen ASCII Gmsh v2.2 subset and hash its full identity."""

    mesh_path = Path(path).resolve()
    try:
        raw_bytes = mesh_path.read_bytes()
        text = raw_bytes.decode("ascii")
    except (OSError, UnicodeDecodeError) as exc:
        raise CrosscheckValidationError("canonical mesh must be readable ASCII") from exc
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    mesh_format = _parse_section(lines, "MeshFormat")
    if mesh_format != ["2.2 0 8"]:
        raise CrosscheckValidationError("mesh must be exactly Gmsh 2.2 ASCII double precision")

    physical = _parse_section(lines, "PhysicalNames")
    try:
        physical_count = int(physical[0])
    except (IndexError, ValueError) as exc:
        raise CrosscheckValidationError("invalid PhysicalNames header") from exc
    if len(physical) != physical_count + 1:
        raise CrosscheckValidationError("PhysicalNames count mismatch")
    parsed_physical: dict[tuple[int, int], str] = {}
    pattern = re.compile(r'^([12])\s+([0-9]+)\s+"([^"]+)"$')
    for row in physical[1:]:
        match = pattern.fullmatch(row)
        if match is None:
            raise CrosscheckValidationError(f"invalid PhysicalNames row: {row}")
        key = (int(match.group(1)), int(match.group(2)))
        if key in parsed_physical:
            raise CrosscheckValidationError("duplicate physical dimension/tag")
        parsed_physical[key] = match.group(3)
    if parsed_physical != EXPECTED_PHYSICAL_NAMES:
        raise CrosscheckValidationError("physical names/tags are not exactly rock/wall/farfield")

    node_section = _parse_section(lines, "Nodes")
    try:
        node_count = int(node_section[0])
    except (IndexError, ValueError) as exc:
        raise CrosscheckValidationError("invalid Nodes header") from exc
    if len(node_section) != node_count + 1 or node_count < 4:
        raise CrosscheckValidationError("Nodes count mismatch or insufficient nodes")
    node_tags: list[int] = []
    nodes: list[list[float]] = []
    for row in node_section[1:]:
        parts = row.split()
        if len(parts) != 4:
            raise CrosscheckValidationError("every v1 node row must contain tag x y z")
        tag = int(parts[0])
        xyz = [float(value) for value in parts[1:]]
        if not np.isfinite(xyz).all() or abs(xyz[2]) > 1.0e-15:
            raise CrosscheckValidationError("mesh nodes must be finite and lie at MOOSE z=0")
        node_tags.append(tag)
        nodes.append(xyz[:2])
    node_tag_array = np.asarray(node_tags, dtype=np.int64)
    nodes_xy = np.asarray(nodes, dtype=np.float64)
    if (
        np.unique(node_tag_array).size != node_count
        or np.unique(nodes_xy, axis=0).shape[0] != node_count
    ):
        raise CrosscheckValidationError("mesh node tags and coordinates must be unique")
    tag_to_row = {int(tag): index for index, tag in enumerate(node_tag_array)}

    element_section = _parse_section(lines, "Elements")
    try:
        element_count = int(element_section[0])
    except (IndexError, ValueError) as exc:
        raise CrosscheckValidationError("invalid Elements header") from exc
    if len(element_section) != element_count + 1:
        raise CrosscheckValidationError("Elements count mismatch")
    line_tags: list[int] = []
    line_physical: list[int] = []
    line_nodes: list[list[int]] = []
    triangle_tags: list[int] = []
    triangles: list[list[int]] = []
    all_element_tags: list[int] = []
    for row in element_section[1:]:
        fields = [int(value) for value in row.split()]
        if len(fields) < 5:
            raise CrosscheckValidationError("invalid Gmsh element row")
        element_tag, element_type, number_of_tags = fields[:3]
        if number_of_tags != 2:
            raise CrosscheckValidationError(
                "every element must carry physical and geometrical tags"
            )
        expected_nodes = 2 if element_type == 1 else 3 if element_type == 2 else None
        if expected_nodes is None or len(fields) != 3 + number_of_tags + expected_nodes:
            raise CrosscheckValidationError("only first-order lines and TRI3 elements are allowed")
        physical_tag = fields[3]
        element_node_tags = fields[3 + number_of_tags :]
        try:
            rows = [tag_to_row[tag] for tag in element_node_tags]
        except KeyError as exc:
            raise CrosscheckValidationError("element references an unknown node tag") from exc
        all_element_tags.append(element_tag)
        if element_type == 1:
            if physical_tag not in (2, 3):
                raise CrosscheckValidationError("line physical tag must be wall or farfield")
            line_tags.append(element_tag)
            line_physical.append(physical_tag)
            line_nodes.append(rows)
        else:
            if physical_tag != 1:
                raise CrosscheckValidationError("every TRI3 element must belong to rock")
            triangle_tags.append(element_tag)
            triangles.append(rows)
    if len(set(all_element_tags)) != len(all_element_tags):
        raise CrosscheckValidationError("Gmsh element tags must be unique")
    if not lines or lines[-1] != "$EndElements":
        raise CrosscheckValidationError("unexpected content after the Elements section")

    line_tag_array = np.asarray(line_tags, dtype=np.int64)
    line_physical_array = np.asarray(line_physical, dtype=np.int64)
    line_array = np.asarray(line_nodes, dtype=np.int64)
    triangle_tag_array = np.asarray(triangle_tags, dtype=np.int64)
    triangle_array = np.asarray(triangles, dtype=np.int64)
    if (
        line_array.ndim != 2
        or line_array.shape[1] != 2
        or triangle_array.ndim != 2
        or triangle_array.shape[1] != 3
    ):
        raise CrosscheckValidationError("mesh must contain non-empty boundary lines and TRI3 rock")
    for physical_tag in (2, 3):
        if np.count_nonzero(line_physical_array == physical_tag) == 0:
            raise CrosscheckValidationError("wall and farfield must both contain line elements")
    if np.any(np.diff(np.sort(line_array, axis=1), axis=1) == 0) or np.any(
        np.diff(np.sort(triangle_array, axis=1), axis=1) == 0
    ):
        raise CrosscheckValidationError("mesh contains repeated local nodes")
    if np.unique(np.sort(line_array, axis=1), axis=0).shape[0] != line_array.shape[0]:
        raise CrosscheckValidationError("mesh contains duplicate boundary lines")
    if np.unique(np.sort(triangle_array, axis=1), axis=0).shape[0] != triangle_array.shape[0]:
        raise CrosscheckValidationError("mesh contains duplicate triangles")
    tri = nodes_xy[triangle_array]
    signed_twice_area = (tri[:, 1, 0] - tri[:, 0, 0]) * (tri[:, 2, 1] - tri[:, 0, 1])
    signed_twice_area -= (tri[:, 1, 1] - tri[:, 0, 1]) * (tri[:, 2, 0] - tri[:, 0, 0])
    if np.any(signed_twice_area <= 0.0):
        raise CrosscheckValidationError(
            "all canonical TRI3 elements must have positive orientation"
        )
    structural_payload = {
        "node_tags": node_tag_array.tolist(),
        "nodes_xy": nodes_xy.tolist(),
        "triangle_tags": triangle_tag_array.tolist(),
        "triangles": triangle_array.tolist(),
        "line_tags": line_tag_array.tolist(),
        "line_physical_tags": line_physical_array.tolist(),
        "lines": line_array.tolist(),
    }
    return CanonicalMesh(
        path=mesh_path,
        file_sha256=_sha256_bytes(raw_bytes),
        structure_sha256=_sha256_bytes(_canonical_json(structural_payload).encode("utf-8")),
        node_tags=node_tag_array,
        nodes_xy=nodes_xy,
        triangle_tags=triangle_tag_array,
        triangles=triangle_array,
        line_tags=line_tag_array,
        line_physical_tags=line_physical_array,
        lines=line_array,
    )


def _repo_root_from_config(config: Mapping[str, Any]) -> Path:
    return Path(str(config["_config_path"])).resolve().parent.parent


def _resolve_repo_path(config: Mapping[str, Any], relative: str) -> Path:
    root = _repo_root_from_config(config)
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise CrosscheckValidationError(f"configured path escapes repository: {relative}") from exc
    return candidate


def load_canonical_mesh(config: Mapping[str, Any]) -> CanonicalMesh:
    return parse_gmsh_v22_ascii(_resolve_repo_path(config, str(config["mesh"]["path"])))


def _facets_from_edges(mesh: Any, edges: IntArray, label: str) -> IntArray:
    lookup = {
        tuple(sorted(map(int, mesh.facets[:, index]))): index
        for index in range(mesh.facets.shape[1])
    }
    result: list[int] = []
    for edge in edges:
        key = tuple(sorted(map(int, edge)))
        if key not in lookup:
            raise CrosscheckValidationError(f"{label} edge is not a scikit-fem facet: {key}")
        result.append(lookup[key])
    values = np.asarray(result, dtype=np.int64)
    if np.unique(values).size != len(result):
        raise CrosscheckValidationError(f"{label} contains duplicate facet mappings")
    return values


def build_local_tunnel_mesh(mesh_data: CanonicalMesh) -> TunnelMesh:
    """Build the local scikit-fem object without changing canonical ordering."""

    try:
        from skfem import MeshTri  # type: ignore[import-not-found]
    except ImportError as exc:
        raise CrosscheckError("local cross-check requires scikit-fem 12.x") from exc
    mesh = MeshTri(mesh_data.nodes_xy.T, mesh_data.triangles.T, validate=True, sort_t=False)
    wall = _facets_from_edges(mesh, mesh_data.boundary_edges(2), WALL)
    farfield = _facets_from_edges(mesh, mesh_data.boundary_edges(3), FARFIELD)
    actual_boundary = np.sort(np.asarray(mesh.boundary_facets(), dtype=np.int64))
    marked = np.sort(np.union1d(wall, farfield))
    if not np.array_equal(actual_boundary, marked) or np.intersect1d(wall, farfield).size:
        raise CrosscheckValidationError(
            "canonical boundary lines do not partition the mesh boundary"
        )
    mesh = mesh.with_boundaries({WALL: wall, FARFIELD: farfield})
    mesh = mesh.with_subdomains({ROCK: np.arange(mesh_data.triangles.shape[0], dtype=np.int64)})
    facet_markers = np.zeros(mesh.facets.shape[1], dtype=np.int64)
    facet_markers[wall] = 2
    facet_markers[farfield] = 3
    return TunnelMesh(
        mesh=mesh,
        nodes=mesh_data.nodes_xy.copy(),
        elements=mesh_data.triangles.copy(),
        boundary_facets={WALL: wall, FARFIELD: farfield},
        facet_markers=facet_markers,
        cell_markers=np.ones(mesh_data.triangles.shape[0], dtype=np.int64),
        physical_tags={ROCK: 1, WALL: 2, FARFIELD: 3},
        outer_bounds=(
            float(mesh_data.nodes_xy[:, 0].min()),
            float(mesh_data.nodes_xy[:, 0].max()),
            float(mesh_data.nodes_xy[:, 1].min()),
            float(mesh_data.nodes_xy[:, 1].max()),
        ),
        metadata={
            "source": "canonical_gmsh_2.2_ascii",
            "mesh_file_sha256": mesh_data.file_sha256,
            "mesh_structure_sha256": mesh_data.structure_sha256,
        },
    )


def _material(config: Mapping[str, Any]) -> AT2Material:
    raw = config["material"]
    return AT2Material(
        young_modulus=float(raw["young_modulus_pa"]),
        poisson_ratio=float(raw["poisson_ratio"]),
        fracture_toughness=float(raw["fracture_toughness_j_per_m2"]),
        length_scale=float(raw["length_scale_m"]),
        residual_stiffness=float(raw["residual_stiffness"]),
    )


def _damage_for_gate(config: Mapping[str, Any], mesh_data: CanonicalMesh, gate: str) -> FloatArray:
    if gate == "intact":
        return np.zeros(mesh_data.nodes_xy.shape[0], dtype=np.float64)
    if gate != "fixed_damage":
        raise CrosscheckValidationError(f"unknown cross-check gate: {gate}")
    coefficients = np.asarray(
        config["fixed_damage_gate"]["damage_coefficients_local_yz"], dtype=np.float64
    )
    damage = coefficients[0] + mesh_data.nodes_xy @ coefficients[1:]
    if not np.isfinite(damage).all() or np.any(damage <= 0.0) or np.any(damage >= 1.0):
        raise CrosscheckValidationError(
            "fixed P1 damage must lie strictly inside (0, 1) on all nodes"
        )
    return np.asarray(damage, dtype=np.float64)


def _average_degradation(damage: FloatArray, triangles: IntArray, residual: float) -> FloatArray:
    local = damage[triangles]
    mean_damage = local.mean(axis=1)
    sum_squares = np.sum(local**2, axis=1)
    pair_sum = local[:, 0] * local[:, 1] + local[:, 0] * local[:, 2]
    pair_sum += local[:, 1] * local[:, 2]
    mean_square = (sum_squares + pair_sum) / 6.0
    return np.asarray(1.0 - 2.0 * mean_damage + mean_square + residual, dtype=np.float64)


def solve_local_case(
    config: Mapping[str, Any],
    mesh_data: CanonicalMesh,
    *,
    gate: str,
    basis_index: int,
) -> LocalCaseResult:
    """Solve one fully released same-mesh local equilibrium case."""

    gate_config = config["intact_gate" if gate == "intact" else "fixed_damage_gate"]
    basis = np.asarray(gate_config["stress_basis_tension_positive_pa"], dtype=np.float64)
    if basis_index < 0 or basis_index >= basis.shape[0]:
        raise CrosscheckValidationError("basis_index is out of range")
    stress = basis[basis_index]
    damage = _damage_for_gate(config, mesh_data, gate)
    tunnel_mesh = build_local_tunnel_mesh(mesh_data)
    material = _material(config)
    options = FractureSolverOptions(
        max_displacement_iterations=16,
        equilibrium_tolerance=1.0e-10,
        tangent_perturbation=1.0e-7,
        raise_on_nonconvergence=True,
    )
    result = solve_fixed_damage_displacement(
        tunnel_mesh,
        material,
        stress,
        load_parameter=1.0,
        damage=damage,
        options=options,
    )
    if not result.converged or not np.isfinite(result.equilibrium_residual):
        raise CrosscheckValidationError("local fixed-damage equilibrium did not converge")
    farfield_facets = np.asarray(tunnel_mesh.boundary_facets[FARFIELD], dtype=np.int64)
    farfield_nodes = np.unique(np.asarray(tunnel_mesh.mesh.facets)[:, farfield_facets])
    internal = np.asarray(result.internal_force, dtype=np.float64).reshape((-1, 2))
    reaction = internal[farfield_nodes].sum(axis=0)
    degradation_average = _average_degradation(
        damage, mesh_data.triangles, material.residual_stiffness
    )
    undegraded_split = miehe_spectral_response(
        result.strain,
        AT2Material(
            material.young_modulus,
            material.poisson_ratio,
            material.fracture_toughness,
            material.length_scale,
            0.0,
        ),
    )
    stress_axis = (
        degradation_average * undegraded_split.stress_positive[:, 0, 0]
        + undegraded_split.stress_negative[:, 0, 0]
    )
    energy_density = degradation_average * result.psi_positive + result.psi_negative
    recomputed_energy = float(mesh_data.triangle_areas @ energy_density)
    energy_scale = max(abs(recomputed_energy), abs(float(result.elastic_energy)), 1.0)
    if abs(recomputed_energy - float(result.elastic_energy)) > 1.0e-12 * energy_scale:
        raise CrosscheckValidationError("local element energy does not reproduce solver energy")
    arrays = (
        result.displacement,
        internal,
        result.strain,
        result.stress,
        stress_axis,
        energy_density,
        reaction,
        damage,
    )
    if not all(np.isfinite(np.asarray(array)).all() for array in arrays):
        raise CrosscheckValidationError("local result contains a non-finite value")
    return LocalCaseResult(
        case_id=f"{gate}-basis-{basis_index}",
        gate=gate,
        applied_stress_yz=stress.copy(),
        nodes_xy=mesh_data.nodes_xy.copy(),
        displacement_xy=np.asarray(result.displacement, dtype=np.float64),
        internal_force_xy=internal,
        farfield_node_rows=np.asarray(farfield_nodes, dtype=np.int64),
        element_centroids_xy=mesh_data.element_centroids_xy,
        strain_engineering=np.asarray(result.strain, dtype=np.float64),
        stress_inplane=np.asarray(result.stress, dtype=np.float64),
        stress_axis=np.asarray(stress_axis, dtype=np.float64),
        energy_density=np.asarray(energy_density, dtype=np.float64),
        # The cross-solver energy comparator is the area integral of the
        # exported element energy density.  It has already been checked above
        # against the solver's independently assembled scalar energy.
        elastic_energy=recomputed_energy,
        reaction_xy=np.asarray(reaction, dtype=np.float64),
        damage=damage,
    )


def _farfield_strain(stress: FloatArray, material: AT2Material) -> FloatArray:
    lame_lambda = material.lame_lambda
    mu = material.shear_modulus
    matrix = np.asarray(
        [[lame_lambda + 2.0 * mu, lame_lambda], [lame_lambda, lame_lambda + 2.0 * mu]],
        dtype=np.float64,
    )
    normal = np.linalg.solve(matrix, stress[:2])
    return np.asarray([normal[0], normal[1], stress[2] / mu], dtype=np.float64)


def render_moose_input(
    config: Mapping[str, Any],
    local_result: LocalCaseResult,
    *,
    mesh_filename: str = "mesh.msh",
    output_base: str = "moose",
) -> str:
    """Render one pinned MOOSE input; any unresolved token is fatal."""

    execution = config["execution"]
    template_key = "intact_template" if local_result.gate == "intact" else "fixed_damage_template"
    template_path = _resolve_repo_path(config, str(execution[template_key]))
    try:
        template = template_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CrosscheckValidationError(f"cannot read MOOSE template: {template_path}") from exc
    strain = _farfield_strain(local_result.applied_stress_yz, _material(config))
    ux = f"({strain[0]:.17g})*x + ({0.5 * strain[2]:.17g})*y"
    uy = f"({0.5 * strain[2]:.17g})*x + ({strain[1]:.17g})*y"
    material = config["material"]
    replacements = {
        "@MESH_FILE@": mesh_filename,
        "@OUTPUT_BASE@": output_base,
        "@UX_EXPRESSION@": ux,
        "@UY_EXPRESSION@": uy,
        "@YOUNG_MODULUS@": f"{float(material['young_modulus_pa']):.17g}",
        "@POISSON_RATIO@": f"{float(material['poisson_ratio']):.17g}",
        "@RESIDUAL_STIFFNESS@": f"{float(material['residual_stiffness']):.17g}",
        "@FRACTURE_TOUGHNESS@": f"{float(material['fracture_toughness_j_per_m2']):.17g}",
        "@LENGTH_SCALE@": f"{float(material['length_scale_m']):.17g}",
        "@DAMAGE_EXPRESSION@": str(config["fixed_damage_gate"]["damage_expression_moose_xy"]),
    }
    rendered = template
    for token, value in replacements.items():
        rendered = rendered.replace(token, value)
    unresolved = sorted(set(UNRESOLVED_TOKEN.findall(rendered)))
    if unresolved:
        raise CrosscheckValidationError(f"unresolved MOOSE template tokens: {unresolved}")
    if "boundary = wall" in rendered or "boundary = 'wall" in rendered:
        raise CrosscheckValidationError("fully released wall must remain a natural boundary")
    required_fragments = (
        "allow_renumbering = false",
        "planar_formulation = PLANE_STRAIN",
        "out_of_plane_direction = z",
        "[Quadrature]",
        "type = GAUSS",
        "order = THIRD",
        "element_order = THIRD",
        "side_order = THIRD",
        "boundary = farfield",
        "type = NodalValueSampler",
        "type = ElementValueSampler",
    )
    if not all(fragment in rendered for fragment in required_fragments):
        raise CrosscheckValidationError("rendered MOOSE input is missing a frozen semantic")
    if "quadrature_order" in rendered:
        raise CrosscheckValidationError(
            "quadrature must be configured under Executioner/Quadrature, not Problem"
        )
    expected_jit_directives = 0 if local_result.gate == "intact" else 2
    if rendered.count("enable_jit = false") != expected_jit_directives:
        raise CrosscheckValidationError(
            "fixed-damage parsed materials must use the frozen bytecode evaluator"
        )
    return rendered


def _purge_ephemeral_parser_cache(case_dir: Path) -> dict[str, Any]:
    """Remove only MOOSE's generated per-case parser cache from a fresh case dir."""

    cache = case_dir / ".jitcache"
    metadata: dict[str, Any] = {
        "path": ".jitcache",
        "generated": False,
        "file_count": 0,
        "total_bytes": 0,
        "purged": True,
        "included_in_evidence": False,
    }
    if not cache.exists():
        return metadata
    if cache.is_symlink() or not cache.is_dir() or cache.resolve().parent != case_dir.resolve():
        raise CrosscheckValidationError("refusing to purge unexpected parser-cache target")
    entries = list(cache.rglob("*"))
    if any(entry.is_symlink() for entry in entries):
        raise CrosscheckValidationError("refusing to purge parser cache containing a symlink")
    files = [entry for entry in entries if entry.is_file()]
    metadata.update(
        {
            "generated": True,
            "file_count": len(files),
            "total_bytes": sum(entry.stat().st_size for entry in files),
        }
    )
    shutil.rmtree(cache)
    if cache.exists():
        raise CrosscheckValidationError("generated parser cache could not be purged")
    return metadata


def _wsl_command(distribution: str, *args: str, cwd_linux: str | None = None) -> list[str]:
    command = ["wsl.exe", "-d", distribution]
    if cwd_linux is not None:
        command.extend(["--cd", cwd_linux])
    command.extend(["-e", *args])
    return command


def _run_checked(command: Sequence[str], *, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise MooseUnavailableError(f"could not execute WSL/MOOSE command: {command[0]}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()[-2000:]
        raise MooseUnavailableError(
            f"WSL/MOOSE command failed with exit {completed.returncode}: {detail}"
        )
    return completed


def _expand_linux_home(distribution: str, path: str) -> str:
    if not path.startswith("~/"):
        if not path.startswith("/"):
            raise CrosscheckValidationError(
                "MOOSE executable must be an absolute Linux path or ~/..."
            )
        return path
    home = _run_checked(_wsl_command(distribution, "/bin/sh", "-c", 'printf %s "$HOME"'))
    home_path = home.stdout.strip()
    if not home_path.startswith("/"):
        raise MooseUnavailableError("WSL did not return an absolute HOME path")
    return f"{home_path}/{path[2:]}"


def _probe_linux_home(distribution: str) -> str:
    result = _run_checked(_wsl_command(distribution, "/bin/sh", "-c", 'printf %s "$HOME"'))
    linux_home = result.stdout.strip()
    if not linux_home.startswith("/") or linux_home == "/":
        raise MooseUnavailableError("WSL did not return a private absolute HOME path")
    return linux_home.rstrip("/")


def probe_moose(config: Mapping[str, Any]) -> MooseEnvironment:
    """Prove the configured MOOSE executable exists, runs, and has a stable hash."""

    execution = config["execution"]
    distribution = str(execution["wsl_distribution"])
    configured_executable = str(execution["moose_executable_linux"])
    linux_home = _probe_linux_home(distribution)
    executable = (
        f"{linux_home}/{configured_executable[2:]}"
        if configured_executable.startswith("~/")
        else configured_executable
    )
    if not executable.startswith("/"):
        raise CrosscheckValidationError("MOOSE executable must resolve to an absolute Linux path")
    _run_checked(_wsl_command(distribution, "/usr/bin/test", "-x", executable))
    version_result = _run_checked(_wsl_command(distribution, executable, "--version"), timeout=60)
    version_lines = [line.strip() for line in version_result.stdout.splitlines() if line.strip()]
    if len(version_lines) != 1 or not version_lines[0].startswith("Application Version:"):
        raise MooseUnavailableError(
            "MOOSE --version output does not match the frozen probe contract"
        )
    application_version = version_lines[0].split(":", 1)[1].strip()
    hash_result = _run_checked(_wsl_command(distribution, "/usr/bin/sha256sum", executable))
    fields = hash_result.stdout.strip().split()
    if len(fields) < 1 or re.fullmatch(r"[0-9a-f]{64}", fields[0]) is None:
        raise MooseUnavailableError("could not parse MOOSE executable SHA-256")
    # modules/combined/combined-opt -> repository root.  This must use POSIX
    # path semantics even when the orchestrator itself runs on Windows.
    source_root = str(PurePosixPath(executable).parents[2])
    git_result = subprocess.run(
        _wsl_command(distribution, "/usr/bin/git", "-C", source_root, "rev-parse", "HEAD"),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    if git_result.returncode != 0 or re.fullmatch(r"[0-9a-f]{40}\s*", git_result.stdout) is None:
        raise MooseUnavailableError("MOOSE source HEAD provenance is unavailable")
    source_commit = git_result.stdout.strip()
    if (
        re.fullmatch(r"[0-9a-f]{7,40}", application_version) is None
        or source_commit[: len(application_version)] != application_version
    ):
        raise MooseUnavailableError(
            "MOOSE application version is not a prefix of the source HEAD commit"
        )
    upstream_result = _run_checked(
        _wsl_command(
            distribution,
            "/usr/bin/git",
            "-C",
            source_root,
            "rev-parse",
            "@{upstream}",
        )
    )
    upstream_commit = upstream_result.stdout.strip()
    if re.fullmatch(r"[0-9a-f]{40}", upstream_commit) is None or upstream_commit != source_commit:
        raise MooseUnavailableError("MOOSE source HEAD does not equal its configured upstream")
    status_result = _run_checked(
        _wsl_command(
            distribution,
            "/usr/bin/git",
            "-C",
            source_root,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        )
    )
    if status_result.stdout.strip():
        raise MooseUnavailableError("MOOSE source tree is not clean")
    return MooseEnvironment(
        distribution=distribution,
        linux_home=linux_home,
        executable_linux=executable,
        application_version=application_version,
        executable_sha256=fields[0],
        source_commit=source_commit,
        upstream_commit=upstream_commit,
        source_tree_clean=True,
    )


def windows_path_to_wsl(path: Path, distribution: str) -> str:
    result = _run_checked(
        _wsl_command(distribution, "/usr/bin/wslpath", "-a", "-u", str(path.resolve()))
    )
    converted = result.stdout.strip()
    if not converted.startswith("/"):
        raise CrosscheckValidationError("wslpath did not return an absolute Linux path")
    return converted


def _read_csv_table(path: Path, required: set[str]) -> dict[str, FloatArray]:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                raise CrosscheckValidationError(f"CSV has no header: {path.name}")
            names = [name.strip() for name in reader.fieldnames]
            if len(set(names)) != len(names):
                raise CrosscheckValidationError(f"CSV has duplicate headers: {path.name}")
            if not required.issubset(names):
                raise CrosscheckValidationError(
                    f"CSV {path.name} lacks required fields {sorted(required - set(names))}"
                )
            columns: dict[str, list[float]] = {name: [] for name in names}
            for row in reader:
                for name in names:
                    columns[name].append(float(row[name]))
    except (OSError, ValueError, TypeError) as exc:
        if isinstance(exc, CrosscheckValidationError):
            raise
        raise CrosscheckValidationError(f"cannot parse numeric MOOSE CSV: {path.name}") from exc
    arrays = {name: np.asarray(values, dtype=np.float64) for name, values in columns.items()}
    if not arrays or any(values.ndim != 1 or values.size == 0 for values in arrays.values()):
        raise CrosscheckValidationError(f"CSV is empty or ragged: {path.name}")
    if not all(np.isfinite(values).all() for values in arrays.values()):
        raise CrosscheckValidationError(f"CSV contains non-finite values: {path.name}")
    return arrays


def _find_sampler_csv(directory: Path, sampler: str, required: set[str]) -> Path:
    candidates: list[Path] = []
    for path in sorted(directory.glob("*.csv")):
        try:
            with path.open("r", encoding="utf-8") as stream:
                header = {field.strip() for field in stream.readline().strip().split(",")}
        except OSError:
            continue
        if required.issubset(header):
            candidates.append(path)
    if len(candidates) != 1:
        raise CrosscheckValidationError(
            f"expected exactly one {sampler} sampler CSV, found {[path.name for path in candidates]}"
        )
    return candidates[0]


def parse_moose_case_output(case_dir: str | Path, *, expect_damage: bool) -> MooseCaseResult:
    """Load MOOSE CSV output; missing, ambiguous, or non-finite fields are fatal."""

    directory = Path(case_dir).resolve()
    node_required = {"id", "x", "y", "z", "disp_x", "disp_y", "resid_x", "resid_y"}
    if expect_damage:
        node_required.add("c")
    element_required = {
        "id",
        "x",
        "y",
        "z",
        "strain_xx",
        "strain_yy",
        "strain_xy",
        "stress_xx",
        "stress_yy",
        "stress_xy",
        "stress_zz",
        "energy_density",
    }
    node_path = _find_sampler_csv(directory, "nodes", node_required)
    element_path = _find_sampler_csv(directory, "elements", element_required)
    nodes = _read_csv_table(node_path, node_required)
    elements = _read_csv_table(element_path, element_required)
    node_ids = nodes["id"]
    element_ids = elements["id"]
    if not np.array_equal(node_ids, np.rint(node_ids)) or not np.array_equal(
        element_ids, np.rint(element_ids)
    ):
        raise CrosscheckValidationError("MOOSE sampler IDs must be exact integers")
    if np.unique(node_ids).size != node_ids.size or np.unique(element_ids).size != element_ids.size:
        raise CrosscheckValidationError("MOOSE sampler IDs must be unique")
    if np.max(np.abs(nodes["z"])) > 1.0e-14 or np.max(np.abs(elements["z"])) > 1.0e-14:
        raise CrosscheckValidationError("MOOSE output violates the z=axis plane")
    return MooseCaseResult(
        node_ids=np.asarray(node_ids, dtype=np.int64),
        nodes_xy=np.column_stack((nodes["x"], nodes["y"])),
        displacement_xy=np.column_stack((nodes["disp_x"], nodes["disp_y"])),
        residual_xy=np.column_stack((nodes["resid_x"], nodes["resid_y"])),
        damage=np.asarray(nodes["c"], dtype=np.float64) if expect_damage else None,
        element_ids=np.asarray(element_ids, dtype=np.int64),
        element_centroids_xy=np.column_stack((elements["x"], elements["y"])),
        strain_engineering=np.column_stack(
            (elements["strain_xx"], elements["strain_yy"], 2.0 * elements["strain_xy"])
        ),
        stress_inplane=np.column_stack(
            (elements["stress_xx"], elements["stress_yy"], elements["stress_xy"])
        ),
        stress_axis=np.asarray(elements["stress_zz"], dtype=np.float64),
        energy_density=np.asarray(elements["energy_density"], dtype=np.float64),
    )


def _bijective_coordinate_map(local: FloatArray, remote: FloatArray, tolerance: float) -> IntArray:
    if local.shape != remote.shape or local.ndim != 2:
        raise CrosscheckValidationError(
            "coordinate tables must have the same two-dimensional shape"
        )
    distances = np.linalg.norm(local[:, None, :] - remote[None, :, :], axis=2)
    mapping = np.argmin(distances, axis=1)
    minimum = distances[np.arange(local.shape[0]), mapping]
    if np.any(minimum > tolerance):
        raise CrosscheckValidationError(
            f"coordinate identity exceeds tolerance; max={float(minimum.max()):.3e}"
        )
    if np.unique(mapping).size != local.shape[0]:
        raise CrosscheckValidationError("coordinate mapping is not bijective")
    close_counts = np.sum(distances <= tolerance, axis=1)
    if np.any(close_counts != 1):
        raise CrosscheckValidationError("coordinate identity is ambiguous within tolerance")
    return np.asarray(mapping, dtype=np.int64)


def _metric(
    local: np.ndarray | float,
    remote: np.ndarray | float,
    *,
    normalization_floor: float,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, Any]:
    left = np.asarray(local, dtype=np.float64)
    right = np.asarray(remote, dtype=np.float64)
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise CrosscheckValidationError("metric inputs must have the same finite shape")
    difference = left - right
    left_norm = float(np.linalg.norm(left.ravel()))
    right_norm = float(np.linalg.norm(right.ravel()))
    difference_norm = float(np.linalg.norm(difference.ravel()))
    reference_norm = max(left_norm, right_norm)
    max_absolute = float(np.max(np.abs(difference), initial=0.0))
    if reference_norm <= normalization_floor:
        primary_error = relative_tolerance * max_absolute / absolute_tolerance
        mode = "absolute_near_zero_scaled_to_primary"
    else:
        primary_error = difference_norm / reference_norm
        mode = "relative_l2"
    return {
        "primary_error": float(primary_error),
        "primary_tolerance": float(relative_tolerance),
        "pass": bool(primary_error <= relative_tolerance),
        "mode": mode,
        "relative_l2": float(difference_norm / max(reference_norm, normalization_floor)),
        "max_absolute": max_absolute,
        "absolute_tolerance": float(absolute_tolerance),
        "local_l2": left_norm,
        "moose_l2": right_norm,
    }


def compare_case(
    config: Mapping[str, Any],
    mesh_data: CanonicalMesh,
    local: LocalCaseResult,
    moose: MooseCaseResult,
) -> dict[str, Any]:
    """Compare one case after strict node and element coordinate bijections."""

    comparison = config["comparison"]
    absolute = comparison["absolute_tolerances"]
    floors = comparison["normalization_floors"]
    relative_tolerance = float(comparison["primary_relative_l2_tolerance"])
    coordinate_tolerance = float(absolute["coordinate_m"])
    node_map = _bijective_coordinate_map(local.nodes_xy, moose.nodes_xy, coordinate_tolerance)
    element_map = _bijective_coordinate_map(
        local.element_centroids_xy, moose.element_centroids_xy, coordinate_tolerance
    )
    moose_u = moose.displacement_xy[node_map]
    moose_residual = moose.residual_xy[node_map]
    moose_strain = moose.strain_engineering[element_map]
    moose_stress = moose.stress_inplane[element_map]
    moose_axis = moose.stress_axis[element_map]
    moose_reported_energy_density = moose.energy_density[element_map]
    if local.gate == "fixed_damage":
        if moose.damage is None:
            raise CrosscheckValidationError("fixed-damage MOOSE output omitted damage")
        moose_damage = moose.damage[node_map]
    else:
        moose_damage = np.zeros_like(local.damage)
    # ComputeLinearElasticPFFractureStress does not populate the elastic_strain
    # property consumed by ElasticEnergyAux, so that raw fixed-damage CSV column
    # is deliberately excluded.  Recompute energy from the exported MOOSE
    # strain and exported P1 damage with the frozen local definition instead.
    material = _material(config)
    moose_split = miehe_spectral_response(
        moose_strain,
        AT2Material(
            material.young_modulus,
            material.poisson_ratio,
            material.fracture_toughness,
            material.length_scale,
            0.0,
        ),
    )
    degradation_average = _average_degradation(
        moose_damage, mesh_data.triangles, material.residual_stiffness
    )
    moose_energy_density = degradation_average * moose_split.psi_positive + moose_split.psi_negative
    moose_energy = float(mesh_data.triangle_areas @ moose_energy_density)
    moose_reaction = moose_residual[local.farfield_node_rows].sum(axis=0)
    metric_specs = {
        "displacement": (
            local.displacement_xy,
            moose_u,
            "displacement_m",
            "displacement_m",
        ),
        "strain": (local.strain_engineering, moose_strain, "strain", "strain"),
        "stress_inplane": (local.stress_inplane, moose_stress, "stress_pa", "stress_pa"),
        "stress_axis": (local.stress_axis, moose_axis, "stress_pa", "stress_pa"),
        "energy_density": (
            local.energy_density,
            moose_energy_density,
            "stress_pa",
            "stress_pa",
        ),
        "elastic_energy": (
            local.elastic_energy,
            moose_energy,
            "energy_j_per_m",
            "energy_j_per_m",
        ),
        "farfield_reaction": (
            local.reaction_xy,
            moose_reaction,
            "reaction_n_per_m",
            "reaction_n_per_m",
        ),
        "farfield_reaction_nodal": (
            local.internal_force_xy[local.farfield_node_rows],
            moose_residual[local.farfield_node_rows],
            "reaction_n_per_m",
            "reaction_n_per_m",
        ),
    }
    metrics = {
        name: _metric(
            local_value,
            moose_value,
            normalization_floor=float(floors[floor_key]),
            absolute_tolerance=float(absolute[absolute_key]),
            relative_tolerance=relative_tolerance,
        )
        for name, (local_value, moose_value, floor_key, absolute_key) in metric_specs.items()
    }
    if local.gate == "fixed_damage":
        metrics["nodal_damage"] = _metric(
            local.damage,
            moose_damage,
            normalization_floor=float(floors["strain"]),
            absolute_tolerance=float(absolute["strain"]),
            relative_tolerance=relative_tolerance,
        )
    else:
        # Only intact ComputeLinearElasticStress + ElasticEnergyAux is the
        # same energy definition and therefore eligible for this auxiliary
        # implementation-path consistency metric.
        metrics["moose_reported_intact_energy_density"] = _metric(
            moose_energy_density,
            moose_reported_energy_density,
            normalization_floor=float(floors["stress_pa"]),
            absolute_tolerance=float(absolute["stress_pa"]),
            relative_tolerance=relative_tolerance,
        )
    passed = all(metric["pass"] for metric in metrics.values())
    return {
        "case_id": local.case_id,
        "gate": local.gate,
        "status": "pass" if passed else "fail",
        "pass": bool(passed),
        "mesh_file_sha256": mesh_data.file_sha256,
        "mesh_structure_sha256": mesh_data.structure_sha256,
        "energy_contract": (
            "exported MOOSE strain + exported nodal P1 damage; raw PF "
            "ElasticEnergyAux column excluded"
            if local.gate == "fixed_damage"
            else "exported MOOSE strain recomputation + intact ElasticEnergyAux cross-check"
        ),
        "node_mapping": {
            "method": "unique_coordinate_bijection",
            "count": int(node_map.size),
            "max_coordinate_error_m": float(
                np.max(
                    np.linalg.norm(local.nodes_xy - moose.nodes_xy[node_map], axis=1), initial=0.0
                )
            ),
        },
        "element_mapping": {
            "method": "unique_centroid_bijection_on_identical_mesh_file",
            "count": int(element_map.size),
            "max_coordinate_error_m": float(
                np.max(
                    np.linalg.norm(
                        local.element_centroids_xy - moose.element_centroids_xy[element_map], axis=1
                    ),
                    initial=0.0,
                )
            ),
        },
        "metrics": metrics,
    }


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(_canonical_json(value) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _save_local_result(path: Path, result: LocalCaseResult) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **result.arrays())
    os.replace(temporary, path)
    return _sha256_file(path)


def _sanitize_log(
    text: str,
    *,
    repo_root: Path,
    repo_root_wsl: str,
    environment: MooseEnvironment,
) -> str:
    sanitized = text.replace(str(repo_root), "<WORKSPACE>")
    sanitized = sanitized.replace(str(repo_root).replace("\\", "/"), "<WORKSPACE>")
    sanitized = sanitized.replace(repo_root_wsl, "<WORKSPACE>")
    sanitized = sanitized.replace(environment.executable_linux, "<MOOSE_EXECUTABLE>")
    for private_root in (environment.linux_home,):
        sanitized = sanitized.replace(private_root, "<WSL_HOME>")
    return sanitized


def _project_source_provenance(config: Mapping[str, Any]) -> dict[str, Any]:
    """Bind the exact local implementation closure and honest Git state."""

    repo_root = _repo_root_from_config(config)
    root_resolved = repo_root.resolve()
    inventory: dict[str, dict[str, Any]] = {}
    package_sources = sorted(
        path.relative_to(repo_root).as_posix()
        for path in (repo_root / "src" / "tunnelgeopt").glob("*.py")
        if path.is_file()
    )
    if not package_sources or "src/tunnelgeopt/__init__.py" not in package_sources:
        raise CrosscheckValidationError("full TunnelGeoPT package source inventory is unavailable")
    implementation_closure = sorted((*PROJECT_NONPACKAGE_INPUTS, *package_sources))
    for relative in implementation_closure:
        path = (repo_root / relative).resolve()
        try:
            path.relative_to(root_resolved)
        except ValueError as exc:
            raise CrosscheckValidationError(
                f"implementation source escaped repository root: {relative}"
            ) from exc
        if not path.is_file() or path.is_symlink():
            raise CrosscheckValidationError(
                f"implementation source is missing or not a regular file: {relative}"
            )
        inventory[relative] = {
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
        }

    def git_output(*args: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        if completed.returncode != 0:
            raise CrosscheckValidationError(
                f"project Git provenance command failed: {' '.join(args)}"
            )
        return completed.stdout

    head = git_output("rev-parse", "HEAD").strip()
    upstream = git_output("rev-parse", "@{upstream}").strip()
    if re.fullmatch(r"[0-9a-f]{40}", head) is None:
        raise CrosscheckValidationError("project Git HEAD is not a full commit hash")
    if re.fullmatch(r"[0-9a-f]{40}", upstream) is None:
        raise CrosscheckValidationError("project Git upstream is not a full commit hash")
    status = git_output("status", "--porcelain=v1", "--untracked-files=all")
    status_lines = [line for line in status.splitlines() if line]
    tracked_changes = sum(not line.startswith("??") for line in status_lines)
    untracked_entries = sum(line.startswith("??") for line in status_lines)
    runtime_versions = {
        distribution: importlib.metadata.version(distribution)
        for distribution in ("numpy", "scipy", "scikit-fem", "tunnelgeopt")
    }
    return {
        "capture_phase": "before artifact directory creation",
        "implementation_closure_method": (
            "full sorted src/tunnelgeopt/*.py package inventory plus runner/config/templates/"
            "pyproject v1"
        ),
        "implementation_files": inventory,
        "repository": {
            "head_commit": head,
            "upstream_commit": upstream,
            "head_equals_upstream": head == upstream,
            "worktree_dirty": bool(status_lines),
            "status_entry_count": len(status_lines),
            "tracked_change_count": tracked_changes,
            "untracked_entry_count": untracked_entries,
            "status_porcelain_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
            "dirty_tree_policy": (
                "dirty state is permitted and reported; exact implementation files are bound "
                "individually by SHA-256"
            ),
        },
        "runtime": {
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "python_executable_version_info": list(sys.version_info[:3]),
            "distributions": runtime_versions,
        },
    }


def run_crosscheck(
    config_path: str | Path,
    artifact_dir: str | Path,
    *,
    run_moose: bool,
) -> dict[str, Any]:
    """Prepare local evidence and optionally execute the real pinned MOOSE binary."""

    config = load_crosscheck_config(config_path)
    mesh_data = load_canonical_mesh(config)
    project_source_provenance = _project_source_provenance(config)
    root = Path(artifact_dir).resolve()
    if root.exists():
        if any(root.iterdir()):
            raise CrosscheckValidationError(
                "artifact_dir must be a new empty directory; evidence runs are immutable"
            )
    else:
        root.mkdir(parents=True, exist_ok=False)
    mesh_copy = root / "canonical_mesh.msh"
    if mesh_copy.exists() and _sha256_file(mesh_copy) != mesh_data.file_sha256:
        raise CrosscheckValidationError("artifact mesh copy exists with a foreign hash")
    shutil.copyfile(mesh_data.path, mesh_copy)
    if _sha256_file(mesh_copy) != mesh_data.file_sha256:
        raise CrosscheckValidationError("artifact mesh copy SHA-256 differs from canonical mesh")

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "gate_id": config["gate_id"],
        "claim_boundary": config["claim_boundary"],
        "status": "prepared_local_only" if not run_moose else "running",
        "config_sha256": config["_config_sha256"],
        "mesh_file_sha256": mesh_data.file_sha256,
        "mesh_structure_sha256": mesh_data.structure_sha256,
        "project_source_provenance": project_source_provenance,
        "mesh_counts": {
            "nodes": int(mesh_data.nodes_xy.shape[0]),
            "tri3": int(mesh_data.triangles.shape[0]),
            "wall_lines": int(np.count_nonzero(mesh_data.line_physical_tags == 2)),
            "farfield_lines": int(np.count_nonzero(mesh_data.line_physical_tags == 3)),
        },
        "coordinate_contract": config["coordinate_contract"],
        "discretization_contract": {
            "displacement": "P1 LAGRANGE",
            "damage": "P1 LAGRANGE fixed auxiliary field",
            "volume_quadrature": "GAUSS THIRD",
            "element_quadrature_order": "THIRD",
            "side_quadrature_order": "THIRD",
            "stress_strain_element_field_projection": (
                "RankTwoAux CONSTANT MONOMIAL JxW-over-volume, then ElementValueSampler"
            ),
            "fixed_damage_energy_comparison": (
                "offline recomputation from exported MOOSE strain and exported MOOSE nodal "
                "P1 damage"
            ),
            "fixed_damage_raw_energy_aux": (
                "hashed diagnostic CSV column only; excluded because the PF stress material "
                "does not populate ElasticEnergyAux elastic_strain"
            ),
            "intact_reported_energy_projection": (
                "ElasticEnergyAux CONSTANT MONOMIAL JxW-over-volume, then ElementValueSampler"
            ),
            "parser_cache_policy": (
                "parsed-material JIT disabled; generated symbolic derivative cache is purged "
                "from each fresh case and recorded as excluded ephemeral state"
            ),
        },
        "cases": [],
    }
    _atomic_json(root / "manifest.json", manifest)
    environment = probe_moose(config) if run_moose else None
    repo_root = _repo_root_from_config(config)
    repo_root_wsl = (
        windows_path_to_wsl(repo_root, environment.distribution)
        if environment is not None
        else "<NO_WSL_WORKSPACE>"
    )
    if environment is not None:
        manifest["moose_environment"] = {
            "application_version": environment.application_version,
            "executable_sha256": environment.executable_sha256,
            "source_commit": environment.source_commit,
            "upstream_commit": environment.upstream_commit,
            "source_tree_clean": environment.source_tree_clean,
            "wsl_distribution": environment.distribution,
            "configured_executable": "<MOOSE_EXECUTABLE>",
        }

    gate_reports: dict[str, Any] = {}
    for gate in ("intact", "fixed_damage"):
        if gate == "fixed_damage" and run_moose and not gate_reports.get("intact", {}).get("pass"):
            gate_reports[gate] = {
                "status": "skipped",
                "pass": False,
                "reason": "conditional gate not run because intact gate did not pass",
            }
            continue
        gate_config = config["intact_gate" if gate == "intact" else "fixed_damage_gate"]
        basis = np.asarray(gate_config["stress_basis_tension_positive_pa"], dtype=np.float64)
        case_reports: list[dict[str, Any]] = []
        for basis_index in range(basis.shape[0]):
            local = solve_local_case(config, mesh_data, gate=gate, basis_index=basis_index)
            case_dir = root / "cases" / local.case_id
            case_dir.mkdir(parents=True, exist_ok=True)
            local_path = case_dir / "local_fields.npz"
            local_sha = _save_local_result(local_path, local)
            shutil.copyfile(mesh_data.path, case_dir / "mesh.msh")
            if _sha256_file(case_dir / "mesh.msh") != mesh_data.file_sha256:
                raise CrosscheckValidationError("per-case MOOSE mesh copy hash mismatch")
            rendered = render_moose_input(config, local)
            input_path = case_dir / "input.i"
            input_path.write_text(rendered, encoding="utf-8", newline="\n")
            case_manifest: dict[str, Any] = {
                "case_id": local.case_id,
                "gate": gate,
                "basis_index": basis_index,
                "applied_stress_tension_positive_pa": basis[basis_index].tolist(),
                "local_fields_sha256": local_sha,
                "mesh_file_sha256": _sha256_file(case_dir / "mesh.msh"),
                "moose_input_sha256": _sha256_file(input_path),
                "status": "prepared_local_only" if not run_moose else "running",
            }
            if run_moose:
                assert environment is not None
                stale_csv = sorted(case_dir.glob("*.csv"))
                if stale_csv:
                    raise CrosscheckValidationError(
                        f"case execution directory is not fresh: {[path.name for path in stale_csv]}"
                    )
                cwd_linux = windows_path_to_wsl(case_dir, environment.distribution)
                command = _wsl_command(
                    environment.distribution,
                    environment.executable_linux,
                    "-i",
                    "input.i",
                    "--n-threads=1",
                    cwd_linux=cwd_linux,
                )
                completed = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=180,
                )
                combined = completed.stdout + ("\n" + completed.stderr if completed.stderr else "")
                sanitized_log = _sanitize_log(
                    combined,
                    repo_root=repo_root,
                    repo_root_wsl=repo_root_wsl,
                    environment=environment,
                )
                log_path = case_dir / "moose.log"
                log_path.write_text(sanitized_log, encoding="utf-8", newline="\n")
                case_manifest["command"] = [
                    "<MOOSE_EXECUTABLE>",
                    "-i",
                    "input.i",
                    "--n-threads=1",
                ]
                case_manifest["ephemeral_parser_cache"] = _purge_ephemeral_parser_cache(case_dir)
                if "Num Threads:             1" not in combined:
                    case_manifest["status"] = "moose_thread_contract_failed"
                    _atomic_json(case_dir / "case_manifest.json", case_manifest)
                    raise CrosscheckValidationError(
                        f"MOOSE thread-count evidence missing for {local.case_id}"
                    )
                case_manifest["moose_log_sha256"] = _sha256_file(log_path)
                case_manifest["moose_returncode"] = int(completed.returncode)
                if completed.returncode != 0:
                    case_manifest["status"] = "moose_failed"
                    _atomic_json(case_dir / "case_manifest.json", case_manifest)
                    raise CrosscheckValidationError(
                        f"MOOSE failed for {local.case_id}; inspect {log_path}"
                    )
                moose = parse_moose_case_output(case_dir, expect_damage=gate == "fixed_damage")
                node_csv = _find_sampler_csv(
                    case_dir,
                    "nodes",
                    {"id", "x", "y", "z", "disp_x", "disp_y", "resid_x", "resid_y"}
                    | ({"c"} if gate == "fixed_damage" else set()),
                )
                element_csv = _find_sampler_csv(
                    case_dir,
                    "elements",
                    {
                        "id",
                        "x",
                        "y",
                        "z",
                        "strain_xx",
                        "strain_yy",
                        "strain_xy",
                        "stress_xx",
                        "stress_yy",
                        "stress_xy",
                        "stress_zz",
                        "energy_density",
                    },
                )
                case_manifest["moose_outputs"] = {
                    "node_csv": {
                        "filename": node_csv.name,
                        "sha256": _sha256_file(node_csv),
                        "size_bytes": node_csv.stat().st_size,
                    },
                    "element_csv": {
                        "filename": element_csv.name,
                        "sha256": _sha256_file(element_csv),
                        "size_bytes": element_csv.stat().st_size,
                    },
                }
                report = compare_case(config, mesh_data, local, moose)
                report["moose_output_sha256"] = {
                    key: value["sha256"] for key, value in case_manifest["moose_outputs"].items()
                }
                case_manifest["status"] = report["status"]
                comparison_path = case_dir / "comparison.json"
                _atomic_json(comparison_path, report)
                case_manifest["comparison_sha256"] = _sha256_file(comparison_path)
                case_reports.append(report)
            _atomic_json(case_dir / "case_manifest.json", case_manifest)
            case_manifest["case_manifest_sha256"] = _sha256_file(case_dir / "case_manifest.json")
            manifest["cases"].append(case_manifest)
        if run_moose:
            gate_pass = len(case_reports) == 3 and all(report["pass"] for report in case_reports)
            gate_reports[gate] = {
                "status": "pass" if gate_pass else "fail",
                "pass": bool(gate_pass),
                "case_count": len(case_reports),
                "maximum_primary_error": float(
                    max(
                        metric["primary_error"]
                        for report in case_reports
                        for metric in report["metrics"].values()
                    )
                ),
                "cases": case_reports,
            }
        else:
            gate_reports[gate] = {
                "status": "prepared_local_only",
                "pass": False,
                "case_count": int(basis.shape[0]),
            }

    overall_pass = bool(
        run_moose
        and gate_reports.get("intact", {}).get("pass")
        and gate_reports.get("fixed_damage", {}).get("pass")
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "gate_id": config["gate_id"],
        "status": "pass" if overall_pass else "prepared_local_only" if not run_moose else "fail",
        "pass": overall_pass,
        "claim_boundary": config["claim_boundary"],
        "primary_relative_l2_tolerance": config["comparison"]["primary_relative_l2_tolerance"],
        "mesh_file_sha256": mesh_data.file_sha256,
        "mesh_structure_sha256": mesh_data.structure_sha256,
        "gates": gate_reports,
        "validated_scope": (
            "intact and fixed-nonuniform-damage same-mesh fixed-u equilibrium only"
            if overall_pass
            else "none; local preparation or failed cross-solver gate"
        ),
        "explicitly_not_validated": [
            "coupled phase-field crack evolution",
            "irreversibility and adaptive retry",
            "path work and energy balance over trajectories",
            "SENT/SENS benchmarks",
            "dynamic rockburst or field prediction",
        ],
    }
    _atomic_json(root / "report.json", report)
    manifest["status"] = report["status"]
    manifest["report_sha256"] = _sha256_file(root / "report.json")
    _atomic_json(root / "manifest.json", manifest)
    return report


__all__ = [
    "CanonicalMesh",
    "CrosscheckError",
    "CrosscheckValidationError",
    "LocalCaseResult",
    "MooseCaseResult",
    "MooseEnvironment",
    "MooseUnavailableError",
    "build_local_tunnel_mesh",
    "compare_case",
    "load_canonical_mesh",
    "load_crosscheck_config",
    "parse_gmsh_v22_ascii",
    "parse_moose_case_output",
    "probe_moose",
    "render_moose_input",
    "run_crosscheck",
    "solve_local_case",
    "windows_path_to_wsl",
]
