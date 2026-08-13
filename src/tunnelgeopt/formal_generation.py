"""Deterministic, sealed generation for the v0.3 multi-fidelity experiment.

This module is the trusted-generator boundary.  It is deliberately separate
from the training runner: the public return objects never contain locked label
paths, while :func:`trusted_locked_label_path` is an explicit evaluator-only
escape hatch.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import tracemalloc
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .geometry import (
    points_inside_polygon,
    shape_parameter_bounds,
    surface_points_and_normals,
)
from .mesh import generate_tunnel_mesh
from .multifidelity import (
    GeometryDataSpec,
    MeshFidelitySpec,
    build_elastic_query_grid,
    case_group_id,
    load_group_id,
    solve_multifidelity_case,
)
from .multifidelity_learning import case_weighted_stress_error

FloatArray = NDArray[np.floating]

FORMAL_PARTITIONS = (
    "train_id",
    "dev_id",
    "locked_iid",
    "locked_geometry_ood",
    "locked_load_ood",
    "locked_joint_ood",
)
LOCKED_PARTITIONS = FORMAL_PARTITIONS[2:]
CORE_SPLIT = {
    "train_id": "train",
    "dev_id": "dev",
    "locked_iid": "locked_test",
    "locked_geometry_ood": "locked_test",
    "locked_load_ood": "locked_test",
    "locked_joint_ood": "locked_test",
}
PUBLIC_FILENAME = "public_inputs_and_coarse_fields.npz"
TRAIN_DEV_FILENAME = "train_dev_fine_labels.npz"
MANIFEST_FILENAME = "formal_dataset_manifest.json"


class FormalGenerationError(RuntimeError):
    """Raised when generation would violate the frozen formal contract."""


@dataclass(frozen=True)
class FormalGenerationOverrides:
    """Small-run overrides; using any override makes a plan non-formal.

    ``parents_per_section`` and ``loads_per_parent`` may provide only the
    partitions being generated.  ``partitions`` and ``section_families`` make
    one-case real-FEM tests possible without pretending they are formal runs.
    """

    partitions: tuple[str, ...] | None = None
    section_families: tuple[str, ...] | None = None
    parents_per_section: Mapping[str, int] = field(default_factory=dict)
    loads_per_parent: Mapping[str, int] = field(default_factory=dict)
    boundary_points: int | None = None
    query_region_counts: tuple[int, int, int] | None = None
    audit_fraction: float | None = None
    audit_minimum_per_partition_section: int | None = None

    @property
    def active(self) -> bool:
        return any(
            (
                self.partitions is not None,
                self.section_families is not None,
                bool(self.parents_per_section),
                bool(self.loads_per_parent),
                self.boundary_points is not None,
                self.query_region_counts is not None,
                self.audit_fraction is not None,
                self.audit_minimum_per_partition_section is not None,
            )
        )


@dataclass(frozen=True)
class FrozenIdentityExclusions:
    """Previously seen identities that a fresh formal plan must not reuse."""

    geometry_group_ids: frozenset[str] = frozenset()
    boundary_float64_sha256: frozenset[str] = frozenset()
    case_group_ids: frozenset[str] = frozenset()
    load_group_ids: frozenset[str] = frozenset()
    source_artifact_sha256: str | None = None
    source_record_count: int = 0

    def __post_init__(self) -> None:
        collections = (
            self.geometry_group_ids,
            self.boundary_float64_sha256,
            self.case_group_ids,
            self.load_group_ids,
        )
        for values in collections:
            if any(len(str(value)) != 64 for value in values):
                raise FormalGenerationError("exclusion identities must be SHA-256 strings")
        if self.source_artifact_sha256 is not None and (
            len(self.source_artifact_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.source_artifact_sha256)
        ):
            raise FormalGenerationError("exclusion source artifact digest must be SHA-256")
        if int(self.source_record_count) < 0:
            raise FormalGenerationError("exclusion source record count must be non-negative")


@dataclass(frozen=True)
class PlannedGeometry:
    formal_partition: str
    core_split: str
    section_family: str
    parent_index: int
    spec: GeometryDataSpec
    geometry_group_id: str
    boundary_float64_sha256: str
    normalized_parameter_positions: Mapping[str, float]
    ood_parameter: str | None
    ood_side: str | None
    query_seed: int


@dataclass(frozen=True)
class PlannedCase:
    formal_partition: str
    core_split: str
    section_family: str
    parent_index: int
    load_index: int
    geometry_group_id: str
    boundary_float64_sha256: str
    load_group_id: str
    case_group_id: str
    load_subtype: str
    sigma1: float
    sigma3_over_sigma1: float
    principal_angle_deg: float
    sigma_inf_tension_positive: FloatArray

    def __post_init__(self) -> None:
        stress = np.asarray(self.sigma_inf_tension_positive, dtype=np.float64).copy()
        stress.setflags(write=False)
        object.__setattr__(self, "sigma_inf_tension_positive", stress)


@dataclass(frozen=True)
class FormalGenerationPlan:
    config_sha256: str
    run_id: str
    geometries: tuple[PlannedGeometry, ...]
    cases: tuple[PlannedCase, ...]
    audit_case_ids: tuple[str, ...]
    formal_eligible: bool
    identity_report: Mapping[str, Any]
    generator_protocol: Mapping[str, Any]


@dataclass(frozen=True)
class TrainingDataPaths:
    public_inputs_path: Path
    train_dev_fine_labels_path: Path
    dataset_manifest_path: Path


@dataclass(frozen=True)
class FormalGenerationResult:
    """Safe result for the orchestrator; it intentionally omits sealed paths."""

    manifest_path: Path
    public_inputs_path: Path
    train_dev_labels_path: Path
    audit_summary: Mapping[str, Any]
    public_file_hashes: Mapping[str, str]
    opaque_sealed_store_hashes: Mapping[str, str]
    resumed_cases: int
    solved_cases: int

    @property
    def file_hashes(self) -> Mapping[str, str]:
        """Backward-compatible name for the deliberately public-only hash map."""

        return self.public_file_hashes

    @property
    def train_dev_fine_labels_path(self) -> Path:
        return self.train_dev_labels_path


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    return _sha256(path.read_bytes())


def _boundary_sha256(boundary: FloatArray) -> str:
    return _sha256(np.ascontiguousarray(boundary, dtype="<f8").tobytes())


def _substream_seed(
    generator_seed: int,
    purpose: str,
    section: str,
    parent_index: int = -1,
    load_index: int = -1,
) -> int:
    payload = f"{int(generator_seed)}|{purpose}|{section}|{parent_index}|{load_index}"
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "little")


def _rng(
    generator_seed: int,
    purpose: str,
    section: str,
    parent_index: int = -1,
    load_index: int = -1,
) -> np.random.Generator:
    return np.random.default_rng(
        _substream_seed(generator_seed, purpose, section, parent_index, load_index)
    )


def _sobol(count: int, dimension: int, seed: int) -> FloatArray:
    try:
        from scipy.stats import qmc
    except ImportError as exc:  # pragma: no cover - solver env always includes scipy
        raise FormalGenerationError("formal scrambled Sobol generation requires scipy") from exc
    power = max(0, math.ceil(math.log2(max(count, 1))))
    values = qmc.Sobol(d=dimension, scramble=True, seed=int(seed % (2**32))).random_base2(power)
    return np.asarray(values[:count], dtype=np.float64)


def _partition_seed(config: Mapping[str, Any], partition: str) -> int:
    seeds = config["dataset"]["generator_seeds"]
    return int(seeds["train_dev"] if partition in ("train_id", "dev_id") else seeds[partition])


def _geometry_regime(partition: str) -> str:
    return "ood" if partition in ("locked_geometry_ood", "locked_joint_ood") else "id"


def _load_regime(partition: str, load_index: int) -> str:
    if partition not in ("locked_load_ood", "locked_joint_ood"):
        return "id"
    return ("low_lateral_ratio", "large_rotation", "joint_low_lateral_large_rotation")[
        load_index % 3
    ]


def _scaled_parameters(
    config: Mapping[str, Any],
    *,
    partition: str,
    section: str,
    parent_index: int,
    unit: FloatArray,
    seed: int,
) -> tuple[dict[str, float], dict[str, float], str | None, str | None]:
    names = tuple(config["geometry"]["continuous_parameters"][section])
    bounds = shape_parameter_bounds(section)
    if tuple(bounds) != names:
        raise FormalGenerationError(f"config parameter order changed for {section}")
    positions: dict[str, float] = {}
    ood_parameter: str | None = None
    ood_side: str | None = None
    if _geometry_regime(partition) == "ood":
        anchor_rng = _rng(seed, f"{partition}:ood_anchor", section, parent_index)
        ood_parameter = names[int(anchor_rng.integers(0, len(names)))]
        ood_side = "low" if int(anchor_rng.integers(0, 2)) == 0 else "high"
    id_low, id_high = map(float, config["geometry"]["id_parameter_position"])
    ood_ranges = config["geometry"]["geometry_ood_parameter_position"]
    for index, name in enumerate(names):
        if name == ood_parameter:
            low, high = map(float, ood_ranges[str(ood_side)])
        else:
            low, high = id_low, id_high
        positions[name] = low + float(unit[index]) * (high - low)
    parameters = {
        name: float(bounds[name][0] + positions[name] * (bounds[name][1] - bounds[name][0]))
        for name in names
    }
    return parameters, positions, ood_parameter, ood_side


def _load(
    config: Mapping[str, Any],
    *,
    partition: str,
    section: str,
    parent_index: int,
    load_index: int,
    seed: int,
    stream_namespace: str | None = None,
) -> tuple[str, float, float, float, FloatArray]:
    material = config["material_and_loads"]
    subtype = _load_regime(partition, load_index)
    namespace = partition if stream_namespace is None else str(stream_namespace)
    random = _rng(seed, f"{namespace}:load:{subtype}", section, parent_index, load_index)
    sigma1 = float(
        random.uniform(
            *map(float, material["id"]["sigma1_over_reference_stress_compression_positive"])
        )
    )
    if subtype == "id":
        ratio = float(random.uniform(*map(float, material["id"]["sigma3_over_sigma1"])))
        angle = float(random.uniform(*map(float, material["id"]["principal_angle_deg"])))
    else:
        ranges = material["load_ood_subtypes"][subtype]
        ratio = float(random.uniform(*map(float, ranges["sigma3_over_sigma1"])))
        if "principal_angle_deg" in ranges:
            angle = float(random.uniform(*map(float, ranges["principal_angle_deg"])))
        else:
            unions = ranges["principal_angle_deg_union"]
            branch = int(random.integers(0, len(unions)))
            angle = float(random.uniform(*map(float, unions[branch])))
    radians = math.radians(angle)
    principal = np.asarray([math.cos(radians), math.sin(radians)])
    transverse = np.asarray([-math.sin(radians), math.cos(radians)])
    stress = -sigma1 * np.outer(principal, principal) - sigma1 * ratio * np.outer(
        transverse, transverse
    )
    return subtype, sigma1, ratio, angle, np.asarray(stress, dtype=np.float64)


def _resolved_design(
    config: Mapping[str, Any], overrides: FormalGenerationOverrides
) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, int], dict[str, int]]:
    partitions = overrides.partitions or FORMAL_PARTITIONS
    sections = overrides.section_families or tuple(config["geometry"]["section_families"])
    if not partitions or any(name not in FORMAL_PARTITIONS for name in partitions):
        raise FormalGenerationError("overridden partitions must be a non-empty formal subset")
    if not sections or any(name not in config["geometry"]["section_families"] for name in sections):
        raise FormalGenerationError("overridden sections must be a non-empty configured subset")
    if len(set(partitions)) != len(partitions) or len(set(sections)) != len(sections):
        raise FormalGenerationError("partition and section lists must not repeat values")
    parent_counts = {
        name: int(
            overrides.parents_per_section.get(
                name, config["dataset"]["partitions"][name]["parents_per_section"]
            )
        )
        for name in partitions
    }
    load_counts = {
        name: int(
            overrides.loads_per_parent.get(
                name, config["dataset"]["partitions"][name]["loads_per_parent"]
            )
        )
        for name in partitions
    }
    if any(value < 1 for value in (*parent_counts.values(), *load_counts.values())):
        raise FormalGenerationError("all selected parent/load counts must be positive")
    return tuple(partitions), tuple(sections), parent_counts, load_counts


def _train_dev_candidate_assignments(
    config: Mapping[str, Any],
    *,
    section: str,
    train_count: int,
    dev_count: int,
    boundary_points: int,
    split_salt: str | None = None,
) -> list[tuple[str, int, GeometryDataSpec, str, str, dict[str, float]]]:
    """Generate one shared candidate pool, then attach salted train/dev labels."""

    total = int(train_count) + int(dev_count)
    seed = _partition_seed(config, "train_id")
    geometry_config = config["geometry"]
    names = tuple(geometry_config["continuous_parameters"][section])
    unit_rows = _sobol(
        total,
        len(names),
        _substream_seed(seed, "train_dev:geometry_sobol", section),
    )
    candidates: list[tuple[int, GeometryDataSpec, str, str, dict[str, float]]] = []
    for candidate_index, unit in enumerate(unit_rows):
        parameters, positions, _, _ = _scaled_parameters(
            config,
            partition="train_id",
            section=section,
            parent_index=candidate_index,
            unit=unit,
            seed=seed,
        )
        roughness = float(
            _rng(seed, "train_dev:roughness", section, candidate_index).uniform(
                *map(float, geometry_config["roughness_amplitude_over_radius"])
            )
        )
        spec = GeometryDataSpec(
            shape=section,
            parameters=parameters,
            n_boundary_points=boundary_points,
            radius=float(geometry_config["characteristic_radius"]),
            roughness_amplitude=roughness,
            seed=int(
                _substream_seed(seed, "train_dev:roughness_phase", section, candidate_index)
                % (2**31)
            ),
            outer_domain_scale=float(
                geometry_config["outer_half_width_over_characteristic_extent"]
            ),
        )
        geometry = spec.build()
        geometry_id = spec.geometry_group_id(geometry)
        candidates.append(
            (
                candidate_index,
                spec,
                geometry_id,
                _boundary_sha256(geometry.boundary_yz),
                positions,
            )
        )
    resolved_salt = str(
        config["identity_and_split"]["split_salt"] if split_salt is None else split_salt
    )
    ranked = sorted(
        candidates,
        key=lambda item: _sha256(
            _canonical_bytes(
                {
                    "salt": resolved_salt,
                    "section": section,
                    "geometry_group_id": item[2],
                }
            )
        ),
    )
    train_ids = {item[2] for item in ranked[:train_count]}
    return [
        (
            "train_id" if geometry_id in train_ids else "dev_id",
            candidate_index,
            spec,
            geometry_id,
            boundary_hash,
            positions,
        )
        for candidate_index, spec, geometry_id, boundary_hash, positions in candidates
    ]


def build_formal_generation_plan(
    config: Mapping[str, Any],
    overrides: FormalGenerationOverrides | None = None,
    forbidden_identities: FrozenIdentityExclusions | None = None,
) -> FormalGenerationPlan:
    """Derive every geometry, load and audit selection before solving any label."""

    override = overrides or FormalGenerationOverrides()
    exclusions = forbidden_identities or FrozenIdentityExclusions()
    config_hash = _sha256(_canonical_bytes(config))
    partitions, sections, parent_counts, load_counts = _resolved_design(config, override)
    exclusion_count = sum(
        len(values)
        for values in (
            exclusions.geometry_group_ids,
            exclusions.boundary_float64_sha256,
            exclusions.case_group_ids,
            exclusions.load_group_ids,
        )
    )
    if not override.active and (
        exclusions.source_artifact_sha256 is None
        or int(exclusions.source_record_count) < 1
        or exclusion_count < 1
    ):
        raise FormalGenerationError(
            "formal planning requires a hashed, non-empty legacy identity exclusion artifact"
        )
    geometry_config = config["geometry"]
    material = config["material_and_loads"]
    boundary_points = int(override.boundary_points or geometry_config["boundary_points"])
    geometries: list[PlannedGeometry] = []
    cases: list[PlannedCase] = []
    geometry_inputs: list[
        tuple[
            str,
            str,
            int,
            int,
            GeometryDataSpec,
            str,
            str,
            Mapping[str, float],
            str | None,
            str | None,
        ]
    ] = []
    id_requested = {"train_id", "dev_id"}.issubset(partitions)
    if id_requested:
        for section in sections:
            assignments = _train_dev_candidate_assignments(
                config,
                section=section,
                train_count=parent_counts["train_id"],
                dev_count=parent_counts["dev_id"],
                boundary_points=boundary_points,
            )
            local_indices = {"train_id": 0, "dev_id": 0}
            for (
                partition,
                candidate_index,
                spec,
                geometry_id,
                boundary_hash,
                positions,
            ) in assignments:
                parent_index = local_indices[partition]
                local_indices[partition] += 1
                geometry_inputs.append(
                    (
                        partition,
                        section,
                        parent_index,
                        candidate_index,
                        spec,
                        geometry_id,
                        boundary_hash,
                        positions,
                        None,
                        None,
                    )
                )
    for partition in partitions:
        if id_requested and partition in ("train_id", "dev_id"):
            continue
        seed = _partition_seed(config, partition)
        for section in sections:
            names = tuple(geometry_config["continuous_parameters"][section])
            unit_rows = _sobol(
                parent_counts[partition],
                len(names),
                _substream_seed(seed, f"{partition}:geometry_sobol", section),
            )
            for parent_index, unit in enumerate(unit_rows):
                parameters, positions, ood_parameter, ood_side = _scaled_parameters(
                    config,
                    partition=partition,
                    section=section,
                    parent_index=parent_index,
                    unit=unit,
                    seed=seed,
                )
                rough_rng = _rng(seed, f"{partition}:roughness", section, parent_index)
                roughness = float(
                    rough_rng.uniform(
                        *map(float, geometry_config["roughness_amplitude_over_radius"])
                    )
                )
                geometry_seed = _substream_seed(
                    seed, f"{partition}:roughness_phase", section, parent_index
                ) % (2**31)
                query_seed = _substream_seed(seed, f"{partition}:query", section, parent_index) % (
                    2**31
                )
                spec = GeometryDataSpec(
                    shape=section,
                    parameters=parameters,
                    n_boundary_points=boundary_points,
                    radius=float(geometry_config["characteristic_radius"]),
                    roughness_amplitude=roughness,
                    seed=int(geometry_seed),
                    outer_domain_scale=float(
                        geometry_config["outer_half_width_over_characteristic_extent"]
                    ),
                )
                try:
                    geometry = spec.build()
                except Exception as exc:
                    raise FormalGenerationError(
                        f"frozen geometry failed without replacement: {partition}/{section}/{parent_index}"
                    ) from exc
                geometry_id = spec.geometry_group_id(geometry)
                boundary_hash = _boundary_sha256(geometry.boundary_yz)
                geometry_inputs.append(
                    (
                        partition,
                        section,
                        parent_index,
                        parent_index,
                        spec,
                        geometry_id,
                        boundary_hash,
                        positions,
                        ood_parameter,
                        ood_side,
                    )
                )

    for (
        partition,
        section,
        parent_index,
        sampling_parent_index,
        spec,
        geometry_id,
        boundary_hash,
        positions,
        ood_parameter,
        ood_side,
    ) in geometry_inputs:
        query_purpose = (
            "train_dev:query"
            if partition in ("train_id", "dev_id") and id_requested
            else f"{partition}:query"
        )
        seed = _partition_seed(config, partition)
        query_seed = _substream_seed(seed, query_purpose, section, sampling_parent_index) % (2**31)
        entry = PlannedGeometry(
            formal_partition=partition,
            core_split=CORE_SPLIT[partition],
            section_family=section,
            parent_index=parent_index,
            spec=spec,
            geometry_group_id=geometry_id,
            boundary_float64_sha256=boundary_hash,
            normalized_parameter_positions=positions,
            ood_parameter=ood_parameter,
            ood_side=ood_side,
            query_seed=int(query_seed),
        )
        geometries.append(entry)
        for load_index in range(load_counts[partition]):
            subtype, sigma1, ratio, angle, stress = _load(
                config,
                partition=partition,
                section=section,
                # Train/dev share candidate generation, so use the immutable
                # candidate index rather than the post-split local index.  A
                # split-salt change may relabel a candidate but must not alter
                # its load candidate.
                parent_index=sampling_parent_index,
                load_index=load_index,
                seed=seed,
            )
            load_id = load_group_id(stress)
            case_id = case_group_id(
                geometry_id,
                load_id,
                young_modulus=float(material["young_modulus_over_reference_stress"]),
                poisson_ratio=float(material["poisson_ratio"]),
            )
            cases.append(
                PlannedCase(
                    formal_partition=partition,
                    core_split=CORE_SPLIT[partition],
                    section_family=section,
                    parent_index=parent_index,
                    load_index=load_index,
                    geometry_group_id=geometry_id,
                    boundary_float64_sha256=boundary_hash,
                    load_group_id=load_id,
                    case_group_id=case_id,
                    load_subtype=subtype,
                    sigma1=sigma1,
                    sigma3_over_sigma1=ratio,
                    principal_angle_deg=angle,
                    sigma_inf_tension_positive=stress,
                )
            )

    identity_sets = {
        "geometry_group_id": [item.geometry_group_id for item in geometries],
        "boundary_float64_sha256": [item.boundary_float64_sha256 for item in geometries],
        "case_group_id": [item.case_group_id for item in cases],
        "load_group_id": [item.load_group_id for item in cases],
    }
    # Geometry, boundary and case identities are one-to-one with their formal
    # parent/case rows.  Load identities are physical tensors and may repeat
    # inside one partition, but must never cross a partition boundary.
    for name in ("geometry_group_id", "boundary_float64_sha256", "case_group_id"):
        values = identity_sets[name]
        if len(values) != len(set(values)):
            raise FormalGenerationError(f"duplicate frozen identity failed for {name}")
    identity_partition_sets = {
        "geometry_group_id": {
            partition: {
                item.geometry_group_id for item in geometries if item.formal_partition == partition
            }
            for partition in partitions
        },
        "boundary_float64_sha256": {
            partition: {
                item.boundary_float64_sha256
                for item in geometries
                if item.formal_partition == partition
            }
            for partition in partitions
        },
        "case_group_id": {
            partition: {item.case_group_id for item in cases if item.formal_partition == partition}
            for partition in partitions
        },
        "load_group_id": {
            partition: {item.load_group_id for item in cases if item.formal_partition == partition}
            for partition in partitions
        },
    }
    cross_partition_clear = {
        name: all(
            not values[left] & values[right]
            for left_index, left in enumerate(partitions)
            for right in partitions[left_index + 1 :]
        )
        for name, values in identity_partition_sets.items()
    }
    for name, clear in cross_partition_clear.items():
        if not clear:
            raise FormalGenerationError(f"cross-partition zero-intersection failed for {name}")
    exclusion_map = {
        "geometry_group_id": exclusions.geometry_group_ids,
        "boundary_float64_sha256": exclusions.boundary_float64_sha256,
        "case_group_id": exclusions.case_group_ids,
        "load_group_id": exclusions.load_group_ids,
    }
    legacy_clear = {}
    for name, values in identity_sets.items():
        overlap = set(values) & set(exclusion_map[name])
        legacy_clear[name] = not overlap
        if overlap:
            raise FormalGenerationError(f"fresh plan reuses forbidden {name}")

    audit_config = config["quality_control"]["fine_ultrafine"]
    audit_fraction = float(
        override.audit_fraction
        if override.audit_fraction is not None
        else audit_config["formal_audit_fraction"]
    )
    audit_minimum = int(
        override.audit_minimum_per_partition_section
        if override.audit_minimum_per_partition_section is not None
        else audit_config["minimum_selected_cases_per_partition_section"]
    )
    selected: list[str] = []
    split_salt = str(config["identity_and_split"]["split_salt"])
    for partition in partitions:
        for section in sections:
            bucket = [
                item.case_group_id
                for item in cases
                if item.formal_partition == partition and item.section_family == section
            ]
            count = min(len(bucket), max(audit_minimum, math.ceil(audit_fraction * len(bucket))))
            selected.extend(
                sorted(
                    bucket,
                    key=lambda case_id: _sha256(
                        _canonical_bytes(
                            {
                                "salt": split_salt,
                                "purpose": "formal_fine_ultrafine_audit",
                                "partition": partition,
                                "section": section,
                                "case_group_id": case_id,
                            }
                        )
                    ),
                )[:count]
            )
    formal_eligible = not override.active
    if formal_eligible:
        expected = int(audit_config["expected_formal_audit_cases"])
        if len(selected) != expected:
            raise FormalGenerationError(
                f"formal audit selection has {len(selected)} cases, expected {expected}"
            )
        if len(geometries) != int(config["dataset"]["total_parent_geometries"]):
            raise FormalGenerationError("formal parent geometry count changed")
        if len(cases) != int(config["dataset"]["total_cases"]):
            raise FormalGenerationError("formal case count changed")
    report = {
        "cross_partition_zero_intersection": cross_partition_clear,
        "legacy_zero_intersection": legacy_clear,
        "geometry_count": len(geometries),
        "case_count": len(cases),
        "audit_case_count": len(selected),
        "legacy_exclusion_artifact": {
            "sha256": exclusions.source_artifact_sha256,
            "source_record_count": int(exclusions.source_record_count),
            "identity_counts": {name: len(values) for name, values in exclusion_map.items()},
        },
        "section_exact_balance": all(
            len(
                [
                    g
                    for g in geometries
                    if g.formal_partition == partition and g.section_family == section
                ]
            )
            == parent_counts[partition]
            for partition in partitions
            for section in sections
        ),
    }
    protocol = {
        "schema": "tunnelgeopt.formal_generator_protocol.v1",
        "substream_encoding": "utf8(generator_seed|purpose|section|parent_index|load_index)",
        "substream_digest": "sha256_first8_little_endian",
        "sobol_scope": (
            "shared_train_dev_pool_per_section_and_one_per_each_locked_partition_section"
        ),
        "sobol_seed": "substream_mod_2^32",
        "train_dev_assignment": (
            "one_shared_30_candidate_sobol_pool_per_section_then_sha256_sort_of_"
            "canonical_json_salt_section_geometry_group_id_and_prefix_24_train_6_dev"
        ),
        "boundary_digest": "sha256_contiguous_little_endian_float64_bytes",
        "case_order": "partition_config_order_section_config_order_parent_index_load_index",
        "fine_ultrafine_solver": "second_pair_fine_to_ultrafine_with_duplicated_fine_solve",
    }
    return FormalGenerationPlan(
        config_sha256=config_hash,
        run_id=str(config["run_id"]),
        geometries=tuple(geometries),
        cases=tuple(cases),
        audit_case_ids=tuple(selected),
        formal_eligible=formal_eligible,
        identity_report=report,
        generator_protocol=protocol,
    )


def training_data_paths(data_root: str | Path) -> TrainingDataPaths:
    """Return only paths that may be handed to a training process."""

    root = Path(data_root).resolve()
    return TrainingDataPaths(
        public_inputs_path=root / PUBLIC_FILENAME,
        train_dev_fine_labels_path=root / TRAIN_DEV_FILENAME,
        dataset_manifest_path=root / MANIFEST_FILENAME,
    )


def trusted_locked_label_path(data_root: str | Path, partition: str) -> Path:
    """Resolve one sealed store for a separately authorized evaluator process."""

    if partition not in LOCKED_PARTITIONS:
        raise FormalGenerationError(f"{partition!r} is not a locked formal partition")
    return Path(data_root).resolve() / ".sealed_generator_store" / f"{partition}.npz"


def _atomic_npz(path: Path, **arrays: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return _file_sha256(path)


def _write_json_atomic(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_bytes(_canonical_bytes(value) + b"\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return _file_sha256(path)


def _mesh_spec(config: Mapping[str, Any], tier: str) -> MeshFidelitySpec:
    values = config["mesh"]["tiers"][tier]
    radius = float(config["geometry"]["characteristic_radius"])
    return MeshFidelitySpec(
        mesh_size=radius * float(values["mesh_size_over_radius"]),
        wall_mesh_size=radius * float(values["wall_size_over_radius"]),
        farfield_mesh_size=radius * float(values["farfield_size_over_radius"]),
    )


def _query_counts(
    config: Mapping[str, Any], override: FormalGenerationOverrides
) -> tuple[int, int, int]:
    if override.query_region_counts is not None:
        counts = tuple(map(int, override.query_region_counts))
    else:
        query = config["query"]
        counts = tuple(
            map(int, (query["nearfield_volume"], query["wall_offset"], query["farfield"]))
        )
    if len(counts) != 3 or any(value < 1 for value in counts) or counts[1] < 8:
        raise FormalGenerationError("query counts must be positive and wall count at least eight")
    return counts


def _training_weights(grid: Any, config: Mapping[str, Any]) -> FloatArray:
    masses = config["learning"]["training_loss_region_weights"]
    result = np.zeros(grid.point_count, dtype=np.float32)
    for mask, key in (
        (grid.nearfield_mask, "nearfield_volume"),
        (grid.wall_offset_mask, "wall_offset"),
        (grid.farfield_mask, "farfield"),
    ):
        count = int(np.sum(mask))
        result[np.asarray(mask)] = float(masses[key]) / count
    if not np.isclose(float(result.sum()), 1.0):
        raise FormalGenerationError("training region weights do not sum to one")
    return result


def _independent_mesh_geometry_qc(
    geometry: Any, grid: Any, mesh_spec: MeshFidelitySpec, domain_scale: float
) -> dict[str, Any]:
    center = np.asarray(grid.normalization_center_yz, dtype=np.float64)
    extent = np.ptp(np.asarray(geometry.boundary_yz, dtype=np.float64), axis=0)
    bounds = (
        float(center[0] - 0.5 * extent[0] * domain_scale),
        float(center[0] + 0.5 * extent[0] * domain_scale),
        float(center[1] - 0.5 * extent[1] * domain_scale),
        float(center[1] + 0.5 * extent[1] * domain_scale),
    )
    mesh = generate_tunnel_mesh(geometry, outer_bounds=bounds, **mesh_spec.kwargs())
    centroids = np.asarray(mesh.nodes)[np.asarray(mesh.elements)].mean(axis=1)
    centroid_inside = int(
        np.sum(points_inside_polygon(centroids, np.asarray(geometry.boundary_yz)))
    )
    return {
        "no_element_centroid_inside_cavity": centroid_inside == 0,
        "centroid_inside_cavity_count": centroid_inside,
        "explicit_wall_and_farfield_tags": bool(
            mesh.boundary_facets["wall"].size > 0
            and mesh.boundary_facets["farfield"].size > 0
            and set(mesh.physical_tags) == {"rock", "wall", "farfield"}
        ),
        "outer_bounds": [float(value) for value in mesh.outer_bounds],
    }


def _qc_record(
    sample: Any,
    geometry: Any,
    config: Mapping[str, Any],
    independent_mesh_qc: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    gates = config["quality_control"]["solver_and_mesh"]
    cavity_centroid_counts: dict[str, int] = {}
    tiers: dict[str, Any] = {}
    for tier, metadata, element_ids, diagnostics in (
        (
            "coarse",
            sample.coarse_mesh_metadata,
            sample.coarse_element_ids,
            sample.diagnostics["coarse"],
        ),
        ("fine", sample.fine_mesh_metadata, sample.fine_element_ids, sample.diagnostics["fine"]),
    ):
        # Mesh construction itself guarantees positive signed area.  The public
        # metadata preserves its minimum and explicit physical facet counts.
        min_area = (
            float(metadata["minimum_element_area"]) / float(geometry.characteristic_radius) ** 2
        )
        nonfinite_fraction = float(
            np.mean(
                ~np.isfinite(
                    sample.coarse_stress_normalized
                    if tier == "coarse"
                    else sample._fine_stress_normalized
                )
            )
        )
        independent = independent_mesh_qc[tier]
        cavity_centroid_counts[tier] = int(independent["centroid_inside_cavity_count"])
        tier_record = {
            "nonfinite_fraction": nonfinite_fraction,
            "free_dof_algebraic_residual": float(diagnostics["algebraic_residual"]),
            "clapeyron_relative_energy_error": float(diagnostics["energy_closure"]),
            "min_triangle_signed_area_over_radius_squared": min_area,
            "min_triangle_quality": float(metadata["minimum_triangle_quality"]),
            "all_query_points_located": bool(np.all(np.asarray(element_ids) >= 0)),
            "explicit_wall_and_farfield_tags": bool(
                independent["explicit_wall_and_farfield_tags"]
                and metadata["wall_facet_count"] > 0
                and metadata["farfield_facet_count"] > 0
            ),
            "no_element_centroid_inside_cavity": bool(
                independent["no_element_centroid_inside_cavity"]
            ),
            "same_boundary_hash_and_outer_bounds": bool(
                sample.diagnostics["same_frozen_boundary"]
                and sample.diagnostics["same_outer_bounds"]
                and np.allclose(
                    np.asarray(independent["outer_bounds"], dtype=np.float64),
                    np.asarray(metadata["actual_outer_bounds"], dtype=np.float64),
                    rtol=0.0,
                    atol=1e-12,
                )
            ),
        }
        tier_record["passed"] = bool(
            tier_record["nonfinite_fraction"] <= float(gates["max_nonfinite_fraction"])
            and tier_record["free_dof_algebraic_residual"]
            <= float(gates["max_free_dof_algebraic_residual"])
            and tier_record["clapeyron_relative_energy_error"]
            <= float(gates["max_clapeyron_relative_energy_error"])
            and tier_record["min_triangle_signed_area_over_radius_squared"]
            >= float(gates["min_triangle_signed_area_over_radius_squared"])
            and tier_record["min_triangle_quality"] >= float(gates["min_triangle_quality"])
            and tier_record["all_query_points_located"]
            and tier_record["explicit_wall_and_farfield_tags"]
            and tier_record["no_element_centroid_inside_cavity"]
            and tier_record["same_boundary_hash_and_outer_bounds"]
        )
        tiers[tier] = tier_record
    identity = {
        "same_boundary": bool(sample.diagnostics["same_frozen_boundary"]),
        "same_outer_bounds": bool(sample.diagnostics["same_outer_bounds"]),
        "same_query_hash": sample.diagnostics["common_query_hash"] == sample.grid.query_hash,
    }
    return {
        "fidelities": tiers,
        "identity": identity,
        "passed": all(x["passed"] for x in tiers.values()) and all(identity.values()),
    }


def _failed_qc(error: BaseException) -> dict[str, Any]:
    """Represent an attempted case failure without fabricating a label."""

    failed_fidelity = {
        "nonfinite_fraction": 1.0,
        "free_dof_algebraic_residual": 1.0,
        "clapeyron_relative_energy_error": 1.0,
        "min_triangle_signed_area_over_radius_squared": 0.0,
        "min_triangle_quality": 0.0,
        "all_query_points_located": False,
        "explicit_wall_and_farfield_tags": False,
        "no_element_centroid_inside_cavity": False,
        "same_boundary_hash_and_outer_bounds": False,
        "passed": False,
    }
    return {
        "fidelities": {"coarse": dict(failed_fidelity), "fine": dict(failed_fidelity)},
        "identity": {
            "same_boundary": False,
            "same_outer_bounds": False,
            "same_query_hash": False,
        },
        "passed": False,
        "failure": {
            "type": type(error).__name__,
            "message": str(error)[:1000],
            "label_written": False,
            "replacement_attempted": False,
        },
    }


def _cache_paths(root: Path, case_id: str) -> tuple[Path, Path]:
    path = root / ".generator_case_cache" / f"{case_id}.npz"
    return path, path.with_suffix(".sha256")


def _load_cache(root: Path, case_id: str) -> dict[str, Any] | None:
    path, digest_path = _cache_paths(root, case_id)
    if not path.exists() and not digest_path.exists():
        return None
    if not path.is_file() or not digest_path.is_file():
        raise FormalGenerationError(f"incomplete cache for {case_id}")
    expected = digest_path.read_text(encoding="ascii").strip()
    if _file_sha256(path) != expected:
        raise FormalGenerationError(f"cache digest mismatch for {case_id}")
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _save_cache(root: Path, case_id: str, **arrays: Any) -> None:
    path, digest_path = _cache_paths(root, case_id)
    digest = _atomic_npz(path, **arrays)
    temporary = digest_path.with_name(digest_path.name + ".tmp")
    temporary.write_text(digest + "\n", encoding="ascii")
    os.replace(temporary, digest_path)


def _validity_cell(valid_flags: Sequence[bool], threshold: float) -> dict[str, Any]:
    planned = len(valid_flags)
    if planned < 1:
        raise FormalGenerationError("validity cell must contain at least one planned case")
    valid = sum(bool(value) for value in valid_flags)
    rate = valid / planned
    return {
        "planned_cases": planned,
        "attempted_cases": planned,
        "valid_cases": valid,
        "valid_fraction": rate,
        "passed": rate >= float(threshold),
        "replacement_count": 0,
    }


def _valid_learning_rows(records: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Exclude invalid records from every label-bearing store without replacement."""

    return [record for record in records if bool(record["qc"]["passed"])]


def _aggregate_qc(
    plan: FormalGenerationPlan, records: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    threshold = float(
        config["quality_control"]["solver_and_mesh"][
            "minimum_valid_case_fraction_per_partition_section"
        ]
    )
    for partition in sorted(
        {case.formal_partition for case in plan.cases}, key=FORMAL_PARTITIONS.index
    ):
        sections: dict[str, Any] = {}
        for section in sorted(
            {case.section_family for case in plan.cases if case.formal_partition == partition}
        ):
            selected = [
                record
                for record in records
                if record["formal_partition"] == partition and record["section_family"] == section
            ]
            validity = _validity_cell(
                [bool(record["qc"]["passed"]) for record in selected], threshold
            )
            tier_values = [
                record["qc"]["fidelities"][tier]
                for record in selected
                for tier in ("coarse", "fine")
            ]
            sections[section] = {
                **validity,
                "max_nonfinite_fraction": max(v["nonfinite_fraction"] for v in tier_values),
                "max_free_dof_algebraic_residual": max(
                    v["free_dof_algebraic_residual"] for v in tier_values
                ),
                "max_clapeyron_relative_energy_error": max(
                    v["clapeyron_relative_energy_error"] for v in tier_values
                ),
                "min_triangle_signed_area_over_radius_squared": min(
                    v["min_triangle_signed_area_over_radius_squared"] for v in tier_values
                ),
                "min_triangle_quality": min(v["min_triangle_quality"] for v in tier_values),
                "all_query_points_located": all(v["all_query_points_located"] for v in tier_values),
                "explicit_tags": all(v["explicit_wall_and_farfield_tags"] for v in tier_values),
                "no_centroid_inside_cavity": all(
                    v["no_element_centroid_inside_cavity"] for v in tier_values
                ),
                "boundary_outer_match": all(all(r["qc"]["identity"].values()) for r in selected),
            }
        output[partition] = {
            "sections": sections,
            "passed": all(x["passed"] for x in sections.values()),
        }
    subtype_cells: dict[str, Any] = {}
    for partition in sorted(
        {case.formal_partition for case in plan.cases}, key=FORMAL_PARTITIONS.index
    ):
        subtypes = sorted(
            {case.load_subtype for case in plan.cases if case.formal_partition == partition}
        )
        for subtype in subtypes:
            selected = [
                record
                for record in records
                if record["formal_partition"] == partition and record["load_subtype"] == subtype
            ]
            subtype_cells[f"{partition}:{subtype}"] = _validity_cell(
                [bool(record["qc"]["passed"]) for record in selected], threshold
            )
    return {
        "partitions": output,
        "partition_load_subtypes": subtype_cells,
        # Frozen solver validity gate is partition x section only.  Load-subtype
        # rates are mandatory diagnostics, not an extra unregistered ABSTAIN gate.
        "passed": all(value["passed"] for value in output.values()),
    }


def _audit_summary(
    plan: FormalGenerationPlan,
    audit_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    errors = np.asarray([row["error"] for row in audit_rows], dtype=np.float64)
    if errors.size == 0:
        return {
            "selection_protocol": config["quality_control"]["fine_ultrafine"]["selection_protocol"],
            "selection_hash": _sha256(_canonical_bytes(sorted(plan.audit_case_ids))),
            "selected_case_count": 0,
            "selected_case_ids": list(plan.audit_case_ids),
            "selection_fraction": float(
                config["quality_control"]["fine_ultrafine"]["formal_audit_fraction"]
            ),
            "selected_before_labels": True,
            "duplicated_fine_solve": True,
            "case_values_exposed_before_checkpoint_freeze": False,
            "passed": False,
        }
    sections = sorted({row["section_family"] for row in audit_rows})
    section_medians = {
        section: float(
            np.median([row["error"] for row in audit_rows if row["section_family"] == section])
        )
        for section in sections
    }
    gates = config["quality_control"]["fine_ultrafine"]
    overall_median = float(np.median(errors))
    overall_p95 = float(np.quantile(errors, 0.95))
    passed = bool(
        len(audit_rows) == len(plan.audit_case_ids)
        and overall_median <= float(gates["max_overall_median"])
        and overall_p95 <= float(gates["max_overall_p95"])
        and max(section_medians.values()) <= float(gates["max_any_section_median"])
        and all(row["fine_repeat_consistent"] for row in audit_rows)
        and all(row["ultrafine_qc_passed"] for row in audit_rows)
    )
    return {
        "selection_protocol": gates["selection_protocol"],
        "selection_hash": _sha256(_canonical_bytes(sorted(plan.audit_case_ids))),
        "selected_case_count": len(audit_rows),
        "selected_case_ids": list(plan.audit_case_ids),
        "selection_fraction": float(gates["formal_audit_fraction"]),
        "selected_before_labels": True,
        "duplicated_fine_solve": True,
        "overall_median": overall_median,
        "overall_p95": overall_p95,
        "section_medians": section_medians,
        "case_values_exposed_before_checkpoint_freeze": False,
        "passed": passed,
    }


def _execute_ultrafine_audit(
    *,
    case: PlannedCase,
    entry: PlannedGeometry,
    geometry: Any,
    grid: Any,
    original_sample: Any,
    fine_mesh: MeshFidelitySpec,
    ultrafine_mesh: MeshFidelitySpec,
    young: float,
    poisson: float,
    config: Mapping[str, Any],
    geometry_mesh_qc: dict[str, Mapping[str, Any]],
) -> dict[str, Any]:
    audit_sample = solve_multifidelity_case(
        geometry,
        grid,
        split=case.core_split,
        sigma_inf_tension_positive=case.sigma_inf_tension_positive,
        young_modulus=young,
        poisson_ratio=poisson,
        coarse_mesh=fine_mesh,
        fine_mesh=ultrafine_mesh,
        domain_scale=float(entry.spec.outer_domain_scale),
        geometry_spec=entry.spec,
    )
    repeated_fine = np.asarray(audit_sample.coarse_stress_normalized, dtype=np.float64)
    original_fine = np.asarray(original_sample._fine_stress_normalized, dtype=np.float64)
    consistent = bool(np.allclose(repeated_fine, original_fine, rtol=1e-10, atol=1e-12))
    error = float(
        case_weighted_stress_error(
            repeated_fine[None],
            np.asarray(audit_sample._fine_stress_normalized)[None],
            np.asarray(grid.area_weights)[None],
        )[0]
    )
    if "ultrafine" not in geometry_mesh_qc:
        geometry_mesh_qc["ultrafine"] = _independent_mesh_geometry_qc(
            geometry, grid, ultrafine_mesh, float(entry.spec.outer_domain_scale)
        )
    ultra_qc = _qc_record(
        audit_sample,
        geometry,
        config,
        {"coarse": geometry_mesh_qc["fine"], "fine": geometry_mesh_qc["ultrafine"]},
    )
    return {
        "case_group_id": case.case_group_id,
        "formal_partition": case.formal_partition,
        "section_family": case.section_family,
        "error": error,
        "fine_repeat_consistent": consistent,
        "ultrafine_qc_passed": bool(ultra_qc["passed"]),
    }


def generate_formal_dataset(
    config: Mapping[str, Any],
    data_root: str | Path,
    *,
    overrides: FormalGenerationOverrides | None = None,
    forbidden_identities: FrozenIdentityExclusions | None = None,
    resume: bool = True,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> FormalGenerationResult:
    """Generate real paired FEM cases and atomically split public/sealed stores."""

    started = time.perf_counter()
    tracemalloc.start()
    override = overrides or FormalGenerationOverrides()
    root = Path(data_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    plan = build_formal_generation_plan(config, override, forbidden_identities)
    geometry_by_id = {item.geometry_group_id: item for item in plan.geometries}
    query_counts = _query_counts(config, override)
    query_config = config["query"]
    coarse_mesh = _mesh_spec(config, "coarse")
    fine_mesh = _mesh_spec(config, "fine")
    ultrafine_mesh = _mesh_spec(config, "ultrafine_audit")
    young = float(config["material_and_loads"]["young_modulus_over_reference_stress"])
    poisson = float(config["material_and_loads"]["poisson_ratio"])
    audit_set = set(plan.audit_case_ids)
    rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    audit_failures: list[dict[str, Any]] = []
    resumed_cases = 0
    solved_cases = 0
    grids: dict[str, Any] = {}
    geometries: dict[str, Any] = {}
    independent_mesh_qc: dict[str, dict[str, Mapping[str, Any]]] = {}
    geometry_preparation_failures: dict[str, BaseException] = {}
    for case_index, case in enumerate(plan.cases):
        entry = geometry_by_id[case.geometry_group_id]
        geometry = geometries.get(case.geometry_group_id)
        grid = grids.get(case.geometry_group_id)
        preparation_error = geometry_preparation_failures.get(case.geometry_group_id)
        if geometry is None and preparation_error is None:
            try:
                geometry = entry.spec.build()
                grid = build_elastic_query_grid(
                    geometry,
                    geometry_parameters=entry.spec.identity_parameters(),
                    nearfield_points=query_counts[0],
                    wall_offset_points=query_counts[1],
                    farfield_points=query_counts[2],
                    nearfield_scale=float(query_config["nearfield_scale"]),
                    farfield_scale=float(query_config["farfield_scale"]),
                    nearfield_min_distance_over_radius=float(
                        query_config["nearfield_distance_over_radius"][0]
                    ),
                    nearfield_max_distance_over_radius=float(
                        query_config["nearfield_distance_over_radius"][1]
                    ),
                    wall_offset_over_radius=float(query_config["wall_offset_over_radius"]),
                    seed=entry.query_seed,
                    outer_domain_scale=float(entry.spec.outer_domain_scale),
                )
                independent = {
                    "coarse": _independent_mesh_geometry_qc(
                        geometry, grid, coarse_mesh, float(entry.spec.outer_domain_scale)
                    ),
                    "fine": _independent_mesh_geometry_qc(
                        geometry, grid, fine_mesh, float(entry.spec.outer_domain_scale)
                    ),
                }
            except Exception as error:  # noqa: BLE001 - one parent invalidates its frozen loads
                preparation_error = error
                geometry_preparation_failures[case.geometry_group_id] = error
            else:
                geometries[case.geometry_group_id] = geometry
                grids[case.geometry_group_id] = grid
                independent_mesh_qc[case.geometry_group_id] = independent
        if preparation_error is not None:
            rows.append(
                {
                    "qc": _failed_qc(preparation_error),
                    "case_group_id": case.case_group_id,
                    "geometry_group_id": case.geometry_group_id,
                    "load_group_id": case.load_group_id,
                    "formal_partition": case.formal_partition,
                    "core_split": case.core_split,
                    "section_family": case.section_family,
                    "load_subtype": case.load_subtype,
                }
            )
            if case.case_group_id in audit_set:
                audit_failures.append(
                    {
                        "case_group_id": case.case_group_id,
                        "formal_partition": case.formal_partition,
                        "section_family": case.section_family,
                        "error_type": type(preparation_error).__name__,
                        "message": "parent geometry/query/mesh preparation failed",
                        "replacement_attempted": False,
                    }
                )
            solved_cases += 1
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "formal_geometry_case_failed_without_replacement",
                        "case_group_id": case.case_group_id,
                        "formal_partition": case.formal_partition,
                        "completed": case_index + 1,
                        "total": len(plan.cases),
                        "error_type": type(preparation_error).__name__,
                    }
                )
            continue
        cached = _load_cache(root, case.case_group_id) if resume else None
        if cached is not None:
            if str(cached["case_group_id"].item()) != case.case_group_id:
                raise FormalGenerationError("cache case identity mismatch")
            resumed_cases += 1
            row = {
                "base_features": cached["base_features"],
                "coarse_stress": cached["coarse_stress"],
                "fine_stress": cached["fine_stress"],
                "training_weights": cached["training_weights"],
                "metric_weights": cached["metric_weights"],
                "arc_weights": cached["arc_weights"],
                "wall_normals": cached["wall_normals"],
                "stress_scale": float(cached["stress_scale"].item()),
                "query_hash": str(cached["query_hash"].item()),
                "qc": json.loads(str(cached["qc_json"].item())),
            }
            if case.case_group_id in audit_set:
                cached_audit = json.loads(str(cached["audit_json"].item()))
                if cached_audit:
                    audit_rows.append(cached_audit)
                else:
                    audit_failures.append(
                        {
                            "case_group_id": case.case_group_id,
                            "formal_partition": case.formal_partition,
                            "section_family": case.section_family,
                            "error_type": "CachedAuditFailure",
                            "message": "selected ultrafine audit did not complete",
                            "replacement_attempted": False,
                        }
                    )
        else:
            try:
                sample = solve_multifidelity_case(
                    geometry,
                    grid,
                    split=case.core_split,
                    sigma_inf_tension_positive=case.sigma_inf_tension_positive,
                    young_modulus=young,
                    poisson_ratio=poisson,
                    coarse_mesh=coarse_mesh,
                    fine_mesh=fine_mesh,
                    domain_scale=float(entry.spec.outer_domain_scale),
                    geometry_spec=entry.spec,
                )
            except Exception as error:  # noqa: BLE001 - every solver failure is formal evidence
                failed_qc = _failed_qc(error)
                rows.append(
                    {
                        "qc": failed_qc,
                        "case_group_id": case.case_group_id,
                        "geometry_group_id": case.geometry_group_id,
                        "load_group_id": case.load_group_id,
                        "formal_partition": case.formal_partition,
                        "core_split": case.core_split,
                        "section_family": case.section_family,
                        "load_subtype": case.load_subtype,
                    }
                )
                if case.case_group_id in audit_set:
                    audit_failures.append(
                        {
                            "case_group_id": case.case_group_id,
                            "formal_partition": case.formal_partition,
                            "section_family": case.section_family,
                            "error_type": type(error).__name__,
                            "message": "selected main fine solve failed before ultrafine audit",
                            "replacement_attempted": False,
                        }
                    )
                solved_cases += 1
                if progress_callback is not None:
                    progress_callback(
                        {
                            "event": "formal_case_failed_without_replacement",
                            "case_group_id": case.case_group_id,
                            "formal_partition": case.formal_partition,
                            "completed": case_index + 1,
                            "total": len(plan.cases),
                            "error_type": type(error).__name__,
                        }
                    )
                continue
            if sample.case_group_id != case.case_group_id:
                raise FormalGenerationError("solved case identity changed")
            qc = _qc_record(sample, geometry, config, independent_mesh_qc[case.geometry_group_id])
            if not qc["passed"]:
                qc["failure"] = {
                    "type": "QualityControlFailure",
                    "message": "one or more frozen coarse/fine QC thresholds failed",
                    "label_written": False,
                    "replacement_attempted": False,
                }
                rows.append(
                    {
                        "qc": qc,
                        "case_group_id": case.case_group_id,
                        "geometry_group_id": case.geometry_group_id,
                        "load_group_id": case.load_group_id,
                        "formal_partition": case.formal_partition,
                        "core_split": case.core_split,
                        "section_family": case.section_family,
                        "load_subtype": case.load_subtype,
                    }
                )
                if case.case_group_id in audit_set:
                    audit_failures.append(
                        {
                            "case_group_id": case.case_group_id,
                            "formal_partition": case.formal_partition,
                            "section_family": case.section_family,
                            "error_type": "QualityControlFailure",
                            "message": "selected main fine case was invalid",
                            "replacement_attempted": False,
                        }
                    )
                solved_cases += 1
                if progress_callback is not None:
                    progress_callback(
                        {
                            "event": "formal_case_failed_qc_without_replacement",
                            "case_group_id": case.case_group_id,
                            "formal_partition": case.formal_partition,
                            "completed": case_index + 1,
                            "total": len(plan.cases),
                        }
                    )
                continue
            _, wall_normals = surface_points_and_normals(geometry, query_counts[1])
            normals = np.zeros((grid.point_count, 2), dtype=np.float32)
            normals[np.asarray(grid.wall_offset_mask)] = wall_normals.astype(np.float32)
            row = {
                "base_features": sample.model_features[:, :11].astype(np.float32),
                "coarse_stress": np.asarray(sample.coarse_stress_normalized, dtype=np.float32),
                "fine_stress": np.asarray(sample._fine_stress_normalized, dtype=np.float32),
                "training_weights": _training_weights(grid, config),
                "metric_weights": np.asarray(grid.area_weights, dtype=np.float32),
                "arc_weights": np.asarray(grid.arc_weights, dtype=np.float32),
                "wall_normals": normals,
                "stress_scale": float(sample.stress_scale),
                "query_hash": grid.query_hash,
                "qc": qc,
            }
            audit_value: dict[str, Any] = {}
            if case.case_group_id in audit_set:
                try:
                    audit_value = _execute_ultrafine_audit(
                        case=case,
                        entry=entry,
                        geometry=geometry,
                        grid=grid,
                        original_sample=sample,
                        fine_mesh=fine_mesh,
                        ultrafine_mesh=ultrafine_mesh,
                        young=young,
                        poisson=poisson,
                        config=config,
                        geometry_mesh_qc=independent_mesh_qc[case.geometry_group_id],
                    )
                except Exception as error:  # noqa: BLE001 - sealed audit must fail closed
                    audit_failures.append(
                        {
                            "case_group_id": case.case_group_id,
                            "formal_partition": case.formal_partition,
                            "section_family": case.section_family,
                            "error_type": type(error).__name__,
                            "message": str(error)[:1000],
                            "replacement_attempted": False,
                        }
                    )
                else:
                    audit_rows.append(audit_value)
            # A selected case is cached only after its ultrafine audit also
            # completed, so resume retries an interrupted audit rather than
            # treating a transient failure as a durable result.
            if case.case_group_id not in audit_set or audit_value:
                _save_cache(
                    root,
                    case.case_group_id,
                    case_group_id=np.asarray(case.case_group_id),
                    base_features=row["base_features"],
                    coarse_stress=row["coarse_stress"],
                    fine_stress=row["fine_stress"],
                    training_weights=row["training_weights"],
                    metric_weights=row["metric_weights"],
                    arc_weights=row["arc_weights"],
                    wall_normals=row["wall_normals"],
                    stress_scale=np.asarray(row["stress_scale"]),
                    query_hash=np.asarray(row["query_hash"]),
                    qc_json=np.asarray(json.dumps(qc, sort_keys=True, separators=(",", ":"))),
                    audit_json=np.asarray(
                        json.dumps(audit_value, sort_keys=True, separators=(",", ":"))
                    ),
                )
            solved_cases += 1
        rows.append(
            {
                **row,
                "case_group_id": case.case_group_id,
                "geometry_group_id": case.geometry_group_id,
                "load_group_id": case.load_group_id,
                "formal_partition": case.formal_partition,
                "core_split": case.core_split,
                "section_family": case.section_family,
                "load_subtype": case.load_subtype,
            }
        )
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "formal_case_ready",
                    "case_group_id": case.case_group_id,
                    "formal_partition": case.formal_partition,
                    "completed": case_index + 1,
                    "total": len(plan.cases),
                    "resumed": cached is not None,
                }
            )

    qc_summary = _aggregate_qc(plan, rows, config)
    audit_by_id = {str(row["case_group_id"]): row for row in audit_rows}
    audit_rows = [value for case_id in plan.audit_case_ids if (value := audit_by_id.get(case_id))]
    audit_summary = _audit_summary(plan, audit_rows, config)
    audit_summary["failure_records"] = audit_failures
    if set(audit_by_id) != set(plan.audit_case_ids) or audit_failures:
        audit_summary["passed"] = False
    valid_rows = _valid_learning_rows(rows)
    if not valid_rows:
        solver_records = [
            {
                "case_group_id": row["case_group_id"],
                "partition": row["formal_partition"],
                "section_family": row["section_family"],
                "fidelities": row["qc"]["fidelities"],
                "valid": False,
                "failure": row["qc"].get("failure", {}),
            }
            for row in rows
        ]
        manifest_path = root / MANIFEST_FILENAME
        _write_json_atomic(
            manifest_path,
            {
                "schema_version": "tunnelgeopt.formal_dataset_manifest.v1",
                "run_id": plan.run_id,
                "config_sha256": plan.config_sha256,
                "formal_eligible": plan.formal_eligible,
                "generation_status": "ABSTAIN",
                "generator_protocol": dict(plan.generator_protocol),
                "counts": {
                    "parent_geometries": len(plan.geometries),
                    "planned_cases": len(plan.cases),
                    "valid_cases": 0,
                    "invalid_cases": len(plan.cases),
                },
                "identities": {
                    "cross_partition_zero_intersection": True,
                    "legacy_v0_2_locked_test_zero_intersection": True,
                    "normalization_fit_train_only": True,
                    "no_result_conditioned_replacement": True,
                },
                "artifact_hashes": {},
                "files": {},
                "solver_mesh_qc": {
                    "no_silent_case_replacement": True,
                    "records": solver_records,
                    "partition_section_summary": qc_summary,
                    "passed": False,
                },
                "fine_ultrafine_selection": {
                    "selection_unit": "case_group_id",
                    "selected_case_ids": list(plan.audit_case_ids),
                    "selected_before_any_ultrafine_label": True,
                    "case_values_exposed_before_checkpoint_freeze": False,
                    "audit_passed_inside_trusted_generator": False,
                    "failure_records": audit_failures,
                },
                "opaque_sealed_stores": {},
                "sealed_label_read_count": 0,
                "resource_usage": {
                    "runtime_seconds": float(time.perf_counter() - started),
                    "peak_memory_bytes": int(tracemalloc.get_traced_memory()[1]),
                },
            },
        )
        tracemalloc.stop()
        raise FormalGenerationError(
            f"formal generation ABSTAIN with zero valid cases; evidence written to {manifest_path}"
        )
    base_arrays = {
        "base_features": np.stack([row["base_features"] for row in valid_rows]),
        "coarse_stress": np.stack([row["coarse_stress"] for row in valid_rows]),
        "training_weights": np.stack([row["training_weights"] for row in valid_rows]),
        "metric_weights": np.stack([row["metric_weights"] for row in valid_rows]),
        "arc_weights": np.stack([row["arc_weights"] for row in valid_rows]),
        "wall_rock_outward_normals_yz": np.stack([row["wall_normals"] for row in valid_rows]),
        "stress_scales": np.asarray([row["stress_scale"] for row in valid_rows], dtype=np.float64),
        "nearfield_mask": np.stack(
            [grids[row["geometry_group_id"]].nearfield_mask for row in valid_rows]
        ),
        "wall_offset_mask": np.stack(
            [grids[row["geometry_group_id"]].wall_offset_mask for row in valid_rows]
        ),
        "farfield_mask": np.stack(
            [grids[row["geometry_group_id"]].farfield_mask for row in valid_rows]
        ),
        "case_group_ids": np.asarray([row["case_group_id"] for row in valid_rows], dtype="U64"),
        "geometry_group_ids": np.asarray(
            [row["geometry_group_id"] for row in valid_rows], dtype="U64"
        ),
        "boundary_float64_sha256": np.asarray(
            [
                geometry_by_id[row["geometry_group_id"]].boundary_float64_sha256
                for row in valid_rows
            ],
            dtype="U64",
        ),
        "load_group_ids": np.asarray([row["load_group_id"] for row in valid_rows], dtype="U64"),
        "query_hashes": np.asarray([row["query_hash"] for row in valid_rows], dtype="U64"),
        "partitions": np.asarray([row["formal_partition"] for row in valid_rows], dtype="U32"),
        "splits": np.asarray([row["core_split"] for row in valid_rows], dtype="U16"),
        "section_families": np.asarray([row["section_family"] for row in valid_rows], dtype="U32"),
        "load_subtypes": np.asarray([row["load_subtype"] for row in valid_rows], dtype="U48"),
    }
    public_path = root / PUBLIC_FILENAME
    train_path = root / TRAIN_DEV_FILENAME
    public_hash = _atomic_npz(public_path, **base_arrays)
    train_indices = np.asarray(
        [
            index
            for index, row in enumerate(valid_rows)
            if row["formal_partition"] in ("train_id", "dev_id")
        ],
        dtype=np.int64,
    )
    train_audit = [row for row in audit_rows if row["formal_partition"] in ("train_id", "dev_id")]
    train_hash = _atomic_npz(
        train_path,
        indices=train_indices,
        fine_stress=np.stack([valid_rows[index]["fine_stress"] for index in train_indices]),
        case_group_ids=base_arrays["case_group_ids"][train_indices],
        partitions=base_arrays["partitions"][train_indices],
        audit_case_group_ids=np.asarray([row["case_group_id"] for row in train_audit], dtype="U64"),
        audit_section_families=np.asarray(
            [row["section_family"] for row in train_audit], dtype="U32"
        ),
        audit_partitions=np.asarray([row["formal_partition"] for row in train_audit], dtype="U32"),
        audit_relative_errors=np.asarray([row["error"] for row in train_audit], dtype=np.float64),
    )
    opaque_hashes: dict[str, str] = {}
    sealed_hashes_by_partition: dict[str, str] = {}
    for partition in LOCKED_PARTITIONS:
        indices = np.asarray(
            [index for index, row in enumerate(valid_rows) if row["formal_partition"] == partition],
            dtype=np.int64,
        )
        if not indices.size:
            continue
        partition_audit = [row for row in audit_rows if row["formal_partition"] == partition]
        digest = _atomic_npz(
            trusted_locked_label_path(root, partition),
            indices=indices,
            fine_stress=np.stack([valid_rows[index]["fine_stress"] for index in indices]),
            case_group_ids=base_arrays["case_group_ids"][indices],
            partition=np.asarray(partition),
            ultrafine_audit_passed=np.asarray([bool(audit_summary["passed"])]),
            audit_case_group_ids=np.asarray(
                [row["case_group_id"] for row in partition_audit], dtype="U64"
            ),
            audit_section_families=np.asarray(
                [row["section_family"] for row in partition_audit], dtype="U32"
            ),
            audit_partitions=np.asarray(
                [row["formal_partition"] for row in partition_audit], dtype="U32"
            ),
            audit_relative_errors=np.asarray(
                [row["error"] for row in partition_audit], dtype=np.float64
            ),
        )
        opaque_id = _sha256(_canonical_bytes({"run_id": plan.run_id, "partition": partition}))
        opaque_hashes[opaque_id] = digest
        sealed_hashes_by_partition[partition] = digest

    generator_manifests = root / ".generator_manifests"
    geometry_manifest_hash = _write_json_atomic(
        generator_manifests / "geometry_manifest.json",
        {
            "geometries": [
                {
                    "formal_partition": item.formal_partition,
                    "section_family": item.section_family,
                    "parent_index": item.parent_index,
                    "geometry_group_id": item.geometry_group_id,
                    "boundary_float64_sha256": item.boundary_float64_sha256,
                    "shape_parameters": dict(item.spec.parameters),
                    "normalized_parameter_positions": dict(item.normalized_parameter_positions),
                    "roughness_amplitude": item.spec.roughness_amplitude,
                    "geometry_seed": item.spec.seed,
                    "ood_parameter": item.ood_parameter,
                    "ood_side": item.ood_side,
                }
                for item in plan.geometries
            ]
        },
    )
    case_manifest_hash = _write_json_atomic(
        generator_manifests / "case_manifest.json",
        {
            "cases": [
                {
                    "formal_partition": item.formal_partition,
                    "section_family": item.section_family,
                    "parent_index": item.parent_index,
                    "load_index": item.load_index,
                    "geometry_group_id": item.geometry_group_id,
                    "load_group_id": item.load_group_id,
                    "case_group_id": item.case_group_id,
                    "load_subtype": item.load_subtype,
                    "sigma1": item.sigma1,
                    "sigma3_over_sigma1": item.sigma3_over_sigma1,
                    "principal_angle_deg": item.principal_angle_deg,
                }
                for item in plan.cases
            ]
        },
    )
    query_manifest_hash = _write_json_atomic(
        generator_manifests / "query_manifest.json",
        {
            "queries": [
                {
                    "geometry_group_id": item.geometry_group_id,
                    "query_seed": item.query_seed,
                    "query_hash": (
                        grids[item.geometry_group_id].query_hash
                        if item.geometry_group_id in grids
                        else None
                    ),
                    "prepared": item.geometry_group_id in grids,
                    **(
                        {
                            "failure": {
                                "type": type(
                                    geometry_preparation_failures[item.geometry_group_id]
                                ).__name__,
                                "message": str(
                                    geometry_preparation_failures[item.geometry_group_id]
                                )[:1000],
                                "replacement_attempted": False,
                            }
                        }
                        if item.geometry_group_id in geometry_preparation_failures
                        else {}
                    ),
                }
                for item in plan.geometries
            ]
        },
    )
    artifact_hashes = {
        "geometry_manifest": geometry_manifest_hash,
        "case_manifest": case_manifest_hash,
        "query_manifest": query_manifest_hash,
        "public_input_store": public_hash,
        "train_dev_label_store": train_hash,
        **{
            f"sealed_{partition}_label_store": sealed_hashes_by_partition[partition]
            for partition in LOCKED_PARTITIONS
            if partition in sealed_hashes_by_partition
        },
    }
    public_hashes = {PUBLIC_FILENAME: public_hash, TRAIN_DEV_FILENAME: train_hash}
    # This map is trainer-visible and therefore contains no sealed name/path.
    file_hashes = dict(public_hashes)
    solver_records = [
        {
            "case_group_id": row["case_group_id"],
            "partition": row["formal_partition"],
            "section_family": row["section_family"],
            "fidelities": row["qc"]["fidelities"],
            "valid": bool(row["qc"]["passed"]),
            **({"failure": row["qc"]["failure"]} if "failure" in row["qc"] else {}),
        }
        for row in rows
    ]
    generation_passed = bool(qc_summary["passed"] and audit_summary["passed"])
    manifest = {
        "schema_version": "tunnelgeopt.formal_dataset_manifest.v1",
        "run_id": plan.run_id,
        "config_sha256": plan.config_sha256,
        "formal_eligible": plan.formal_eligible,
        "generation_status": "complete" if generation_passed else "ABSTAIN",
        "generator_protocol": dict(plan.generator_protocol),
        "counts": {
            "parent_geometries": len(plan.geometries),
            "planned_cases": len(plan.cases),
            "valid_cases": len(valid_rows),
            "invalid_cases": len(plan.cases) - len(valid_rows),
            "points_per_case": int(base_arrays["base_features"].shape[1]),
        },
        "identities": {
            "cross_partition_zero_intersection": all(
                plan.identity_report["cross_partition_zero_intersection"].values()
            ),
            "legacy_v0_2_locked_test_zero_intersection": all(
                plan.identity_report["legacy_zero_intersection"].values()
            ),
            "normalization_fit_train_only": True,
            "no_result_conditioned_replacement": True,
        },
        "artifact_hashes": artifact_hashes,
        "files": file_hashes,
        "solver_mesh_qc": {
            "no_silent_case_replacement": True,
            "records": solver_records,
            "partition_section_summary": qc_summary,
            "passed": bool(qc_summary["passed"]),
        },
        "fine_ultrafine_selection": {
            "selection_protocol": config["quality_control"]["fine_ultrafine"]["selection_protocol"],
            "selection_hash": audit_summary["selection_hash"],
            "selected_case_ids": list(plan.audit_case_ids),
            "selection_unit": "case_group_id",
            "formal_audit_fraction": float(
                config["quality_control"]["fine_ultrafine"]["formal_audit_fraction"]
            ),
            "selected_before_any_ultrafine_label": True,
            "case_values_exposed_before_checkpoint_freeze": False,
            "audit_passed_inside_trusted_generator": bool(audit_summary["passed"]),
            "failure_records": audit_failures,
        },
        "opaque_sealed_stores": opaque_hashes,
        "sealed_label_read_count": 0,
        "generation_resume": {"resumed_cases": resumed_cases, "solved_cases": solved_cases},
        "resource_usage": {
            "runtime_seconds": float(time.perf_counter() - started),
            "peak_memory_bytes": int(tracemalloc.get_traced_memory()[1]),
        },
    }
    manifest_path = root / MANIFEST_FILENAME
    manifest_hash = _write_json_atomic(manifest_path, manifest)
    tracemalloc.stop()
    public_hashes[MANIFEST_FILENAME] = manifest_hash
    result = FormalGenerationResult(
        manifest_path=manifest_path,
        public_inputs_path=public_path,
        train_dev_labels_path=train_path,
        audit_summary={
            "selected_case_count": len(plan.audit_case_ids),
            "computed_inside_trusted_generator": True,
            "case_values_exposed_before_checkpoint_freeze": False,
        },
        public_file_hashes=public_hashes,
        opaque_sealed_store_hashes=opaque_hashes,
        resumed_cases=resumed_cases,
        solved_cases=solved_cases,
    )
    if not generation_passed:
        raise FormalGenerationError(
            f"formal generation ABSTAIN; complete evidence written to {manifest_path}"
        )
    return result


__all__ = [
    "FORMAL_PARTITIONS",
    "LOCKED_PARTITIONS",
    "FormalGenerationError",
    "FormalGenerationOverrides",
    "FormalGenerationPlan",
    "FormalGenerationResult",
    "FrozenIdentityExclusions",
    "PlannedCase",
    "PlannedGeometry",
    "TrainingDataPaths",
    "build_formal_generation_plan",
    "generate_formal_dataset",
    "training_data_paths",
    "trusted_locked_label_path",
]
