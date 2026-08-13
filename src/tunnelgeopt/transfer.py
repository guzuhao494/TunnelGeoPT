"""Leakage-safe analytic Kirsch transfer-learning infrastructure.

This module is deliberately restricted to a circular, homogeneous, linear-
elastic opening with closed-form Kirsch labels.  It is a pipeline screen for
conditional geometric pre-training; it is not evidence about non-circular
tunnels, fracture, damage, rockburst, or field performance.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .kirsch import kirsch_stress

try:  # PyTorch is an optional ``learn`` dependency.
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised in core-only installations
    torch = None
    nn = None


FloatArray = NDArray[np.floating]
IntArray = NDArray[np.integer]
BoolArray = NDArray[np.bool_]

SPLIT_NAMES = ("train", "dev", "locked_test")
PRETRAIN_METHODS = (
    "static_geometry_80",
    "random_lift_80",
    "shuffled_stress_lift_80",
    "stress_lift_80",
)
ALL_METHODS = ("scratch_80", "scratch_100", *PRETRAIN_METHODS)


class TransferContractError(ValueError):
    """Raised when the frozen analytic-transfer contract is violated."""


@dataclass(frozen=True)
class AnalyticLoadCase:
    """One canonical circular-opening far-field load case."""

    ordinal: int
    sigma_ratio: float
    azimuth_rad: float
    sigma1: float
    radius: float
    stratum: int
    case_group_id: str
    split: str = "unassigned"

    @property
    def principal_direction(self) -> FloatArray:
        return np.asarray(
            [math.sin(self.azimuth_rad), math.cos(self.azimuth_rad)], dtype=np.float64
        )

    @property
    def farfield_tensor(self) -> FloatArray:
        e1 = self.principal_direction
        e3 = np.asarray([e1[1], -e1[0]], dtype=np.float64)
        return self.sigma1 * np.outer(e1, e1) + self.sigma1 * self.sigma_ratio * np.outer(e3, e3)

    @property
    def condition_vector(self) -> FloatArray:
        e1 = self.principal_direction
        return np.asarray([0.0, e1[0], e1[1], 1.0 - self.sigma_ratio], dtype=np.float32)


@dataclass(frozen=True)
class QueryGrid:
    """The common 512-point circular query grid and its geometric features."""

    points_yz: FloatArray
    x: FloatArray
    normals: FloatArray
    vector_distance: FloatArray
    annulus_mask: BoolArray
    wall_mask: BoolArray
    farfield_mask: BoolArray


@dataclass
class DatasetAccessAudit:
    """Mutable, data-layer evidence for label materialization and access."""

    materialized_cases: dict[str, int] = field(
        default_factory=lambda: {name: 0 for name in SPLIT_NAMES}
    )
    label_case_reads: dict[str, int] = field(
        default_factory=lambda: {name: 0 for name in SPLIT_NAMES}
    )
    denied_locked_test_accesses: int = 0
    locked_test_unlocked: bool = False
    frozen_checkpoint_count_authorized: int = 0
    events: list[dict[str, Any]] = field(default_factory=list)

    def snapshot(self) -> dict[str, Any]:
        """Return an immutable JSON-native copy suitable for run evidence."""

        return {
            "materialized_cases": dict(self.materialized_cases),
            "label_case_reads": dict(self.label_case_reads),
            "denied_locked_test_accesses": int(self.denied_locked_test_accesses),
            "locked_test_unlocked": bool(self.locked_test_unlocked),
            "frozen_checkpoint_count_authorized": int(self.frozen_checkpoint_count_authorized),
            "events": [dict(event) for event in self.events],
        }


@dataclass
class AnalyticDataset:
    """Canonical cases with split-aware, audited, lazily materialized labels."""

    cases: tuple[AnalyticLoadCase, ...]
    grid: QueryGrid
    _labels_by_case: dict[int, FloatArray] = field(default_factory=dict, repr=False)
    access_audit: DatasetAccessAudit = field(default_factory=DatasetAccessAudit)

    def indices(self, split: str) -> IntArray:
        if split not in SPLIT_NAMES:
            raise TransferContractError(f"unknown split {split!r}")
        return np.asarray(
            [index for index, case in enumerate(self.cases) if case.split == split],
            dtype=np.int64,
        )

    @property
    def expected_label_shape(self) -> tuple[int, int, int]:
        return (len(self.cases), self.grid.x.shape[0], 3)

    def _validate_indices(self, indices: Sequence[int]) -> list[int]:
        result = [int(index) for index in indices]
        if any(index < 0 or index >= len(self.cases) for index in result):
            raise TransferContractError("label access contains an out-of-range case index")
        return result

    def materialize_split(self, split: str, *, purpose: str) -> None:
        """Generate exact labels for one split, enforcing the locked-test gate."""

        if split not in SPLIT_NAMES:
            raise TransferContractError(f"unknown split {split!r}")
        if split == "locked_test" and not self.access_audit.locked_test_unlocked:
            self.access_audit.denied_locked_test_accesses += 1
            self.access_audit.events.append(
                {"event": "denied_materialization", "split": split, "purpose": purpose}
            )
            raise TransferContractError(
                "locked_test labels cannot be materialized before every checkpoint is frozen"
            )
        new_indices = [
            int(index) for index in self.indices(split) if int(index) not in self._labels_by_case
        ]
        for index in new_indices:
            self._labels_by_case[index] = _kirsch_case_labels(self.cases[index], self.grid).astype(
                np.float32
            )
        self.access_audit.materialized_cases[split] += len(new_indices)
        self.access_audit.events.append(
            {
                "event": "materialized",
                "split": split,
                "purpose": purpose,
                "new_case_count": len(new_indices),
            }
        )

    def authorize_locked_test(
        self,
        frozen_checkpoint_ids: Sequence[str],
        *,
        expected_checkpoint_count: int,
    ) -> None:
        """Unlock test labels only after an exact, unique checkpoint set froze."""

        identities = [str(identity) for identity in frozen_checkpoint_ids]
        if expected_checkpoint_count <= 0:
            raise TransferContractError("expected_checkpoint_count must be positive")
        if len(identities) != expected_checkpoint_count:
            raise TransferContractError(
                "locked_test authorization requires every expected frozen checkpoint"
            )
        if len(set(identities)) != len(identities) or any(not identity for identity in identities):
            raise TransferContractError(
                "locked_test authorization requires unique non-empty checkpoint identities"
            )
        if self.access_audit.materialized_cases["locked_test"] != 0:
            raise TransferContractError("locked_test labels were materialized before authorization")
        if self.access_audit.label_case_reads["locked_test"] != 0:
            raise TransferContractError("locked_test labels were read before authorization")
        self.access_audit.locked_test_unlocked = True
        self.access_audit.frozen_checkpoint_count_authorized = len(identities)
        self.access_audit.events.append(
            {
                "event": "locked_test_authorized",
                "frozen_checkpoint_count": len(identities),
            }
        )

    def labels_for(self, indices: Sequence[int], *, purpose: str) -> FloatArray:
        """Read labels through the split gate and record the real case access."""

        requested = self._validate_indices(indices)
        if not requested:
            raise TransferContractError("label access requires at least one case")
        splits = {self.cases[index].split for index in requested}
        if "locked_test" in splits and not self.access_audit.locked_test_unlocked:
            self.access_audit.denied_locked_test_accesses += 1
            self.access_audit.events.append(
                {
                    "event": "denied_read",
                    "split": "locked_test",
                    "purpose": purpose,
                    "requested_case_count": sum(
                        self.cases[index].split == "locked_test" for index in requested
                    ),
                }
            )
            raise TransferContractError(
                "locked_test labels cannot be read before every checkpoint is frozen"
            )
        for split in sorted(splits):
            if any(
                index not in self._labels_by_case
                for index in requested
                if self.cases[index].split == split
            ):
                self.materialize_split(split, purpose=purpose)
        for split in splits:
            self.access_audit.label_case_reads[split] += sum(
                self.cases[index].split == split for index in requested
            )
        self.access_audit.events.append(
            {
                "event": "label_read",
                "purpose": purpose,
                "case_count": len(requested),
                "splits": sorted(splits),
            }
        )
        result = np.stack([self._labels_by_case[index] for index in requested]).astype(
            np.float32, copy=False
        )
        if result.shape != (len(requested), self.grid.x.shape[0], 3):
            raise TransferContractError("stored labels violate the case tensor contract")
        return result

    def access_snapshot(self) -> dict[str, Any]:
        return self.access_audit.snapshot()


@dataclass
class TrainingResult:
    """A frozen best-dev checkpoint plus reconstructable training history."""

    state_dict: dict[str, Any]
    best_epoch: int
    epochs_run: int
    best_dev_primary: float
    history: list[dict[str, float]]


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def config_sha256(path: str | Path) -> str:
    """Hash the exact frozen config bytes used by a run."""

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_transfer_config(path: str | Path) -> dict[str, Any]:
    """Load and strictly validate the preregistered analytic smoke config."""

    path = Path(path)
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TransferContractError(f"could not read strict JSON config {path}: {exc}") from exc
    required = {
        "schema_version",
        "config_name",
        "status",
        "scope",
        "claim_exclusions",
        "research_question",
        "dataset",
        "split",
        "model",
        "methods",
        "pretraining",
        "high_fidelity_fraction",
        "optimization",
        "metrics",
        "smoke_go_no_go",
        "formal_gate_unchanged",
    }
    if set(config) != required:
        missing = sorted(required - set(config))
        extra = sorted(set(config) - required)
        raise TransferContractError(
            f"config top-level keys differ; missing={missing}, extra={extra}"
        )

    dataset = config["dataset"]
    split = config["split"]
    model = config["model"]
    pretraining = config["pretraining"]
    optimization = config["optimization"]
    metrics = config["metrics"]
    expected_literals = {
        "schema_version": "0.2.0",
        "config_name": "kirsch_analytic_transfer_smoke",
        "status": "preregistered_before_run",
        "scope": "pipeline_and_directional_pretraining_screen_on_analytic_circle_only",
    }
    for key, expected in expected_literals.items():
        if config.get(key) != expected:
            raise TransferContractError(f"{key} must remain {expected!r}")
    if dataset.get("source") != "closed_form_kirsch_stress" or dataset.get("section") != "circle":
        raise TransferContractError("this runner accepts only closed-form Kirsch circle data")
    if dataset.get("sampling") != "deterministic_scrambled_sobol":
        raise TransferContractError("dataset sampling must be deterministic_scrambled_sobol")
    if int(dataset.get("case_count", -1)) != 240:
        raise TransferContractError("the frozen smoke requires exactly 240 load cases")
    point_counts = dataset.get("points_per_case", {})
    if point_counts != {"annulus_volume": 384, "wall": 64, "farfield": 64, "total": 512}:
        raise TransferContractError("the frozen smoke requires the exact 384/64/64 point contract")
    if (
        split.get("unit") != "case_group_id"
        or split.get("case_hash") != "sha256_of_canonical_load_case"
    ):
        raise TransferContractError("split must use canonical case_group_id hashes")
    if split.get("assignment") != "hash_order_with_global_largest_remainder":
        raise TransferContractError("unexpected split assignment rule")
    if split.get("counts") != {"train": 168, "dev": 36, "locked_test": 36}:
        raise TransferContractError("the frozen split must be 168/36/36")
    if split.get("pretraining_access") != "train_cases_only":
        raise TransferContractError("pretraining access must be train_cases_only")
    if tuple(config.get("methods", ())) != ALL_METHODS:
        raise TransferContractError(f"methods must be exactly {ALL_METHODS}")
    if model.get("family") != "shared_deepsets_point_operator":
        raise TransferContractError("model family must be shared_deepsets_point_operator")
    if (
        model.get("point_input_width"),
        model.get("pretrain_output_width"),
        model.get("finetune_output_width"),
    ) != (11, 9, 3):
        raise TransferContractError("model widths must remain 11 -> 9/3")
    if not model.get("replace_output_head_before_finetune") or not model.get(
        "full_backbone_finetune"
    ):
        raise TransferContractError("head replacement and full-backbone fine-tuning are mandatory")
    if int(pretraining.get("trajectory_steps", -1)) != 3:
        raise TransferContractError("the frozen target has exactly three trajectory steps")
    if pretraining.get("principal_direction_condition") != "[0,sin(alpha),cos(alpha),1-K]":
        raise TransferContractError("principal condition convention changed")
    if pretraining.get("stress_lift_step_magnitude") != "base_step_times_1_minus_K":
        raise TransferContractError("stress-lift magnitude convention changed")
    if pretraining.get("static_target") != "t0_vector_distance_repeated_to_nine_channels":
        raise TransferContractError("static target convention changed")
    if optimization.get("case_batch_size") != 8 or optimization.get("optimizer") != "adamw":
        raise TransferContractError("case-batch optimizer contract changed")
    if optimization.get("early_stopping", {}).get("split") != "dev":
        raise TransferContractError("early stopping may use only dev")
    if metrics.get("primary") != "case_mean_stress_frobenius_relative_l2":
        raise TransferContractError("primary metric contract changed")
    if metrics.get("bootstrap", {}).get("unit") != "case_group_id":
        raise TransferContractError("bootstrap unit must be case_group_id")
    excluded = set(config["claim_exclusions"])
    mandatory_exclusions = {
        "noncircular_tunnel_transfer",
        "fracture",
        "damage",
        "rockburst",
        "field_validity",
    }
    if not mandatory_exclusions.issubset(excluded):
        raise TransferContractError("claim exclusions may not remove the analytic-scope safeguards")
    return config


def _sobol_points(count: int, dimension: int, seed: int) -> FloatArray:
    try:
        from scipy.stats import qmc
    except ImportError as exc:  # pragma: no cover - environment gate
        raise RuntimeError("SciPy is required for deterministic scrambled Sobol sampling") from exc
    exponent = math.ceil(math.log2(max(1, count)))
    sampler = qmc.Sobol(d=dimension, scramble=True, seed=int(seed))
    return np.asarray(sampler.random_base2(exponent)[:count], dtype=np.float64)


def _canonical_case_payload(
    *, ordinal: int, sigma_ratio: float, azimuth_rad: float, sigma1: float, radius: float
) -> dict[str, Any]:
    return {
        "azimuth_rad": float(azimuth_rad),
        "ordinal": int(ordinal),
        "radius": float(radius),
        "section": "circle",
        "sigma1_tension_positive": float(sigma1),
        "sigma3_over_sigma1": float(sigma_ratio),
    }


def _global_largest_remainder_allocations(
    stratum_sizes: Mapping[int, int], desired: Mapping[str, int]
) -> dict[int, dict[str, int]]:
    """Balance per-stratum Hamilton allocations against exact global counts.

    The dynamic program distributes each stratum's residual cases to the split
    with the largest fractional quota while enforcing the preregistered global
    totals exactly.  This avoids an after-the-fact unstratified correction.
    """

    total = sum(stratum_sizes.values())
    if total != sum(desired.values()):
        raise TransferContractError("stratum and global split totals differ")
    ratios = {name: desired[name] / total for name in SPLIT_NAMES}
    strata = sorted(stratum_sizes)
    floors: dict[int, dict[str, int]] = {}
    residuals: dict[int, int] = {}
    fractions: dict[int, dict[str, float]] = {}
    for stratum in strata:
        quotas = {name: stratum_sizes[stratum] * ratios[name] for name in SPLIT_NAMES}
        floors[stratum] = {name: math.floor(quotas[name]) for name in SPLIT_NAMES}
        fractions[stratum] = {name: quotas[name] - floors[stratum][name] for name in SPLIT_NAMES}
        residuals[stratum] = stratum_sizes[stratum] - sum(floors[stratum].values())
    deficits = tuple(
        desired[name] - sum(floors[stratum][name] for stratum in strata) for name in SPLIT_NAMES
    )
    # state -> (score, chosen split-index tuples)
    states: dict[tuple[int, int, int], tuple[float, tuple[tuple[int, ...], ...]]] = {
        (0, 0, 0): (0.0, ())
    }
    for stratum in strata:
        next_states: dict[tuple[int, int, int], tuple[float, tuple[tuple[int, ...], ...]]] = {}
        for picked in itertools.combinations(range(3), residuals[stratum]):
            increment = tuple(int(index in picked) for index in range(3))
            score_delta = sum(fractions[stratum][SPLIT_NAMES[index]] for index in picked)
            for state, (score, choices) in states.items():
                new_state = tuple(state[index] + increment[index] for index in range(3))
                if any(new_state[index] > deficits[index] for index in range(3)):
                    continue
                candidate = (score + score_delta, choices + (picked,))
                current = next_states.get(new_state)
                if (
                    current is None
                    or candidate[0] > current[0] + 1e-15
                    or (
                        math.isclose(candidate[0], current[0], rel_tol=0.0, abs_tol=1e-15)
                        and candidate[1] < current[1]
                    )
                ):
                    next_states[new_state] = candidate
        states = next_states
    if deficits not in states:
        raise TransferContractError("could not reconcile stratified largest-remainder counts")
    choices = states[deficits][1]
    result: dict[int, dict[str, int]] = {}
    for stratum, picked in zip(strata, choices, strict=True):
        result[stratum] = {
            name: floors[stratum][name] + int(index in picked)
            for index, name in enumerate(SPLIT_NAMES)
        }
    return result


def build_load_cases(config: Mapping[str, Any]) -> tuple[AnalyticLoadCase, ...]:
    """Create 240 Sobol load cases and the frozen 168/36/36 case split."""

    dataset = config["dataset"]
    split_config = config["split"]
    case_count = int(dataset["case_count"])
    sample = _sobol_points(case_count, 2, int(dataset["sampling_seed"]))
    ratio_min = float(dataset["sigma3_over_sigma1"]["min"])
    ratio_max = float(dataset["sigma3_over_sigma1"]["max"])
    azimuth_min = float(dataset["sigma1_azimuth_rad"]["min"])
    azimuth_max = float(dataset["sigma1_azimuth_rad"]["max_exclusive"])
    sigma1 = float(dataset["sigma1_tension_positive"])
    radius = float(dataset["radius"])
    unsplit: list[AnalyticLoadCase] = []
    for ordinal, unit in enumerate(sample):
        ratio = ratio_min + (ratio_max - ratio_min) * float(unit[0])
        azimuth = azimuth_min + (azimuth_max - azimuth_min) * float(unit[1])
        ratio_bin = min(3, int(4.0 * (ratio - ratio_min) / (ratio_max - ratio_min)))
        azimuth_sector = min(3, int(4.0 * (azimuth - azimuth_min) / (azimuth_max - azimuth_min)))
        stratum = 4 * ratio_bin + azimuth_sector
        payload = _canonical_case_payload(
            ordinal=ordinal,
            sigma_ratio=ratio,
            azimuth_rad=azimuth,
            sigma1=sigma1,
            radius=radius,
        )
        unsplit.append(
            AnalyticLoadCase(
                ordinal=ordinal,
                sigma_ratio=ratio,
                azimuth_rad=azimuth,
                sigma1=sigma1,
                radius=radius,
                stratum=stratum,
                case_group_id=_sha256_json(payload),
            )
        )
    if len({case.case_group_id for case in unsplit}) != case_count:
        raise TransferContractError("canonical case hashes are not unique")
    stratum_sizes = {
        stratum: sum(case.stratum == stratum for case in unsplit) for stratum in range(16)
    }
    desired = {name: int(split_config["counts"][name]) for name in SPLIT_NAMES}
    allocations = _global_largest_remainder_allocations(stratum_sizes, desired)
    split_by_id: dict[str, str] = {}
    for stratum in range(16):
        ordered = sorted(
            (case for case in unsplit if case.stratum == stratum),
            key=lambda case: case.case_group_id,
        )
        cursor = 0
        for split in SPLIT_NAMES:
            count = allocations[stratum][split]
            for case in ordered[cursor : cursor + count]:
                split_by_id[case.case_group_id] = split
            cursor += count
    cases = tuple(replace(case, split=split_by_id[case.case_group_id]) for case in unsplit)
    actual = {name: sum(case.split == name for case in cases) for name in SPLIT_NAMES}
    if actual != desired:
        raise TransferContractError(f"split count mismatch: expected {desired}, got {actual}")
    return cases


def build_query_grid(config: Mapping[str, Any]) -> QueryGrid:
    """Build the exact 384 annulus + 64 wall + 64 far-field point grid."""

    dataset = config["dataset"]
    counts = dataset["points_per_case"]
    seed = int(dataset["sampling_seed"])
    radius = float(dataset["radius"])
    inner_ratio, outer_ratio = map(float, dataset["annulus_radius_over_radius"])
    far_ratio = float(dataset["farfield_radius_over_radius"])
    annulus_unit = _sobol_points(int(counts["annulus_volume"]), 2, seed + 1)
    annulus_radius = radius * np.sqrt(
        inner_ratio**2 + annulus_unit[:, 0] * (outer_ratio**2 - inner_ratio**2)
    )
    annulus_angle = 2.0 * np.pi * annulus_unit[:, 1]
    annulus = np.column_stack(
        [annulus_radius * np.sin(annulus_angle), annulus_radius * np.cos(annulus_angle)]
    )
    wall_unit = _sobol_points(int(counts["wall"]), 1, seed + 2)[:, 0]
    wall_angle = 2.0 * np.pi * wall_unit
    wall = radius * np.column_stack([np.sin(wall_angle), np.cos(wall_angle)])
    far_unit = _sobol_points(int(counts["farfield"]), 1, seed + 3)[:, 0]
    far_angle = 2.0 * np.pi * far_unit
    farfield = far_ratio * radius * np.column_stack([np.sin(far_angle), np.cos(far_angle)])
    points = np.vstack([annulus, wall, farfield]).astype(np.float64)
    radial = np.linalg.norm(points, axis=1)
    normals = points / radial[:, None]
    distance = radial - radius
    vector_distance = distance[:, None] * normals
    total = int(counts["total"])
    annulus_mask = np.zeros(total, dtype=bool)
    wall_mask = np.zeros(total, dtype=bool)
    farfield_mask = np.zeros(total, dtype=bool)
    a_end = int(counts["annulus_volume"])
    w_end = a_end + int(counts["wall"])
    annulus_mask[:a_end] = True
    wall_mask[a_end:w_end] = True
    farfield_mask[w_end:] = True
    # GeoPT-compatible x: xyz, unsigned distance, xyz direction.  Rock-query
    # directions point toward the closest wall; wall points carry the rock-side normal.
    geometry_direction = -normals.copy()
    geometry_direction[wall_mask] = normals[wall_mask]
    x = np.column_stack(
        [
            np.zeros(total),
            points,
            distance,
            np.zeros(total),
            geometry_direction,
        ]
    ).astype(np.float32)
    if x.shape != (total, 7) or total != 512:
        raise TransferContractError("query grid violates the 512 x 7 contract")
    return QueryGrid(
        points_yz=points,
        x=x,
        normals=normals,
        vector_distance=vector_distance,
        annulus_mask=annulus_mask,
        wall_mask=wall_mask,
        farfield_mask=farfield_mask,
    )


def _kirsch_case_labels(case: AnalyticLoadCase, grid: QueryGrid) -> FloatArray:
    tensor = case.farfield_tensor
    result = kirsch_stress(
        grid.points_yz[:, 0],
        grid.points_yz[:, 1],
        radius=case.radius,
        sigma_x=float(tensor[0, 0]),
        sigma_y=float(tensor[1, 1]),
        tau_xy=float(tensor[0, 1]),
        return_cartesian=True,
    )
    scale = abs(case.sigma1)
    return (
        np.column_stack([result["sigma_xx"], result["sigma_yy"], result["tau_xy"]]).astype(
            np.float64
        )
        / scale
    )


def build_analytic_dataset(config: Mapping[str, Any]) -> AnalyticDataset:
    """Build cases/grid and materialize train/dev labels, never locked test."""

    cases = build_load_cases(config)
    grid = build_query_grid(config)
    dataset = AnalyticDataset(cases=cases, grid=grid)
    dataset.materialize_split("train", purpose="initial_train_build")
    dataset.materialize_split("dev", purpose="initial_dev_build")
    audit = dataset.access_snapshot()
    if audit["materialized_cases"] != {"train": 168, "dev": 36, "locked_test": 0}:
        raise TransferContractError("initial label materialization violated the split gate")
    return dataset


def _embed_vectors(vectors_yz: FloatArray) -> FloatArray:
    return np.column_stack([np.zeros(vectors_yz.shape[0]), vectors_yz])


def _vector_distance(points: FloatArray, radius: float) -> FloatArray:
    radial = np.linalg.norm(points, axis=1)
    normals = points / np.maximum(radial[:, None], 1e-15)
    distance = radial - radius
    distance[np.abs(distance) <= 1e-12 * max(radius, 1.0)] = 0.0
    return distance[:, None] * normals


def _advance_and_stick(
    positions: FloatArray,
    displacement: FloatArray,
    *,
    radius: float,
    stuck: BoolArray,
) -> tuple[FloatArray, BoolArray]:
    """Advance line segments and stick exactly at the first circular-wall hit."""

    updated = positions.copy()
    active = ~stuck
    proposed = positions[active] + displacement[active]
    start = positions[active]
    step = displacement[active]
    aa = np.sum(step * step, axis=1)
    bb = 2.0 * np.sum(start * step, axis=1)
    cc = np.sum(start * start, axis=1) - radius**2
    discriminant = bb**2 - 4.0 * aa * cc
    valid_disc = (aa > 1e-20) & (discriminant >= 0.0)
    sqrt_disc = np.sqrt(np.maximum(discriminant, 0.0))
    safe_aa = np.where(aa > 1e-20, aa, 1.0)
    roots = np.column_stack(
        [(-bb - sqrt_disc) / (2.0 * safe_aa), (-bb + sqrt_disc) / (2.0 * safe_aa)]
    )
    roots = np.where((roots >= 0.0) & (roots <= 1.0) & valid_disc[:, None], roots, np.inf)
    first = np.min(roots, axis=1)
    hit = np.isfinite(first)
    active_indices = np.flatnonzero(active)
    updated[active_indices] = proposed
    if np.any(hit):
        hit_indices = active_indices[hit]
        hit_points = start[hit] + first[hit, None] * step[hit]
        hit_points *= radius / np.maximum(np.linalg.norm(hit_points, axis=1, keepdims=True), 1e-15)
        updated[hit_indices] = hit_points
        stuck = stuck.copy()
        stuck[hit_indices] = True
    # A proposed endpoint inside the cavity necessarily crossed the wall; the
    # quadratic branch above should catch it, but keep a strict numeric guard.
    inside = np.linalg.norm(updated, axis=1) < radius * (1.0 - 1e-10)
    if np.any(inside):
        raise TransferContractError("trajectory entered the circular cavity without sticking")
    return updated, stuck


def stress_lift_case(case: AnalyticLoadCase, grid: QueryGrid) -> tuple[FloatArray, FloatArray]:
    """Create the task-specific Stress-Lift condition and three-step target.

    With rock-side closest-wall normal ``n`` and principal direction ``e1``:
    ``q = K + (1-K)(e1 dot n)^2`` and ``v = -smax*q*n``, where the normalized
    base step is one radius and therefore ``smax = R(1-K)``.  A point sticks at
    the first wall hit; wall points are fixed from the start.
    """

    count = grid.points_yz.shape[0]
    condition = np.repeat(case.condition_vector[None, :], count, axis=0)
    condition[grid.wall_mask, 3] = 0.0
    positions = grid.points_yz.copy()
    stuck = grid.wall_mask.copy()
    targets: list[FloatArray] = []
    e1 = case.principal_direction
    smax = case.radius * (1.0 - case.sigma_ratio)
    for step_index in range(3):
        targets.append(_embed_vectors(_vector_distance(positions, case.radius)))
        if step_index == 2:
            break
        normals = positions / np.linalg.norm(positions, axis=1, keepdims=True)
        q = case.sigma_ratio + (1.0 - case.sigma_ratio) * np.sum(normals * e1, axis=1) ** 2
        displacement = -smax * q[:, None] * normals
        displacement[stuck] = 0.0
        positions, stuck = _advance_and_stick(
            positions, displacement, radius=case.radius, stuck=stuck
        )
    return condition.astype(np.float32), np.concatenate(targets, axis=1).astype(np.float32)


def random_lift_case(
    case: AnalyticLoadCase,
    grid: QueryGrid,
    *,
    seed: int,
) -> tuple[FloatArray, FloatArray]:
    """Create a direction-random control with the exact Stress-Lift magnitude marginal."""

    rng = np.random.default_rng(int(seed))
    angles = rng.uniform(0.0, 2.0 * np.pi, grid.points_yz.shape[0])
    directions = np.column_stack([np.sin(angles), np.cos(angles)])
    magnitude = 1.0 - case.sigma_ratio
    condition = np.column_stack(
        [np.zeros(len(angles)), directions, np.full(len(angles), magnitude)]
    )
    condition[grid.wall_mask, 3] = 0.0
    positions = grid.points_yz.copy()
    stuck = grid.wall_mask.copy()
    targets: list[FloatArray] = []
    displacement = case.radius * magnitude * directions
    displacement[grid.wall_mask] = 0.0
    for step_index in range(3):
        targets.append(_embed_vectors(_vector_distance(positions, case.radius)))
        if step_index == 2:
            break
        step = displacement.copy()
        step[stuck] = 0.0
        positions, stuck = _advance_and_stick(positions, step, radius=case.radius, stuck=stuck)
    return condition.astype(np.float32), np.concatenate(targets, axis=1).astype(np.float32)


def static_geometry_case(case: AnalyticLoadCase, grid: QueryGrid) -> tuple[FloatArray, FloatArray]:
    """Create the static vector-distance control repeated over three target blocks."""

    del case
    condition = np.zeros((grid.x.shape[0], 4), dtype=np.float32)
    t0 = _embed_vectors(grid.vector_distance).astype(np.float32)
    return condition, np.tile(t0, (1, 3))


def deterministic_derangement(size: int, seed: int) -> IntArray:
    """Return a deterministic Sattolo single-cycle permutation with no fixed points."""

    if size < 2:
        raise TransferContractError("a derangement requires at least two cases")
    rng = np.random.default_rng(int(seed))
    permutation = np.arange(size, dtype=np.int64)
    for index in range(size - 1, 0, -1):
        other = int(rng.integers(0, index))
        permutation[index], permutation[other] = permutation[other], permutation[index]
    if np.any(permutation == np.arange(size)) or len(np.unique(permutation)) != size:
        raise TransferContractError("internal derangement construction failed")
    return permutation


def _assert_train_only(dataset: AnalyticDataset, indices: Sequence[int], purpose: str) -> None:
    offending = [
        dataset.cases[int(index)].case_group_id
        for index in indices
        if dataset.cases[int(index)].split != "train"
    ]
    if offending:
        raise TransferContractError(f"{purpose} attempted to read non-train cases: {offending[:3]}")


def build_pretraining_arrays(
    dataset: AnalyticDataset,
    train_indices: Sequence[int],
    method: str,
    *,
    seed: int,
) -> tuple[FloatArray, FloatArray, dict[str, Any]]:
    """Build one pretraining task using train cases only."""

    if method not in PRETRAIN_METHODS:
        raise TransferContractError(f"{method!r} is not a pretraining method")
    indices = np.asarray(train_indices, dtype=np.int64)
    _assert_train_only(dataset, indices, "pretraining")
    base_x = dataset.grid.x
    conditions: list[FloatArray] = []
    targets: list[FloatArray] = []
    if method == "static_geometry_80":
        for index in indices:
            condition, target = static_geometry_case(dataset.cases[int(index)], dataset.grid)
            conditions.append(condition)
            targets.append(target)
        partner = None
    elif method == "random_lift_80":
        for position, index in enumerate(indices):
            condition, target = random_lift_case(
                dataset.cases[int(index)], dataset.grid, seed=int(seed) * 1_000_003 + position
            )
            conditions.append(condition)
            targets.append(target)
        partner = None
    else:
        stress_pairs = [
            stress_lift_case(dataset.cases[int(index)], dataset.grid) for index in indices
        ]
        conditions = [pair[0] for pair in stress_pairs]
        targets = [pair[1] for pair in stress_pairs]
        partner = None
        if method == "shuffled_stress_lift_80":
            partner = deterministic_derangement(len(indices), int(seed) + 91_337)
            # Only the condition is shuffled.  The original target remains in
            # place, intentionally creating a contradictory negative control.
            conditions = [conditions[int(source)] for source in partner]
    features = np.stack(
        [np.concatenate([base_x, condition], axis=1) for condition in conditions]
    ).astype(np.float32)
    target_array = np.stack(targets).astype(np.float32)
    metadata = {
        "method": method,
        "case_group_ids": [dataset.cases[int(index)].case_group_id for index in indices],
        "splits_read": sorted({dataset.cases[int(index)].split for index in indices}),
        "derangement": partner.tolist() if partner is not None else None,
        "condition_only_shuffled": method == "shuffled_stress_lift_80",
    }
    if features.shape[2] != 11 or target_array.shape[2] != 9:
        raise TransferContractError("pretraining arrays violate 11 -> 9 contract")
    return features, target_array, metadata


def nested_train_indices(dataset: AnalyticDataset, fraction: float) -> IntArray:
    """Return a hash-ordered, case-level nested subset no larger than ``fraction``."""

    if not 0.0 < fraction <= 1.0:
        raise TransferContractError("training fraction must lie in (0, 1]")
    train = sorted(dataset.indices("train").tolist(), key=lambda i: dataset.cases[i].case_group_id)
    count = len(train) if fraction == 1.0 else math.floor(len(train) * fraction)
    if count < 1:
        raise TransferContractError("training fraction produced an empty case subset")
    return np.asarray(train[:count], dtype=np.int64)


def build_finetuning_arrays(
    dataset: AnalyticDataset,
    indices: Sequence[int],
) -> tuple[FloatArray, FloatArray]:
    """Build correctly conditioned stress-surrogate arrays for whole cases."""

    features = []
    requested = [int(index) for index in indices]
    for index in indices:
        case = dataset.cases[int(index)]
        condition = np.repeat(case.condition_vector[None, :], dataset.grid.x.shape[0], axis=0)
        features.append(np.concatenate([dataset.grid.x, condition], axis=1))
    result_x = np.stack(features).astype(np.float32)
    result_y = dataset.labels_for(requested, purpose="finetuning_or_evaluation")
    if result_x.shape[1:] != (512, 11) or result_y.shape[1:] != (512, 3):
        raise TransferContractError("fine-tuning arrays violate case tensor contract")
    return result_x, result_y


def build_conditioned_features(
    dataset: AnalyticDataset,
    indices: Sequence[int],
) -> FloatArray:
    """Build model features without touching labels (used by equivariance checks)."""

    features = []
    for index in indices:
        case = dataset.cases[int(index)]
        condition = np.repeat(case.condition_vector[None, :], dataset.grid.x.shape[0], axis=0)
        features.append(np.concatenate([dataset.grid.x, condition], axis=1))
    result = np.stack(features).astype(np.float32)
    if result.shape[1:] != (512, 11):
        raise TransferContractError("conditioned features violate the case tensor contract")
    return result


if nn is not None:

    class DeepSetsPointOperator(nn.Module):
        """Shared pointwise/global-context backbone used by every method."""

        def __init__(
            self,
            *,
            input_width: int,
            hidden_width: int,
            global_context_blocks: int,
            output_width: int,
        ) -> None:
            super().__init__()
            self.input_width = int(input_width)
            self.hidden_width = int(hidden_width)
            self.global_context_blocks = int(global_context_blocks)
            self.input_projection = nn.Sequential(
                nn.Linear(self.input_width, self.hidden_width), nn.GELU()
            )
            self.context_blocks = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(2 * self.hidden_width, self.hidden_width),
                        nn.GELU(),
                        nn.Linear(self.hidden_width, self.hidden_width),
                        nn.GELU(),
                    )
                    for _ in range(self.global_context_blocks)
                ]
            )
            self.context_norms = nn.ModuleList(
                [nn.LayerNorm(self.hidden_width) for _ in range(self.global_context_blocks)]
            )
            self.output_head = nn.Linear(self.hidden_width, int(output_width))

        def forward(self, points: Any) -> Any:
            state = self.input_projection(points)
            for block, norm in zip(self.context_blocks, self.context_norms, strict=True):
                context = state.mean(dim=1, keepdim=True).expand_as(state)
                state = norm(state + block(torch.cat([state, context], dim=-1)))
            return self.output_head(state)

        def replace_output_head(self, output_width: int, *, seed: int) -> None:
            """Discard the pretraining head and deterministically initialize a new head."""

            device = self.output_head.weight.device
            devices = [device.index] if device.type == "cuda" and device.index is not None else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(int(seed))
                self.output_head = nn.Linear(self.hidden_width, int(output_width)).to(device)

else:

    class DeepSetsPointOperator:  # pragma: no cover - core-only installation
        def __init__(self, **_: Any) -> None:
            raise RuntimeError("PyTorch is required; install TunnelGeoPT with the learn extra")


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("PyTorch is required; install TunnelGeoPT with the learn extra")
    return torch


def resolve_device(requested: str = "auto") -> str:
    """Resolve ``auto`` to CUDA when a real CUDA device is available."""

    framework = _require_torch()
    if requested == "auto":
        return "cuda" if framework.cuda.is_available() else "cpu"
    if requested == "cuda" and not framework.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if requested not in {"cpu", "cuda"}:
        raise TransferContractError("device must be auto, cpu, or cuda")
    return requested


def _seed_everything(seed: int) -> None:
    framework = _require_torch()
    random.seed(int(seed))
    np.random.seed(int(seed))
    framework.manual_seed(int(seed))
    if framework.cuda.is_available():
        framework.cuda.manual_seed_all(int(seed))
    framework.use_deterministic_algorithms(True, warn_only=True)


def make_model(config: Mapping[str, Any], *, output_width: int, seed: int, device: str) -> Any:
    """Create the one shared architecture with deterministic initialization."""

    _seed_everything(seed)
    model_config = config["model"]
    model = DeepSetsPointOperator(
        input_width=int(model_config["point_input_width"]),
        hidden_width=int(model_config["hidden_width"]),
        global_context_blocks=int(model_config["global_context_blocks"]),
        output_width=int(output_width),
    )
    return model.to(device)


def save_cpu_checkpoint_atomic(
    model: Any,
    path: str | Path,
    config: Mapping[str, Any],
    *,
    seed: int,
    metadata: Mapping[str, Any],
) -> str:
    """Atomically persist a CPU-only frozen state dict and return its SHA-256."""

    framework = _require_torch()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    model_config = config["model"]
    cpu_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    payload = {
        "format_version": 1,
        "model": {
            "input_width": int(model_config["point_input_width"]),
            "hidden_width": int(model_config["hidden_width"]),
            "global_context_blocks": int(model_config["global_context_blocks"]),
            "output_width": int(model_config["finetune_output_width"]),
        },
        "initialization_seed": int(seed),
        "metadata": dict(metadata),
        "state_dict": cpu_state,
    }
    try:
        framework.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    if not destination.is_file() or destination.stat().st_size <= 0:
        raise TransferContractError("atomic checkpoint write produced no durable file")
    return digest


def load_checkpoint_payload(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate a CPU checkpoint without constructing a model."""

    framework = _require_torch()
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        payload = framework.load(source, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise TransferContractError(f"could not load checkpoint {source}: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "format_version",
        "model",
        "initialization_seed",
        "metadata",
        "state_dict",
    }:
        raise TransferContractError(f"checkpoint {source} has an invalid envelope")
    if payload["format_version"] != 1 or not isinstance(payload["state_dict"], dict):
        raise TransferContractError(f"checkpoint {source} has an unsupported format")
    if any(
        getattr(value, "device", None).type != "cpu" for value in payload["state_dict"].values()
    ):
        raise TransferContractError(f"checkpoint {source} contains non-CPU tensors")
    return payload


def checkpoint_identity(path: str | Path) -> str:
    """Return the content identity of a durable frozen checkpoint."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    return hashlib.sha256(source.read_bytes()).hexdigest()


def load_model_checkpoint(
    path: str | Path,
    config: Mapping[str, Any],
    *,
    device: str,
    expected_metadata: Mapping[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Reconstruct one fine-tuning model from a validated CPU checkpoint."""

    payload = load_checkpoint_payload(path)
    model_config = config["model"]
    expected_model = {
        "input_width": int(model_config["point_input_width"]),
        "hidden_width": int(model_config["hidden_width"]),
        "global_context_blocks": int(model_config["global_context_blocks"]),
        "output_width": int(model_config["finetune_output_width"]),
    }
    if payload["model"] != expected_model:
        raise TransferContractError(f"checkpoint {path} model contract does not match config")
    metadata = payload["metadata"]
    if not isinstance(metadata, dict):
        raise TransferContractError(f"checkpoint {path} metadata must be a mapping")
    if expected_metadata is not None:
        for key, expected in expected_metadata.items():
            if metadata.get(key) != expected:
                raise TransferContractError(
                    f"checkpoint {path} metadata {key!r} does not match this run"
                )
    model = make_model(
        config,
        output_width=int(model_config["finetune_output_width"]),
        seed=int(payload["initialization_seed"]),
        device="cpu",
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, metadata


def _torch_case_batches(
    features: FloatArray,
    targets: FloatArray,
    *,
    batch_size: int,
    device: str,
    shuffle_seed: int | None,
) -> Sequence[tuple[Any, Any]]:
    framework = _require_torch()
    case_count = features.shape[0]
    if shuffle_seed is None:
        order = np.arange(case_count)
    else:
        order = np.random.default_rng(int(shuffle_seed)).permutation(case_count)
    batches = []
    for start in range(0, case_count, batch_size):
        batch = order[start : start + batch_size]
        batches.append(
            (
                framework.as_tensor(features[batch], dtype=framework.float32, device=device),
                framework.as_tensor(targets[batch], dtype=framework.float32, device=device),
            )
        )
    return batches


def one_training_step(
    model: Any,
    features: FloatArray,
    targets: FloatArray,
    config: Mapping[str, Any],
    *,
    device: str,
) -> float:
    """Run one real case-batch optimizer step for the CLI dry run."""

    framework = _require_torch()
    optimization = config["optimization"]
    optimizer = framework.optim.AdamW(
        model.parameters(),
        lr=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
    )
    batch_x, batch_y = _torch_case_batches(
        features,
        targets,
        batch_size=int(optimization["case_batch_size"]),
        device=device,
        shuffle_seed=None,
    )[0]
    optimizer.zero_grad(set_to_none=True)
    prediction = model(batch_x)
    loss = framework.mean((prediction - batch_y) ** 2)
    loss.backward()
    optimizer.step()
    value = float(loss.detach().cpu())
    if not math.isfinite(value):
        raise TransferContractError("non-finite loss in dry-run optimizer step")
    return value


def stress_frobenius_relative_l2(prediction: FloatArray, target: FloatArray) -> FloatArray:
    """Compute the preregistered tensor Frobenius relative L2 per case."""

    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[-1] != 3:
        raise TransferContractError("stress tensors must have matching [case, point, 3] shapes")
    delta = prediction - target
    numerator = np.sqrt(
        np.sum(delta[..., 0] ** 2 + delta[..., 1] ** 2 + 2.0 * delta[..., 2] ** 2, axis=1)
    )
    denominator = np.sqrt(
        np.sum(target[..., 0] ** 2 + target[..., 1] ** 2 + 2.0 * target[..., 2] ** 2, axis=1)
    )
    if np.any(denominator <= 0.0):
        raise TransferContractError("relative stress metric has a zero denominator")
    return numerator / denominator


def _components_to_tensor(components: FloatArray) -> FloatArray:
    tensor = np.empty((*components.shape[:-1], 2, 2), dtype=np.float64)
    tensor[..., 0, 0] = components[..., 0]
    tensor[..., 1, 1] = components[..., 1]
    tensor[..., 0, 1] = components[..., 2]
    tensor[..., 1, 0] = components[..., 2]
    return tensor


def _tensor_to_components(tensor: FloatArray) -> FloatArray:
    return np.stack([tensor[..., 0, 0], tensor[..., 1, 1], tensor[..., 0, 1]], axis=-1)


def case_metrics(
    prediction: FloatArray,
    target: FloatArray,
    grid: QueryGrid,
) -> dict[str, FloatArray]:
    """Compute primary, wall, far-field, and peak metrics case by case."""

    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    primary = stress_frobenius_relative_l2(prediction, target)
    farfield = stress_frobenius_relative_l2(
        prediction[:, grid.farfield_mask], target[:, grid.farfield_mask]
    )
    pred_tensor = _components_to_tensor(prediction[:, grid.wall_mask])
    target_tensor = _components_to_tensor(target[:, grid.wall_mask])
    normals = grid.normals[grid.wall_mask]
    pred_traction = np.einsum("cpij,pj->cpi", pred_tensor, normals)
    target_traction = np.einsum("cpij,pj->cpi", target_tensor, normals)
    traction_numerator = np.sqrt(np.sum((pred_traction - target_traction) ** 2, axis=(1, 2)))
    # The exact traction is zero, so use the wall target tensor norm as the
    # non-degenerate reference scale for a dimensionless traction violation.
    traction_denominator = np.sqrt(np.sum(target_tensor**2, axis=(1, 2, 3)))
    wall = traction_numerator / np.maximum(traction_denominator, 1e-15)
    tangents = np.column_stack([-normals[:, 1], normals[:, 0]])
    pred_hoop = np.einsum("pi,cpij,pj->cp", tangents, pred_tensor, tangents)
    target_hoop = np.einsum("pi,cpij,pj->cp", tangents, target_tensor, tangents)
    pred_peak = np.max(np.abs(pred_hoop), axis=1)
    target_peak = np.max(np.abs(target_hoop), axis=1)
    peak = np.abs(pred_peak - target_peak) / np.maximum(target_peak, 1e-15)
    result = {
        "stress_frobenius_relative_l2": primary,
        "wall_traction_relative_l2": wall,
        "farfield_stress_relative_l2": farfield,
        "peak_wall_hoop_stress_relative_error": peak,
    }
    if not all(np.isfinite(values).all() for values in result.values()):
        raise TransferContractError("non-finite evaluation metric")
    return result


def paired_stratified_bootstrap(
    candidate: FloatArray,
    reference: FloatArray,
    strata: IntArray,
    *,
    replicates: int,
    confidence: float,
    seed: int,
) -> dict[str, float]:
    """Paired case-level bootstrap of the equal-case mean error ratio."""

    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    strata = np.asarray(strata)
    if candidate.ndim != 1 or candidate.shape != reference.shape or strata.shape != candidate.shape:
        raise TransferContractError("bootstrap inputs must be aligned one-dimensional case arrays")
    if np.any(reference <= 0.0) or replicates <= 0 or not 0.0 < confidence < 1.0:
        raise TransferContractError("invalid bootstrap denominator or settings")
    groups = [np.flatnonzero(strata == value) for value in np.unique(strata)]
    rng = np.random.default_rng(int(seed))
    ratios = np.empty(int(replicates), dtype=np.float64)
    for replicate in range(int(replicates)):
        sampled = np.concatenate(
            [group[rng.integers(0, len(group), size=len(group))] for group in groups]
        )
        ratios[replicate] = candidate[sampled].mean() / reference[sampled].mean()
    alpha = 1.0 - confidence
    return {
        "center_ratio": float(candidate.mean() / reference.mean()),
        "lower": float(np.quantile(ratios, alpha / 2.0)),
        "upper": float(np.quantile(ratios, 1.0 - alpha / 2.0)),
        "replicates": int(replicates),
        "confidence": float(confidence),
    }


def _predict(model: Any, features: FloatArray, *, case_batch_size: int, device: str) -> FloatArray:
    framework = _require_torch()
    model.eval()
    predictions = []
    with framework.no_grad():
        for start in range(0, features.shape[0], case_batch_size):
            batch = framework.as_tensor(
                features[start : start + case_batch_size], dtype=framework.float32, device=device
            )
            predictions.append(model(batch).cpu().numpy())
    result = np.concatenate(predictions).astype(np.float64)
    if not np.isfinite(result).all():
        raise TransferContractError("model produced non-finite predictions")
    return result


def train_pretraining(
    model: Any,
    features: FloatArray,
    targets: FloatArray,
    config: Mapping[str, Any],
    *,
    seed: int,
    device: str,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, float]]:
    """Train a fixed-budget pretext task; it has no access to dev or test."""

    framework = _require_torch()
    optimization = config["optimization"]
    optimizer = framework.optim.AdamW(
        model.parameters(),
        lr=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
    )
    history: list[dict[str, float]] = []
    for epoch in range(int(optimization["max_epochs"])):
        model.train()
        weighted_loss = 0.0
        seen = 0
        for batch_x, batch_y in _torch_case_batches(
            features,
            targets,
            batch_size=int(optimization["case_batch_size"]),
            device=device,
            shuffle_seed=int(seed) * 1000 + epoch,
        ):
            optimizer.zero_grad(set_to_none=True)
            loss = framework.mean((model(batch_x) - batch_y) ** 2)
            loss.backward()
            optimizer.step()
            weighted_loss += float(loss.detach().cpu()) * batch_x.shape[0]
            seen += batch_x.shape[0]
        epoch_loss = weighted_loss / seen
        if not math.isfinite(epoch_loss):
            raise TransferContractError("non-finite pretraining loss")
        history.append({"epoch": float(epoch), "train_mse": epoch_loss})
        if progress is not None:
            progress({"phase": "pretrain", "epoch": epoch, "train_mse": epoch_loss})
    return history


def train_finetuning(
    model: Any,
    train_features: FloatArray,
    train_targets: FloatArray,
    dev_features: FloatArray,
    dev_targets: FloatArray,
    config: Mapping[str, Any],
    *,
    seed: int,
    device: str,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> TrainingResult:
    """Full-backbone fine-tune with dev-only early stopping."""

    framework = _require_torch()
    optimization = config["optimization"]
    stopping = optimization["early_stopping"]
    optimizer = framework.optim.AdamW(
        model.parameters(),
        lr=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
    )
    best_state: dict[str, Any] | None = None
    best_value = math.inf
    best_epoch = -1
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(int(optimization["max_epochs"])):
        model.train()
        weighted_loss = 0.0
        seen = 0
        for batch_x, batch_y in _torch_case_batches(
            train_features,
            train_targets,
            batch_size=int(optimization["case_batch_size"]),
            device=device,
            shuffle_seed=int(seed) * 10_000 + epoch,
        ):
            optimizer.zero_grad(set_to_none=True)
            loss = framework.mean((model(batch_x) - batch_y) ** 2)
            loss.backward()
            optimizer.step()
            weighted_loss += float(loss.detach().cpu()) * batch_x.shape[0]
            seen += batch_x.shape[0]
        train_mse = weighted_loss / seen
        dev_prediction = _predict(
            model,
            dev_features,
            case_batch_size=int(optimization["case_batch_size"]),
            device=device,
        )
        dev_primary = float(stress_frobenius_relative_l2(dev_prediction, dev_targets).mean())
        if not math.isfinite(train_mse) or not math.isfinite(dev_primary):
            raise TransferContractError("non-finite fine-tuning metric")
        history.append({"epoch": float(epoch), "train_mse": train_mse, "dev_primary": dev_primary})
        if dev_primary < best_value - float(stopping["min_delta"]):
            best_value = dev_primary
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if progress is not None:
            progress(
                {
                    "phase": "finetune",
                    "epoch": epoch,
                    "train_mse": train_mse,
                    "dev_primary": dev_primary,
                    "best_epoch": best_epoch,
                }
            )
        if stale >= int(stopping["patience"]):
            break
    if best_state is None:
        raise TransferContractError("fine-tuning never produced a finite dev checkpoint")
    model.load_state_dict(best_state)
    return TrainingResult(
        state_dict=best_state,
        best_epoch=best_epoch,
        epochs_run=len(history),
        best_dev_primary=best_value,
        history=history,
    )


def rotation_equivariance_errors(
    model: Any,
    dataset: AnalyticDataset,
    indices: Sequence[int],
    *,
    case_batch_size: int,
    device: str,
    angle_rad: float = math.pi / 2.0,
) -> FloatArray:
    """Compare predictions under a 90-degree circle/load tensor rotation."""

    features = build_conditioned_features(dataset, indices)
    original = _predict(model, features, case_batch_size=case_batch_size, device=device)
    cosine, sine = math.cos(angle_rad), math.sin(angle_rad)
    rotation = np.asarray([[cosine, sine], [-sine, cosine]], dtype=np.float64)
    rotated_features = features.copy()
    rotated_features[..., 1:3] = np.einsum("ij,cpj->cpi", rotation, features[..., 1:3])
    rotated_features[..., 5:7] = np.einsum("ij,cpj->cpi", rotation, features[..., 5:7])
    for position, index in enumerate(indices):
        case = dataset.cases[int(index)]
        alpha = (case.azimuth_rad + angle_rad) % np.pi
        condition = np.asarray(
            [0.0, math.sin(alpha), math.cos(alpha), 1.0 - case.sigma_ratio], dtype=np.float32
        )
        rotated_features[position, :, 7:11] = condition
    rotated_prediction = _predict(
        model, rotated_features, case_batch_size=case_batch_size, device=device
    )
    original_tensor = _components_to_tensor(original)
    expected_tensor = np.einsum("ij,cpjk,lk->cpil", rotation, original_tensor, rotation)
    expected = _tensor_to_components(expected_tensor)
    return stress_frobenius_relative_l2(rotated_prediction, expected)


def train_method(
    dataset: AnalyticDataset,
    config: Mapping[str, Any],
    method: str,
    seed: int,
    *,
    device: str,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Train one fair shared-backbone method without reading locked test."""

    if method not in ALL_METHODS:
        raise TransferContractError(f"unknown method {method!r}")
    train_full = nested_train_indices(dataset, 1.0)
    train_80 = nested_train_indices(dataset, float(config["high_fidelity_fraction"]["candidate"]))
    finetune_indices = train_full if method == "scratch_100" else train_80
    dev_indices = dataset.indices("dev")
    train_x, train_y = build_finetuning_arrays(dataset, finetune_indices)
    dev_x, dev_y = build_finetuning_arrays(dataset, dev_indices)
    model_config = config["model"]
    pretraining_meta = None
    pretraining_history: list[dict[str, float]] = []
    if method in PRETRAIN_METHODS:
        model = make_model(
            config,
            output_width=int(model_config["pretrain_output_width"]),
            seed=seed,
            device=device,
        )
        pretrain_x, pretrain_y, pretraining_meta = build_pretraining_arrays(
            dataset, train_full, method, seed=seed
        )
        pretraining_history = train_pretraining(
            model,
            pretrain_x,
            pretrain_y,
            config,
            seed=seed,
            device=device,
            progress=progress,
        )
        model.replace_output_head(int(model_config["finetune_output_width"]), seed=seed + 500_000)
    else:
        model = make_model(
            config,
            output_width=int(model_config["finetune_output_width"]),
            seed=seed,
            device=device,
        )
    finetuning = train_finetuning(
        model,
        train_x,
        train_y,
        dev_x,
        dev_y,
        config,
        seed=seed,
        device=device,
        progress=progress,
    )
    audit = {
        "method": method,
        "seed": int(seed),
        "pretraining_case_count": len(train_full) if method in PRETRAIN_METHODS else 0,
        "pretraining_splits_read": pretraining_meta["splits_read"] if pretraining_meta else [],
        "finetuning_case_count": len(finetune_indices),
        "finetuning_case_group_ids": [
            dataset.cases[int(i)].case_group_id for i in finetune_indices
        ],
        "selection_split": "dev",
        "pretraining_epochs": len(pretraining_history),
        "finetuning_epochs": finetuning.epochs_run,
        "best_epoch": finetuning.best_epoch,
        "best_dev_primary": finetuning.best_dev_primary,
    }
    return model, audit


def evaluate_locked_test(
    model: Any,
    dataset: AnalyticDataset,
    config: Mapping[str, Any],
    *,
    device: str,
) -> dict[str, Any]:
    """Perform the single post-freeze locked-test inference for one checkpoint."""

    test_indices = dataset.indices("locked_test")
    features, targets = build_finetuning_arrays(dataset, test_indices)
    batch_size = int(config["optimization"]["case_batch_size"])
    forward_batches_per_pass = math.ceil(len(test_indices) / batch_size)
    prediction = _predict(model, features, case_batch_size=batch_size, device=device)
    metrics = case_metrics(prediction, targets, dataset.grid)
    metrics["rotation_equivariance_relative_error"] = rotation_equivariance_errors(
        model,
        dataset,
        test_indices,
        case_batch_size=batch_size,
        device=device,
    )
    return {
        "case_group_ids": [dataset.cases[int(index)].case_group_id for index in test_indices],
        "strata": [dataset.cases[int(index)].stratum for index in test_indices],
        "per_case": {key: values.tolist() for key, values in metrics.items()},
        "means": {key: float(np.mean(values)) for key, values in metrics.items()},
        "access_counts": {
            "evaluation_calls": 1,
            "locked_test_label_case_reads": len(test_indices),
            "locked_test_model_forward_passes": 3,
            "locked_test_model_forward_batches": 3 * forward_batches_per_pass,
        },
    }


def dry_run_contract(
    dataset: AnalyticDataset,
    config: Mapping[str, Any],
    *,
    device: str,
) -> dict[str, Any]:
    """Exercise all method paths with one real train-only optimizer batch each.

    This is wiring evidence only.  It never performs locked-test inference and
    therefore cannot be interpreted as a model-effect result.
    """

    framework = _require_torch()
    train_full = nested_train_indices(dataset, 1.0)
    train_80 = nested_train_indices(dataset, float(config["high_fidelity_fraction"]["candidate"]))
    method_reports: dict[str, Any] = {}
    model_config = config["model"]
    for method in ALL_METHODS:
        finetune_indices = train_full if method == "scratch_100" else train_80
        train_x, train_y = build_finetuning_arrays(dataset, finetune_indices)
        if method in PRETRAIN_METHODS:
            model = make_model(
                config,
                output_width=int(model_config["pretrain_output_width"]),
                seed=17,
                device=device,
            )
            pretrain_x, pretrain_y, metadata = build_pretraining_arrays(
                dataset, train_full, method, seed=17
            )
            pretrain_loss = one_training_step(model, pretrain_x, pretrain_y, config, device=device)
            old_head_id = id(model.output_head)
            model.replace_output_head(int(model_config["finetune_output_width"]), seed=500_017)
            head_replaced = id(model.output_head) != old_head_id
            splits_read = metadata["splits_read"]
            derangement_fixed_points = (
                None
                if metadata["derangement"] is None
                else int(
                    np.sum(
                        np.asarray(metadata["derangement"])
                        == np.arange(len(metadata["derangement"]))
                    )
                )
            )
        else:
            model = make_model(
                config,
                output_width=int(model_config["finetune_output_width"]),
                seed=17,
                device=device,
            )
            pretrain_loss = None
            head_replaced = False
            splits_read = []
            derangement_fixed_points = None
        finetune_loss = one_training_step(model, train_x, train_y, config, device=device)
        method_reports[method] = {
            "pretraining_loss_one_batch": pretrain_loss,
            "finetuning_loss_one_batch": finetune_loss,
            "pretraining_splits_read": splits_read,
            "finetuning_case_count": len(finetune_indices),
            "output_width_after_setup": int(model.output_head.out_features),
            "head_replaced": head_replaced,
            "derangement_fixed_points": derangement_fixed_points,
        }
        del model
        if device == "cuda":
            framework.cuda.empty_cache()
    split_counts = {name: len(dataset.indices(name)) for name in SPLIT_NAMES}
    return {
        "status": "dry_run_passed",
        "claim_scope": "analytic_circle_pipeline_wiring_only",
        "claim_exclusions": config["claim_exclusions"],
        "case_count": len(dataset.cases),
        "points_per_case": int(dataset.grid.x.shape[0]),
        "expected_label_shape": list(dataset.expected_label_shape),
        "split_counts": split_counts,
        "train_80_case_count": len(train_80),
        "device": device,
        "cuda_device_name": framework.cuda.get_device_name(0) if device == "cuda" else None,
        "locked_test_inference_count": 0,
        "dataset_access_audit": dataset.access_snapshot(),
        "methods": method_reports,
    }


def summarize_full_gate(
    results: Mapping[str, Mapping[int, Mapping[str, Any]]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the preregistered smoke gate after all locked-test runs exist."""

    bootstrap = config["metrics"]["bootstrap"]
    gate_config = config["smoke_go_no_go"]
    seeds = [int(seed) for seed in config["optimization"]["training_seeds"]]
    candidate_name = str(gate_config["candidate"])
    reference_name = str(gate_config["reference"])
    per_seed: dict[str, Any] = {}
    candidate_matrix = []
    reference_matrix = []
    for seed in seeds:
        candidate_result = results[candidate_name][seed]
        reference_result = results[reference_name][seed]
        candidate = np.asarray(candidate_result["per_case"]["stress_frobenius_relative_l2"])
        reference = np.asarray(reference_result["per_case"]["stress_frobenius_relative_l2"])
        strata = np.asarray(candidate_result["strata"])
        interval = paired_stratified_bootstrap(
            candidate,
            reference,
            strata,
            replicates=int(bootstrap["replicates"]),
            confidence=float(bootstrap["confidence"]),
            seed=int(bootstrap["seed"]) + seed,
        )
        interval["passes_ratio_gate"] = bool(
            interval["upper"] <= float(gate_config["max_upper_95_percent_ci_error_ratio"])
        )
        per_seed[str(seed)] = interval
        candidate_matrix.append(candidate)
        reference_matrix.append(reference)
    seed_passes = sum(item["passes_ratio_gate"] for item in per_seed.values())

    def mean_primary(method: str) -> float:
        return float(
            np.mean(
                [results[method][seed]["means"]["stress_frobenius_relative_l2"] for seed in seeds]
            )
        )

    candidate_center = mean_primary(candidate_name)
    better_static = candidate_center < mean_primary("static_geometry_80")
    better_random = candidate_center < mean_primary("random_lift_80")
    shuffled_passes = []
    for seed in seeds:
        shuffled = np.asarray(
            results["shuffled_stress_lift_80"][seed]["per_case"]["stress_frobenius_relative_l2"]
        )
        reference = np.asarray(
            results[reference_name][seed]["per_case"]["stress_frobenius_relative_l2"]
        )
        strata = np.asarray(results[reference_name][seed]["strata"])
        interval = paired_stratified_bootstrap(
            shuffled,
            reference,
            strata,
            replicates=int(bootstrap["replicates"]),
            confidence=float(bootstrap["confidence"]),
            seed=int(bootstrap["seed"]) + seed + 100_000,
        )
        shuffled_passes.append(
            interval["upper"] <= float(gate_config["max_upper_95_percent_ci_error_ratio"])
        )
    violation_deltas = {}
    for key in ("wall_traction_relative_l2", "farfield_stress_relative_l2"):
        candidate_value = float(
            np.mean([results[candidate_name][seed]["means"][key] for seed in seeds])
        )
        reference_value = float(
            np.mean([results[reference_name][seed]["means"][key] for seed in seeds])
        )
        violation_deltas[key] = candidate_value - reference_value
    physics_ok = max(violation_deltas.values()) <= float(
        gate_config["max_absolute_wall_or_farfield_violation_increase"]
    )
    passed = bool(
        seed_passes >= int(gate_config["minimum_seed_pass_count"])
        and better_static
        and better_random
        and not any(shuffled_passes)
        and physics_ok
    )
    return {
        "status": "go" if passed else "no_go",
        "passing_scope": gate_config["passing_scope"],
        "per_seed_candidate_vs_reference": per_seed,
        "seed_pass_count": int(seed_passes),
        "candidate_center_better_than_static": bool(better_static),
        "candidate_center_better_than_random": bool(better_random),
        "shuffled_control_passes_same_gate": [bool(value) for value in shuffled_passes],
        "wall_farfield_violation_deltas": violation_deltas,
        "physics_violation_gate_passed": bool(physics_ok),
        "formal_gate_status": "not_evaluated_by_this_smoke",
        "claim_exclusions": config["claim_exclusions"],
    }
