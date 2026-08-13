"""Auditable learning utilities for coarse-to-fine elastic stress correction.

This module is deliberately independent of the v0.2 circle/Kirsch transfer
pipeline.  It learns a numerical correction between two discretizations of
the same linear-elastic problem; it does not model damage or rockburst.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

try:  # Keep the elastic/core installation importable without PyTorch.
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - core-only installation
    torch = None
    nn = None

FloatArray = NDArray[np.floating]
IntArray = NDArray[np.integer]

METHODS = (
    "scratch",
    "direct_coarse",
    "residual_coarse",
    "mismatched_coarse",
)


class LearningContractError(ValueError):
    """Raised when an experiment violates the multi-fidelity contract."""


@dataclass(frozen=True)
class LearningBatch:
    """Aligned case tensors and immutable grouping metadata."""

    base_features: FloatArray  # [case, point, 11]
    coarse_stress: FloatArray  # [case, point, 3]
    fine_stress: FloatArray  # [case, point, 3]
    weights: FloatArray  # [case, point]
    geometry_group_ids: tuple[str, ...]
    section_families: tuple[str, ...]
    case_group_ids: tuple[str, ...]
    splits: tuple[str, ...]

    def __post_init__(self) -> None:
        base = np.asarray(self.base_features)
        coarse = np.asarray(self.coarse_stress)
        fine = np.asarray(self.fine_stress)
        weights = np.asarray(self.weights)
        if base.ndim != 3 or base.shape[-1] != 11:
            raise LearningContractError("base_features must have shape [C, P, 11]")
        if coarse.shape != (*base.shape[:2], 3) or fine.shape != coarse.shape:
            raise LearningContractError("coarse/fine stress must have shape [C, P, 3]")
        if weights.shape != base.shape[:2]:
            raise LearningContractError("weights must have shape [C, P]")
        if not all(np.isfinite(value).all() for value in (base, coarse, fine, weights)):
            raise LearningContractError("learning tensors must be finite")
        if np.any(weights < 0.0) or np.any(weights.sum(axis=1) <= 0.0):
            raise LearningContractError(
                "query weights must be non-negative with positive mass in every case"
            )
        case_count = base.shape[0]
        metadata = (
            self.geometry_group_ids,
            self.section_families,
            self.case_group_ids,
            self.splits,
        )
        if any(len(values) != case_count for values in metadata):
            raise LearningContractError("metadata length must equal the case count")
        if len(set(self.case_group_ids)) != case_count:
            raise LearningContractError("case_group_ids must be unique")
        if any(not value for values in metadata for value in values):
            raise LearningContractError("metadata identifiers must be non-empty")
        geometry_splits: dict[str, str] = {}
        for geometry_id, split in zip(self.geometry_group_ids, self.splits, strict=True):
            previous = geometry_splits.setdefault(geometry_id, split)
            if previous != split:
                raise LearningContractError("one geometry_group_id crosses dataset splits")


@dataclass(frozen=True)
class TrainingOutcome:
    """One dev-selected training result with a CPU state dictionary."""

    state_dict: Mapping[str, Any]
    best_epoch: int
    epochs_run: int
    best_dev_error: float
    history: tuple[Mapping[str, float], ...]


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _stable_fraction_rank(identifier: str, salt: str) -> str:
    return _sha256_text(f"{salt}:{identifier}")


def nested_geometry_subsets(
    geometry_group_ids: Sequence[str],
    section_families: Sequence[str],
    *,
    fractions: Sequence[float] = (0.25, 0.5, 0.75, 1.0),
    salt: str,
) -> dict[float, tuple[str, ...]]:
    """Select section-balanced, parent-geometry-level, strictly nested subsets."""

    if len(geometry_group_ids) != len(section_families) or not geometry_group_ids:
        raise LearningContractError("geometry ids and sections must be aligned and non-empty")
    if not salt:
        raise LearningContractError("subset salt must be non-empty")
    if tuple(sorted({float(value) for value in fractions})) != tuple(
        float(value) for value in fractions
    ):
        raise LearningContractError("fractions must be unique and increasing")
    if any(not 0.0 < float(value) <= 1.0 for value in fractions):
        raise LearningContractError("fractions must lie in (0, 1]")
    geometry_section: dict[str, str] = {}
    for geometry_id, section in zip(geometry_group_ids, section_families, strict=True):
        previous = geometry_section.setdefault(str(geometry_id), str(section))
        if previous != section:
            raise LearningContractError("one geometry belongs to multiple section families")
    by_section: dict[str, list[str]] = {}
    for geometry_id, section in geometry_section.items():
        by_section.setdefault(section, []).append(geometry_id)
    for section, identifiers in by_section.items():
        by_section[section] = sorted(
            identifiers, key=lambda value: _stable_fraction_rank(value, f"{salt}:{section}")
        )
    result: dict[float, tuple[str, ...]] = {}
    previous: set[str] = set()
    for fraction in fractions:
        selected: set[str] = set()
        for identifiers in by_section.values():
            count = (
                len(identifiers)
                if fraction == 1.0
                else max(1, math.floor(len(identifiers) * fraction))
            )
            selected.update(identifiers[:count])
        if not previous.issubset(selected):  # defensive: selection is prefix-based
            raise LearningContractError("geometry subsets are not nested")
        result[float(fraction)] = tuple(sorted(selected))
        previous = selected
    return result


def mismatched_coarse_indices(section_families: Sequence[str], *, seed: int) -> IntArray:
    """Create a deterministic within-section derangement for a negative control."""

    sections = np.asarray(tuple(section_families), dtype=object)
    if sections.ndim != 1 or sections.size == 0:
        raise LearningContractError("section_families must be a non-empty sequence")
    permutation = np.empty(sections.size, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    for section in sorted(set(sections.tolist())):
        indices = np.flatnonzero(sections == section)
        if indices.size < 2:
            raise LearningContractError("mismatched control needs at least two cases per section")
        shift = int(rng.integers(1, indices.size))
        permutation[indices] = np.roll(indices, shift)
    if np.any(permutation == np.arange(sections.size)):
        raise LearningContractError("mismatched coarse permutation contains a fixed point")
    return permutation


def method_arrays(
    batch: LearningBatch,
    method: str,
    *,
    mismatch_indices: ArrayLike | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return model features, optimization target, and coarse reconstruction base."""

    if method not in METHODS:
        raise LearningContractError(f"unknown method {method!r}")
    coarse = np.asarray(batch.coarse_stress, dtype=np.float32)
    if method == "scratch":
        input_coarse = np.zeros_like(coarse)
        target = np.asarray(batch.fine_stress, dtype=np.float32)
        reconstruction_base = np.zeros_like(coarse)
    elif method == "direct_coarse":
        input_coarse = coarse
        target = np.asarray(batch.fine_stress, dtype=np.float32)
        reconstruction_base = np.zeros_like(coarse)
    else:
        input_coarse = coarse
        if method == "mismatched_coarse":
            if mismatch_indices is None:
                raise LearningContractError("mismatched method requires explicit indices")
            permutation = np.asarray(mismatch_indices, dtype=np.int64)
            if permutation.shape != (coarse.shape[0],):
                raise LearningContractError("mismatch_indices must have one entry per case")
            if np.any(permutation < 0) or np.any(permutation >= coarse.shape[0]):
                raise LearningContractError("mismatch_indices contain an out-of-range value")
            input_coarse = coarse[permutation]
        reconstruction_base = input_coarse
        target = np.asarray(batch.fine_stress, dtype=np.float32) - input_coarse
    features = np.concatenate(
        [np.asarray(batch.base_features, dtype=np.float32), input_coarse], axis=-1
    )
    if features.shape[-1] != 14 or target.shape[-1] != 3:
        raise LearningContractError("method arrays violate the 14-to-3 tensor contract")
    return features, target, reconstruction_base


def reconstruct_fine_prediction(
    raw_prediction: ArrayLike, reconstruction_base: ArrayLike
) -> FloatArray:
    """Convert a direct output or residual output into a fine stress prediction."""

    prediction = np.asarray(raw_prediction, dtype=np.float64)
    base = np.asarray(reconstruction_base, dtype=np.float64)
    if prediction.shape != base.shape or prediction.ndim != 3 or prediction.shape[-1] != 3:
        raise LearningContractError("prediction and reconstruction base must match [C,P,3]")
    result = prediction + base
    if not np.isfinite(result).all():
        raise LearningContractError("reconstructed prediction is non-finite")
    return result


def case_weighted_stress_error(
    prediction: ArrayLike, target: ArrayLike, weights: ArrayLike
) -> FloatArray:
    """Area/query-weighted tensor Frobenius relative L2 for each case."""

    prediction_array = np.asarray(prediction, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    weight_array = np.asarray(weights, dtype=np.float64)
    if prediction_array.shape != target_array.shape or prediction_array.ndim != 3:
        raise LearningContractError("prediction and target must match [C,P,3]")
    if prediction_array.shape[-1] != 3 or weight_array.shape != prediction_array.shape[:2]:
        raise LearningContractError("weights must align with stress query points")
    if not all(
        np.isfinite(value).all() for value in (prediction_array, target_array, weight_array)
    ):
        raise LearningContractError("metric inputs must be finite")
    if np.any(weight_array < 0.0) or np.any(weight_array.sum(axis=1) <= 0.0):
        raise LearningContractError(
            "metric weights must be non-negative with positive mass in every case"
        )
    multiplier = np.asarray([1.0, 1.0, 2.0])
    numerator = np.sum(
        weight_array[..., None] * multiplier * (prediction_array - target_array) ** 2,
        axis=(1, 2),
    )
    denominator = np.sum(
        weight_array[..., None] * multiplier * target_array**2,
        axis=(1, 2),
    )
    if np.any(denominator <= np.finfo(float).tiny):
        raise LearningContractError("fine stress norm is zero")
    return np.sqrt(numerator / denominator)


def section_balanced_geometry_mean(
    case_errors: ArrayLike,
    geometry_group_ids: Sequence[str],
    section_families: Sequence[str],
) -> tuple[float, dict[str, float], dict[str, float]]:
    """Average loads within geometry, then average geometries and sections equally."""

    errors = np.asarray(case_errors, dtype=np.float64)
    if (
        errors.ndim != 1
        or len(geometry_group_ids) != errors.size
        or len(section_families) != errors.size
    ):
        raise LearningContractError("case errors and grouping metadata must align")
    if not np.isfinite(errors).all() or np.any(errors < 0.0):
        raise LearningContractError("case errors must be finite and non-negative")
    geometry_values: dict[str, list[float]] = {}
    geometry_section: dict[str, str] = {}
    for value, geometry_id, section in zip(
        errors, geometry_group_ids, section_families, strict=True
    ):
        geometry_values.setdefault(geometry_id, []).append(float(value))
        previous = geometry_section.setdefault(geometry_id, section)
        if previous != section:
            raise LearningContractError("one geometry belongs to multiple section families")
    geometry_means = {key: float(np.mean(values)) for key, values in geometry_values.items()}
    section_groups: dict[str, list[float]] = {}
    for geometry_id, value in geometry_means.items():
        section_groups.setdefault(geometry_section[geometry_id], []).append(value)
    section_means = {key: float(np.mean(values)) for key, values in section_groups.items()}
    return float(np.mean(list(section_means.values()))), geometry_means, section_means


if nn is not None:

    class MultiFidelityPointOperator(nn.Module):
        """The shared 14-channel backbone for every learned baseline."""

        def __init__(
            self,
            *,
            input_width: int = 14,
            hidden_width: int = 64,
            global_context_blocks: int = 3,
            output_width: int = 3,
        ) -> None:
            super().__init__()
            if input_width != 14 or output_width != 3:
                raise LearningContractError("v0.3 shared model contract is fixed at 14-to-3")
            self.input_width = input_width
            self.hidden_width = int(hidden_width)
            self.global_context_blocks = int(global_context_blocks)
            self.input_projection = nn.Sequential(
                nn.Linear(input_width, self.hidden_width), nn.GELU()
            )
            self.blocks = nn.ModuleList(
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
            self.norms = nn.ModuleList(
                [nn.LayerNorm(self.hidden_width) for _ in range(self.global_context_blocks)]
            )
            self.head = nn.Linear(self.hidden_width, output_width)

        def forward(self, points: Any) -> Any:
            state = self.input_projection(points)
            for block, norm in zip(self.blocks, self.norms, strict=True):
                context = state.mean(dim=1, keepdim=True).expand_as(state)
                state = norm(state + block(torch.cat([state, context], dim=-1)))
            return self.head(state)

else:

    class MultiFidelityPointOperator:  # pragma: no cover - core-only install
        def __init__(self, **_: Any) -> None:
            raise RuntimeError("PyTorch is required; install TunnelGeoPT with the learn extra")


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("PyTorch is required; install TunnelGeoPT with the learn extra")
    return torch


def _seed_everything(seed: int) -> None:
    framework = _require_torch()
    random.seed(int(seed))
    np.random.seed(int(seed))
    framework.manual_seed(int(seed))
    if framework.cuda.is_available():
        framework.cuda.manual_seed_all(int(seed))
    framework.use_deterministic_algorithms(True, warn_only=True)


def make_model(
    model_config: Mapping[str, Any], *, seed: int, device: str
) -> MultiFidelityPointOperator:
    """Construct the shared model with deterministic initialization."""

    _seed_everything(seed)
    model = MultiFidelityPointOperator(
        input_width=int(model_config.get("point_input_width", 14)),
        hidden_width=int(model_config.get("hidden_width", 64)),
        global_context_blocks=int(model_config.get("global_context_blocks", 3)),
        output_width=int(model_config.get("output_width", 3)),
    )
    return model.to(device)


def _weighted_mse(prediction: Any, target: Any, weights: Any) -> Any:
    multiplier = torch.as_tensor([1.0, 1.0, 2.0], device=prediction.device)
    numerator = torch.sum(weights[..., None] * multiplier * (prediction - target) ** 2)
    denominator = torch.sum(weights) * 4.0
    return numerator / denominator


def train_with_dev_selection(
    model: Any,
    train_features: FloatArray,
    train_targets: FloatArray,
    train_weights: FloatArray,
    dev_features: FloatArray,
    dev_fine: FloatArray,
    dev_reconstruction_base: FloatArray,
    dev_weights: FloatArray,
    *,
    seed: int,
    device: str,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    min_delta: float,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> TrainingOutcome:
    """Train one method and select only with the dev fine-stress metric."""

    framework = _require_torch()
    if max_epochs <= 0 or patience <= 0 or batch_size <= 0:
        raise LearningContractError("training counts must be positive")
    optimizer = framework.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    best_state: dict[str, Any] | None = None
    best_epoch = -1
    best_dev = math.inf
    stale = 0
    history: list[Mapping[str, float]] = []
    for epoch in range(int(max_epochs)):
        model.train()
        order = np.random.default_rng(int(seed) * 100_000 + epoch).permutation(
            train_features.shape[0]
        )
        cumulative = 0.0
        seen = 0
        for start in range(0, order.size, batch_size):
            indices = order[start : start + batch_size]
            batch_x = framework.as_tensor(
                train_features[indices], dtype=framework.float32, device=device
            )
            batch_y = framework.as_tensor(
                train_targets[indices], dtype=framework.float32, device=device
            )
            batch_w = framework.as_tensor(
                train_weights[indices], dtype=framework.float32, device=device
            )
            optimizer.zero_grad(set_to_none=True)
            loss = _weighted_mse(model(batch_x), batch_y, batch_w)
            loss.backward()
            optimizer.step()
            cumulative += float(loss.detach().cpu()) * indices.size
            seen += indices.size
        model.eval()
        dev_raw: list[FloatArray] = []
        with framework.no_grad():
            for start in range(0, dev_features.shape[0], batch_size):
                values = framework.as_tensor(
                    dev_features[start : start + batch_size],
                    dtype=framework.float32,
                    device=device,
                )
                dev_raw.append(model(values).detach().cpu().numpy())
        dev_prediction = reconstruct_fine_prediction(
            np.concatenate(dev_raw), dev_reconstruction_base
        )
        dev_error = float(case_weighted_stress_error(dev_prediction, dev_fine, dev_weights).mean())
        train_loss = cumulative / seen
        record = {"epoch": float(epoch), "train_mse": train_loss, "dev_error": dev_error}
        history.append(record)
        if progress is not None:
            progress(record)
        if dev_error < best_dev - float(min_delta):
            best_dev = dev_error
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None or not math.isfinite(best_dev):
        raise LearningContractError("training produced no finite dev-selected checkpoint")
    model.load_state_dict(best_state, strict=True)
    return TrainingOutcome(
        state_dict=best_state,
        best_epoch=best_epoch,
        epochs_run=len(history),
        best_dev_error=best_dev,
        history=tuple(history),
    )


def save_checkpoint_atomic(
    outcome: TrainingOutcome,
    path: str | Path,
    *,
    method: str,
    fraction: float,
    seed: int,
    model_config: Mapping[str, Any],
    config_sha256: str,
    train_geometry_ids: Sequence[str],
) -> str:
    """Persist a CPU checkpoint and its exact experiment identity atomically."""

    framework = _require_torch()
    if method not in METHODS:
        raise LearningContractError("checkpoint method is unknown")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    payload = {
        "format_version": 1,
        "method": method,
        "fine_fraction": float(fraction),
        "seed": int(seed),
        "model_config": dict(model_config),
        "config_sha256": str(config_sha256),
        "train_geometry_ids": tuple(sorted(train_geometry_ids)),
        "best_epoch": int(outcome.best_epoch),
        "epochs_run": int(outcome.epochs_run),
        "best_dev_error": float(outcome.best_dev_error),
        "state_dict": dict(outcome.state_dict),
    }
    try:
        framework.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return hashlib.sha256(destination.read_bytes()).hexdigest()


def checkpoint_payload(path: str | Path) -> dict[str, Any]:
    """Load a CPU-only checkpoint envelope for audit or reconstruction."""

    framework = _require_torch()
    try:
        payload = framework.load(Path(path), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise LearningContractError(f"could not load checkpoint: {exc}") from exc
    required = {
        "format_version",
        "method",
        "fine_fraction",
        "seed",
        "model_config",
        "config_sha256",
        "train_geometry_ids",
        "best_epoch",
        "epochs_run",
        "best_dev_error",
        "state_dict",
    }
    if not isinstance(payload, dict) or set(payload) != required or payload["format_version"] != 1:
        raise LearningContractError("checkpoint envelope is invalid")
    if any(value.device.type != "cpu" for value in payload["state_dict"].values()):
        raise LearningContractError("checkpoint tensors must be on CPU")
    return payload


def predict(model: Any, features: FloatArray, *, batch_size: int, device: str) -> FloatArray:
    """Run deterministic batched model inference."""

    framework = _require_torch()
    model.eval()
    result = []
    with framework.no_grad():
        for start in range(0, features.shape[0], batch_size):
            batch = framework.as_tensor(
                features[start : start + batch_size], dtype=framework.float32, device=device
            )
            result.append(model(batch).detach().cpu().numpy())
    values = np.concatenate(result).astype(np.float64)
    if not np.isfinite(values).all():
        raise LearningContractError("model prediction is non-finite")
    return values


def load_model_from_checkpoint(path: str | Path, *, device: str) -> tuple[Any, dict[str, Any]]:
    """Reconstruct one model from a validated CPU checkpoint."""

    payload = checkpoint_payload(path)
    model = make_model(payload["model_config"], seed=int(payload["seed"]), device="cpu")
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, payload


def hierarchical_paired_bootstrap(
    candidate: ArrayLike,
    reference: ArrayLike,
    seeds: Sequence[int],
    geometry_group_ids: Sequence[str],
    section_families: Sequence[str],
    *,
    replicates: int,
    confidence: float,
    bootstrap_seed: int,
) -> dict[str, float]:
    """Paired bootstrap over training seeds, then geometries within section."""

    candidate_array = np.asarray(candidate, dtype=np.float64)
    reference_array = np.asarray(reference, dtype=np.float64)
    if candidate_array.shape != reference_array.shape or candidate_array.ndim != 2:
        raise LearningContractError("candidate/reference must align as [seed, geometry]")
    if candidate_array.shape != (len(seeds), len(geometry_group_ids)):
        raise LearningContractError("bootstrap metadata does not align with errors")
    if len(section_families) != len(geometry_group_ids):
        raise LearningContractError("bootstrap sections do not align with geometries")
    if np.any(reference_array <= 0.0) or not all(
        np.isfinite(value).all() for value in (candidate_array, reference_array)
    ):
        raise LearningContractError("bootstrap errors must be finite with positive reference")
    if replicates <= 0 or not 0.0 < confidence < 1.0:
        raise LearningContractError("invalid bootstrap settings")
    sections = np.asarray(section_families, dtype=object)
    section_indices = [np.flatnonzero(sections == value) for value in sorted(set(sections))]

    def aggregate(
        values: FloatArray, selected_seeds: IntArray, selected_geometry: IntArray
    ) -> float:
        section_means = []
        cursor = 0
        for indices in section_indices:
            draw = selected_geometry[cursor : cursor + len(indices)]
            cursor += len(indices)
            section_means.append(float(values[np.ix_(selected_seeds, draw)].mean()))
        return float(np.mean(section_means))

    original_seeds = np.arange(len(seeds), dtype=np.int64)
    original_geometry = np.concatenate(section_indices)
    center_candidate = aggregate(candidate_array, original_seeds, original_geometry)
    center_reference = aggregate(reference_array, original_seeds, original_geometry)
    rng = np.random.default_rng(int(bootstrap_seed))
    ratios = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled_seeds = rng.integers(0, len(seeds), size=len(seeds))
        sampled_geometry = np.concatenate(
            [
                indices[rng.integers(0, len(indices), size=len(indices))]
                for indices in section_indices
            ]
        )
        ratios[replicate] = aggregate(candidate_array, sampled_seeds, sampled_geometry) / aggregate(
            reference_array, sampled_seeds, sampled_geometry
        )
    alpha = 1.0 - confidence
    return {
        "center_ratio": center_candidate / center_reference,
        "lower": float(np.quantile(ratios, alpha / 2.0)),
        "upper": float(np.quantile(ratios, 1.0 - alpha / 2.0)),
        "one_sided_upper": float(np.quantile(ratios, confidence)),
        "replicates": float(replicates),
        "confidence": float(confidence),
    }


def write_json_atomic(path: str | Path, value: Mapping[str, Any]) -> str:
    """Write canonical JSON and return SHA-256 for durable run manifests."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, destination)
    return hashlib.sha256(destination.read_bytes()).hexdigest()


__all__ = [
    "METHODS",
    "LearningBatch",
    "LearningContractError",
    "MultiFidelityPointOperator",
    "TrainingOutcome",
    "case_weighted_stress_error",
    "checkpoint_payload",
    "hierarchical_paired_bootstrap",
    "load_model_from_checkpoint",
    "make_model",
    "method_arrays",
    "mismatched_coarse_indices",
    "nested_geometry_subsets",
    "predict",
    "reconstruct_fine_prediction",
    "save_checkpoint_atomic",
    "section_balanced_geometry_mean",
    "train_with_dev_selection",
    "write_json_atomic",
]
