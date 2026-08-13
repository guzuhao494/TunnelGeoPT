"""Physics-structured coarse-to-fine stress residual operator.

The module is deliberately limited to the v0.3/v0.4 two-dimensional elastic
task.  It does not model damage, fracture, rockburst, or field observations.

For the strict model, a geometry-only network produces a linear map ``A(g)``
and the stress correction is ``A(g) d``.  ``d`` contains the normalized
far-field tensor, the coarse stress, and an optional *linear* case mean.  No
bias, activation, normalization, or data-dependent attention is applied on the
dynamic action path.  Consequently the dimensional correction obeys physical
load superposition when the per-case stress normalization is undone.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .multifidelity_learning import (
    LearningContractError,
    TrainingOutcome,
    case_weighted_stress_error,
)

try:  # Keep the non-learning installation importable.
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - core-only installation
    torch = None
    nn = None

FloatArray = NDArray[np.floating]


@dataclass(frozen=True)
class StructuredResidualConfig:
    """Frozen architecture switches for the structured residual operator."""

    hidden_width: int = 64
    global_context_blocks: int = 3
    strict_load_linearity: bool = True
    local_tensor_frame: bool = True
    exact_zero_init_coarse_gate: bool = True
    dynamic_context: str = "pointwise_global_mean"
    attention_heads: int = 4
    attention_key_width: int = 16
    matrix_scale: float = 1.0

    def __post_init__(self) -> None:
        if int(self.hidden_width) <= 0 or int(self.global_context_blocks) < 0:
            raise LearningContractError("structured model widths must be positive")
        if self.dynamic_context not in {
            "pointwise",
            "pointwise_global_mean",
            "pointwise_geometry_attention",
        }:
            raise LearningContractError("unknown structured dynamic context")
        if int(self.attention_heads) <= 0 or int(self.attention_key_width) <= 0:
            raise LearningContractError("structured attention dimensions must be positive")
        if not math.isfinite(float(self.matrix_scale)) or float(self.matrix_scale) <= 0.0:
            raise LearningContractError("matrix_scale must be positive and finite")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> StructuredResidualConfig:
        """Construct from the explicit v0.4 architecture/model records."""

        return cls(
            hidden_width=int(values.get("hidden_width", 64)),
            global_context_blocks=int(values.get("global_context_blocks", 3)),
            strict_load_linearity=bool(values.get("strict_load_linearity", True)),
            local_tensor_frame=bool(values.get("local_tensor_frame", True)),
            exact_zero_init_coarse_gate=bool(values.get("exact_zero_init_coarse_gate", True)),
            dynamic_context=str(values.get("dynamic_context", "pointwise_global_mean")),
            attention_heads=int(values.get("attention_heads", 4)),
            attention_key_width=int(values.get("attention_key_width", 16)),
            matrix_scale=float(values.get("matrix_scale", 1.0)),
        )


def _as_finite_array(values: ArrayLike, label: str) -> FloatArray:
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
        raise LearningContractError(f"{label} must be a finite floating array")
    return array


def pack_structured_features(
    features14: ArrayLike,
    wall_normals_yz: ArrayLike,
    wall_offset_mask: ArrayLike,
) -> FloatArray:
    """Append a deterministic hybrid frame and wall mask to ``[... ,14]``.

    The formal generator stores the actual rock-outward wall normal only at
    wall-offset queries.  Those normals take precedence there.  Away from the
    wall-offset ring, ``-g_yz`` from the nearest-boundary vector defines a
    *nearest-distance frame*; it is not described as an exact physical wall
    normal.  The returned extra channels are ``[frame_y, frame_z, is_wall]``.
    """

    features = _as_finite_array(features14, "features14")
    supplied = _as_finite_array(wall_normals_yz, "wall_normals_yz")
    mask = np.asarray(wall_offset_mask)
    if features.ndim < 2 or features.shape[-1] != 14:
        raise LearningContractError("features14 must have shape [...,P,14]")
    if supplied.shape != (*features.shape[:-1], 2):
        raise LearningContractError("wall normals must align with features14")
    if mask.shape != features.shape[:-1] or mask.dtype != np.bool_:
        raise LearningContractError("wall_offset_mask must be boolean and align with points")

    nearest = -np.asarray(features[..., 5:7], dtype=np.float64)
    nearest_norm = np.linalg.norm(nearest, axis=-1, keepdims=True)
    if np.any(nearest_norm <= 1e-8):
        raise LearningContractError("nearest-distance frame contains a zero direction")
    nearest /= nearest_norm

    supplied64 = np.asarray(supplied, dtype=np.float64)
    supplied_norm = np.linalg.norm(supplied64, axis=-1, keepdims=True)
    if np.any(supplied_norm[mask] <= 1e-8):
        raise LearningContractError("wall-offset query is missing its physical normal")
    physical = supplied64 / np.maximum(supplied_norm, 1e-12)
    if np.any(np.sum(physical[mask] * nearest[mask], axis=-1) <= 0.0):
        raise LearningContractError("wall and nearest-distance frames have opposite orientation")

    frame = nearest
    frame[mask] = physical[mask]
    packed = np.concatenate(
        [
            np.asarray(features, dtype=np.float32),
            frame.astype(np.float32),
            mask[..., None].astype(np.float32),
        ],
        axis=-1,
    )
    if packed.shape != (*features.shape[:-1], 17) or not np.isfinite(packed).all():
        raise LearningContractError("structured feature packing failed")
    return packed


def stress_global_to_local(stress: ArrayLike, normals_yz: ArrayLike) -> FloatArray:
    """Rotate symmetric ``[yy,zz,yz]`` tensors into ``[nn,tt,nt]``."""

    values = _as_finite_array(stress, "stress")
    normals = _as_finite_array(normals_yz, "normals_yz")
    if values.shape[-1] != 3 or normals.shape != (*values.shape[:-1], 2):
        raise LearningContractError("stress and normal tensor shapes do not align")
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    if np.any(norm <= 1e-8):
        raise LearningContractError("local frame normal must be nonzero")
    normal = normals / norm
    tangent = np.stack([-normal[..., 1], normal[..., 0]], axis=-1)
    yy, zz, yz = (values[..., index] for index in range(3))
    ny, nz = normal[..., 0], normal[..., 1]
    ty, tz = tangent[..., 0], tangent[..., 1]
    return np.stack(
        [
            ny * ny * yy + nz * nz * zz + 2.0 * ny * nz * yz,
            ty * ty * yy + tz * tz * zz + 2.0 * ty * tz * yz,
            ny * ty * yy + nz * tz * zz + (ny * tz + nz * ty) * yz,
        ],
        axis=-1,
    )


def stress_local_to_global(stress: ArrayLike, normals_yz: ArrayLike) -> FloatArray:
    """Rotate symmetric ``[nn,tt,nt]`` tensors back to ``[yy,zz,yz]``."""

    values = _as_finite_array(stress, "stress")
    normals = _as_finite_array(normals_yz, "normals_yz")
    if values.shape[-1] != 3 or normals.shape != (*values.shape[:-1], 2):
        raise LearningContractError("stress and normal tensor shapes do not align")
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    if np.any(norm <= 1e-8):
        raise LearningContractError("local frame normal must be nonzero")
    normal = normals / norm
    tangent = np.stack([-normal[..., 1], normal[..., 0]], axis=-1)
    nn_value, tt_value, nt_value = (values[..., index] for index in range(3))
    ny, nz = normal[..., 0], normal[..., 1]
    ty, tz = tangent[..., 0], tangent[..., 1]
    return np.stack(
        [
            ny * ny * nn_value + ty * ty * tt_value + 2.0 * ny * ty * nt_value,
            nz * nz * nn_value + tz * tz * tt_value + 2.0 * nz * tz * nt_value,
            ny * nz * nn_value + ty * tz * tt_value + (ny * tz + nz * ty) * nt_value,
        ],
        axis=-1,
    )


def _torch_global_to_local(stress: Any, normal: Any) -> Any:
    tangent = torch.stack([-normal[..., 1], normal[..., 0]], dim=-1)
    yy, zz, yz = stress.unbind(dim=-1)
    ny, nz = normal.unbind(dim=-1)
    ty, tz = tangent.unbind(dim=-1)
    return torch.stack(
        [
            ny * ny * yy + nz * nz * zz + 2.0 * ny * nz * yz,
            ty * ty * yy + tz * tz * zz + 2.0 * ty * tz * yz,
            ny * ty * yy + nz * tz * zz + (ny * tz + nz * ty) * yz,
        ],
        dim=-1,
    )


def _torch_local_to_global(stress: Any, normal: Any) -> Any:
    tangent = torch.stack([-normal[..., 1], normal[..., 0]], dim=-1)
    nn_value, tt_value, nt_value = stress.unbind(dim=-1)
    ny, nz = normal.unbind(dim=-1)
    ty, tz = tangent.unbind(dim=-1)
    return torch.stack(
        [
            ny * ny * nn_value + ty * ty * tt_value + 2.0 * ny * ty * nt_value,
            nz * nz * nn_value + tz * tz * tt_value + 2.0 * nz * tz * nt_value,
            ny * nz * nn_value + ty * tz * tt_value + (ny * tz + nz * ty) * nt_value,
        ],
        dim=-1,
    )


if nn is not None:

    class StructuredLinearResidualOperator(nn.Module):
        """Geometry-conditioned stress correction with an auditable action path."""

        def __init__(self, config: StructuredResidualConfig) -> None:
            super().__init__()
            self.config = config
            context_copies = {
                "pointwise": 1,
                "pointwise_global_mean": 2,
                "pointwise_geometry_attention": 1 + int(config.attention_heads),
            }[config.dynamic_context]
            dynamic_width = 7 * context_copies
            self.dynamic_width = dynamic_width
            # Local-frame geometry is invariant to a proper global rotation:
            # radial/tangential coordinates, distance, and the wall-ring flag.
            static_width = 4 if config.local_tensor_frame else 6
            # The non-strict ablation mixes only the seven pointwise dynamic
            # channels into its nonlinear encoder.  Context expansion happens
            # later and is relevant only to the strict matrix action.
            projection_width = static_width if config.strict_load_linearity else static_width + 7
            width = int(config.hidden_width)
            self.input_projection = nn.Sequential(nn.Linear(projection_width, width), nn.GELU())
            self.blocks = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(2 * width, width),
                        nn.GELU(),
                        nn.Linear(width, width),
                        nn.GELU(),
                    )
                    for _ in range(int(config.global_context_blocks))
                ]
            )
            self.norms = nn.ModuleList(
                [nn.LayerNorm(width) for _ in range(int(config.global_context_blocks))]
            )
            if config.dynamic_context == "pointwise_geometry_attention":
                attention_width = int(config.attention_heads) * int(config.attention_key_width)
                self.attention_queries = nn.Linear(width, attention_width, bias=True)
                self.attention_keys = nn.Linear(width, attention_width, bias=True)
            else:
                self.attention_queries = None
                self.attention_keys = None
            output_width = 3 * dynamic_width if config.strict_load_linearity else 3
            self.head = nn.Linear(width, output_width)
            initial_gain = 0.0 if config.exact_zero_init_coarse_gate else 1.0
            self.correction_gain = nn.Parameter(torch.full((3,), initial_gain))

        def _static_and_point_dynamic(self, packed: Any) -> tuple[Any, Any, Any]:
            if packed.ndim != 3 or packed.shape[-1] != 17:
                raise LearningContractError("structured model expects [case,point,17]")
            position = packed[..., 1:3]
            distance = packed[..., 3:4]
            condition = packed[..., 7:11]
            coarse = packed[..., 11:14]
            normal = packed[..., 14:16]
            normal = normal / torch.linalg.vector_norm(normal, dim=-1, keepdim=True).clamp_min(1e-8)
            wall = packed[..., 16:17]
            tangent = torch.stack([-normal[..., 1], normal[..., 0]], dim=-1)
            radial = torch.sum(position * normal, dim=-1, keepdim=True)
            tangential = torch.sum(position * tangent, dim=-1, keepdim=True)
            if self.config.local_tensor_frame:
                static = torch.cat([radial, tangential, distance, wall], dim=-1)
                condition3 = _torch_global_to_local(condition[..., :3], normal)
                coarse3 = _torch_global_to_local(coarse, normal)
            else:
                static = torch.cat([position, distance, normal, wall], dim=-1)
                condition3 = condition[..., :3]
                coarse3 = coarse
            point_dynamic = torch.cat([condition3, condition[..., 3:4], coarse3], dim=-1)
            return static, point_dynamic, normal

        def _dynamic_context(self, state: Any, point_dynamic: Any) -> Any:
            if self.config.dynamic_context == "pointwise":
                return point_dynamic
            if self.config.dynamic_context == "pointwise_global_mean":
                mean_dynamic = point_dynamic.mean(dim=1, keepdim=True).expand_as(point_dynamic)
                return torch.cat([point_dynamic, mean_dynamic], dim=-1)
            heads = int(self.config.attention_heads)
            key_width = int(self.config.attention_key_width)
            queries = self.attention_queries(state).view(*state.shape[:2], heads, key_width)
            keys = self.attention_keys(state).view(*state.shape[:2], heads, key_width)
            logits = torch.einsum("bphk,bqhk->bhpq", queries, keys) / math.sqrt(key_width)
            # These weights depend on geometry state only.  Acting on the raw
            # dynamic tensor is therefore still a linear map of load/coarse.
            weights = torch.softmax(logits, dim=-1)
            attended = torch.einsum("bhpq,bqv->bphv", weights, point_dynamic)
            return torch.cat([point_dynamic, attended.flatten(start_dim=-2)], dim=-1)

        def forward(self, packed: Any) -> Any:
            static, point_dynamic, normal = self._static_and_point_dynamic(packed)
            # In the strict branch, only geometry enters the nonlinear encoder.
            encoder_input = (
                static
                if self.config.strict_load_linearity
                else torch.cat([static, point_dynamic], dim=-1)
            )
            state = self.input_projection(encoder_input)
            for block, norm in zip(self.blocks, self.norms, strict=True):
                context = state.mean(dim=1, keepdim=True).expand_as(state)
                state = norm(state + block(torch.cat([state, context], dim=-1)))
            if self.config.strict_load_linearity:
                dynamic = self._dynamic_context(state, point_dynamic)
                matrix = self.config.matrix_scale * torch.tanh(self.head(state))
                matrix = matrix.view(*state.shape[:2], 3, self.dynamic_width)
                residual = torch.einsum("bpij,bpj->bpi", matrix, dynamic)
            else:
                residual = self.config.matrix_scale * self.head(state)
            residual = torch.tanh(self.correction_gain)[None, None, :] * residual
            if self.config.local_tensor_frame:
                residual = _torch_local_to_global(residual, normal)
            return residual

else:

    class StructuredLinearResidualOperator:  # pragma: no cover - core-only install
        def __init__(self, _: StructuredResidualConfig) -> None:
            raise RuntimeError("PyTorch is required; install TunnelGeoPT with the learn extra")


def _seed_everything(seed: int) -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required; install TunnelGeoPT with the learn extra")
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(True, warn_only=True)


def make_structured_residual_model(
    config: StructuredResidualConfig | Mapping[str, Any],
    *,
    seed: int,
    device: str,
) -> StructuredLinearResidualOperator:
    """Create a deterministic structured model on ``device``."""

    values = (
        config
        if isinstance(config, StructuredResidualConfig)
        else StructuredResidualConfig.from_mapping(config)
    )
    _seed_everything(seed)
    return StructuredLinearResidualOperator(values).to(device)


def train_structured_with_dev_selection(
    model: Any,
    train_features: FloatArray,
    train_targets: FloatArray,
    train_weights: FloatArray,
    dev_features: FloatArray,
    dev_fine: FloatArray,
    dev_coarse: FloatArray,
    dev_metric_weights: FloatArray,
    *,
    seed: int,
    device: str,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    min_delta: float,
    loss_mode: str = "residual_relative_per_case",
) -> TrainingOutcome:
    """Train a structured residual and select by the formal fine-stress metric."""

    if torch is None:
        raise RuntimeError("PyTorch is required; install TunnelGeoPT with the learn extra")
    arrays = tuple(
        np.asarray(value)
        for value in (
            train_features,
            train_targets,
            train_weights,
            dev_features,
            dev_fine,
            dev_coarse,
            dev_metric_weights,
        )
    )
    if arrays[0].ndim != 3 or arrays[0].shape[-1] != 17:
        raise LearningContractError("structured training features must be [C,P,17]")
    if arrays[1].shape != (*arrays[0].shape[:2], 3):
        raise LearningContractError("structured residual target must align with features")
    if arrays[2].shape != arrays[0].shape[:2]:
        raise LearningContractError("structured training weights must align with features")
    if arrays[3].ndim != 3 or arrays[3].shape[-1] != 17:
        raise LearningContractError("structured dev features must be [C,P,17]")
    if arrays[4].shape != (*arrays[3].shape[:2], 3) or arrays[5].shape != arrays[4].shape:
        raise LearningContractError("structured dev stress arrays must align")
    if arrays[6].shape != arrays[3].shape[:2]:
        raise LearningContractError("structured dev metric weights must align")
    if not all(np.isfinite(value).all() for value in arrays):
        raise LearningContractError("structured training arrays must be finite")
    if loss_mode not in {"residual_relative_per_case", "weighted_mse"}:
        raise LearningContractError("unknown structured loss mode")
    if min(batch_size, max_epochs, patience) <= 0:
        raise LearningContractError("structured training counts must be positive")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    multiplier = torch.as_tensor([1.0, 1.0, 2.0], dtype=torch.float32, device=device)
    best_state: dict[str, Any] | None = None
    best_epoch = -1
    best_dev = math.inf
    stale = 0
    history: list[Mapping[str, float]] = []
    for epoch in range(int(max_epochs)):
        model.train()
        order = np.random.default_rng(int(seed) * 100_000 + epoch).permutation(arrays[0].shape[0])
        cumulative = 0.0
        seen = 0
        for start in range(0, order.size, int(batch_size)):
            indices = order[start : start + int(batch_size)]
            features = torch.as_tensor(arrays[0][indices], dtype=torch.float32, device=device)
            target = torch.as_tensor(arrays[1][indices], dtype=torch.float32, device=device)
            weights = torch.as_tensor(arrays[2][indices], dtype=torch.float32, device=device)
            prediction = model(features)
            numerator = torch.sum(
                weights[..., None] * multiplier * (prediction - target) ** 2,
                dim=(1, 2),
            )
            if loss_mode == "residual_relative_per_case":
                denominator = torch.sum(
                    weights[..., None] * multiplier * target**2, dim=(1, 2)
                ).clamp_min(1e-8)
                loss = torch.mean(numerator / denominator)
            else:
                loss = torch.sum(numerator) / (torch.sum(weights) * 4.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            cumulative += float(loss.detach().cpu()) * len(indices)
            seen += len(indices)

        model.eval()
        raw: list[FloatArray] = []
        with torch.no_grad():
            for start in range(0, arrays[3].shape[0], int(batch_size)):
                features = torch.as_tensor(
                    arrays[3][start : start + int(batch_size)],
                    dtype=torch.float32,
                    device=device,
                )
                raw.append(model(features).detach().cpu().numpy())
        fine_prediction = np.asarray(arrays[5], dtype=np.float64) + np.concatenate(raw)
        dev_error = float(
            np.mean(case_weighted_stress_error(fine_prediction, arrays[4], arrays[6]))
        )
        train_loss = cumulative / max(seen, 1)
        history.append({"epoch": float(epoch), "train_loss": train_loss, "dev_error": dev_error})
        if dev_error < best_dev - float(min_delta):
            best_dev = dev_error
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= int(patience):
            break
    if best_state is None or not math.isfinite(best_dev):
        raise LearningContractError("structured training produced no finite checkpoint")
    model.load_state_dict(best_state, strict=True)
    return TrainingOutcome(
        state_dict=best_state,
        best_epoch=best_epoch,
        epochs_run=len(history),
        best_dev_error=best_dev,
        history=tuple(history),
    )


__all__ = [
    "StructuredLinearResidualOperator",
    "StructuredResidualConfig",
    "make_structured_residual_model",
    "pack_structured_features",
    "stress_global_to_local",
    "stress_local_to_global",
    "train_structured_with_dev_selection",
]
