from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tunnelgeopt.multifidelity_learning import LearningContractError
from tunnelgeopt.structured_residual import (
    StructuredResidualConfig,
    make_structured_residual_model,
    pack_structured_features,
    stress_global_to_local,
    stress_local_to_global,
    train_structured_with_dev_selection,
)


def _tensor_rotate(stress: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    tensor = np.empty((*stress.shape[:-1], 2, 2), dtype=np.float64)
    tensor[..., 0, 0] = stress[..., 0]
    tensor[..., 1, 1] = stress[..., 1]
    tensor[..., 0, 1] = stress[..., 2]
    tensor[..., 1, 0] = stress[..., 2]
    rotated = np.einsum("ij,...jk,lk->...il", rotation, tensor, rotation)
    return np.stack([rotated[..., 0, 0], rotated[..., 1, 1], rotated[..., 0, 1]], -1)


def _scale(stress: np.ndarray) -> float:
    return math.sqrt(float(stress[0] ** 2 + stress[1] ** 2 + 2.0 * stress[2] ** 2))


def _geometry_features(points: np.ndarray, normal: np.ndarray) -> np.ndarray:
    count = len(points)
    result = np.zeros((count, 14), dtype=np.float32)
    result[:, 1:3] = points
    result[:, 3] = np.linspace(0.1, 1.0, count)
    result[:, 5:7] = -normal
    return result


def _packed_case(
    geometry: np.ndarray,
    normal: np.ndarray,
    condition_unscaled: np.ndarray,
    coarse_unscaled: np.ndarray,
    wall_normal: np.ndarray,
    wall_mask: np.ndarray,
) -> tuple[np.ndarray, float]:
    scale = _scale(condition_unscaled[:3])
    features = geometry.copy()
    features[:, 7:11] = condition_unscaled / scale
    features[:, 11:14] = coarse_unscaled / scale
    return (
        pack_structured_features(
            features[None], wall_normal[None].astype(np.float32), wall_mask[None]
        ),
        scale,
    )


def test_tensor_local_roundtrip_preserves_engineering_shear() -> None:
    rng = np.random.default_rng(4)
    stress = rng.normal(size=(3, 11, 3))
    angle = rng.uniform(-np.pi, np.pi, size=(3, 11))
    normal = np.stack([np.cos(angle), np.sin(angle)], axis=-1)
    local = stress_global_to_local(stress, normal)
    restored = stress_local_to_global(local, normal)
    np.testing.assert_allclose(restored, stress, rtol=1e-12, atol=1e-12)


def test_pack_uses_stored_physical_wall_normal_and_distance_frame_elsewhere() -> None:
    points = np.asarray([[1.0, 0.2], [0.2, 1.0], [-0.8, 0.3]])
    distance_normal = points / np.linalg.norm(points, axis=1, keepdims=True)
    features = _geometry_features(points, distance_normal)
    wall_mask = np.asarray([True, False, True])
    physical = np.zeros((3, 2), dtype=np.float32)
    physical[0] = np.asarray([0.8, 0.6])
    physical[2] = np.asarray([-0.6, 0.8])
    packed = pack_structured_features(features[None], physical[None], wall_mask[None])
    np.testing.assert_allclose(packed[0, wall_mask, 14:16], physical[wall_mask], atol=1e-7)
    np.testing.assert_allclose(packed[0, ~wall_mask, 14:16], distance_normal[~wall_mask], atol=1e-7)
    np.testing.assert_array_equal(packed[0, :, 16].astype(bool), wall_mask)


def test_pack_rejects_missing_or_flipped_wall_normal() -> None:
    points = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    normal = points.copy()
    features = _geometry_features(points, normal)
    mask = np.asarray([[True, False]])
    with pytest.raises(LearningContractError, match="missing"):
        pack_structured_features(features[None], np.zeros((1, 2, 2)), mask)
    supplied = np.zeros((1, 2, 2), dtype=np.float32)
    supplied[0, 0] = -normal[0]
    with pytest.raises(LearningContractError, match="opposite"):
        pack_structured_features(features[None], supplied, mask)


def test_exact_zero_initialization_reconstructs_coarse() -> None:
    rng = np.random.default_rng(8)
    points = rng.normal(size=(9, 2))
    normal = points / np.linalg.norm(points, axis=1, keepdims=True)
    features = _geometry_features(points, normal)
    features[:, 7:14] = rng.normal(size=(9, 7))
    packed = pack_structured_features(
        features[None], np.zeros((1, 9, 2), dtype=np.float32), np.zeros((1, 9), bool)
    )
    model = make_structured_residual_model(
        StructuredResidualConfig(exact_zero_init_coarse_gate=True), seed=17, device="cpu"
    )
    output = model(torch.as_tensor(packed)).detach().numpy()
    np.testing.assert_array_equal(output, np.zeros_like(output))


@pytest.mark.parametrize(
    "dynamic_context", ["pointwise", "pointwise_global_mean", "pointwise_geometry_attention"]
)
def test_strict_operator_obeys_denormalized_physical_superposition(
    dynamic_context: str,
) -> None:
    rng = np.random.default_rng(12)
    points = rng.normal(size=(13, 2))
    normal = points / np.linalg.norm(points, axis=1, keepdims=True)
    geometry = _geometry_features(points, normal)
    mask = np.zeros(13, dtype=bool)
    supplied = np.zeros((13, 2), dtype=np.float32)
    far1 = np.asarray([-4.0, -1.2, 0.7, -1.3])
    far2 = np.asarray([-0.8, -3.5, -0.4, -1.1])
    coarse1 = rng.normal(size=(13, 3)) * 2.0
    coarse2 = rng.normal(size=(13, 3)) * 1.5
    packed1, scale1 = _packed_case(geometry, normal, far1, coarse1, supplied, mask)
    packed2, scale2 = _packed_case(geometry, normal, far2, coarse2, supplied, mask)
    packed12, scale12 = _packed_case(
        geometry, normal, far1 + far2, coarse1 + coarse2, supplied, mask
    )
    model = make_structured_residual_model(
        StructuredResidualConfig(
            strict_load_linearity=True,
            dynamic_context=dynamic_context,
            attention_heads=2,
            attention_key_width=4,
        ),
        seed=19,
        device="cpu",
    )
    with torch.no_grad():
        model.correction_gain.fill_(0.4)
        out1 = model(torch.as_tensor(packed1)).numpy() * scale1
        out2 = model(torch.as_tensor(packed2)).numpy() * scale2
        out12 = model(torch.as_tensor(packed12)).numpy() * scale12
    np.testing.assert_allclose(out12, out1 + out2, rtol=2e-5, atol=2e-5)


def test_each_single_factor_ablation_constructs_and_runs() -> None:
    rng = np.random.default_rng(21)
    points = rng.normal(size=(2, 7, 2))
    directions = points / np.linalg.norm(points, axis=-1, keepdims=True)
    features = rng.normal(size=(2, 7, 14)).astype(np.float32)
    features[..., 1:3] = points
    features[..., 5:7] = -directions
    packed = pack_structured_features(
        features, np.zeros((2, 7, 2), dtype=np.float32), np.zeros((2, 7), bool)
    )
    configurations = (
        StructuredResidualConfig(strict_load_linearity=False),
        StructuredResidualConfig(local_tensor_frame=False),
        StructuredResidualConfig(exact_zero_init_coarse_gate=False),
    )
    for index, config in enumerate(configurations):
        model = make_structured_residual_model(config, seed=40 + index, device="cpu")
        output = model(torch.as_tensor(packed))
        assert output.shape == (2, 7, 3)
        assert torch.isfinite(output).all()


def test_local_model_is_equivariant_for_rotated_non_circular_queries() -> None:
    points = np.asarray([[1.8, 0.1], [0.9, 0.7], [-0.4, 1.1], [-1.5, 0.2], [-0.6, -0.8]])
    normal = np.asarray([[0.98, 0.2], [0.7, 0.71], [-0.3, 0.95], [-0.99, 0.1], [-0.5, -0.86]])
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    geometry = _geometry_features(points, normal)
    wall_mask = np.asarray([True, False, True, False, True])
    supplied = np.zeros((5, 2), dtype=np.float32)
    supplied[wall_mask] = normal[wall_mask]
    condition = np.asarray([-3.2, -1.1, 0.5, -1.075])
    coarse = np.asarray(
        [
            [-2.1, -1.0, 0.2],
            [-2.4, -0.9, 0.3],
            [-1.5, -1.3, -0.1],
            [-2.0, -0.8, 0.4],
            [-1.7, -1.4, -0.2],
        ]
    )
    packed, _ = _packed_case(geometry, normal, condition, coarse, supplied, wall_mask)

    angle = 0.63
    rotation = np.asarray([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    rotated_points = points @ rotation.T
    rotated_normal = normal @ rotation.T
    rotated_geometry = _geometry_features(rotated_points, rotated_normal)
    rotated_supplied = supplied @ rotation.T
    rotated_condition3 = _tensor_rotate(condition[None, :3], rotation)[0]
    rotated_condition = np.r_[rotated_condition3, condition[3]]
    rotated_coarse = _tensor_rotate(coarse, rotation)
    rotated_packed, _ = _packed_case(
        rotated_geometry,
        rotated_normal,
        rotated_condition,
        rotated_coarse,
        rotated_supplied,
        wall_mask,
    )
    model = make_structured_residual_model(
        StructuredResidualConfig(local_tensor_frame=True), seed=23, device="cpu"
    )
    with torch.no_grad():
        model.correction_gain.fill_(0.35)
        original = model(torch.as_tensor(packed)).numpy()[0]
        rotated = model(torch.as_tensor(rotated_packed)).numpy()[0]
    np.testing.assert_allclose(rotated, _tensor_rotate(original, rotation), rtol=2e-5, atol=2e-5)


def test_structured_training_smoke_returns_cpu_checkpoint() -> None:
    rng = np.random.default_rng(31)
    features = rng.normal(size=(6, 8, 14)).astype(np.float32)
    direction = rng.normal(size=(6, 8, 2))
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
    features[..., 5:7] = -direction
    normals = np.zeros((6, 8, 2), dtype=np.float32)
    mask = np.zeros((6, 8), dtype=bool)
    packed = pack_structured_features(features, normals, mask)
    coarse = features[..., 11:14].copy()
    residual = (0.01 * features[..., 7:10]).astype(np.float32)
    fine = coarse + residual
    weights = np.full((6, 8), 1.0 / 8.0, dtype=np.float32)
    model = make_structured_residual_model(
        StructuredResidualConfig(hidden_width=8, global_context_blocks=1),
        seed=37,
        device="cpu",
    )
    outcome = train_structured_with_dev_selection(
        model,
        packed[:4],
        residual[:4],
        weights[:4],
        packed[4:],
        fine[4:],
        coarse[4:],
        weights[4:],
        seed=37,
        device="cpu",
        learning_rate=1e-3,
        weight_decay=1e-4,
        batch_size=2,
        max_epochs=3,
        patience=2,
        min_delta=1e-8,
    )
    assert outcome.epochs_run >= 1
    assert math.isfinite(outcome.best_dev_error)
    assert all(value.device.type == "cpu" for value in outcome.state_dict.values())
