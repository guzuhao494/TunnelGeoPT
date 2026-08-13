from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import numpy as np
import pytest

from tunnelgeopt.fracture_loading import (
    WALL_ZONE_IDS,
    FractureLoadError,
    compile_phase1_load_schedule,
)
from tunnelgeopt.fracture_validation import load_fracture_phase1_config
from tunnelgeopt.geometry import make_parametric_tunnel_boundary


def _polygon_mesh(nodes_yz: np.ndarray, wall_ids: np.ndarray | None = None) -> SimpleNamespace:
    nodes = np.asarray(nodes_yz, dtype=np.float64)
    facets = np.vstack(
        [np.arange(nodes.shape[0], dtype=np.int64), np.roll(np.arange(nodes.shape[0]), -1)]
    )
    ids = np.arange(nodes.shape[0], dtype=np.int64) if wall_ids is None else wall_ids
    return SimpleNamespace(
        nodes=nodes,
        mesh=SimpleNamespace(facets=facets),
        boundary_facets={"wall": np.asarray(ids, dtype=np.int64)},
    )


def _circular_mesh(*, angular_step_deg: float = 0.25) -> SimpleNamespace:
    count = round(360.0 / angular_step_deg)
    node_angle_deg = np.arange(count, dtype=np.float64) * angular_step_deg
    node_angle_deg -= 0.5 * angular_step_deg
    angle = np.radians(node_angle_deg)
    return _polygon_mesh(np.column_stack([np.cos(angle), np.sin(angle)]))


def _state(path_id: str, s: float, *, ucs_scale: float = 1.0):
    schedule = compile_phase1_load_schedule(
        load_fracture_phase1_config(), path_id, ucs_scale, _circular_mesh()
    )
    return schedule.state_at(s)


def _extreme_facet_indices(schedule) -> dict[str, int]:
    points = schedule.wall_facet_midpoints_yz
    center = schedule.wall_perimeter_centroid_yz
    candidates: dict[str, np.ndarray] = {
        "crown": np.flatnonzero(points[:, 0] >= points[:, 0].max() - 1e-10),
        "right_sidewall": np.flatnonzero(points[:, 1] >= points[:, 1].max() - 1e-10),
        "invert": np.flatnonzero(points[:, 0] <= points[:, 0].min() + 1e-10),
        "left_sidewall": np.flatnonzero(points[:, 1] <= points[:, 1].min() + 1e-10),
    }
    return {
        "crown": int(
            candidates["crown"][np.argmin(abs(points[candidates["crown"], 1] - center[1]))]
        ),
        "right_sidewall": int(
            candidates["right_sidewall"][
                np.argmin(abs(points[candidates["right_sidewall"], 0] - center[0]))
            ]
        ),
        "invert": int(
            candidates["invert"][np.argmin(abs(points[candidates["invert"], 1] - center[1]))]
        ),
        "left_sidewall": int(
            candidates["left_sidewall"][
                np.argmin(abs(points[candidates["left_sidewall"], 0] - center[0]))
            ]
        ),
    }


def test_yz_axes_principal_angle_and_tension_positive_sign_are_explicit() -> None:
    angle_zero = _state("p1", 0.375, ucs_scale=2.0)
    assert angle_zero.farfield_stress_tension_positive_yz == pytest.approx(
        np.asarray([[-0.9, 0.0], [0.0, -0.63]])
    )

    rotated = _state("p3", 0.75, ucs_scale=2.0)
    angle = np.radians(30.0)
    major_yz = np.asarray([np.cos(angle), np.sin(angle)])
    minor_yz = np.asarray([-np.sin(angle), np.cos(angle)])
    expected = -(1.1 * np.outer(major_yz, major_yz) + 0.605 * np.outer(minor_yz, minor_yz))
    assert rotated.principal_angle_deg == pytest.approx(30.0)
    assert rotated.farfield_stress_tension_positive_yz == pytest.approx(expected)
    assert rotated.farfield_stress_tension_positive_yz[0, 1] < 0.0
    assert np.linalg.eigvalsh(rotated.farfield_stress_tension_positive_yz) == pytest.approx(
        [-1.1, -0.605]
    )


def test_p2_interpolates_stored_controls_before_tensor_conversion_at_0375() -> None:
    state = _state("p2", 0.375)
    assert state.sigma1_over_UCS == pytest.approx(0.60)
    assert state.sigma3_over_sigma1 == pytest.approx(0.675)
    assert state.principal_angle_deg == pytest.approx(0.0)
    assert state.wall_zone_release == pytest.approx([0.275] * 4)
    assert state.farfield_stress_tension_positive_yz == pytest.approx(
        np.asarray([[-0.60, 0.0], [0.0, -0.405]])
    )


def test_p3_supports_arbitrary_non_output_parameter() -> None:
    state = _state("p3", 0.613)
    interpolation_fraction = (0.613 - 0.5) / 0.25
    assert state.sigma1_over_UCS == pytest.approx(0.55)
    assert state.sigma3_over_sigma1 == pytest.approx(0.55)
    assert state.principal_angle_deg == pytest.approx(30.0 * interpolation_fraction)
    assert state.wall_release == pytest.approx(
        [0.45 + interpolation_fraction * (0.70 - 0.45)] * state.wall_facet_ids.size
    )


def test_p4_zone_controls_and_extreme_facet_values_at_0375() -> None:
    schedule = compile_phase1_load_schedule(
        load_fracture_phase1_config(), "p4", 1.0, _circular_mesh()
    )
    state = schedule.state_at(0.375)
    assert state.wall_zone_ids == WALL_ZONE_IDS
    assert state.wall_zone_release == pytest.approx([0.875, 0.25, 0.0, 0.25])
    extremes = _extreme_facet_indices(schedule)
    for zone_index, zone in enumerate(WALL_ZONE_IDS):
        facet_index = extremes[zone]
        expected_weights = np.zeros(4)
        expected_weights[zone_index] = 1.0
        assert state.wall_zone_weights[facet_index] == pytest.approx(expected_weights)
        assert state.wall_release[facet_index] == pytest.approx(state.wall_zone_release[zone_index])


@pytest.mark.parametrize("shape", ["circle", "horseshoe", "straight_wall_arch"])
def test_actual_geometry_extremes_use_vertical_y_and_horizontal_z(shape: str) -> None:
    geometry = make_parametric_tunnel_boundary(shape, n_points=256)
    schedule = compile_phase1_load_schedule(
        load_fracture_phase1_config(), "p4", 1.0, _polygon_mesh(geometry.boundary_yz)
    )
    extremes = _extreme_facet_indices(schedule)
    for zone_index, zone in enumerate(WALL_ZONE_IDS):
        weights = schedule.wall_zone_weights[extremes[zone]]
        assert weights[zone_index] == pytest.approx(1.0)
        assert np.count_nonzero(weights > 0.0) == 1


def test_p4_transition_is_exact_five_degree_continuous_convex_partition() -> None:
    schedule = compile_phase1_load_schedule(
        load_fracture_phase1_config(), "p4", 1.0, _circular_mesh()
    )
    relative = schedule.wall_facet_midpoints_yz - schedule.wall_perimeter_centroid_yz
    theta = np.mod(np.degrees(np.arctan2(relative[:, 1], relative[:, 0])), 360.0)
    for angle, expected in {
        42.5: [1.0, 0.0, 0.0, 0.0],
        43.75: [0.75, 0.25, 0.0, 0.0],
        45.0: [0.5, 0.5, 0.0, 0.0],
        46.25: [0.25, 0.75, 0.0, 0.0],
        47.5: [0.0, 1.0, 0.0, 0.0],
    }.items():
        facet_index = int(np.argmin(abs(theta - angle)))
        assert theta[facet_index] == pytest.approx(angle, abs=1e-12)
        assert schedule.wall_zone_weights[facet_index] == pytest.approx(expected, abs=1e-13)
    assert schedule.wall_zone_weights.sum(axis=1) == pytest.approx(
        np.ones(schedule.wall_facet_ids.size), abs=1e-14
    )
    assert np.all(schedule.wall_zone_weights >= 0.0)
    assert np.all(schedule.wall_zone_weights <= 1.0)


def test_facet_ids_remain_aligned_under_marker_order_and_edge_orientation() -> None:
    config = load_fracture_phase1_config()
    original_mesh = _circular_mesh(angular_step_deg=2.5)
    original = compile_phase1_load_schedule(config, "p4", 1.0, original_mesh)
    permutation = np.arange(original.wall_facet_ids.size - 1, -1, -1, dtype=np.int64)
    reordered_mesh = SimpleNamespace(
        nodes=original_mesh.nodes.copy(),
        mesh=SimpleNamespace(facets=original_mesh.mesh.facets[::-1].copy()),
        boundary_facets={"wall": permutation},
    )
    reordered = compile_phase1_load_schedule(config, "p4", 1.0, reordered_mesh)
    assert np.array_equal(reordered.wall_facet_ids, permutation)
    assert reordered.wall_perimeter_centroid_yz == pytest.approx(
        original.wall_perimeter_centroid_yz
    )
    original_row_by_id = {
        int(facet_id): index for index, facet_id in enumerate(original.wall_facet_ids)
    }
    original_state = original.state_at(0.375)
    reordered_state = reordered.state_at(0.375)
    original_release_by_facet_id = dict(
        zip(original_state.wall_facet_ids.tolist(), original_state.wall_release.tolist())
    )
    reordered_release_by_facet_id = dict(
        zip(reordered_state.wall_facet_ids.tolist(), reordered_state.wall_release.tolist())
    )
    assert reordered_release_by_facet_id == pytest.approx(original_release_by_facet_id)
    for reordered_row, facet_id in enumerate(reordered.wall_facet_ids):
        original_row = original_row_by_id[int(facet_id)]
        assert reordered.wall_zone_weights[reordered_row] == pytest.approx(
            original.wall_zone_weights[original_row]
        )
        assert reordered_state.wall_release[reordered_row] == pytest.approx(
            original_state.wall_release[original_row]
        )


def test_public_state_rejects_stress_inconsistent_with_principal_controls() -> None:
    state = _state("p3", 0.613, ucs_scale=2.0)
    inconsistent_stress = state.farfield_stress_tension_positive_yz.copy()
    inconsistent_stress[0, 0] += 1e-6
    with pytest.raises(FractureLoadError, match="inconsistent with ucs_scale"):
        replace(state, farfield_stress_tension_positive_yz=inconsistent_stress)


@pytest.mark.parametrize("path_id", ["p1", "p2", "p3", "p4"])
def test_every_facet_release_is_monotone_and_s1_is_fully_released(path_id: str) -> None:
    schedule = compile_phase1_load_schedule(
        load_fracture_phase1_config(), path_id, 1.0, _circular_mesh(angular_step_deg=2.5)
    )
    release = np.stack([schedule.state_at(index / 40.0).wall_release for index in range(41)])
    assert np.all(np.diff(release, axis=0) >= -1e-14)
    assert release[-1] == pytest.approx(np.ones(release.shape[1]))


def test_state_arrays_and_ids_are_immutable_and_schedule_requires_positive_finite_ucs() -> None:
    schedule = compile_phase1_load_schedule(
        load_fracture_phase1_config(), "p4", 1.0, _circular_mesh(angular_step_deg=2.5)
    )
    state = schedule.state_at(0.375)
    with pytest.raises(ValueError, match="read-only"):
        state.wall_release[0] = 0.0
    with pytest.raises(ValueError, match="read-only"):
        state.wall_facet_ids[0] = 999
    with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
        state.wall_release.setflags(write=True)
    with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
        state.wall_facet_ids.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        state.path_id = "p1"  # type: ignore[misc]

    for invalid in (0.0, -1.0, np.inf, np.nan, True):
        with pytest.raises(FractureLoadError, match="ucs_scale"):
            compile_phase1_load_schedule(
                load_fracture_phase1_config(), "p1", invalid, _circular_mesh()
            )
    for invalid_s in (-0.01, 1.01, np.inf, np.nan, True):
        with pytest.raises(FractureLoadError, match="s"):
            schedule.state_at(invalid_s)
