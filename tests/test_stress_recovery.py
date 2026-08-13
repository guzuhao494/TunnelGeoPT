import numpy as np
import pytest

from tunnelgeopt.stress_recovery import (
    StressRecoveryError,
    preserve_baseline_traction_with_tangential_correction,
    recover_stress_at_queries,
)


def _grid_mesh(size: int = 5) -> tuple[np.ndarray, np.ndarray]:
    nodes = np.asarray([(x, y) for y in range(size) for x in range(size)], dtype=np.float64)
    elements: list[tuple[int, int, int]] = []
    for y in range(size - 1):
        for x in range(size - 1):
            lower_left = y * size + x
            lower_right = lower_left + 1
            upper_left = lower_left + size
            upper_right = upper_left + 1
            elements.extend(
                [
                    (lower_left, lower_right, upper_right),
                    (lower_left, upper_right, upper_left),
                ]
            )
    return nodes, np.asarray(elements, dtype=np.int64)


def _affine_stress(points: np.ndarray) -> np.ndarray:
    intercept = np.asarray([2.0, -3.0, 0.75])
    gradient_y = np.asarray([0.4, -0.7, 1.2])
    gradient_z = np.asarray([-1.1, 0.25, 0.6])
    return intercept + points[:, :1] * gradient_y + points[:, 1:] * gradient_z


def test_constant_stress_is_reproduced_exactly_with_or_without_element_ids() -> None:
    nodes, elements = _grid_mesh()
    constant = np.asarray([7.0, -2.0, 0.5])
    stress = np.repeat(constant[None, :], elements.shape[0], axis=0)
    points = np.asarray([[1.8, 1.4], [2.2, 2.6], [0.1, 3.7]])

    automatic = recover_stress_at_queries(nodes, elements, stress, points)
    explicit = recover_stress_at_queries(
        nodes,
        elements,
        stress,
        points,
        np.asarray([10, 21, 25], dtype=np.int64),
    )

    np.testing.assert_allclose(
        automatic,
        np.repeat(constant[None, :], points.shape[0], axis=0),
        rtol=0.0,
        atol=2e-14,
    )
    np.testing.assert_allclose(explicit, automatic, rtol=0.0, atol=2e-14)


def test_affine_field_is_reproduced_inside_a_triangle_with_full_rank_node_patches() -> None:
    nodes, elements = _grid_mesh()
    centroids = nodes[elements].mean(axis=1)
    element_stress = _affine_stress(centroids)
    # Element 10 has vertices (1,1), (2,1), (2,2), all strictly interior.
    points = np.asarray([[1.8, 1.4], [1.25, 1.1], [1.5, 1.5]])
    ids = np.full(points.shape[0], 10, dtype=np.int64)

    recovered = recover_stress_at_queries(nodes, elements, element_stress, points, ids)

    np.testing.assert_allclose(recovered, _affine_stress(points), rtol=2e-13, atol=2e-13)


def test_recovery_is_linear_in_element_stress() -> None:
    nodes, elements = _grid_mesh()
    rng = np.random.default_rng(907)
    first = rng.normal(size=(elements.shape[0], 3))
    second = rng.normal(size=(elements.shape[0], 3))
    points = np.asarray([[1.8, 1.4], [2.2, 2.6], [0.1, 3.7]])

    recovered_sum = recover_stress_at_queries(nodes, elements, first + second, points)
    separate_sum = recover_stress_at_queries(nodes, elements, first, points) + (
        recover_stress_at_queries(nodes, elements, second, points)
    )
    scaled = recover_stress_at_queries(nodes, elements, -3.25 * first, points)

    np.testing.assert_allclose(recovered_sum, separate_sum, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(
        scaled,
        -3.25 * recover_stress_at_queries(nodes, elements, first, points),
        rtol=2e-13,
        atol=2e-13,
    )


def test_rank_deficient_single_triangle_falls_back_to_constant_average() -> None:
    nodes = np.asarray([[0.0, 0.0], [2.0, 0.0], [0.0, 1.0]])
    elements = np.asarray([[0, 1, 2]], dtype=np.int64)
    stress = np.asarray([[4.0, -6.0, 1.25]])
    points = np.asarray([[0.1, 0.1], [0.8, 0.05], [0.2, 0.6]])

    recovered = recover_stress_at_queries(nodes, elements, stress, points)

    np.testing.assert_allclose(
        recovered,
        np.repeat(stress, points.shape[0], axis=0),
        rtol=0.0,
        atol=1e-14,
    )


def _stress_tensor(stress: np.ndarray) -> np.ndarray:
    tensor = np.empty((stress.shape[0], 2, 2), dtype=np.float64)
    tensor[:, 0, 0] = stress[:, 0]
    tensor[:, 1, 1] = stress[:, 1]
    tensor[:, 0, 1] = stress[:, 2]
    tensor[:, 1, 0] = stress[:, 2]
    return tensor


def test_tangential_correction_preserves_baseline_traction_and_recovered_tt() -> None:
    baseline = np.asarray([[2.0, -1.0, 0.4], [0.5, 3.0, -0.7]])
    recovered = np.asarray([[-4.0, 5.0, 1.2], [2.0, -3.0, 0.9]])
    normals = np.asarray([[1.0, 0.0], [3.0 / 5.0, 4.0 / 5.0]])
    tangents = np.column_stack([-normals[:, 1], normals[:, 0]])

    output = preserve_baseline_traction_with_tangential_correction(baseline, recovered, normals)
    baseline_tensor = _stress_tensor(baseline)
    recovered_tensor = _stress_tensor(recovered)
    output_tensor = _stress_tensor(output)

    np.testing.assert_allclose(
        np.einsum("pij,pj->pi", output_tensor, normals),
        np.einsum("pij,pj->pi", baseline_tensor, normals),
        rtol=0.0,
        atol=3e-15,
    )
    np.testing.assert_allclose(
        np.einsum("pi,pij,pj->p", tangents, output_tensor, tangents),
        np.einsum("pi,pij,pj->p", tangents, recovered_tensor, tangents),
        rtol=0.0,
        atol=3e-15,
    )


def test_tangential_correction_is_linear_in_both_stress_arguments() -> None:
    rng = np.random.default_rng(501)
    first_base = rng.normal(size=(8, 3))
    first_recovered = rng.normal(size=(8, 3))
    second_base = rng.normal(size=(8, 3))
    second_recovered = rng.normal(size=(8, 3))
    angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    normals = np.column_stack([np.cos(angles), np.sin(angles)])

    combined = preserve_baseline_traction_with_tangential_correction(
        first_base + second_base,
        first_recovered + second_recovered,
        normals,
    )
    separate = preserve_baseline_traction_with_tangential_correction(
        first_base, first_recovered, normals
    ) + preserve_baseline_traction_with_tangential_correction(
        second_base, second_recovered, normals
    )

    np.testing.assert_allclose(combined, separate, rtol=2e-14, atol=2e-14)


def test_tangential_correction_rejects_nonunit_normals() -> None:
    with pytest.raises(StressRecoveryError, match="unit length"):
        preserve_baseline_traction_with_tangential_correction(
            np.zeros((1, 3)), np.ones((1, 3)), np.asarray([[2.0, 0.0]])
        )


@pytest.mark.parametrize(
    ("nodes", "elements", "stress", "points", "ids", "message"),
    [
        (
            np.zeros((3, 3)),
            np.asarray([[0, 1, 2]]),
            np.zeros((1, 3)),
            np.zeros((1, 2)),
            None,
            "nodes_yz",
        ),
        (
            np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
            np.asarray([[0.0, 1.0, 2.0]]),
            np.zeros((1, 3)),
            np.zeros((1, 2)),
            None,
            "integer",
        ),
        (
            np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
            np.asarray([[0, 1, 3]]),
            np.zeros((1, 3)),
            np.zeros((1, 2)),
            None,
            "out-of-range",
        ),
        (
            np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
            np.asarray([[0, 1, 2]]),
            np.zeros((1, 3)),
            np.zeros((1, 2)),
            None,
            "degenerate",
        ),
        (
            np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
            np.asarray([[0, 1, 2]]),
            np.zeros((1, 3)),
            np.zeros((1, 2)),
            None,
            "referenced",
        ),
        (
            np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
            np.asarray([[0, 1, 2]]),
            np.zeros((2, 3)),
            np.zeros((1, 2)),
            None,
            "element_stress",
        ),
        (
            np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
            np.asarray([[0, 1, 2]]),
            np.asarray([[np.nan, 0.0, 0.0]]),
            np.zeros((1, 2)),
            None,
            "finite",
        ),
        (
            np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
            np.asarray([[0, 1, 2]]),
            np.zeros((1, 3)),
            np.zeros((1, 3)),
            None,
            "query_points_yz",
        ),
        (
            np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
            np.asarray([[0, 1, 2]]),
            np.zeros((1, 3)),
            np.asarray([[2.0, 2.0]]),
            None,
            "outside",
        ),
        (
            np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
            np.asarray([[0, 1, 2]]),
            np.zeros((1, 3)),
            np.asarray([[0.2, 0.2]]),
            np.asarray([1]),
            "out-of-range",
        ),
        (
            np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
            np.asarray([[0, 1, 2]]),
            np.zeros((1, 3)),
            np.asarray([[0.2, 0.2]]),
            np.asarray([0.0]),
            "integer",
        ),
        (
            np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
            np.asarray([[0, 1, 2]]),
            np.zeros((1, 3)),
            np.asarray([[0.2, 0.2]]),
            np.asarray([0, 0]),
            "shape",
        ),
        (
            np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
            np.asarray([[0, 1, 2]]),
            np.zeros((1, 3)),
            np.asarray([[0.2, 0.2]]),
            np.asarray([0]),
            "never",
        ),
    ],
)
def test_invalid_inputs_are_rejected(
    nodes: np.ndarray,
    elements: np.ndarray,
    stress: np.ndarray,
    points: np.ndarray,
    ids: np.ndarray | None,
    message: str,
) -> None:
    if message == "never":
        # Turn the final otherwise-valid row into a supplied-element mismatch.
        points = np.asarray([[0.8, 0.8]])
        message = "disagrees"
    with pytest.raises(StressRecoveryError, match=message):
        recover_stress_at_queries(nodes, elements, stress, points, ids)
