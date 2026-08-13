from __future__ import annotations

import numpy as np
import pytest

from tunnelgeopt.load_basis import (
    LoadBasisError,
    fit_linear_stress_response_basis,
)


def _basis(point_count: int = 7) -> np.ndarray:
    rng = np.random.default_rng(20260813)
    return rng.normal(size=(point_count, 3, 3))


def test_three_independent_loads_recover_exact_nine_channel_basis() -> None:
    coefficients = _basis()
    loads = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.5]])
    responses = np.einsum("ki,poi->kpo", loads, coefficients)

    fitted = fit_linear_stress_response_basis(loads, responses)

    np.testing.assert_allclose(fitted.coefficients, coefficients, rtol=1e-14, atol=1e-14)
    assert fitted.coefficients.shape == (7, 3, 3)
    assert fitted.relative_fit_residual < 1e-14


def test_overdetermined_fit_predicts_unseen_load_and_preserves_superposition() -> None:
    coefficients = _basis(11)
    fit_loads = np.asarray([[1.0, 0.2, 0.1], [0.1, 1.0, -0.2], [0.3, 0.4, 0.7], [-0.5, 0.2, 0.6]])
    responses = np.einsum("ki,poi->kpo", fit_loads, coefficients)
    fitted = fit_linear_stress_response_basis(fit_loads, responses)
    first = np.asarray([0.7, -0.3, 0.2])
    second = np.asarray([-0.1, 0.6, 0.4])

    combined = fitted.predict(first + second)[0]
    separate = fitted.predict(first)[0] + fitted.predict(second)[0]

    np.testing.assert_allclose(combined, separate, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(
        fitted.predict(first)[0],
        np.einsum("i,poi->po", first, coefficients),
        rtol=2e-14,
        atol=2e-14,
    )


def test_common_positive_normalization_preserves_the_basis_prediction() -> None:
    coefficients = _basis(5)
    dimensional_loads = np.asarray([[-8.0, -3.0, 1.0], [-4.0, -9.0, -2.0], [-7.0, -6.0, 3.0]])
    scales = np.sqrt(
        dimensional_loads[:, 0] ** 2
        + dimensional_loads[:, 1] ** 2
        + 2.0 * dimensional_loads[:, 2] ** 2
    )
    dimensional_responses = np.einsum("ki,poi->kpo", dimensional_loads, coefficients)
    normalized_loads = dimensional_loads / scales[:, None]
    normalized_responses = dimensional_responses / scales[:, None, None]
    fitted = fit_linear_stress_response_basis(normalized_loads, normalized_responses)
    unseen = np.asarray([-5.0, -2.0, 0.75])
    unseen_scale = np.sqrt(unseen[0] ** 2 + unseen[1] ** 2 + 2.0 * unseen[2] ** 2)

    prediction = fitted.predict(unseen / unseen_scale)[0] * unseen_scale

    np.testing.assert_allclose(
        prediction,
        np.einsum("i,poi->po", unseen, coefficients),
        rtol=3e-14,
        atol=3e-14,
    )


@pytest.mark.parametrize(
    ("loads", "responses", "message"),
    [
        (np.zeros((2, 3)), np.zeros((2, 4, 3)), "K >= 3"),
        (np.zeros((3, 2)), np.zeros((3, 4, 3)), "load_vectors"),
        (np.zeros((3, 3)), np.zeros((3, 4, 2)), "stress_responses"),
        (
            np.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
            np.zeros((3, 4, 3)),
            "span all three",
        ),
        (np.eye(3), np.full((3, 4, 3), np.nan), "finite"),
    ],
)
def test_invalid_or_rank_deficient_inputs_are_rejected(
    loads: np.ndarray, responses: np.ndarray, message: str
) -> None:
    with pytest.raises(LoadBasisError, match=message):
        fit_linear_stress_response_basis(loads, responses)


def test_explicit_condition_limit_is_fail_closed() -> None:
    loads = np.asarray([[1.0, 0.0, 0.0], [1.0, 1e-8, 0.0], [1.0, 0.0, 1e-8]])
    responses = np.einsum("ki,poi->kpo", loads, _basis(3))
    with pytest.raises(LoadBasisError, match="ill-conditioned"):
        fit_linear_stress_response_basis(loads, responses, maximum_condition_number=1e6)
