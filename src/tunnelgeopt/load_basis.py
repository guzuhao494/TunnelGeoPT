"""Exact load-axis factorization for the linear-elastic tunnel layer.

For a fixed geometry, mesh and query grid, the current plane-strain solver is
linear in the three independent in-plane far-field stress components
``[yy, zz, yz]``.  A response basis therefore stores a ``3 x 3`` linear map at
every query point: three output stress components for three input load
components.  This module fits and evaluates that map without introducing a
learned or stress-dependent nonlinearity.

The contract applies only to the homogeneous small-strain linear-elastic
layer.  It must not be extended to damage, contact, plasticity or fracture
without a new validation argument.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.floating]


class LoadBasisError(ValueError):
    """Raised when a response-basis input violates the linear contract."""


def _loads(value: ArrayLike, *, minimum_rows: int = 1) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2 or array.shape[1] != 3 or array.shape[0] < minimum_rows:
        raise LoadBasisError(f"load_vectors must have shape [K, 3] with K >= {minimum_rows}")
    if not np.isfinite(array).all():
        raise LoadBasisError("load_vectors must contain only finite values")
    return array


def _responses(value: ArrayLike, *, load_count: int) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 3 or array.shape[0] != load_count or array.shape[2] != 3:
        raise LoadBasisError("stress_responses must have shape [K, P, 3]")
    if array.shape[1] < 1 or not np.isfinite(array).all():
        raise LoadBasisError("stress_responses must be finite and contain at least one point")
    return array


@dataclass(frozen=True)
class LinearStressResponseBasis:
    """A per-query map from ``[sigma_yy, sigma_zz, tau_yz]`` to stress."""

    coefficients: FloatArray
    fit_load_count: int
    load_rank: int
    load_condition_number: float
    relative_fit_residual: float

    def __post_init__(self) -> None:
        coefficients = np.asarray(self.coefficients, dtype=np.float64).copy()
        if coefficients.ndim != 3 or coefficients.shape[1:] != (3, 3):
            raise LoadBasisError("coefficients must have shape [P, 3, 3]")
        if coefficients.shape[0] < 1 or not np.isfinite(coefficients).all():
            raise LoadBasisError("coefficients must be finite and contain at least one point")
        if int(self.fit_load_count) < 3 or int(self.load_rank) != 3:
            raise LoadBasisError("a response basis requires at least three rank-three loads")
        if not np.isfinite(self.load_condition_number) or self.load_condition_number < 1.0:
            raise LoadBasisError("load_condition_number must be finite and at least one")
        if not np.isfinite(self.relative_fit_residual) or self.relative_fit_residual < 0.0:
            raise LoadBasisError("relative_fit_residual must be finite and non-negative")
        coefficients.setflags(write=False)
        object.__setattr__(self, "coefficients", coefficients)

    @property
    def point_count(self) -> int:
        return int(self.coefficients.shape[0])

    def predict(self, load_vectors: ArrayLike) -> FloatArray:
        """Evaluate the response basis for one or more normalized loads."""

        loads = _loads(load_vectors)
        prediction = np.einsum("ki,poi->kpo", loads, self.coefficients)
        if not np.isfinite(prediction).all():
            raise RuntimeError("linear response-basis prediction became non-finite")
        return prediction


def fit_linear_stress_response_basis(
    load_vectors: ArrayLike,
    stress_responses: ArrayLike,
    *,
    rank_tolerance: float | None = None,
    maximum_condition_number: float | None = None,
) -> LinearStressResponseBasis:
    """Fit the unique least-squares linear stress map for one fixed geometry.

    Inputs and outputs may both be divided by the same prescribed far-field
    stress norm.  This preserves the linear map and matches the v0.3 dataset's
    normalized ``condition[:3]`` and normalized stress fields.
    """

    loads = _loads(load_vectors, minimum_rows=3)
    responses = _responses(stress_responses, load_count=loads.shape[0])
    if rank_tolerance is not None and (
        not np.isfinite(rank_tolerance) or not 0.0 < float(rank_tolerance) < 1.0
    ):
        raise LoadBasisError("rank_tolerance must be None or lie in (0, 1)")
    rank = int(np.linalg.matrix_rank(loads, tol=rank_tolerance))
    if rank != 3:
        raise LoadBasisError("load_vectors must span all three in-plane stress components")
    condition = float(np.linalg.cond(loads))
    if not np.isfinite(condition):
        raise LoadBasisError("load basis condition number is not finite")
    if maximum_condition_number is not None:
        limit = float(maximum_condition_number)
        if not np.isfinite(limit) or limit < 1.0:
            raise LoadBasisError("maximum_condition_number must be finite and at least one")
        if condition > limit:
            raise LoadBasisError(
                f"load basis is too ill-conditioned ({condition:.6g} > {limit:.6g})"
            )

    flattened = responses.reshape(responses.shape[0], -1)
    solution, _, solved_rank, _ = np.linalg.lstsq(loads, flattened, rcond=rank_tolerance)
    if int(solved_rank) != 3:
        raise LoadBasisError("least-squares solver did not retain rank three")
    coefficients = solution.T.reshape(responses.shape[1], 3, 3)
    fitted = np.einsum("ki,poi->kpo", loads, coefficients)
    denominator = float(np.linalg.norm(responses))
    residual = float(np.linalg.norm(fitted - responses) / max(denominator, np.finfo(float).tiny))
    return LinearStressResponseBasis(
        coefficients=coefficients,
        fit_load_count=int(loads.shape[0]),
        load_rank=rank,
        load_condition_number=condition,
        relative_fit_residual=residual,
    )


__all__ = [
    "LinearStressResponseBasis",
    "LoadBasisError",
    "fit_linear_stress_response_basis",
]
