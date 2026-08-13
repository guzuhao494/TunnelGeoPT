"""Analytical Kirsch stress solution used as the first physics validation anchor."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.floating]


def kirsch_stress(
    x: ArrayLike,
    y: ArrayLike,
    *,
    radius: float,
    sigma_x: float,
    sigma_y: float,
    tau_xy: float = 0.0,
    return_cartesian: bool = False,
) -> dict[str, FloatArray]:
    """Evaluate stresses around a circular traction-free opening.

    Positive input stresses are tensile under the classical elasticity sign
    convention.  Rock-mechanics compression-positive data must therefore be
    converted explicitly by the caller and recorded in metadata.
    """

    if radius <= 0.0:
        raise ValueError("radius must be positive")
    x_arr, y_arr = np.broadcast_arrays(
        np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    )
    r = np.hypot(x_arr, y_arr)
    if np.any(r < radius * (1.0 - 1e-12)):
        raise ValueError("Kirsch solution is defined only in the rock domain r >= radius")
    theta = np.arctan2(y_arr, x_arr)
    a2_r2 = (radius / r) ** 2
    a4_r4 = a2_r2**2
    cos_2t = np.cos(2.0 * theta)
    sin_2t = np.sin(2.0 * theta)
    mean = 0.5 * (sigma_x + sigma_y)
    difference = 0.5 * (sigma_x - sigma_y)

    radial = (
        mean * (1.0 - a2_r2)
        + difference * (1.0 - 4.0 * a2_r2 + 3.0 * a4_r4) * cos_2t
        + tau_xy * (1.0 - 4.0 * a2_r2 + 3.0 * a4_r4) * sin_2t
    )
    hoop = (
        mean * (1.0 + a2_r2)
        - difference * (1.0 + 3.0 * a4_r4) * cos_2t
        - tau_xy * (1.0 + 3.0 * a4_r4) * sin_2t
    )
    shear = (
        -difference * (1.0 + 2.0 * a2_r2 - 3.0 * a4_r4) * sin_2t
        + tau_xy * (1.0 + 2.0 * a2_r2 - 3.0 * a4_r4) * cos_2t
    )
    result: dict[str, FloatArray] = {
        "r": r,
        "theta": theta,
        "sigma_rr": radial,
        "sigma_tt": hoop,
        "tau_rt": shear,
    }
    if return_cartesian:
        cosine = np.cos(theta)
        sine = np.sin(theta)
        result.update(
            {
                "sigma_xx": radial * cosine**2 - 2.0 * shear * sine * cosine + hoop * sine**2,
                "sigma_yy": radial * sine**2 + 2.0 * shear * sine * cosine + hoop * cosine**2,
                "tau_xy": (radial - hoop) * sine * cosine + shear * (cosine**2 - sine**2),
            }
        )
    return result
