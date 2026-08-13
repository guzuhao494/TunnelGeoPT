import numpy as np
import pytest

from tunnelgeopt.kirsch import kirsch_stress


def test_traction_free_circular_boundary() -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 129)
    result = kirsch_stress(
        np.cos(theta),
        np.sin(theta),
        radius=1.0,
        sigma_x=10.0,
        sigma_y=4.0,
        tau_xy=2.0,
    )
    assert np.max(np.abs(result["sigma_rr"])) < 1e-12
    assert np.max(np.abs(result["tau_rt"])) < 1e-12


def test_uniaxial_boundary_stress_concentration_is_three() -> None:
    result = kirsch_stress(
        np.array([0.0]),
        np.array([1.0]),
        radius=1.0,
        sigma_x=10.0,
        sigma_y=0.0,
    )
    assert result["sigma_tt"][0] == pytest.approx(30.0)


def test_far_field_recovers_applied_cartesian_stress() -> None:
    result = kirsch_stress(
        np.array([1000.0, 0.0]),
        np.array([0.0, 1000.0]),
        radius=1.0,
        sigma_x=12.0,
        sigma_y=7.0,
        tau_xy=1.5,
        return_cartesian=True,
    )
    assert np.allclose(result["sigma_xx"], 12.0, atol=5e-5)
    assert np.allclose(result["sigma_yy"], 7.0, atol=5e-5)
    assert np.allclose(result["tau_xy"], 1.5, atol=5e-5)


def test_points_inside_cavity_are_rejected() -> None:
    with pytest.raises(ValueError, match="rock domain"):
        kirsch_stress(0.5, 0.0, radius=1.0, sigma_x=1.0, sigma_y=1.0)
