import hashlib
from dataclasses import asdict

import numpy as np
import pytest

from tunnelgeopt.geometry import (
    canonical_shape_parameters,
    make_parametric_tunnel_boundary,
    make_tunnel_boundary,
    nearest_boundary_vectors,
    points_inside_polygon,
    sample_rock_points,
    shape_parameter_bounds,
    surface_points_and_normals,
)


@pytest.mark.parametrize("shape", ["circle", "horseshoe", "straight_wall_arch"])
def test_procedural_boundaries_and_sampling(shape: str) -> None:
    geometry = make_tunnel_boundary(shape, n_points=96, roughness_amplitude=0.01, seed=7)
    assert geometry.boundary_yz.shape == (96, 2)
    rock = sample_rock_points(geometry, 256, seed=11)
    assert not points_inside_polygon(rock, geometry.boundary_yz).any()
    distance, direction, nearest = nearest_boundary_vectors(rock, geometry.boundary_yz)
    assert np.all(distance > 0.0)
    assert np.allclose(np.linalg.norm(direction, axis=1), 1.0, atol=1e-10)
    assert nearest.shape == rock.shape


def test_circle_surface_normals_point_away_from_cavity() -> None:
    geometry = make_tunnel_boundary("circle", n_points=128)
    points, normals = surface_points_and_normals(geometry, 64)
    radial = points / np.linalg.norm(points, axis=1, keepdims=True)
    assert np.all(np.sum(radial * normals, axis=1) > 0.99)


def test_invalid_roughness_is_rejected() -> None:
    with pytest.raises(ValueError, match="roughness_amplitude"):
        make_tunnel_boundary("circle", roughness_amplitude=0.2)


@pytest.mark.parametrize(
    ("shape", "expected_sha256"),
    [
        ("circle", "a67f41bf4e33ec709a05f9f5f9ebf81a7d66a67a92366ee5ac68722a6724175c"),
        ("horseshoe", "903b88afefc9e85f71dd0f7ec07484a642458a262236165097b4cdee17c58669"),
        (
            "straight_wall_arch",
            "be8f6a3dec470d00cc03608746388f9b9e943f4347c7e27b00bf751cb00eb924",
        ),
    ],
)
def test_historical_canonical_boundary_is_bitwise_preserved(
    shape: str, expected_sha256: str
) -> None:
    boundary = make_tunnel_boundary(shape, n_points=96, seed=7).boundary_yz
    assert hashlib.sha256(boundary.tobytes()).hexdigest() == expected_sha256


def test_geometry_remains_dataclass_serializable() -> None:
    payload = asdict(make_tunnel_boundary("circle", n_points=16))
    assert payload["shape_parameters"] == canonical_shape_parameters("circle")


@pytest.mark.parametrize(
    ("shape", "parameter_updates"),
    [
        ("circle", {"axis_ratio": 1.3, "superellipse_exponent": 3.0}),
        (
            "horseshoe",
            {"span_height_ratio": 0.95, "sidewall_height_ratio": 1.15, "crown_shape": 2.8},
        ),
        (
            "straight_wall_arch",
            {
                "span_height_ratio": 1.15,
                "springline_height_ratio": 0.35,
                "crown_rise_span": 1.0,
            },
        ),
    ],
)
def test_each_shape_macro_parameter_changes_normalized_boundary(
    shape: str, parameter_updates: dict[str, float]
) -> None:
    canonical = make_parametric_tunnel_boundary(shape, n_points=192)
    defaults = canonical_shape_parameters(shape)
    assert set(defaults) == set(shape_parameter_bounds(shape))
    for name, value in parameter_updates.items():
        varied = make_parametric_tunnel_boundary(shape, parameters={name: value}, n_points=192)
        assert not np.allclose(varied.boundary_yz, canonical.boundary_yz)
        assert varied.shape_parameters[name] == value
        assert np.isfinite(varied.boundary_yz).all()


def test_parametric_boundary_rejects_unknown_invalid_and_non_finite_parameters() -> None:
    with pytest.raises(ValueError, match="unknown .* shape parameter"):
        make_parametric_tunnel_boundary("circle", parameters={"not_a_parameter": 1.0})
    with pytest.raises(ValueError, match="axis_ratio"):
        make_parametric_tunnel_boundary("circle", parameters={"axis_ratio": 0.1})
    with pytest.raises(ValueError, match="finite"):
        make_parametric_tunnel_boundary("circle", parameters={"axis_ratio": np.nan})
    with pytest.raises(TypeError, match="real number"):
        make_parametric_tunnel_boundary("circle", parameters={"axis_ratio": True})


def test_roughness_uses_complete_large_seed_reproducibly() -> None:
    seed = (1 << 80) + 12345
    first = make_parametric_tunnel_boundary(
        "horseshoe", n_points=128, roughness_amplitude=0.04, seed=seed
    )
    repeat = make_parametric_tunnel_boundary(
        "horseshoe", n_points=128, roughness_amplitude=0.04, seed=seed
    )
    truncated = make_parametric_tunnel_boundary(
        "horseshoe", n_points=128, roughness_amplitude=0.04, seed=12345
    )
    assert np.array_equal(first.boundary_yz, repeat.boundary_yz)
    assert not np.array_equal(first.boundary_yz, truncated.boundary_yz)
    assert first.seed == seed
