import numpy as np
import pytest

from tunnelgeopt.geometry import (
    make_tunnel_boundary,
    nearest_boundary_vectors,
    points_inside_polygon,
    sample_rock_points,
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
