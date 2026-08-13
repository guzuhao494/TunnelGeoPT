from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gmsh")
pytest.importorskip("skfem")
from skfem import ElementTriP1, FacetBasis

from tunnelgeopt.geometry import make_tunnel_boundary, points_inside_polygon
from tunnelgeopt.mesh import FARFIELD, ROCK, WALL, generate_tunnel_mesh


@pytest.mark.parametrize("shape", ["circle", "horseshoe", "straight_wall_arch"])
def test_gmsh_mesh_excludes_hole_and_labels_complete_boundary(shape: str) -> None:
    geometry = make_tunnel_boundary(shape, n_points=48, radius=1.0)
    tunnel_mesh = generate_tunnel_mesh(
        geometry,
        domain_scale=4.0,
        mesh_size=0.55,
        wall_mesh_size=0.16,
    )

    assert tunnel_mesh.nodes.ndim == 2 and tunnel_mesh.nodes.shape[1] == 2
    assert tunnel_mesh.elements.ndim == 2 and tunnel_mesh.elements.shape[1] == 3
    assert tunnel_mesh.elements.min() >= 0
    assert tunnel_mesh.elements.max() < tunnel_mesh.nodes.shape[0]
    centers = tunnel_mesh.nodes[tunnel_mesh.elements].mean(axis=1)
    assert not points_inside_polygon(centers, geometry.boundary_yz).any()

    assert set(tunnel_mesh.boundary_facets) == {WALL, FARFIELD}
    wall = tunnel_mesh.boundary_facets[WALL]
    farfield = tunnel_mesh.boundary_facets[FARFIELD]
    assert wall.size > 0 and farfield.size > 0
    assert np.intersect1d(wall, farfield).size == 0
    assert np.array_equal(
        np.sort(np.union1d(wall, farfield)),
        np.sort(tunnel_mesh.mesh.boundary_facets()),
    )
    assert np.array_equal(tunnel_mesh.mesh.boundaries[WALL], wall)
    assert np.array_equal(tunnel_mesh.mesh.boundaries[FARFIELD], farfield)
    assert np.all(tunnel_mesh.facet_markers[wall] == tunnel_mesh.physical_tags[WALL])
    assert np.all(tunnel_mesh.facet_markers[farfield] == tunnel_mesh.physical_tags[FARFIELD])
    assert np.all(tunnel_mesh.cell_markers == tunnel_mesh.physical_tags[ROCK])

    triangles = tunnel_mesh.nodes[tunnel_mesh.elements]
    first_edge = triangles[:, 1] - triangles[:, 0]
    second_edge = triangles[:, 2] - triangles[:, 0]
    twice_area = first_edge[:, 0] * second_edge[:, 1] - first_edge[:, 1] * second_edge[:, 0]
    assert np.all(twice_area > 0.0)
    assert tunnel_mesh.metadata["minimum_element_area"] > 0.0
    assert tunnel_mesh.metadata["minimum_triangle_quality"] > 0.0
    assert tunnel_mesh.metadata["element_order"] == 1
    assert tunnel_mesh.metadata["coordinate_order"] == ["y", "z"]
    if shape == "circle":
        wall_basis = FacetBasis(
            tunnel_mesh.mesh,
            ElementTriP1(),
            facets=tunnel_mesh.boundary_facets[WALL],
        )
        radial_projection = np.sum(
            np.asarray(wall_basis.global_coordinates()) * np.asarray(wall_basis.normals),
            axis=0,
        )
        assert np.all(radial_projection < 0.0)


def test_mesh_accepts_reversed_closed_loop_and_explicit_bounds() -> None:
    geometry = make_tunnel_boundary("circle", n_points=32, radius=1.0)
    counterclockwise_closed = np.vstack([geometry.boundary_yz[::-1], geometry.boundary_yz[-1]])
    tunnel_mesh = generate_tunnel_mesh(
        counterclockwise_closed,
        outer_bounds=(-3.0, 3.0, -2.5, 2.5),
        mesh_size=0.6,
        wall_mesh_size=0.2,
    )

    assert tunnel_mesh.outer_bounds == (-3.0, 3.0, -2.5, 2.5)
    assert tunnel_mesh.boundary_facets[WALL].size >= 32
    assert tunnel_mesh.boundary_facets[FARFIELD].size >= 4


def test_mesh_rejects_cavity_touching_farfield() -> None:
    geometry = make_tunnel_boundary("circle", n_points=32, radius=1.0)
    with pytest.raises(ValueError, match="strictly inside"):
        generate_tunnel_mesh(
            geometry,
            outer_bounds=(-1.0, 2.0, -2.0, 2.0),
            mesh_size=0.4,
        )
