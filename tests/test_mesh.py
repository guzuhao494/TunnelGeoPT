from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gmsh")
pytest.importorskip("skfem")
from skfem import ElementTriP1, FacetBasis

from tunnelgeopt import mesh as mesh_module
from tunnelgeopt.geometry import make_tunnel_boundary, points_inside_polygon
from tunnelgeopt.mesh import FARFIELD, ROCK, WALL, generate_tunnel_mesh


def _independent_polyline_distance(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    result = np.full(points.shape[0], np.inf)
    for start, end in zip(polyline, np.roll(polyline, -1, axis=0), strict=True):
        direction = end - start
        fraction = np.clip((points - start) @ direction / (direction @ direction), 0.0, 1.0)
        closest = start + fraction[:, None] * direction
        result = np.minimum(result, np.linalg.norm(points - closest, axis=1))
    return result


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


def test_default_mesh_path_is_unchanged_when_nearfield_is_omitted() -> None:
    geometry = make_tunnel_boundary("circle", n_points=24, radius=1.0)
    default_mesh = generate_tunnel_mesh(
        geometry,
        domain_scale=3.5,
        mesh_size=0.65,
        wall_mesh_size=0.24,
    )
    explicit_disabled_mesh = generate_tunnel_mesh(
        geometry,
        domain_scale=3.5,
        mesh_size=0.65,
        wall_mesh_size=0.24,
        nearfield_distance=None,
        nearfield_mesh_size=None,
        fracture_length_scale=None,
        nearfield_transition_width=None,
    )

    assert np.array_equal(default_mesh.nodes, explicit_disabled_mesh.nodes)
    assert np.array_equal(default_mesh.elements, explicit_disabled_mesh.elements)
    assert default_mesh.metadata == explicit_disabled_mesh.metadata
    assert not any(key.startswith("nearfield_") for key in default_mesh.metadata)


@pytest.mark.parametrize("shape", ["circle", "horseshoe", "straight_wall_arch"])
def test_nearfield_distance_threshold_refines_and_records_h_over_ell_audit(shape: str) -> None:
    geometry = make_tunnel_boundary(shape, n_points=32, radius=1.0)
    coarse = generate_tunnel_mesh(
        geometry,
        domain_scale=4.0,
        mesh_size=0.60,
        wall_mesh_size=0.30,
        farfield_mesh_size=0.60,
    )
    refined = generate_tunnel_mesh(
        geometry,
        domain_scale=4.0,
        mesh_size=0.60,
        wall_mesh_size=0.30,
        farfield_mesh_size=0.60,
        nearfield_distance=0.70,
        nearfield_mesh_size=0.18,
        fracture_length_scale=0.72,
        nearfield_transition_width=0.45,
    )

    coarse_triangles = coarse.nodes[coarse.elements]
    coarse_centroids = coarse_triangles.mean(axis=1)
    coarse_distance = _independent_polyline_distance(
        coarse_centroids,
        geometry.boundary_yz,
    )
    coarse_edges = np.linalg.norm(
        coarse_triangles - np.roll(coarse_triangles, -1, axis=1),
        axis=2,
    )
    coarse_band_max = float(coarse_edges[coarse_distance <= 0.70].max())

    refined_triangles = refined.nodes[refined.elements]
    refined_distance = _independent_polyline_distance(
        refined_triangles.mean(axis=1),
        geometry.boundary_yz,
    )
    refined_edges = np.linalg.norm(
        refined_triangles - np.roll(refined_triangles, -1, axis=1),
        axis=2,
    )
    independent_band_mask = refined_distance <= 0.70
    independent_band_max = float(refined_edges[independent_band_mask].max())

    metadata = refined.metadata
    assert metadata["nearfield_enabled"] is True
    assert metadata["nearfield_gmsh_fields"] == ["Distance", "Threshold"]
    assert metadata["nearfield_distance"] == pytest.approx(0.70)
    assert metadata["nearfield_guard_width"] == pytest.approx(0.36)
    assert metadata["nearfield_transition_start_distance"] == pytest.approx(1.06)
    assert metadata["nearfield_transition_end_distance"] == pytest.approx(1.51)
    assert metadata["nearfield_audited_element_count"] == int(independent_band_mask.sum())
    assert metadata["nearfield_actual_maximum_edge"] == pytest.approx(independent_band_max)
    assert metadata["nearfield_actual_maximum_edge"] < 0.65 * coarse_band_max
    assert metadata["nearfield_actual_maximum_edge"] <= metadata["nearfield_audit_limit"]
    assert metadata["nearfield_actual_maximum_edge"] <= 0.85 * metadata["nearfield_mesh_size"]
    assert metadata["nearfield_audit_passed"] is True
    assert metadata["nearfield_requested_h_over_ell"] == pytest.approx(0.25)
    assert metadata["nearfield_actual_maximum_h_over_ell"] == pytest.approx(
        metadata["nearfield_actual_maximum_edge"] / 0.72
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"nearfield_distance": 0.5}, "must be supplied together"),
        (
            {
                "nearfield_distance": 0.5,
                "nearfield_mesh_size": np.nan,
                "fracture_length_scale": 0.5,
            },
            "finite and positive",
        ),
        (
            {
                "nearfield_distance": 0.5,
                "nearfield_mesh_size": 0.6,
                "fracture_length_scale": 0.5,
            },
            "smaller than farfield_mesh_size",
        ),
        (
            {
                "nearfield_distance": 0.5,
                "nearfield_mesh_size": 0.15,
                "fracture_length_scale": 0.5,
                "nearfield_transition_width": 0.0,
            },
            "finite and positive",
        ),
    ],
)
def test_nearfield_rejects_incomplete_or_invalid_parameters(
    kwargs: dict[str, float],
    message: str,
) -> None:
    geometry = make_tunnel_boundary("circle", n_points=24, radius=1.0)
    with pytest.raises(ValueError, match=message):
        generate_tunnel_mesh(
            geometry,
            domain_scale=3.5,
            mesh_size=0.60,
            farfield_mesh_size=0.60,
            **kwargs,
        )


def test_nearfield_fails_closed_when_generated_edges_exceed_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry = make_tunnel_boundary("circle", n_points=24, radius=1.0)
    monkeypatch.setattr(mesh_module, "_NEARFIELD_GMSH_TARGET_FACTOR", 1.5)

    with pytest.raises(RuntimeError, match="near-field maximum edge audit failed"):
        generate_tunnel_mesh(
            geometry,
            domain_scale=3.5,
            mesh_size=0.60,
            wall_mesh_size=0.60,
            farfield_mesh_size=0.60,
            nearfield_distance=0.60,
            nearfield_mesh_size=0.18,
            fracture_length_scale=0.72,
            nearfield_transition_width=0.40,
        )
