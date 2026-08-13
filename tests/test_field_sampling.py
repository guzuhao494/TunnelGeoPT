import numpy as np
import pytest

from tunnelgeopt.field_sampling import (
    ElementLookup,
    locate_elements,
    sample_piecewise_constant,
)
from tunnelgeopt.geometry import make_tunnel_boundary
from tunnelgeopt.mesh import generate_tunnel_mesh


def test_same_physical_points_are_located_independently_in_coarse_and_fine_meshes() -> None:
    geometry = make_tunnel_boundary("circle", n_points=32)
    common = {
        "outer_bounds": (-3.0, 3.0, -3.0, 3.0),
        "farfield_mesh_size": 0.7,
    }
    coarse = generate_tunnel_mesh(
        geometry,
        mesh_size=0.7,
        wall_mesh_size=0.35,
        **common,
    )
    fine = generate_tunnel_mesh(
        geometry,
        mesh_size=0.38,
        wall_mesh_size=0.18,
        **common,
    )
    points = np.asarray([[2.0, 0.0], [-2.0, 0.0], [0.0, 2.0], [0.0, -2.0], [2.0, 2.0]])

    coarse_ids = locate_elements(ElementLookup.from_mesh(coarse), points, raise_outside=True)
    fine_ids = locate_elements(fine.nodes, fine.elements, points, raise_outside=True)

    assert np.all(coarse_ids >= 0)
    assert np.all(fine_ids >= 0)
    assert coarse.elements.shape[0] != fine.elements.shape[0]
    coarse_values = np.full((coarse.elements.shape[0], 3), [7.0, -2.0, 0.5])
    fine_values = np.full((fine.elements.shape[0], 3), [7.0, -2.0, 0.5])
    assert np.array_equal(
        sample_piecewise_constant(coarse_values, coarse_ids),
        sample_piecewise_constant(fine_values, fine_ids),
    )


def test_edge_point_is_validated_and_outside_point_has_explicit_failure_modes() -> None:
    skfem = pytest.importorskip("skfem")
    mesh = skfem.MeshTri.init_sqsymmetric()
    lookup = ElementLookup.from_mesh(mesh)
    points = np.asarray([[0.5, 0.5], [2.0, 2.0]])

    ids = locate_elements(lookup, points, chunk_size=2)
    assert ids[0] >= 0  # shared vertex/edge remains a valid point
    assert ids[1] == -1
    with pytest.raises(ValueError, match="outside"):
        locate_elements(lookup, points, raise_outside=True)
    with pytest.raises(ValueError, match="outside"):
        sample_piecewise_constant(np.arange(mesh.t.shape[1]), ids)

    sampled = sample_piecewise_constant(
        np.arange(mesh.t.shape[1]), ids, allow_outside=True, fill_value=np.nan
    )
    assert np.isfinite(sampled[0])
    assert np.isnan(sampled[1])


def test_sampling_validates_connectivity_queries_and_value_indices() -> None:
    with pytest.raises(ValueError, match=r"shape \[N, 2\]"):
        ElementLookup.from_arrays(np.zeros((4, 3)), np.asarray([[0, 1, 2]]))
    with pytest.raises(ValueError, match=r"shape \[P, 2\]"):
        locate_elements(
            np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
            np.asarray([[0, 1, 2]]),
            np.zeros((2, 3)),
        )
    with pytest.raises(ValueError, match="out-of-range"):
        sample_piecewise_constant(np.ones((2, 3)), np.asarray([2]))
