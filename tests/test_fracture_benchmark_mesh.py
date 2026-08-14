from __future__ import annotations

from types import MappingProxyType

import numpy as np
import pytest

pytest.importorskip("gmsh")
pytest.importorskip("skfem")

from tunnelgeopt.fracture_benchmark_mesh import (
    BOTTOM,
    BOUNDARY_LABELS,
    BOUNDARY_NODE_LABELS,
    BULK,
    LEFT_LOWER,
    LEFT_UPPER,
    LOADING_MODES,
    MESH_TIERS,
    NOTCH_LOWER,
    NOTCH_TIP,
    NOTCH_UPPER,
    PHYSICAL_LABELS,
    RIGHT,
    TOP,
    _audit_zero_width_slit,
    benchmark_mesh_plan,
    generate_fracture_benchmark_mesh,
)


def _face_nodes(benchmark_mesh, label: str) -> set[int]:
    facets = benchmark_mesh.boundary_facets[label]
    return {int(value) for value in benchmark_mesh.mesh.facets[:, facets].ravel()}


def test_frozen_plans_have_exact_tiers_corridors_and_stable_hashes() -> None:
    assert MESH_TIERS == ("coarse", "medium", "fine")
    assert LOADING_MODES == ("sent", "sens")
    assert [benchmark_mesh_plan(tier=tier).target_h_mm for tier in MESH_TIERS] == [
        0.0075,
        0.00375,
        0.001875,
    ]
    assert [benchmark_mesh_plan(tier=tier).farfield_h_mm for tier in MESH_TIERS] == [
        0.03,
        0.015,
        0.0075,
    ]

    sent = benchmark_mesh_plan(loading="SENT", tier="coarse")
    sens = benchmark_mesh_plan(loading="sens", tier="coarse")
    assert sent.notch_polyline_yz_mm == ((0.5, 0.0), (0.5, 0.5))
    assert sens.notch_polyline_yz_mm == sent.notch_polyline_yz_mm
    assert sent.notch_band_half_width_mm == 0.05
    assert sens.notch_band_half_width_mm == 0.05
    assert sent.propagation_corridor_polyline_yz_mm == ((0.5, 0.5), (0.5, 1.0))
    assert sent.propagation_corridor_half_width_mm == 0.10
    assert sens.propagation_corridor_polyline_yz_mm == ((0.5, 0.5), (0.0, 1.0))
    assert sens.propagation_corridor_half_width_mm == 0.15
    assert sent.plan_sha256 == benchmark_mesh_plan(loading="sent", tier="coarse").plan_sha256
    assert len(sent.plan_sha256) == 64
    assert sent.plan_sha256 != sens.plan_sha256

    with pytest.raises(ValueError, match="loading must be"):
        benchmark_mesh_plan(loading="mixed")
    with pytest.raises(ValueError, match="tier must be"):
        benchmark_mesh_plan(tier="production")


def test_coarse_sent_is_an_exact_zero_width_double_face_slit() -> None:
    benchmark_mesh = generate_fracture_benchmark_mesh(loading="sent", tier="coarse")
    mesh = benchmark_mesh.mesh
    nodes = benchmark_mesh.nodes

    assert tuple(benchmark_mesh.physical_tags) == PHYSICAL_LABELS
    assert set(benchmark_mesh.boundary_facets) == set(BOUNDARY_LABELS)
    assert set(benchmark_mesh.boundary_nodes) == set(BOUNDARY_NODE_LABELS)
    assert set(benchmark_mesh.physical_entity_tags) == set(PHYSICAL_LABELS)
    assert set(benchmark_mesh.identity["physical_entity_dimensions"]) == set(PHYSICAL_LABELS)
    assert dict(benchmark_mesh.identity["physical_tags"]) == dict(benchmark_mesh.physical_tags)
    assert dict(benchmark_mesh.identity["physical_entity_tags"]) == dict(
        benchmark_mesh.physical_entity_tags
    )
    all_marked = np.concatenate(
        [benchmark_mesh.boundary_facets[label] for label in BOUNDARY_LABELS]
    )
    assert np.unique(all_marked).size == all_marked.size
    assert np.array_equal(np.sort(all_marked), np.sort(mesh.boundary_facets()))

    upper = _face_nodes(benchmark_mesh, NOTCH_UPPER)
    lower = _face_nodes(benchmark_mesh, NOTCH_LOWER)
    shared = upper & lower
    assert len(shared) == 1
    tip = next(iter(shared))
    assert nodes[tip] == pytest.approx((0.5, 0.5), abs=2.0e-12)
    assert np.array_equal(benchmark_mesh.boundary_nodes[NOTCH_TIP], [tip])
    assert benchmark_mesh.identity["physical_entity_dimensions"][NOTCH_TIP] == 0
    assert all(
        benchmark_mesh.identity["physical_entity_dimensions"][label] == 1
        for label in BOUNDARY_LABELS
    )

    upper_distinct = sorted(upper - shared, key=lambda index: nodes[index, 1])
    lower_distinct = sorted(lower - shared, key=lambda index: nodes[index, 1])
    assert len(upper_distinct) == len(lower_distinct) >= 2
    assert set(upper_distinct).isdisjoint(lower_distinct)
    assert np.allclose(
        nodes[np.asarray(upper_distinct)],
        nodes[np.asarray(lower_distinct)],
        rtol=0.0,
        atol=2.0e-12,
    )
    assert nodes[upper_distinct[0]] == pytest.approx((0.5, 0.0), abs=2.0e-12)
    assert nodes[lower_distinct[0]] == pytest.approx((0.5, 0.0), abs=2.0e-12)

    left_upper = _face_nodes(benchmark_mesh, LEFT_UPPER)
    left_lower = _face_nodes(benchmark_mesh, LEFT_LOWER)
    assert not (left_upper & left_lower)
    assert left_upper & upper == {upper_distinct[0]}
    assert left_lower & lower == {lower_distinct[0]}
    assert not (left_upper & lower)
    assert not (left_lower & upper)
    assert np.all(nodes[np.asarray(sorted(left_upper)), 0] >= 0.5 - 2.0e-12)
    assert np.all(nodes[np.asarray(sorted(left_lower)), 0] <= 0.5 + 2.0e-12)
    assert np.allclose(
        nodes[np.asarray(sorted(left_upper | left_lower)), 1],
        0.0,
        rtol=0.0,
        atol=2.0e-12,
    )

    for label in (NOTCH_UPPER, NOTCH_LOWER):
        facets = benchmark_mesh.boundary_facets[label]
        assert np.all(np.sum(mesh.f2t[:, facets] >= 0, axis=0) == 1)
        face_coordinates = nodes[mesh.facets[:, facets]]
        assert np.allclose(face_coordinates[..., 0], 0.5, rtol=0.0, atol=2.0e-12)
        assert face_coordinates[..., 1].min() >= -2.0e-12
        assert face_coordinates[..., 1].max() <= 0.5 + 2.0e-12

    triangles = nodes[benchmark_mesh.elements]
    first = triangles[:, 1] - triangles[:, 0]
    second = triangles[:, 2] - triangles[:, 0]
    twice_area = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    assert np.all(twice_area > 0.0)
    assert 0.5 * twice_area.sum() == pytest.approx(1.0, abs=2.0e-12)

    assert np.all(
        benchmark_mesh.facet_markers[benchmark_mesh.boundary_facets[TOP]]
        == benchmark_mesh.physical_tags[TOP]
    )
    assert np.all(
        benchmark_mesh.facet_markers[benchmark_mesh.boundary_facets[BOTTOM]]
        == benchmark_mesh.physical_tags[BOTTOM]
    )
    assert np.all(
        benchmark_mesh.facet_markers[benchmark_mesh.boundary_facets[LEFT_UPPER]]
        == benchmark_mesh.physical_tags[LEFT_UPPER]
    )
    assert np.all(
        benchmark_mesh.facet_markers[benchmark_mesh.boundary_facets[LEFT_LOWER]]
        == benchmark_mesh.physical_tags[LEFT_LOWER]
    )
    assert np.all(
        benchmark_mesh.facet_markers[benchmark_mesh.boundary_facets[RIGHT]]
        == benchmark_mesh.physical_tags[RIGHT]
    )
    assert np.all(benchmark_mesh.cell_markers == benchmark_mesh.physical_tags[BULK])
    assert benchmark_mesh.metadata["topology_audit_passed"] is True
    assert benchmark_mesh.metadata["zero_width_double_face_slit_audit_passed"] is True
    assert benchmark_mesh.metadata["slit_face_shared_node_count"] == 1
    assert benchmark_mesh.metadata["notch_tip_point_entity_audit_passed"] is True
    assert benchmark_mesh.metadata["left_split_mouth_contact_audit_passed"] is True
    assert (
        benchmark_mesh.metadata["notch_tip_physical_tag"] == benchmark_mesh.physical_tags[NOTCH_TIP]
    )
    assert (
        benchmark_mesh.metadata["notch_tip_gmsh_entity_tag"]
        == (benchmark_mesh.physical_entity_tags[NOTCH_TIP][0])
    )
    assert benchmark_mesh.metadata["slit_distinct_coincident_node_pair_count"] == len(
        upper_distinct
    )


@pytest.mark.parametrize("loading", LOADING_MODES)
@pytest.mark.parametrize("tier", MESH_TIERS)
def test_all_real_gmsh_tiers_pass_frozen_corridor_hmax_audit(loading: str, tier: str) -> None:
    benchmark_mesh = generate_fracture_benchmark_mesh(loading=loading, tier=tier)
    metadata = benchmark_mesh.metadata
    plan = benchmark_mesh.plan

    assert metadata["loading"] == loading
    assert metadata["tier"] == tier
    assert metadata["target_h_mm"] == plan.target_h_mm
    assert metadata["corridor_audited_element_count"] > 0
    assert metadata["notch_band_audited_element_count"] > 0
    assert metadata["propagation_corridor_audited_element_count"] > 0
    assert metadata["corridor_actual_hmax_mm"] <= 1.15 * plan.target_h_mm + 3.0e-14
    assert metadata["corridor_actual_hmax_mm"] <= metadata["corridor_hmax_limit_mm"]
    assert metadata["corridor_hmax_audit_passed"] is True
    assert len(benchmark_mesh.identity["topology_sha256"]) == 64


def test_topology_hash_is_deterministic_for_repeated_real_gmsh_generation() -> None:
    first = generate_fracture_benchmark_mesh(loading="sent", tier="coarse")
    second = generate_fracture_benchmark_mesh(loading="sent", tier="coarse")

    assert first.plan.plan_sha256 == second.plan.plan_sha256
    assert first.identity["topology_sha256"] == second.identity["topology_sha256"]
    assert first.recompute_topology_sha256() == first.identity["topology_sha256"]
    assert second.recompute_topology_sha256() == second.identity["topology_sha256"]
    assert first.metadata["node_count"] == second.metadata["node_count"]
    assert first.metadata["element_count"] == second.metadata["element_count"]


def test_identity_and_physical_metadata_are_immutable() -> None:
    benchmark_mesh = generate_fracture_benchmark_mesh(loading="sens", tier="coarse")

    assert isinstance(benchmark_mesh.identity, MappingProxyType)
    assert isinstance(benchmark_mesh.metadata, MappingProxyType)
    assert isinstance(benchmark_mesh.physical_tags, MappingProxyType)
    assert isinstance(benchmark_mesh.physical_entity_tags, MappingProxyType)
    with pytest.raises(TypeError):
        benchmark_mesh.identity["topology_sha256"] = "tampered"  # type: ignore[index]
    with pytest.raises(TypeError):
        benchmark_mesh.physical_tags[TOP] = 999  # type: ignore[index]
    with pytest.raises(ValueError):
        benchmark_mesh.nodes[0, 0] = -1.0
    with pytest.raises(ValueError):
        benchmark_mesh.boundary_facets[TOP][0] = -1
    with pytest.raises(ValueError):
        benchmark_mesh.boundary_nodes[NOTCH_TIP][0] = -1
    with pytest.raises(ValueError):
        benchmark_mesh.mesh.p[0, 0] = -1.0
    with pytest.raises(ValueError):
        benchmark_mesh.mesh.t[0, 0] = -1
    with pytest.raises(ValueError):
        benchmark_mesh.mesh.facets[0, 0] = -1
    with pytest.raises(ValueError):
        benchmark_mesh.mesh.f2t[0, 0] = -1
    with pytest.raises(TypeError):
        benchmark_mesh.mesh.boundaries[TOP] = np.asarray([0], dtype=np.int64)
    assert benchmark_mesh.recompute_topology_sha256() == benchmark_mesh.identity["topology_sha256"]


def test_topology_audit_fails_closed_if_the_two_slit_faces_are_merged() -> None:
    benchmark_mesh = generate_fracture_benchmark_mesh(loading="sent", tier="coarse")
    merged = dict(benchmark_mesh.boundary_facets)
    merged[NOTCH_LOWER] = merged[NOTCH_UPPER]

    # The real generator already passed.  Deliberately corrupting the label
    # topology demonstrates that a merged face cannot be silently accepted.
    with pytest.raises(RuntimeError, match="overlap"):
        _audit_zero_width_slit(
            mesh=benchmark_mesh.mesh,
            nodes=benchmark_mesh.nodes,
            elements=benchmark_mesh.elements,
            boundary_facets=merged,
            boundary_nodes=benchmark_mesh.boundary_nodes,
            interface_facets=np.asarray([], dtype=np.int64),
            plan=benchmark_mesh.plan,
        )


def test_invalid_gmsh_algorithm_is_rejected_before_generation() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        generate_fracture_benchmark_mesh(gmsh_algorithm=0)
