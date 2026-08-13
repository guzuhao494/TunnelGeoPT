from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from tunnelgeopt.elastic_schema import (
    ARRAY_KEYS,
    SI_UNITS,
    ElasticSchemaValidationError,
    elastic_record_from_result,
    load_elastic_record,
    save_elastic_record,
    save_elastic_result,
)

CASE_GROUP_ID = "a" * 64
MESH_ID = "b" * 64
CONFIG_HASH = "c" * 64


@pytest.fixture
def elastic_result() -> SimpleNamespace:
    nodes = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
    elements = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    gradient = np.asarray([[0.03, -0.02], [0.04, 0.01]])
    displacement = nodes @ gradient.T + np.asarray([0.2, -0.4])
    strain = np.tile(np.asarray([0.03, 0.01, 0.02]), (2, 1))
    delta_stress = np.tile(np.asarray([3.0, 1.0, 1.0]), (2, 1))
    sigma_inf = np.asarray([[-10.0, 2.0], [2.0, -5.0]])
    sigma_vector = np.asarray([-10.0, -5.0, 2.0])
    total_stress = delta_stress + sigma_vector
    energy_density = 0.5 * np.sum(strain * delta_stress, axis=1)
    area = np.asarray([0.5, 0.5])
    centers = nodes[elements].mean(axis=1)
    # Lexicographically sorted mesh facets are:
    # [0,1], [0,2], [0,3], [1,2], [2,3].
    return SimpleNamespace(
        nodes=nodes,
        elements=elements,
        displacement=displacement,
        strain=strain,
        delta_stress=delta_stress,
        total_stress=total_stress,
        sigma_inf=sigma_inf,
        sigma_xx=np.asarray([-2.0, -2.0]),
        energy_density=energy_density,
        element_area=area,
        element_centers=centers,
        energy=float(np.sum(energy_density * area)),
        external_work=0.12,
        algebraic_residual=1.0e-12,
        residual_norm=2.0e-12,
        energy_closure=3.0e-12,
        energy_discretization_error=4.0e-12,
        stiffness_symmetry_error=5.0e-16,
        boundary_facets={
            "wall": np.asarray([0, 3], dtype=np.int64),
            "farfield": np.asarray([2, 4], dtype=np.int64),
        },
        physical_tags={"rock": 1, "wall": 1, "farfield": 2},
        material={
            "young_modulus": 100.0,
            "poisson_ratio": 0.0,
            "lame_lambda": 0.0,
            "shear_modulus": 50.0,
        },
        sigma_xx_inf=-2.0,
        mesh_metadata={
            "formulation": "P1_vector_small_strain_plane_strain",
            "coordinate_order": ["y", "z"],
        },
    )


@pytest.fixture
def record(elastic_result: SimpleNamespace):
    return elastic_record_from_result(
        elastic_result,
        case_group_id=CASE_GROUP_ID,
        mesh_id=MESH_ID,
        config_hash=CONFIG_HASH,
        env={"python": "3.13.5", "solver": "synthetic-test"},
        meta={"section_family": "circle", "publication": "calculation"},
    )


def test_result_conversion_is_float64_and_preserves_explicit_boundary_topology(record) -> None:
    record.validate()
    assert record.dtype == np.dtype(np.float64)
    assert record.nodes.shape == (4, 2)
    assert record.elements.shape == (2, 3)
    assert record.wall_facets.shape == (2, 2)
    assert record.farfield_facets.shape == (2, 2)
    assert np.array_equal(record.wall_facets, [[0, 1], [1, 2]])
    assert np.array_equal(record.farfield_facets, [[0, 3], [2, 3]])
    assert record.u.shape == (4, 2)
    assert record.strain.shape == record.stress.shape == (2, 3)
    assert record.sigma_xx.shape == record.energy_density.shape == record.area.shape == (2,)
    assert record.centers.shape == (2, 2)
    assert set(record.arrays()) == set(ARRAY_KEYS)
    assert all(
        forbidden not in name
        for name in record.arrays()
        for forbidden in ("damage", "velocity", "dissipation")
    )


def test_save_load_roundtrip_checks_file_and_semantic_hashes(tmp_path, record) -> None:
    paths = save_elastic_record(tmp_path / "case", record)
    metadata = json.loads(paths.meta.read_text(encoding="utf-8"))

    assert len(metadata["arrays_file_sha256"]) == 64
    assert len(metadata["content_sha256"]) == 64
    assert len(metadata["mesh_content_sha256"]) == 64
    assert set(metadata["array_manifest"]) == set(ARRAY_KEYS)

    loaded = load_elastic_record(paths.case_dir)
    assert loaded.case_group_id == CASE_GROUP_ID
    assert loaded.mesh_id == MESH_ID
    assert loaded.config_hash == CONFIG_HASH
    assert loaded.env == record.env
    for name in ARRAY_KEYS:
        assert np.array_equal(getattr(loaded, name), getattr(record, name))


def test_default_is_strict_float64_and_float32_requires_explicit_publication(
    tmp_path, elastic_result
) -> None:
    record32 = elastic_record_from_result(
        elastic_result,
        case_group_id=CASE_GROUP_ID,
        config_hash=CONFIG_HASH,
        env={"purpose": "explicit-float32-publication"},
        publication_dtype=np.float32,
    )
    with pytest.raises(ElasticSchemaValidationError, match="expected float64"):
        record32.validate()
    record32.validate(expected_dtype=np.float32)

    paths = save_elastic_record(tmp_path / "case32", record32, expected_dtype=np.float32)
    with pytest.raises(ElasticSchemaValidationError, match="expected float64"):
        load_elastic_record(paths.case_dir)
    loaded = load_elastic_record(paths.case_dir, expected_dtype=np.float32)
    assert loaded.dtype == np.dtype(np.float32)


def test_existing_record_is_protected_and_overwrite_replaces_both_files(tmp_path, record) -> None:
    case_dir = tmp_path / "case"
    save_elastic_record(case_dir, record)
    changed = replace(record, env={"solver": "replacement"})
    with pytest.raises(FileExistsError, match="protected"):
        save_elastic_record(case_dir, changed)

    save_elastic_record(case_dir, changed, overwrite=True)
    assert load_elastic_record(case_dir).env == {"solver": "replacement"}
    assert not (case_dir / ".elastic-schema.lock").exists()


def test_arrays_file_corruption_is_detected_before_loading(tmp_path, record) -> None:
    paths = save_elastic_record(tmp_path / "case", record)
    with paths.arrays.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ElasticSchemaValidationError, match="SHA-256"):
        load_elastic_record(paths.case_dir)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda item: replace(item, sign_convention="compression_positive"),
            "sign_convention",
        ),
        (
            lambda item: replace(item, stress_component_order=("zz", "yy", "yz")),
            "stress_component_order",
        ),
        (
            lambda item: replace(item, units={**SI_UNITS, "stress": "MPa"}),
            "SI unit contract",
        ),
        (
            lambda item: replace(
                item,
                stress=np.asarray(item.stress) + np.asarray([1.0, 0.0, 0.0]),
            ),
            "stress must be total",
        ),
        (
            lambda item: replace(
                item,
                elements=np.asarray([[0, 1, 8], [0, 2, 3]], dtype=np.int64),
            ),
            "outside",
        ),
        (
            lambda item: replace(item, wall_facets=np.asarray([[0, 2]], dtype=np.int64)),
            "complete mesh boundary",
        ),
        (
            lambda item: replace(item, sigma_xx=np.asarray([np.nan, -2.0])),
            "non-finite",
        ),
    ],
)
def test_shape_index_finite_component_sign_and_unit_checks(record, mutator, message) -> None:
    invalid = mutator(record)
    with pytest.raises(ElasticSchemaValidationError, match=message):
        invalid.validate()


@pytest.mark.parametrize("forbidden", ["damage", "velocity", "total_dissipation"])
def test_non_elastic_placeholder_fields_are_rejected(record, forbidden) -> None:
    invalid = replace(record, meta={forbidden: 0.0})
    with pytest.raises(ElasticSchemaValidationError, match="outside the linear-elastic schema"):
        invalid.validate()


def test_save_elastic_result_convenience_path(tmp_path, elastic_result) -> None:
    paths = save_elastic_result(
        tmp_path / "from-result",
        elastic_result,
        case_group_id=CASE_GROUP_ID,
        mesh_id=MESH_ID,
        config_hash=CONFIG_HASH,
        env={"test": True},
    )
    loaded = load_elastic_record(paths.case_dir)
    assert loaded.meta == {}
    assert loaded.diagnostics["energy"] == pytest.approx(0.06)
