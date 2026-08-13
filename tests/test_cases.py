from __future__ import annotations

import json
from copy import deepcopy

import pytest

from tunnelgeopt.cases import (
    CaseValidationError,
    build_case_manifest,
    canonical_json,
    case_group_id,
    freeze_case_splits,
    inherit_derived_splits,
    largest_remainder_counts,
    load_case_manifest,
    verify_case_manifest,
    write_case_manifest,
)


def make_case(family: str = "circle", seed: int = 0) -> dict:
    return {
        "section_family": family,
        "section_parameters": {
            "radius": 1.0 + 0.01 * seed,
            "roughness_amplitude": 0.01,
        },
        "material_field_seed": seed,
        "joint_network_seed": 10_000 + seed,
        "dimensionless_material_parameters": {
            "young_modulus_ratio": 100.0,
            "poisson_ratio": 0.25,
        },
        "initial_stress_tensor": [[1.0, 0.1, 0.0], [0.1, 0.8, 0.0], [0.0, 0.0, 0.6]],
        "stress_orientation": 30.0,
        "excavation_schedule": [[0.0, 0.0], [1.0, 1.0]],
        "unloading_schedule": [[0.0, 1.0], [1.0, 0.0]],
    }


def test_canonical_json_is_order_independent_and_numeric_canonical() -> None:
    first = {"z": -0.0, "a": {"b": 1.0, "a": 2}}
    second = {"a": {"a": 2.0, "b": 1}, "z": 0}

    assert canonical_json(first) == canonical_json(second) == '{"a":{"a":2,"b":1},"z":0}'


def test_physics_changes_parent_id_but_derived_solver_fields_do_not() -> None:
    base = make_case()
    derived_variant = {
        **base,
        "mesh": {"element_size_over_radius": 0.05},
        "fidelity": "high",
        "solver_restart": 7,
    }
    material_variant = deepcopy(base)
    material_variant["dimensionless_material_parameters"]["poisson_ratio"] = 0.26

    assert case_group_id(base) == case_group_id(derived_variant)
    assert case_group_id(base) != case_group_id(material_variant)


def test_each_parent_identity_component_changes_id() -> None:
    base = make_case()
    variants = []

    section_family = deepcopy(base)
    section_family["section_family"] = "horseshoe"
    variants.append(section_family)
    section_parameters = deepcopy(base)
    section_parameters["section_parameters"]["radius"] = 1.1
    variants.append(section_parameters)
    material_seed = deepcopy(base)
    material_seed["material_field_seed"] = 1
    variants.append(material_seed)
    joint_seed = deepcopy(base)
    joint_seed["joint_network_seed"] += 1
    variants.append(joint_seed)
    material = deepcopy(base)
    material["dimensionless_material_parameters"]["young_modulus_ratio"] = 101.0
    variants.append(material)
    stress = deepcopy(base)
    stress["initial_stress_tensor"][0][0] = 1.1
    variants.append(stress)
    orientation = deepcopy(base)
    orientation["stress_orientation"] = 31.0
    variants.append(orientation)
    excavation = deepcopy(base)
    excavation["excavation_schedule"][1][1] = 0.9
    variants.append(excavation)
    unloading = deepcopy(base)
    unloading["unloading_schedule"][1][1] = 0.1
    variants.append(unloading)

    base_id = case_group_id(base)
    assert all(case_group_id(variant) != base_id for variant in variants)


def test_tensor_voigt_and_matrix_forms_have_same_identity() -> None:
    matrix = make_case()
    voigt = deepcopy(matrix)
    voigt["initial_stress_tensor"] = [1.0, 0.8, 0.6, 0.1, 0.0, 0.0]

    assert case_group_id(matrix) == case_group_id(voigt)


def test_largest_remainder_examples_are_frozen() -> None:
    assert largest_remainder_counts(6) == {"train": 4, "dev": 1, "locked_test": 1}
    assert largest_remainder_counts(128) == {"train": 90, "dev": 19, "locked_test": 19}
    assert sum(largest_remainder_counts(1).values()) == 1


def test_three_section_split_is_stratified_and_deterministic() -> None:
    cases = [
        make_case(family, family_index * 100 + seed)
        for family_index, family in enumerate(("circle", "horseshoe", "straight_wall_arch"))
        for seed in range(6)
    ]
    first = freeze_case_splits(cases)
    second = freeze_case_splits(list(reversed(cases)))

    assert first == second
    for family in ("circle", "horseshoe", "straight_wall_arch"):
        family_splits = [record["split"] for record in first if record["section_family"] == family]
        assert family_splits.count("train") == 4
        assert family_splits.count("dev") == 1
        assert family_splits.count("locked_test") == 1
        family_ids = [
            record["case_group_id"] for record in first if record["section_family"] == family
        ]
        assert family_ids == sorted(family_ids)


def test_duplicate_parent_case_is_rejected_even_with_different_mesh() -> None:
    first = make_case()
    duplicate_child_view = {**first, "mesh": "fine", "fidelity": "high"}

    with pytest.raises(CaseValidationError, match="duplicate parent"):
        freeze_case_splits([first, duplicate_child_view])


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda case: case["section_parameters"].__setitem__("radius", 0.0), "radius"),
        (
            lambda case: case["dimensionless_material_parameters"].__setitem__(
                "poisson_ratio", 0.5
            ),
            "poisson_ratio",
        ),
        (
            lambda case: case["initial_stress_tensor"][0].__setitem__(0, float("nan")),
            "finite",
        ),
        (lambda case: case.__setitem__("stress_orientation", 180.0), "orientation"),
        (lambda case: case.__setitem__("material_field_seed", -1), "material_field_seed"),
    ],
)
def test_non_finite_and_out_of_range_physics_is_rejected(mutate, message: str) -> None:
    case = make_case()
    mutate(case)

    with pytest.raises(CaseValidationError, match=message):
        case_group_id(case)


def test_asymmetric_stress_and_nonmonotone_schedule_time_are_rejected() -> None:
    asymmetric = make_case()
    asymmetric["initial_stress_tensor"][0][1] = 0.2
    with pytest.raises(CaseValidationError, match="symmetric"):
        case_group_id(asymmetric)

    reversed_time = make_case()
    reversed_time["excavation_schedule"] = [[1.0, 0.0], [0.0, 1.0]]
    with pytest.raises(CaseValidationError, match="strictly increasing"):
        case_group_id(reversed_time)


def test_signed_axis_orientation_and_relative_roughness_match_frozen_config() -> None:
    negative = make_case()
    negative["stress_orientation"] = -45.0
    negative["section_parameters"] = {
        "radius": 1.0,
        "roughness_amplitude_over_radius": 0.0,
    }
    equivalent = deepcopy(negative)
    equivalent["stress_orientation"] = 135.0

    assert case_group_id(negative) == case_group_id(equivalent)


def test_derived_records_inherit_parent_split_and_parent_id() -> None:
    parents = freeze_case_splits([make_case(seed=seed) for seed in range(6)])
    parent = parents[0]
    children = inherit_derived_splits(
        [
            {
                "case_group_id": parent["case_group_id"],
                "mesh": {"element_size_over_radius": 0.05},
                "fidelity": "b_elastic",
                "solver_restart": 2,
            }
        ],
        parents,
    )

    assert children[0]["case_group_id"] == parent["case_group_id"]
    assert children[0]["split"] == parent["split"]
    assert len(children[0]["derived_record_id"]) == 64
    assert len(children[0]["content_hash"]) == 64


def test_derived_split_conflict_or_orphan_and_duplicates_are_rejected() -> None:
    parents = freeze_case_splits([make_case(seed=seed) for seed in range(6)])
    parent = parents[0]
    record = {
        "case_group_id": parent["case_group_id"],
        "mesh": "coarse",
        "fidelity": "b_elastic",
    }
    wrong_split = "locked_test" if parent["split"] != "locked_test" else "train"
    with pytest.raises(CaseValidationError, match="conflicts"):
        inherit_derived_splits([{**record, "split": wrong_split}], parents)
    with pytest.raises(CaseValidationError, match="unknown parent"):
        inherit_derived_splits([{**record, "case_group_id": "0" * 64}], parents)
    with pytest.raises(CaseValidationError, match="duplicate derived"):
        inherit_derived_splits([record, record], parents)


def test_manifest_hashes_are_deterministic_and_round_trip(tmp_path) -> None:
    cases = [
        make_case(family, family_index * 10 + seed)
        for family_index, family in enumerate(("circle", "horseshoe", "straight_wall_arch"))
        for seed in range(6)
    ]
    first = build_case_manifest(cases, metadata={"generator_version": "test"})
    second = build_case_manifest(list(reversed(cases)), metadata={"generator_version": "test"})

    assert first == second
    assert len(first["content_hash"]) == 64
    assert len(first["manifest_hash"]) == 64
    assert first["content_hash"] != first["manifest_hash"]
    verify_case_manifest(first)

    path = write_case_manifest(tmp_path / "cases.json", first)
    assert load_case_manifest(path) == first
    assert json.loads(path.read_text(encoding="utf-8")) == first


def test_manifest_tampering_is_detected_at_record_and_envelope_levels() -> None:
    manifest = build_case_manifest([make_case(seed=seed) for seed in range(6)])
    split_tampered = deepcopy(manifest)
    split_tampered["cases"][0]["split"] = "locked_test"
    with pytest.raises(CaseValidationError, match="case records"):
        verify_case_manifest(split_tampered)

    metadata_tampered = deepcopy(manifest)
    metadata_tampered["metadata"]["untracked"] = True
    with pytest.raises(CaseValidationError, match="content_hash"):
        verify_case_manifest(metadata_tampered)
