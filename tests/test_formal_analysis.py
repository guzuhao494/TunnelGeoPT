from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import tunnelgeopt.formal_analysis as analysis

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "multifidelity_formal.json"
SECTIONS = ("circle", "horseshoe", "straight_wall_arch")
REAL_BOOTSTRAP_RATIO = analysis._bootstrap_ratio  # type: ignore[attr-defined]


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _key(method: str, fraction: float, seed: int) -> str:
    return f"{method}__f{round(fraction * 100):03d}__seed{seed}"


def _specs(config: dict[str, Any]) -> list[tuple[str, float, int]]:
    return [
        (method, float(fraction), int(seed))
        for seed in config["learning"]["training_seeds"]
        for method, fractions in config["learning"]["method_fraction_matrix"].items()
        for fraction in fractions
    ]


def _case_layout(config: dict[str, Any], partition: str) -> tuple[list[str], ...]:
    spec = config["dataset"]["partitions"][partition]
    cases: list[str] = []
    parents: list[str] = []
    sections: list[str] = []
    subtypes: list[str] = []
    ood_subtypes = tuple(config["material_and_loads"]["load_ood_subtypes"])
    for section in SECTIONS:
        for parent_index in range(spec["parents_per_section"]):
            parent = f"{partition}-{section}-parent-{parent_index:03d}"
            for load_index in range(spec["loads_per_parent"]):
                cases.append(f"{parent}-load-{load_index:02d}")
                parents.append(parent)
                sections.append(section)
                if partition in {"locked_load_ood", "locked_joint_ood"}:
                    subtypes.append(ood_subtypes[load_index % len(ood_subtypes)])
                else:
                    subtypes.append("id")
    return cases, parents, sections, subtypes


def _solver_record(case: str, partition: str, section: str) -> dict[str, Any]:
    fidelity = {
        "nonfinite_fraction": 0.0,
        "free_dof_algebraic_residual": 1e-11,
        "clapeyron_relative_energy_error": 1e-11,
        "min_triangle_signed_area_over_radius_squared": 1e-6,
        "min_triangle_quality": 0.10,
        "explicit_wall_and_farfield_tags": True,
        "no_element_centroid_inside_cavity": True,
        "same_boundary_hash_and_outer_bounds": True,
        "all_query_points_located": True,
    }
    return {
        "case_group_id": case,
        "partition": partition,
        "section_family": section,
        "valid": True,
        "fidelities": {"coarse": dict(fidelity), "fine": dict(fidelity)},
    }


def _passing_evidence() -> tuple[dict[str, Any], ...]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config_hash = analysis.canonical_sha256(config)
    run_id = config["run_id"]
    specifications = _specs(config)
    registry_records = {
        _key(method, fraction, seed): {
            "sha256": _sha(f"checkpoint:{method}:{fraction}:{seed}"),
            "training_contract_sha256": _sha(f"contract:{method}:{fraction}:{seed}"),
            "config_sha256": config_hash,
        }
        for method, fraction, seed in specifications
    }
    registry_hash = analysis.canonical_sha256(
        {
            "identity": "tunnelgeopt.checkpoint_registry.v1",
            "checkpoint_ids": [
                registry_records[_key(method, fraction, seed)]["sha256"]
                for method, fraction, seed in specifications
            ],
        }
    )

    all_records = []
    all_case_metadata: list[tuple[str, str, str]] = []
    partition_layouts: dict[str, tuple[list[str], ...]] = {}
    for partition in config["dataset"]["partitions"]:
        layout = _case_layout(config, partition)
        partition_layouts[partition] = layout
        for case, section in zip(layout[0], layout[2], strict=True):
            all_records.append(_solver_record(case, partition, section))
            all_case_metadata.append((case, partition, section))

    # Exactly ceil(20%) inside every partition-section cell: 144 total.
    selected: list[tuple[str, str, str]] = []
    for partition, spec in config["dataset"]["partitions"].items():
        cases, _, sections, _ = partition_layouts[partition]
        per_cell = int(np.ceil(0.2 * spec["parents_per_section"] * spec["loads_per_parent"]))
        for section in SECTIONS:
            matching = [
                case for case, observed in zip(cases, sections, strict=True) if observed == section
            ]
            selected.extend((case, partition, section) for case in matching[:per_cell])

    dataset_manifest: dict[str, Any] = {
        "schema": "tunnelgeopt.formal_dataset_manifest.v1",
        "run_id": run_id,
        "config_sha256": config_hash,
        "identities": {
            "cross_partition_zero_intersection": True,
            "legacy_v0_2_locked_test_zero_intersection": True,
            "normalization_fit_train_only": True,
            "no_result_conditioned_replacement": True,
        },
        "artifact_hashes": {
            "geometry_manifest": _sha("geometry"),
            "case_manifest": _sha("case"),
            "query_manifest": _sha("query"),
            "public_input_store": _sha("public"),
            "train_dev_label_store": _sha("train-dev"),
            **{
                f"sealed_{partition}_label_store": _sha(f"sealed:{partition}")
                for partition in analysis.LOCKED_PARTITIONS
            },
        },
        "solver_mesh_qc": {
            "no_silent_case_replacement": True,
            "records": all_records,
        },
        "fine_ultrafine_selection": {
            "formal_audit_fraction": 0.2,
            "selection_protocol": config["quality_control"]["fine_ultrafine"]["selection_protocol"],
            "selection_unit": "case_group_id",
            "selected_before_any_ultrafine_label": True,
            "case_values_exposed_before_checkpoint_freeze": False,
            "selected_case_ids": [value[0] for value in selected],
        },
    }

    sealed_metrics: dict[str, Any] = {
        "schema": "tunnelgeopt.multifidelity.sealed_metrics.v1",
        "run_id": run_id,
        "config_sha256": config_hash,
        "backend": "formal",
        "effect_claim_allowed": True,
        "registry_hash": registry_hash,
        "partitions": {},
        "fine_ultrafine_audit": {
            "case_group_ids": [value[0] for value in selected],
            "partitions": [value[1] for value in selected],
            "section_families": [value[2] for value in selected],
            "relative_errors": [0.02] * len(selected),
        },
        "resource_usage": {
            "generation": {"runtime_seconds": 1.0, "peak_memory_bytes": 1024},
            "training": {
                _key(method, fraction, seed): {
                    "runtime_seconds": 1.0,
                    "peak_memory_bytes": 2048,
                }
                for method, fraction, seed in specifications
            },
            "evaluation": {
                partition: {"runtime_seconds": 1.0, "peak_memory_bytes": 1024}
                for partition in analysis.LOCKED_PARTITIONS
            },
        },
    }
    error_by_method = {
        "scratch": 0.10,
        "direct_coarse": 0.10,
        "residual_coarse": 0.055,
        "mismatched_coarse": 0.13,
    }
    for partition in analysis.LOCKED_PARTITIONS:
        cases, parents, sections, subtypes = partition_layouts[partition]
        size = len(cases)
        checkpoints = {}
        for method, fraction, seed in specifications:
            key = _key(method, fraction, seed)
            # Tiny deterministic within-parent variation keeps summary reports non-degenerate
            # while every ratio remains exactly its intended constant.
            base = error_by_method[method]
            if method == "residual_coarse":
                base += 0.02 * (1.0 - fraction)
            values = [base * (1.0 + 0.001 * (index % 7)) for index in range(size)]
            checkpoints[key] = {
                "method": method,
                "fine_fraction": fraction,
                "seed": seed,
                "checkpoint_sha256": registry_records[key]["sha256"],
                "case_errors": values,
                "nonfinite_prediction_count": 0,
            }
        wall_residual = {
            "by_seed": {
                str(seed): {
                    "traction_discrepancy_by_case": [0.04] * size,
                    "resultant_discrepancy_by_case": [0.02] * size,
                }
                for seed in config["learning"]["training_seeds"]
            }
        }
        sealed_metrics["partitions"][partition] = {
            "case_group_ids": cases,
            "geometry_group_ids": parents,
            "section_families": sections,
            "load_subtypes": subtypes,
            "coarse_only_case_errors": [0.10] * size,
            "checkpoints": checkpoints,
            "checkpoint_evaluation_counts": {key: 1 for key in checkpoints},
            "wall_offset_physics": {
                "coarse_only": {
                    "traction_discrepancy_by_case": [0.08] * size,
                    "resultant_discrepancy_by_case": [0.04] * size,
                },
                "residual_coarse_0.5": wall_residual,
            },
        }

    source_sha256 = {
        path: (_sha("config-file") if path == "configs/multifidelity_formal.json" else _sha(path))
        for path in analysis.IMPLEMENTATION_SOURCE_PATHS
    }
    implementation_manifest = {
        "schema": "tunnelgeopt.formal_implementation_manifest.v1",
        "run_id": run_id,
        "config_sha256": config_hash,
        "effect_claim_allowed": True,
        "recorded_at_utc": "2026-08-13T00:00:00+00:00",
        "source_provenance": {
            "git_head": "1" * 40,
            "upstream_ref": "origin/codex/formal-v0.3",
            "upstream_head": "1" * 40,
            "head_matches_upstream": True,
            "worktree_clean_before_prepare": True,
            "remote_url_sanitized": "https://github.com/example/TunnelGeoPT.git",
            "all_sources_tracked": True,
            "source_sha256": source_sha256,
        },
        "environment": {
            "python": "3.12.0",
            "platform": "Windows-11",
            "numpy": "2.0.0",
            "scipy": "1.14.0",
            "skfem": "11.0.0",
            "gmsh": "4.13.0",
            "torch": "2.8.0+cu128",
            "cuda_runtime": "12.8",
            "cuda_available": True,
            "device_requested": "cuda",
            "device_name": "NVIDIA GeForce RTX 5070 Ti Laptop GPU",
            "device_total_memory_bytes": 12 * 1024**3,
            "driver_version": "580.00",
        },
    }

    access_state: dict[str, Any] = {
        "schema": "tunnelgeopt.formal_analysis_access_state.v1",
        "run_id": run_id,
        "config_sha256": config_hash,
        "config_frozen_before_generation": True,
        "locked_labels_opened_before_checkpoint_freeze": False,
        "locked_labels_used_for_tuning": False,
        "trainer_received_locked_label_path": False,
        "access_log_append_only": True,
        "denied_premature_sealed_accesses": 0,
        "sealed_partition_open_counts": {partition: 1 for partition in analysis.LOCKED_PARTITIONS},
        "checkpoint_evaluation_counts": {
            f"{partition}:{key}": 1
            for partition in analysis.LOCKED_PARTITIONS
            for key in registry_records
        },
        "checkpoint_registry": {
            "frozen": True,
            "config_sha256": config_hash,
            "checkpoint_count": 35,
            "registry_hash": registry_hash,
            "checkpoints": registry_records,
        },
        "abstain_reasons": [],
        "implementation_manifest": implementation_manifest,
        "hashes": {},
    }
    access_state["hashes"] = {
        "config_canonical_sha256": config_hash,
        "dataset_manifest_canonical_sha256": analysis.canonical_sha256(dataset_manifest),
        "sealed_metrics_canonical_sha256": analysis.canonical_sha256(sealed_metrics),
        "config_file_sha256": _sha("config-file"),
        "dataset_manifest_file_sha256": _sha("dataset-file"),
        "sealed_metrics_file_sha256": _sha("metrics-file"),
        "access_log_file_sha256": _sha("access-log"),
        "checkpoint_manifest_file_sha256": _sha("checkpoint-manifest"),
        "checkpoint_registry_file_sha256": _sha("checkpoint-registry"),
        "implementation_manifest_file_sha256": analysis._canonical_json_file_sha256(  # type: ignore[attr-defined]
            implementation_manifest
        ),
        "prepare_manifest_file_sha256": _sha("prepare-manifest"),
    }
    return config, sealed_metrics, dataset_manifest, access_state


@pytest.fixture(scope="module")
def evidence() -> tuple[dict[str, Any], ...]:
    return _passing_evidence()


@pytest.fixture(autouse=True)
def fast_deterministic_bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Preserve gate semantics while making 13 mutation tests fast.

    A separate test exercises the real 20,000-replicate implementation and its
    parent/load aggregation.  Mutation tests need only isolate decision gates.
    """

    def deterministic(
        candidate: np.ndarray,
        reference: np.ndarray,
        geometry_ids: list[str],
        sections: list[str],
        section_order: list[str],
        *,
        replicates: int,
        seed: int,
    ) -> dict[str, float | int]:
        del seed
        numerator, *_ = analysis._aggregate_parent_section(  # type: ignore[attr-defined]
            candidate, geometry_ids, sections, section_order
        )
        denominator, *_ = analysis._aggregate_parent_section(  # type: ignore[attr-defined]
            reference, geometry_ids, sections, section_order
        )
        ratio = numerator / denominator
        return {
            "center_ratio": ratio,
            "lower_95": ratio,
            "upper_95": ratio,
            "one_sided_upper_95": ratio,
            "interval_width": 0.0,
            "replicates": replicates,
        }

    monkeypatch.setattr(analysis, "_bootstrap_ratio", deterministic)


def _run(values: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    return analysis.evaluate_formal_decision(*copy.deepcopy(values))


def _rebind_config(values: tuple[dict[str, Any], ...]) -> None:
    config, metrics, manifest, state = values
    digest = analysis.canonical_sha256(config)
    metrics["config_sha256"] = digest
    manifest["config_sha256"] = digest
    state["config_sha256"] = digest
    state["checkpoint_registry"]["config_sha256"] = digest
    for record in state["checkpoint_registry"]["checkpoints"].values():
        record["config_sha256"] = digest
    state["hashes"]["config_canonical_sha256"] = digest
    state["hashes"]["dataset_manifest_canonical_sha256"] = analysis.canonical_sha256(manifest)
    state["hashes"]["sealed_metrics_canonical_sha256"] = analysis.canonical_sha256(metrics)


def _rebind_implementation(state: dict[str, Any]) -> None:
    state["hashes"]["implementation_manifest_file_sha256"] = analysis._canonical_json_file_sha256(  # type: ignore[attr-defined]
        state["implementation_manifest"]
    )


def _residual_records(metrics: dict[str, Any], partition: str) -> list[dict[str, Any]]:
    return [
        record
        for record in metrics["partitions"][partition]["checkpoints"].values()
        if record["method"] == "residual_coarse" and record["fine_fraction"] == 0.5
    ]


def test_complete_passing_fixture_is_go_and_joint_ood_is_report_only(evidence) -> None:
    assert len(analysis.IMPLEMENTATION_SOURCE_PATHS) == 20
    assert {
        "src/tunnelgeopt/__init__.py",
        "src/tunnelgeopt/cases.py",
        "src/tunnelgeopt/elastic_schema.py",
        "src/tunnelgeopt/elastic_validation.py",
        "src/tunnelgeopt/kirsch.py",
        "src/tunnelgeopt/lift.py",
        "src/tunnelgeopt/schema.py",
    }.issubset(analysis.IMPLEMENTATION_SOURCE_PATHS)
    decision = _run(evidence)
    assert decision["classification"] == "GO"
    assert decision["effect_claim_allowed"] is True
    assert decision["mandatory_reports"]["complete"] is True
    assert decision["mandatory_reports"]["dataset_qc"]["invalid_cases"] == []
    assert decision["results"]["locked_joint_ood"]["report_only"] is True
    assert decision["decision_payload_sha256"] == analysis.canonical_sha256(
        {key: value for key, value in decision.items() if key != "decision_payload_sha256"}
    )


def test_mocked_formal_runner_40hex_provenance_and_file_encoding_are_accepted(
    evidence, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = ROOT / "scripts" / "run_multifidelity_formal.py"
    spec = importlib.util.spec_from_file_location("formal_runner_for_analysis_test", script)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)
    instance = runner.FormalExperimentRunner(
        config_path=CONFIG_PATH,
        approval_path=ROOT / "configs" / "multifidelity_formal_approval.json",
        exclusions_path=ROOT / "configs" / "multifidelity_seen_identity_exclusions.json",
        output_dir=tmp_path / "formal-provenance",
        backend="formal",
        device="cuda",
    )
    head = "a" * 40

    def fake_git(*arguments: str, description: str) -> str:
        del description
        if arguments in (("rev-parse", "HEAD"), ("rev-parse", "@{upstream}")):
            return head
        if arguments == (
            "rev-parse",
            "--abbrev-ref",
            "--symbolic-full-name",
            "@{upstream}",
        ):
            return "origin/codex/v0.3-multifidelity"
        if arguments[:3] == ("status", "--porcelain=v1", "--untracked-files=all"):
            return ""
        if arguments[:2] == ("ls-files", "--"):
            return "\n".join(runner.IMPLEMENTATION_SOURCE_PATHS)
        if arguments[:2] == ("remote", "get-url"):
            return "https://token@example.invalid/org/repo.git"
        raise AssertionError(arguments)

    monkeypatch.setattr(runner, "_git_output", fake_git)
    monkeypatch.setattr(runner, "_module_version", lambda name: f"{name}-test")
    monkeypatch.setattr(
        runner,
        "_cuda_environment",
        lambda: {
            "torch": "2.11.0+cu128",
            "cuda_runtime": "12.8",
            "cuda_available": True,
            "device_name": "NVIDIA GeForce RTX 5070 Ti Laptop GPU",
            "device_total_memory_bytes": 12_820_480_000,
            "driver_version": "596.49",
        },
    )
    implementation = instance._implementation_manifest()
    assert implementation["source_provenance"]["git_head"] == head
    assert implementation["source_provenance"]["upstream_head"] == head
    implementation_path = tmp_path / "implementation_manifest.json"
    implementation_digest = runner._atomic_json(implementation_path, implementation)
    assert (
        implementation_digest
        == hashlib.sha256(runner._canonical_bytes(implementation) + b"\n").hexdigest()
    )

    values = copy.deepcopy(evidence)
    values[3]["implementation_manifest"] = implementation
    values[3]["hashes"]["config_file_sha256"] = runner._file_sha256(CONFIG_PATH)
    values[3]["hashes"]["implementation_manifest_file_sha256"] = implementation_digest
    decision = analysis.evaluate_formal_decision(*values)
    assert decision["classification"] == "GO"


def test_primary_upper95_failure_is_no_go(evidence) -> None:
    values = copy.deepcopy(evidence)
    for record in _residual_records(values[1], "locked_iid"):
        record["case_errors"] = [0.11] * len(record["case_errors"])
    values[3]["hashes"]["sealed_metrics_canonical_sha256"] = analysis.canonical_sha256(values[1])
    decision = analysis.evaluate_formal_decision(*values)
    assert decision["classification"] == "NO_GO"
    assert any(
        item["code"] == "PRIMARY_UPPER95_GATE_FAILED" for item in decision["effect_failures"]
    )


def test_seed_stability_requires_four_same_seed_rs_and_rd(evidence) -> None:
    values = copy.deepcopy(evidence)
    records = _residual_records(values[1], "locked_geometry_ood")
    for record in records[:2]:
        record["case_errors"] = [0.12] * len(record["case_errors"])
    values[3]["hashes"]["sealed_metrics_canonical_sha256"] = analysis.canonical_sha256(values[1])
    decision = analysis.evaluate_formal_decision(*values)
    assert decision["classification"] == "NO_GO"
    assert any(item["code"] == "SEED_STABILITY_GATE_FAILED" for item in decision["effect_failures"])


def test_iid_section_max_and_two_of_three_strict_gates(evidence) -> None:
    values = copy.deepcopy(evidence)
    partition = values[1]["partitions"]["locked_iid"]
    for record in _residual_records(values[1], "locked_iid"):
        record["case_errors"] = [
            (0.111 if section == "circle" else 0.103) for section in partition["section_families"]
        ]
    values[3]["hashes"]["sealed_metrics_canonical_sha256"] = analysis.canonical_sha256(values[1])
    decision = analysis.evaluate_formal_decision(*values)
    codes = {item["code"] for item in decision["effect_failures"]}
    assert decision["classification"] == "NO_GO"
    assert "IID_SECTION_MAX_GATE_FAILED" in codes
    assert "IID_SECTION_STRICT_COUNT_GATE_FAILED" in codes


def test_each_load_ood_subtype_is_gated_against_both_full_baselines(evidence) -> None:
    values = copy.deepcopy(evidence)
    partition = values[1]["partitions"]["locked_load_ood"]
    for record in _residual_records(values[1], "locked_load_ood"):
        record["case_errors"] = [
            0.12 if subtype == "large_rotation" else 0.06 for subtype in partition["load_subtypes"]
        ]
    values[3]["hashes"]["sealed_metrics_canonical_sha256"] = analysis.canonical_sha256(values[1])
    decision = analysis.evaluate_formal_decision(*values)
    assert decision["classification"] == "NO_GO"
    assert any(
        item["code"] == "LOAD_OOD_SUBTYPE_GATE_FAILED" for item in decision["effect_failures"]
    )


@pytest.mark.parametrize("mode", ["absolute_cap", "coarse_nonworsening"])
def test_wall_offset_absolute_and_coarse_relative_gates_are_both_required(evidence, mode) -> None:
    values = copy.deepcopy(evidence)
    wall = values[1]["partitions"]["locked_iid"]["wall_offset_physics"]
    if mode == "absolute_cap":
        for item in wall["residual_coarse_0.5"]["by_seed"].values():
            item["traction_discrepancy_by_case"] = [0.101] * len(
                item["traction_discrepancy_by_case"]
            )
    else:
        wall["coarse_only"]["traction_discrepancy_by_case"] = [0.01] * len(
            wall["coarse_only"]["traction_discrepancy_by_case"]
        )
        for item in wall["residual_coarse_0.5"]["by_seed"].values():
            item["traction_discrepancy_by_case"] = [0.02] * len(
                item["traction_discrepancy_by_case"]
            )
    values[3]["hashes"]["sealed_metrics_canonical_sha256"] = analysis.canonical_sha256(values[1])
    decision = analysis.evaluate_formal_decision(*values)
    assert decision["classification"] == "NO_GO"
    assert any(item["code"] == "WALL_OFFSET_GATE_FAILED" for item in decision["effect_failures"])


def test_nonfinite_or_missing_metric_is_abstain(evidence) -> None:
    values = copy.deepcopy(evidence)
    record = next(iter(values[1]["partitions"]["locked_iid"]["checkpoints"].values()))
    record["nonfinite_prediction_count"] = 1
    values[3]["hashes"]["sealed_metrics_canonical_sha256"] = analysis.canonical_sha256(values[1])
    decision = analysis.evaluate_formal_decision(*values)
    assert decision["classification"] == "ABSTAIN"
    assert decision["effect_claim_allowed"] is False


def test_solver_mesh_partition_section_rate_below_95_is_abstain(evidence) -> None:
    values = copy.deepcopy(evidence)
    records = [
        record
        for record in values[2]["solver_mesh_qc"]["records"]
        if record["partition"] == "locked_iid" and record["section_family"] == "circle"
    ]
    for record in records[:3]:
        record["valid"] = False
    values[3]["hashes"]["dataset_manifest_canonical_sha256"] = analysis.canonical_sha256(values[2])
    decision = analysis.evaluate_formal_decision(*values)
    assert decision["classification"] == "ABSTAIN"
    assert "95%" in decision["validity_failures"][0]["detail"]


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("free_dof_algebraic_residual", 2e-9),
        ("clapeyron_relative_energy_error", 2e-9),
        ("min_triangle_signed_area_over_radius_squared", 1e-13),
        ("min_triangle_quality", 0.019),
        ("all_query_points_located", False),
    ],
)
def test_each_numeric_solver_mesh_gate_can_force_abstain(evidence, field, bad_value) -> None:
    values = copy.deepcopy(evidence)
    record = values[2]["solver_mesh_qc"]["records"][0]
    for fidelity in record["fidelities"].values():
        fidelity[field] = bad_value
    # A single train-circle failure remains above 95%, so fail the whole cell.
    for record in values[2]["solver_mesh_qc"]["records"]:
        if record["partition"] == "train_id" and record["section_family"] == "circle":
            for fidelity in record["fidelities"].values():
                fidelity[field] = bad_value
    values[3]["hashes"]["dataset_manifest_canonical_sha256"] = analysis.canonical_sha256(values[2])
    assert analysis.evaluate_formal_decision(*values)["classification"] == "ABSTAIN"


@pytest.mark.parametrize(
    ("values", "expected_text"),
    [
        ([0.031] * 144, "3%/5%/4%"),
        ([0.01] * 136 + [0.10] * 8, "3%/5%/4%"),
    ],
)
def test_fine_ultrafine_overall_median_and_p95_are_abstain(evidence, values, expected_text) -> None:
    mutated = copy.deepcopy(evidence)
    mutated[1]["fine_ultrafine_audit"]["relative_errors"] = values
    mutated[3]["hashes"]["sealed_metrics_canonical_sha256"] = analysis.canonical_sha256(mutated[1])
    decision = analysis.evaluate_formal_decision(*mutated)
    assert decision["classification"] == "ABSTAIN"
    assert expected_text in decision["validity_failures"][0]["detail"]


def test_fine_ultrafine_any_section_median_above_four_percent_is_abstain(evidence) -> None:
    values = copy.deepcopy(evidence)
    audit = values[1]["fine_ultrafine_audit"]
    audit["relative_errors"] = [
        0.041 if section == "circle" else 0.01 for section in audit["section_families"]
    ]
    values[3]["hashes"]["sealed_metrics_canonical_sha256"] = analysis.canonical_sha256(values[1])
    assert analysis.evaluate_formal_decision(*values)["classification"] == "ABSTAIN"


@pytest.mark.parametrize(
    "mutation",
    ["premature_open", "repeat_partition", "repeat_checkpoint", "registry_contract_hash"],
)
def test_access_checkpoint_and_evaluation_contract_failures_are_abstain(evidence, mutation) -> None:
    values = copy.deepcopy(evidence)
    state = values[3]
    if mutation == "premature_open":
        state["locked_labels_opened_before_checkpoint_freeze"] = True
    elif mutation == "repeat_partition":
        state["sealed_partition_open_counts"]["locked_iid"] = 2
    elif mutation == "repeat_checkpoint":
        first = next(iter(state["checkpoint_evaluation_counts"]))
        state["checkpoint_evaluation_counts"][first] = 2
    else:
        first = next(iter(state["checkpoint_registry"]["checkpoints"].values()))
        first["training_contract_sha256"] = "bad"
    assert analysis.evaluate_formal_decision(*values)["classification"] == "ABSTAIN"


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_manifest_hash",
        "manifest_digest_mismatch",
        "dirty_worktree",
        "unpushed_head",
        "untracked_source",
        "missing_critical_source",
        "config_source_mismatch",
        "cuda_unavailable",
        "missing_environment_field",
    ],
)
def test_source_provenance_contract_failures_are_abstain(evidence, mutation) -> None:
    values = copy.deepcopy(evidence)
    state = values[3]
    implementation = state["implementation_manifest"]
    provenance = implementation["source_provenance"]
    if mutation == "missing_manifest_hash":
        state["hashes"].pop("prepare_manifest_file_sha256")
    elif mutation == "manifest_digest_mismatch":
        state["hashes"]["implementation_manifest_file_sha256"] = _sha("wrong")
    elif mutation == "dirty_worktree":
        provenance["worktree_clean_before_prepare"] = False
        _rebind_implementation(state)
    elif mutation == "unpushed_head":
        provenance["upstream_head"] = "2" * 40
        provenance["head_matches_upstream"] = False
        _rebind_implementation(state)
    elif mutation == "untracked_source":
        provenance["all_sources_tracked"] = False
        _rebind_implementation(state)
    elif mutation == "missing_critical_source":
        provenance["source_sha256"].pop("src/tunnelgeopt/field_sampling.py")
        _rebind_implementation(state)
    elif mutation == "config_source_mismatch":
        provenance["source_sha256"]["configs/multifidelity_formal.json"] = _sha("wrong")
        _rebind_implementation(state)
    elif mutation == "cuda_unavailable":
        implementation["environment"]["cuda_available"] = False
        _rebind_implementation(state)
    else:
        implementation["environment"].pop("driver_version")
        _rebind_implementation(state)
    decision = analysis.evaluate_formal_decision(*values)
    assert decision["classification"] == "ABSTAIN"
    assert decision["effect_claim_allowed"] is False


def test_missing_learning_curve_or_resource_field_is_abstain(evidence) -> None:
    values = copy.deepcopy(evidence)
    values[1]["resource_usage"]["training"].pop(next(iter(values[1]["resource_usage"]["training"])))
    values[3]["hashes"]["sealed_metrics_canonical_sha256"] = analysis.canonical_sha256(values[1])
    assert analysis.evaluate_formal_decision(*values)["classification"] == "ABSTAIN"


def test_primary_ci_width_above_point_one_is_abstain(
    evidence, monkeypatch: pytest.MonkeyPatch
) -> None:
    def wide(*args, replicates: int, **kwargs) -> dict[str, float | int]:
        del args, kwargs
        return {
            "center_ratio": 0.6,
            "lower_95": 0.54,
            "upper_95": 0.66,
            "one_sided_upper_95": 0.65,
            "interval_width": 0.12,
            "replicates": replicates,
        }

    monkeypatch.setattr(analysis, "_bootstrap_ratio", wide)
    decision = _run(evidence)
    assert decision["classification"] == "ABSTAIN"
    assert any(
        item["code"] == "BOOTSTRAP_INTERVAL_TOO_WIDE" for item in decision["validity_failures"]
    )


@pytest.mark.parametrize("mutation", ["registry_hash", "config_gate"])
def test_registry_and_frozen_config_gate_authentication_fail_closed(evidence, mutation) -> None:
    values = copy.deepcopy(evidence)
    if mutation == "registry_hash":
        values[3]["checkpoint_registry"]["registry_hash"] = _sha("wrong-registry")
    else:
        values[0]["scientific_decision"]["upper_95_ci_gates"]["locked_iid"]["R_s"] = 9.0
        _rebind_config(values)
    assert analysis.evaluate_formal_decision(*values)["classification"] == "ABSTAIN"


def test_joint_ood_effect_and_ci_width_do_not_change_primary_go(
    evidence, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = copy.deepcopy(evidence)
    for record in _residual_records(values[1], "locked_joint_ood"):
        record["case_errors"] = [100.0] * len(record["case_errors"])
    values[3]["hashes"]["sealed_metrics_canonical_sha256"] = analysis.canonical_sha256(values[1])

    def joint_wide(
        candidate: np.ndarray,
        reference: np.ndarray,
        geometry_ids: list[str],
        sections: list[str],
        section_order: list[str],
        *,
        replicates: int,
        seed: int,
    ) -> dict[str, float | int]:
        del seed
        numerator, *_ = analysis._aggregate_parent_section(  # type: ignore[attr-defined]
            candidate, geometry_ids, sections, section_order
        )
        denominator, *_ = analysis._aggregate_parent_section(  # type: ignore[attr-defined]
            reference, geometry_ids, sections, section_order
        )
        ratio = numerator / denominator
        return {
            "center_ratio": ratio,
            "lower_95": ratio - (0.1 if ratio > 100 else 0.0),
            "upper_95": ratio + (0.1 if ratio > 100 else 0.0),
            "one_sided_upper_95": ratio,
            "interval_width": 0.2 if ratio > 100 else 0.0,
            "replicates": replicates,
        }

    monkeypatch.setattr(analysis, "_bootstrap_ratio", joint_wide)
    decision = analysis.evaluate_formal_decision(*values)
    assert decision["classification"] == "GO"
    assert decision["results"]["locked_joint_ood"]["comparisons"]["R_s"]["center_ratio"] > 100


def test_real_bootstrap_is_20k_seed_parent_hierarchical_and_averages_loads_first() -> None:
    geometry = [
        f"{section}-p{parent}" for section in SECTIONS for parent in range(3) for _ in range(2)
    ]
    sections = [section for section in SECTIONS for _parent in range(3) for _ in range(2)]
    reference = np.ones((5, len(geometry)), dtype=np.float64)
    candidate = reference * 0.7
    result = REAL_BOOTSTRAP_RATIO(
        candidate,
        reference,
        geometry,
        sections,
        SECTIONS,
        replicates=20_000,
        seed=71,
    )
    assert result["replicates"] == 20_000
    assert result["center_ratio"] == pytest.approx(0.7)
    assert result["one_sided_upper_95"] == pytest.approx(0.7)
    assert result["interval_width"] < 1e-12


def test_section_gate_is_parent_equal_when_valid_load_counts_differ() -> None:
    """One parent with one valid load must not be downweighted by another's four loads."""

    partition = analysis.PartitionMetric(
        name="locked_iid",
        case_group_ids=("a0", "b0", "b1", "b2", "b3"),
        geometry_group_ids=("parent-a", "parent-b", "parent-b", "parent-b", "parent-b"),
        section_families=("circle",) * 5,
        load_subtypes=("id",) * 5,
        coarse_errors=np.ones(5, dtype=np.float64),
        checkpoints={},
        wall_offset={},
    )
    reference = np.ones((5, 5), dtype=np.float64)
    candidate = np.zeros((5, 5), dtype=np.float64)
    candidate[:, 0] = 1.0
    ratio = analysis._slice_ratio(  # type: ignore[attr-defined]
        candidate,
        reference,
        partition,
        {"sections": ("circle",)},
        section_order=("circle",),
    )
    assert float(candidate.mean() / reference.mean()) == pytest.approx(0.2)
    assert ratio == pytest.approx(0.5)
