from __future__ import annotations

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_v04_development.py"
CONFIG = ROOT / "configs" / "multifidelity_v04_development.json"
EXPLORATORY_RECORD = (
    ROOT / "artifacts" / "analysis" / "v04-structured-prototype-stop" / "exploratory_record.json"
)
SPEC = importlib.util.spec_from_file_location("run_v04_development", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _write_config(tmp_path: Path, config: dict) -> Path:
    path = tmp_path / "development.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def _inputs() -> tuple[dict, str, runner.DevelopmentData, dict, dict]:
    config, digest = runner.load_development_config(CONFIG)
    data, checkpoint_manifest, audit = runner.audit_and_load_inputs(config, digest)
    return config, digest, data, checkpoint_manifest, audit


def test_protocol_is_seen_only_stopped_and_cannot_claim_validation() -> None:
    config, digest = runner.load_development_config(CONFIG)
    assert len(digest) == 64
    assert config["status"] == "implementation_stop_pending_pivot"
    assert config["effect_claim_allowed"] is False
    assert config["independent_validation_claim_allowed"] is False
    assert config["execution_authorization"] == {
        "validate_only_authorized": True,
        "tiny_mock_authorized": True,
        "real_cross_fit_authorized": False,
        "reason": (
            "structured prototypes did not satisfy the frozen launch margins; "
            "a pivot decision is required before expensive compute"
        ),
    }
    assert config["seen_data_contract"]["former_locked_partitions_are_seen"] is True
    assert config["seen_data_contract"]["new_locked_partition_count"] == 0
    assert config["seen_data_contract"]["generator_invocation_allowed"] is False


def test_structured_diagnostics_each_toggle_one_switch_without_causal_claim() -> None:
    config, _ = runner.load_development_config(CONFIG)
    assert config["model"]["hidden_width"] == 64
    assert config["model"]["global_context_blocks"] == 3
    champion = config["architecture"]["champion"]
    switches = (
        "strict_load_linearity",
        "local_tensor_frame",
        "exact_zero_init_coarse_gate",
    )
    assert all(champion[name] is True for name in switches)
    for index, ablation in enumerate(config["architecture"]["ablations"]):
        assert [ablation[name] for name in switches].count(False) == 1
        assert ablation[switches[index]] is False
        mapping = runner._model_mapping(config, ablation)
        assert mapping[switches[index]] is False
        assert sum(mapping[name] is False for name in switches) == 1


def test_real_structured_pack_and_switches_reach_the_model() -> None:
    torch = pytest.importorskip("torch")
    config, _ = runner.load_development_config(CONFIG)
    point_count = 8
    features14 = np.zeros((2, point_count, 14), dtype=np.float32)
    features14[..., 1] = np.linspace(-0.5, 0.5, point_count)
    features14[..., 2] = 0.25
    features14[..., 3] = 0.2
    features14[..., 5] = -1.0
    features14[..., 7:11] = 0.1
    features14[..., 11:14] = 0.2
    normals = np.zeros((2, point_count, 2), dtype=np.float32)
    wall = np.zeros((2, point_count), dtype=bool)
    wall[:, :2] = True
    normals[:, :2, 0] = 1.0
    module = runner._structured_module()
    packed = module.pack_structured_features(features14, normals, wall)
    assert packed.shape == (2, point_count, 17)
    champion = runner._model_mapping(
        config, runner._architecture_by_name(config, "structured_linear_residual")
    )
    model = module.make_structured_residual_model(champion, seed=7, device="cpu")
    assert sum(parameter.numel() for parameter in model.parameters()) == 40685
    assert model.config.strict_load_linearity is True
    assert model.config.local_tensor_frame is True
    assert model.config.exact_zero_init_coarse_gate is True
    with torch.no_grad():
        prediction = model(torch.as_tensor(packed))
    assert torch.equal(prediction, torch.zeros_like(prediction))
    expected_counts = {
        "ablate_strict_load_linearity": 38598,
        "ablate_local_tensor_frame": 40813,
        "ablate_zero_init_coarse_gate": 40685,
    }
    for method, field in zip(
        runner.ABLATIONS,
        (
            "strict_load_linearity",
            "local_tensor_frame",
            "exact_zero_init_coarse_gate",
        ),
        strict=True,
    ):
        mapping = runner._model_mapping(config, runner._architecture_by_name(config, method))
        ablated = module.make_structured_residual_model(mapping, seed=7, device="cpu")
        assert getattr(ablated.config, field) is False
        assert (
            sum(parameter.numel() for parameter in ablated.parameters()) == expected_counts[method]
        )
    generic = runner.make_model(
        {
            "point_input_width": 14,
            "hidden_width": 64,
            "global_context_blocks": 3,
            "output_width": 3,
        },
        seed=7,
        device="cpu",
    )
    assert sum(parameter.numel() for parameter in generic.parameters()) == 38787
    disclosures = config["architecture"]["comparison_disclosures"]
    assert disclosures["equal_parameter_count"] is False
    assert disclosures["loss_functions_matched"] is False
    assert disclosures["structured_model_has_additional_derived_information"] is True
    assert disclosures["causal_component_attribution_allowed"] is False


def test_input_audit_authenticates_705_cases_without_opening_seen_stress_values() -> None:
    _, _, data, _, audit = _inputs()
    assert data.case_count == 705
    assert len(data.parent_ids) == 195
    assert int(data.fine_available.sum()) == 360
    assert np.isfinite(data.fine_stress[data.fine_available]).all()
    assert np.isnan(data.fine_stress[~data.fine_available]).all()
    assert audit["train_dev_fine_label_case_reads"] == 360
    assert audit["former_locked_seen_fine_label_case_reads"] == 0
    assert audit["former_locked_seen_values_opened"] is False
    assert audit["development_partition_case_counts"] == {
        "train_id": 288,
        "dev_id": 72,
        "seen_iid": 120,
        "seen_geometry_ood": 90,
        "seen_load_ood": 90,
        "seen_joint_ood": 45,
    }


def test_fold_manifest_is_parent_grouped_budget_matched_and_complete() -> None:
    config, digest, data, _, _ = _inputs()
    manifest = runner.build_fold_manifest(data, config, digest)
    folds = manifest["folds"]
    assert [len(value["oof_parent_ids"]) for value in folds] == [15, 15, 15, 15, 12]
    assert [len(value["available_non_oof_train_parent_ids"]) for value in folds] == [
        57,
        57,
        57,
        57,
        60,
    ]
    occurrences: dict[str, int] = {}
    for fold in folds:
        oof = set(fold["oof_parent_ids"])
        optimizer = set(fold["optimizer_fine50_parent_ids"])
        normalizer = set(fold["normalization_fit_parent_ids"])
        dev = set(fold["fixed_dev_parent_ids"])
        assert len(optimizer) == 36
        assert optimizer == normalizer
        assert len(dev) == 18
        assert not (oof & optimizer or oof & dev or optimizer & dev)
        assert fold["role_counts"]["optimizer_fine50"] == {
            "train_id:circle": 12,
            "train_id:horseshoe": 12,
            "train_id:straight_wall_arch": 12,
        }
        assert fold["role_counts"]["fixed_dev"] == {
            "dev_id:circle": 6,
            "dev_id:horseshoe": 6,
            "dev_id:straight_wall_arch": 6,
        }
        for parent in oof:
            occurrences[parent] = occurrences.get(parent, 0) + 1
    assert len(occurrences) == 72
    assert set(occurrences.values()) == {1}
    assert manifest["each_parent_exactly_one_oof_fold"] is True


def test_final_identity_contract_recovers_exact_v03_36_train_and_18_dev() -> None:
    config, _, data, checkpoint_manifest, _ = _inputs()
    identity = runner.final_fit_identity_contract(data, checkpoint_manifest, config)
    assert identity["optimizer_parent_count"] == 36
    assert identity["optimizer_parents_per_section"] == {
        "circle": 12,
        "horseshoe": 12,
        "straight_wall_arch": 12,
    }
    assert identity["early_stopping_parent_count"] == 18
    assert identity["early_stopping_parents_per_section"] == {
        "circle": 6,
        "horseshoe": 6,
        "straight_wall_arch": 6,
    }
    assert identity["former_locked_optimizer_intersection_count"] == 0
    assert identity["former_locked_normalization_intersection_count"] == 0
    assert identity["former_locked_early_stopping_intersection_count"] == 0


def test_seen_stress_labels_are_fail_closed_until_explicit_post_freeze_open() -> None:
    config, _, data, _, _ = _inputs()
    seen_parent = next(
        parent
        for parent, partition in zip(
            data.geometry_group_ids, data.development_partitions, strict=True
        )
        if partition == "seen_iid"
    )
    rows = runner._row_indices(data, (seen_parent,))
    with pytest.raises(runner.DevelopmentProtocolError, match="unopened fine labels"):
        runner._batch_for_rows(data, rows, split="illegal_train")
    opened, audit = runner.open_seen_stress_labels(data, config)
    assert int(opened.fine_available.sum()) == 705
    assert np.isfinite(opened.fine_stress).all()
    assert set(audit) == {
        "seen_iid",
        "seen_geometry_ood",
        "seen_load_ood",
        "seen_joint_ood",
    }
    assert all(value["role"] == "seen_post_selection_stress_only" for value in audit.values())


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (
            lambda config: config["architecture"]["ablations"][0].__setitem__(
                "local_tensor_frame", False
            ),
            "one-switch diagnostic",
        ),
        (
            lambda config: config["launch_gates"]["cross_fit_architecture_gate"].__setitem__(
                "champion_to_each_ablation_max_point_ratio", 1.01
            ),
            "ablation margin",
        ),
        (
            lambda config: config["seen_data_contract"].__setitem__(
                "new_locked_partition_count", 1
            ),
            "seen-data or no-new-locked",
        ),
        (
            lambda config: config["execution_authorization"].__setitem__(
                "real_cross_fit_authorized", True
            ),
            "must remain unauthorized",
        ),
        (
            lambda config: config["architecture"]["comparison_disclosures"][
                "parameter_counts"
            ].__setitem__("structured_linear_residual", 40684),
            "candidate-comparison confound disclosures",
        ),
    ],
)
def test_config_rejects_ablation_threshold_locked_and_authorization_drift(
    tmp_path: Path, mutator, match: str
) -> None:
    config = deepcopy(_config())
    mutator(config)
    with pytest.raises(runner.DevelopmentProtocolError, match=match):
        runner.load_development_config(_write_config(tmp_path, config))


def test_source_hash_drift_is_rejected_before_any_label_use(tmp_path: Path) -> None:
    config = deepcopy(_config())
    config["source_experiment"]["files"]["public_inputs"]["sha256"] = "0" * 64
    parsed, digest = runner.load_development_config(_write_config(tmp_path, config))
    with pytest.raises(runner.DevelopmentProtocolError, match="source artifact hash mismatch"):
        runner.audit_and_load_inputs(parsed, digest)


def test_exploratory_stop_record_is_conversation_only_and_not_replayable() -> None:
    record = json.loads(EXPLORATORY_RECORD.read_text(encoding="utf-8"))
    assert record["provenance_quality"] == "conversation_record_only"
    assert record["raw_training_logs_available"] is False
    assert record["raw_checkpoints_available"] is False
    assert record["replayable_from_this_record"] is False
    assert record["effect_claim_allowed"] is False
    assert record["independent_validation_claim_allowed"] is False
    assert record["formal_go_no_go_claim_allowed"] is False
    probe = record["reported_exploratory_observations"]["structured_residual_seed_probe"]
    assert probe["mean_values"] == [
        0.031587,
        0.032346,
        0.031707,
        0.033407,
        0.037223,
        0.037635,
    ]
    assert probe["candidate_to_coarse_point_ratio_approximate"] == {
        "seen_iid": 0.945,
        "seen_geometry_ood": 0.971,
        "seen_load_ood": 0.965,
    }
    assert record["execution_counts"]["production_v04_crossfit_runs"] == 0
    assert record["execution_counts"]["new_v04_locked_cases_generated"] == 0
    assert record["decision"] == {
        "classification": "IMPLEMENTATION_STOP_PENDING_PIVOT",
        "real_cross_fit_authorized": False,
        "reason": ("available exploratory evidence did not satisfy the frozen launch requirements"),
        "required_next_transition": "explicit_pivot_decision_before_any_real_crossfit",
    }


def test_validate_only_is_read_only_and_reports_stop_boundary(tmp_path: Path) -> None:
    missing_output = tmp_path / "must-not-exist"
    result = runner.execute(
        config_path=CONFIG,
        output_dir=missing_output,
        device="cpu",
        validate_only=True,
    )
    assert result["status"] == "validated"
    assert result["real_cross_fit_authorized"] is False
    assert result["former_locked_seen_fine_label_case_reads"] == 0
    assert result["new_locked_data_created"] is False
    assert not missing_output.exists()


def test_real_run_is_rejected_before_output_or_preflight(tmp_path: Path) -> None:
    output = tmp_path / "real"
    with pytest.raises(runner.DevelopmentProtocolError, match="stopped pending"):
        runner.execute(
            config_path=CONFIG,
            output_dir=output,
            device="cuda",
            tiny_mock=False,
        )
    assert not output.exists()


def test_complete_tiny_chain_stays_stop_and_records_freeze_before_seen_open(
    tmp_path: Path,
) -> None:
    output = tmp_path / "tiny"
    result = runner.execute(
        config_path=CONFIG,
        output_dir=output,
        device="cpu",
        tiny_mock=True,
    )
    assert result["classification"] == "IMPLEMENTATION_STOP_PENDING_PIVOT"
    decision = json.loads((output / "launch_decision.json").read_text(encoding="utf-8"))
    assert decision["implementation_stop_pending_pivot"] is True
    assert decision["real_cross_fit_authorized"] is False
    assert decision["may_draft_new_locked_preregistration"] is False
    assert decision["effect_claim_allowed"] is False
    assert decision["independent_validation_claim_allowed"] is False
    final = json.loads((output / "final_fit_manifest.json").read_text(encoding="utf-8"))
    opened = json.loads((output / "seen_stress_open_audit.json").read_text(encoding="utf-8"))
    assert final["checkpoints_frozen_before_seen_stress_labels_open"] is True
    assert final["former_locked_label_reads_before_checkpoint_freeze"] == 0
    assert opened["checkpoints_frozen_before_open"] is True
    assert opened["opened_case_count"] == 345
    events = [
        json.loads(line)["event"]
        for line in (output / "access_audit.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert events.index("final_development_checkpoints_frozen") < events.index(
        "former_locked_seen_labels_opened"
    )
    assert not any("new_locked" in path.name.lower() for path in output.rglob("*"))
