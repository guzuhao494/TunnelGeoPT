from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from tunnelgeopt.multifidelity_learning import LearningContractError

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_multifidelity_formal.py"
SPEC = importlib.util.spec_from_file_location("run_multifidelity_formal", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner_module
SPEC.loader.exec_module(runner_module)


def _base_config() -> dict:
    return json.loads((ROOT / "configs" / "multifidelity_formal.json").read_text(encoding="utf-8"))


def _tiny_runner(tmp_path: Path, *, run_id: str = "tiny-mock-formal-run"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = runner_module.make_tiny_mock_config(_base_config(), run_id=run_id)
    config_path = tmp_path / "tiny_config.json"
    approval_path = tmp_path / "approval.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    runner_module.write_tiny_mock_approval(approval_path, config)
    instance = runner_module.FormalExperimentRunner(
        config_path=config_path,
        approval_path=approval_path,
        output_dir=tmp_path / "run",
        backend="tiny-mock",
        device="cpu",
    )
    return instance, config_path, approval_path


def _run_through(instance, phase: str) -> None:
    for candidate in runner_module.PHASES:
        instance.run_phase(candidate)
        if candidate == phase:
            return


def test_real_frozen_config_requires_external_hash_bound_approval(tmp_path: Path) -> None:
    config = _base_config()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    approval = {
        "schema": "tunnelgeopt.formal_execution_approval.v1",
        "run_id": config["run_id"],
        "config_sha256": "0" * 64,
        "config_frozen": True,
        "development_convergence_audit_passed": True,
        "ultrafine_development_audit_passed": True,
        "formal_execution_authorized": True,
    }
    approval_path = tmp_path / "approval.json"
    approval_path.write_text(json.dumps(approval), encoding="utf-8")
    with pytest.raises(runner_module.FormalRunError, match="complete config hash"):
        runner_module.FormalExperimentRunner(
            config_path=config_path,
            approval_path=approval_path,
            output_dir=tmp_path / "run",
        )


def test_checkpoint_matrix_is_exactly_35_and_seven_per_seed() -> None:
    specifications = runner_module._expected_checkpoint_specs(_base_config())
    assert len(specifications) == 35
    by_seed: dict[int, list[tuple[str, float]]] = {}
    for value in specifications:
        by_seed.setdefault(value["seed"], []).append((value["method"], value["fraction"]))
    assert len(by_seed) == 5
    assert all(len(values) == 7 for values in by_seed.values())
    assert set(next(iter(by_seed.values()))) == {
        ("scratch", 1.0),
        ("direct_coarse", 1.0),
        ("residual_coarse", 0.25),
        ("residual_coarse", 0.5),
        ("residual_coarse", 0.75),
        ("residual_coarse", 1.0),
        ("mismatched_coarse", 0.5),
    }


def test_training_paths_never_return_or_name_a_sealed_store(tmp_path: Path) -> None:
    instance, _, _ = _tiny_runner(tmp_path)
    instance.run_phase("prepare")
    instance.run_phase("generate")
    public_path, train_path = instance._training_paths()
    assert public_path.name == runner_module.PUBLIC_FILENAME
    assert train_path.name == runner_module.TRAIN_DEV_FILENAME
    assert all("sealed" not in str(path).lower() for path in (public_path, train_path))
    state = json.loads(instance.paths.state.read_text(encoding="utf-8"))
    artifact_names = state["phases"]["generate"]["artifacts"]
    assert all("sealed" not in value.lower() for value in artifact_names)
    assert state["sealed_partition_open_counts"] == {
        partition: 0 for partition in runner_module.LOCKED_PARTITIONS
    }


def test_training_and_primary_metric_weights_are_distinct(tmp_path: Path) -> None:
    instance, _, _ = _tiny_runner(tmp_path)
    instance.run_phase("prepare")
    instance.run_phase("generate")
    public_path, _ = instance._training_paths()
    public = runner_module._load_npz(public_path, "tiny public store")
    training = public["training_weights"]
    metric = public["metric_weights"]
    near = public["nearfield_mask"]
    wall = public["wall_offset_mask"]
    far = public["farfield_mask"]
    assert not np.array_equal(training, metric)
    assert np.allclose(training[:, near[0]].sum(axis=1), 0.8)
    assert np.allclose(training[:, wall[0]].sum(axis=1), 0.15)
    assert np.allclose(training[:, far[0]].sum(axis=1), 0.05)
    assert np.allclose(metric[:, near[0]].sum(axis=1), 1.0)
    assert np.all(metric[:, wall[0] | far[0]] == 0.0)


def test_premature_sealed_open_is_denied_and_audited(tmp_path: Path) -> None:
    instance, _, _ = _tiny_runner(tmp_path)
    instance.run_phase("prepare")
    with pytest.raises(runner_module.SealedAccessError, match="cannot be opened"):
        instance.open_sealed_partition("locked_iid")
    state = json.loads(instance.paths.state.read_text(encoding="utf-8"))
    assert state["denied_premature_sealed_accesses"] == 1
    assert state["sealed_partition_open_counts"]["locked_iid"] == 0


def test_fake_declared_fraction_is_rejected_by_training_contract(tmp_path: Path) -> None:
    instance, _, _ = _tiny_runner(tmp_path)
    instance.run_phase("prepare")
    instance.run_phase("generate")
    _, batch, _ = instance._load_training_inputs()
    subsets = instance._nested_subsets(batch)
    with pytest.raises(LearningContractError, match="declared fine fraction disagrees"):
        runner_module.build_training_contract(
            batch,
            method="residual_coarse",
            config_sha256=instance.config_sha256,
            train_geometry_selector=subsets[0.5],
            expected_fine_fraction=0.75,
        )


def test_resume_skips_completed_phase_and_completed_checkpoints(tmp_path: Path) -> None:
    instance, config_path, approval_path = _tiny_runner(tmp_path)
    instance.run_phase("prepare")
    instance.run_phase("generate")
    first = instance.run_phase("train")
    assert first["status"] == "completed"
    checkpoint_manifest = json.loads(
        (instance.paths.checkpoints / runner_module.CHECKPOINT_MANIFEST_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    hashes = {key: value["sha256"] for key, value in checkpoint_manifest["checkpoints"].items()}
    restarted = runner_module.FormalExperimentRunner(
        config_path=config_path,
        approval_path=approval_path,
        output_dir=instance.paths.root,
        backend="tiny-mock",
        device="cpu",
    )
    assert restarted.run_phase("train")["status"] == "already_completed"
    resumed_manifest = json.loads(
        (instance.paths.checkpoints / runner_module.CHECKPOINT_MANIFEST_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    assert hashes == {
        key: value["sha256"] for key, value in resumed_manifest["checkpoints"].items()
    }
    state = json.loads(instance.paths.state.read_text(encoding="utf-8"))
    assert state["phases"]["train"]["attempts"] == 1


def test_each_sealed_file_opens_once_and_every_checkpoint_evaluates_once(
    tmp_path: Path,
) -> None:
    instance, _, _ = _tiny_runner(tmp_path)
    _run_through(instance, "evaluate")
    state = json.loads(instance.paths.state.read_text(encoding="utf-8"))
    assert state["sealed_partition_open_counts"] == {
        partition: 1 for partition in runner_module.LOCKED_PARTITIONS
    }
    assert len(state["checkpoint_evaluation_counts"]) == 4 * 35
    assert set(state["checkpoint_evaluation_counts"].values()) == {1}
    metrics = json.loads(
        (instance.paths.evaluation / "sealed_metrics.json").read_text(encoding="utf-8")
    )
    assert set(metrics["partitions"]) == set(runner_module.LOCKED_PARTITIONS)
    assert all(len(value["checkpoints"]) == 35 for value in metrics["partitions"].values())
    assert instance.run_phase("evaluate")["status"] == "already_completed"


def test_sealed_labels_cannot_change_training_checkpoints(tmp_path: Path) -> None:
    first, _, _ = _tiny_runner(tmp_path / "a", run_id="tiny-mock-a")
    second, _, _ = _tiny_runner(tmp_path / "b", run_id="tiny-mock-a")
    for instance in (first, second):
        instance.run_phase("prepare")
        instance.run_phase("generate")
    sealed_path = second.paths.data / runner_module.SEALED_FILENAMES["locked_iid"]
    sealed = runner_module._load_npz(sealed_path, "test sealed archive")
    sealed["fine_stress"] = sealed["fine_stress"] + 777.0
    runner_module._atomic_npz(sealed_path, **sealed)
    # Deliberately update only the trusted generator manifest hash; training sees
    # neither this path nor its contents.
    manifest_path = second.paths.data / runner_module.DATASET_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][runner_module.SEALED_FILENAMES["locked_iid"]] = runner_module._file_sha256(
        sealed_path
    )
    runner_module._atomic_json(manifest_path, manifest)
    state = json.loads(second.paths.state.read_text(encoding="utf-8"))
    generate_artifacts = state["phases"]["generate"]["artifacts"]
    generate_artifacts[f"data/{runner_module.DATASET_MANIFEST_FILENAME}"] = (
        runner_module._file_sha256(manifest_path)
    )
    runner_module._atomic_json(second.paths.state, state)
    first.run_phase("train")
    second.run_phase("train")
    first_manifest = json.loads(
        (first.paths.checkpoints / runner_module.CHECKPOINT_MANIFEST_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    second_manifest = json.loads(
        (second.paths.checkpoints / runner_module.CHECKPOINT_MANIFEST_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    assert [value["sha256"] for value in first_manifest["checkpoints"].values()] == [
        value["sha256"] for value in second_manifest["checkpoints"].values()
    ]


def test_interrupted_evaluation_abstains_instead_of_reopening_sealed(tmp_path: Path) -> None:
    instance, _, _ = _tiny_runner(tmp_path)
    _run_through(instance, "train")
    state = json.loads(instance.paths.state.read_text(encoding="utf-8"))
    state["phases"]["evaluate"]["status"] = "in_progress"
    runner_module._atomic_json(instance.paths.state, state)
    with pytest.raises(runner_module.FormalAbstain, match="interrupted"):
        instance.run_phase("evaluate")
    state = json.loads(instance.paths.state.read_text(encoding="utf-8"))
    assert state["phases"]["evaluate"]["status"] == "abstained"


def test_tiny_mock_analysis_is_always_abstain(tmp_path: Path) -> None:
    instance, _, _ = _tiny_runner(tmp_path)
    _run_through(instance, "analyze")
    decision = json.loads((instance.paths.analysis / "decision.json").read_text(encoding="utf-8"))
    assert decision["classification"] == "ABSTAIN"
    assert decision["effect_claim_allowed"] is False
