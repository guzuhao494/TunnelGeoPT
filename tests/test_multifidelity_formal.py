from __future__ import annotations

import ast
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tunnelgeopt.multifidelity_learning import LearningContractError

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_multifidelity_formal.py"
TRAIN_WORKER = ROOT / "scripts" / "run_multifidelity_train_worker.py"
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


def test_implementation_manifest_has_exact_sources_environment_and_hashes(
    tmp_path: Path,
) -> None:
    instance, _, _ = _tiny_runner(tmp_path)
    result = instance.run_phase("prepare")
    implementation_path = instance.paths.root / runner_module.IMPLEMENTATION_MANIFEST_FILENAME
    implementation = json.loads(implementation_path.read_text(encoding="utf-8"))
    assert set(implementation) == {
        "schema",
        "run_id",
        "config_sha256",
        "effect_claim_allowed",
        "recorded_at_utc",
        "source_provenance",
        "environment",
    }
    provenance = implementation["source_provenance"]
    assert set(provenance) == {
        "git_head",
        "upstream_ref",
        "upstream_head",
        "head_matches_upstream",
        "worktree_clean_before_prepare",
        "remote_url_sanitized",
        "all_sources_tracked",
        "source_sha256",
    }
    assert set(provenance["source_sha256"]) == set(runner_module.IMPLEMENTATION_SOURCE_PATHS)
    assert all(
        len(value) == 64 and set(value) <= set("0123456789abcdef")
        for value in provenance["source_sha256"].values()
    )
    assert set(implementation["environment"]) == {
        "python",
        "platform",
        "numpy",
        "scipy",
        "skfem",
        "gmsh",
        "torch",
        "cuda_runtime",
        "cuda_available",
        "device_requested",
        "device_name",
        "device_total_memory_bytes",
        "driver_version",
    }
    assert result["artifacts"][runner_module.IMPLEMENTATION_MANIFEST_FILENAME] == (
        runner_module._file_sha256(implementation_path)
    )
    prepare = json.loads((instance.paths.root / "prepare_manifest.json").read_text("utf-8"))
    assert prepare["implementation_manifest_file_sha256"] == runner_module._file_sha256(
        implementation_path
    )


@pytest.mark.parametrize(
    "relative",
    [
        "scripts/run_multifidelity_formal.py",
        "scripts/run_multifidelity_train_worker.py",
        "configs/multifidelity_formal.json",
    ],
)
def test_prepare_after_source_tamper_refuses_next_phase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative: str,
) -> None:
    instance, _, _ = _tiny_runner(tmp_path)
    instance.run_phase("prepare")
    original = instance._source_hashes
    monkeypatch.setattr(
        instance,
        "_source_hashes",
        lambda: {**original(), relative: "f" * 64},
    )
    with pytest.raises(runner_module.FormalAbstain, match="source hash changed"):
        instance.run_phase("generate")


def test_prepare_after_environment_change_refuses_next_phase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance, _, _ = _tiny_runner(tmp_path)
    instance.run_phase("prepare")
    original = runner_module._environment_manifest(instance.device)
    monkeypatch.setattr(
        runner_module,
        "_environment_manifest",
        lambda device: {**original, "driver_version": "changed-after-prepare"},
    )
    with pytest.raises(runner_module.FormalAbstain, match="environment changed"):
        instance.run_phase("generate")


@pytest.mark.parametrize(
    ("status", "upstream", "message"),
    [
        (" M scripts/run_multifidelity_formal.py", "a" * 40, "clean git worktree"),
        ("", "b" * 40, "configured upstream"),
    ],
)
def test_formal_prepare_rejects_dirty_or_unpushed_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: str,
    upstream: str,
    message: str,
) -> None:
    instance = runner_module.FormalExperimentRunner(
        config_path=ROOT / "configs" / "multifidelity_formal.json",
        approval_path=ROOT / "configs" / "multifidelity_formal_approval.json",
        exclusions_path=ROOT / "configs" / "multifidelity_seen_identity_exclusions.json",
        output_dir=tmp_path / "formal-provenance",
        backend="formal",
        device="cuda",
    )
    head = "a" * 40

    def fake_git(*arguments: str, description: str) -> str:
        del description
        if arguments == ("rev-parse", "HEAD"):
            return head
        if arguments == (
            "rev-parse",
            "--abbrev-ref",
            "--symbolic-full-name",
            "@{upstream}",
        ):
            return "origin/codex/v0.3-multifidelity"
        if arguments == ("rev-parse", "@{upstream}"):
            return upstream
        if arguments[:3] == ("status", "--porcelain=v1", "--untracked-files=all"):
            return status
        if arguments[:2] == ("ls-files", "--"):
            return "\n".join(runner_module.IMPLEMENTATION_SOURCE_PATHS)
        if arguments[:2] == ("remote", "get-url"):
            return "https://token@example.invalid/org/repo.git"
        raise AssertionError(arguments)

    monkeypatch.setattr(runner_module, "_git_output", fake_git)
    with pytest.raises(runner_module.FormalRunError, match=message):
        instance.run_phase("prepare")


def test_formal_prepare_accepts_clean_pushed_git_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = runner_module.FormalExperimentRunner(
        config_path=ROOT / "configs" / "multifidelity_formal.json",
        approval_path=ROOT / "configs" / "multifidelity_formal_approval.json",
        exclusions_path=ROOT / "configs" / "multifidelity_seen_identity_exclusions.json",
        output_dir=tmp_path / "formal-provenance-ok",
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
            return "\n".join(runner_module.IMPLEMENTATION_SOURCE_PATHS)
        if arguments[:2] == ("remote", "get-url"):
            return "https://secret-token@example.invalid/org/repo.git"
        raise AssertionError(arguments)

    monkeypatch.setattr(runner_module, "_git_output", fake_git)
    monkeypatch.setattr(
        runner_module,
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
    result = instance.run_phase("prepare")
    assert result["status"] == "completed"
    implementation = json.loads(
        (instance.paths.root / runner_module.IMPLEMENTATION_MANIFEST_FILENAME).read_text("utf-8")
    )
    provenance = implementation["source_provenance"]
    assert provenance["git_head"] == provenance["upstream_head"] == head
    assert provenance["head_matches_upstream"] is True
    assert provenance["worktree_clean_before_prepare"] is True
    assert provenance["all_sources_tracked"] is True
    assert provenance["remote_url_sanitized"] == "https://example.invalid/org/repo.git"


def test_prepare_captures_clean_state_before_repo_output_is_created(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = runner_module.FormalExperimentRunner(
        config_path=ROOT / "configs" / "multifidelity_formal.json",
        approval_path=ROOT / "configs" / "multifidelity_formal_approval.json",
        exclusions_path=ROOT / "configs" / "multifidelity_seen_identity_exclusions.json",
        output_dir=ROOT / "artifacts" / "experiment" / "provenance-order-test",
        backend="formal",
        device="cuda",
    )
    observations: list[tuple[bool, bool]] = []

    def preflight() -> dict:
        observations.append((instance.paths.state.exists(), instance.paths.access_log.exists()))
        return {
            "schema": "tunnelgeopt.formal_implementation_manifest.v1",
            "run_id": instance.config["run_id"],
            "config_sha256": instance.config_sha256,
            "effect_claim_allowed": True,
            "recorded_at_utc": runner_module._now(),
            "source_provenance": {
                "git_head": "a" * 40,
                "upstream_ref": "origin/codex/v0.3-multifidelity",
                "upstream_head": "a" * 40,
                "head_matches_upstream": True,
                "worktree_clean_before_prepare": True,
                "remote_url_sanitized": "https://example.invalid/org/repo.git",
                "all_sources_tracked": True,
                "source_sha256": instance._source_hashes(),
            },
            "environment": {
                "python": sys.version,
                "platform": "test",
                "numpy": np.__version__,
                "scipy": None,
                "skfem": None,
                "gmsh": None,
                "torch": "test",
                "cuda_runtime": "test",
                "cuda_available": True,
                "device_requested": "cuda",
                "device_name": "test GPU",
                "device_total_memory_bytes": 1,
                "driver_version": "test",
            },
        }

    monkeypatch.setattr(instance, "_implementation_manifest", preflight)
    try:
        result = instance.run_phase("prepare")
        assert result["status"] == "completed"
        assert observations == [(False, False)]
    finally:
        if instance.paths.root.is_dir():
            shutil.rmtree(instance.paths.root)


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


def test_frozen_exclusions_are_hash_bound_and_have_exact_zero_intersection(
    tmp_path: Path,
) -> None:
    from tunnelgeopt.formal_generation import build_formal_generation_plan

    instance = runner_module.FormalExperimentRunner(
        config_path=ROOT / "configs" / "multifidelity_formal.json",
        approval_path=ROOT / "configs" / "multifidelity_formal_approval.json",
        exclusions_path=ROOT / "configs" / "multifidelity_seen_identity_exclusions.json",
        output_dir=tmp_path / "plan-only",
        backend="formal",
        device="cuda",
    )
    plan = build_formal_generation_plan(
        instance.config,
        forbidden_identities=instance.forbidden_identities,
    )
    identities = {
        "geometry_group_id": {value.geometry_group_id for value in plan.geometries},
        "boundary_float64_sha256": {value.boundary_float64_sha256 for value in plan.geometries},
        "case_group_id": {value.case_group_id for value in plan.cases},
        "load_group_id": {value.load_group_id for value in plan.cases},
    }
    excluded = {
        "geometry_group_id": instance.forbidden_identities.geometry_group_ids,
        "boundary_float64_sha256": instance.forbidden_identities.boundary_float64_sha256,
        "case_group_id": instance.forbidden_identities.case_group_ids,
        "load_group_id": instance.forbidden_identities.load_group_ids,
    }
    assert plan.formal_eligible is True
    assert (len(plan.geometries), len(plan.cases), len(plan.audit_case_ids)) == (195, 705, 144)
    assert {name: len(values) for name, values in excluded.items()} == {
        "geometry_group_id": 42,
        "boundary_float64_sha256": 43,
        "case_group_id": 102,
        "load_group_id": 84,
    }
    assert instance.forbidden_identities.source_record_count == 271
    assert all(not (identities[name] & excluded[name]) for name in identities)
    assert all(plan.identity_report["legacy_zero_intersection"].values())
    assert all(plan.identity_report["cross_partition_zero_intersection"].values())


def test_training_worker_source_and_import_contract_are_minimal() -> None:
    source = TRAIN_WORKER.read_text(encoding="utf-8")
    lowered = source.lower()
    for forbidden in (
        "sealed",
        "locked",
        "formal_generation",
        "evaluator",
        "dataset_manifest",
        "approval",
        "trusted_locked_label_path",
    ):
        assert forbidden not in lowered
    tree = ast.parse(source)
    tunnelgeopt_imports = [
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module
        and node.module.startswith("tunnelgeopt")
    ]
    assert tunnelgeopt_imports == ["tunnelgeopt.multifidelity_learning"]


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


def test_generation_90_percent_validity_evidence_abstains_and_blocks_train(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tunnelgeopt.formal_generation as generation

    instance = runner_module.FormalExperimentRunner(
        config_path=ROOT / "configs" / "multifidelity_formal.json",
        approval_path=ROOT / "configs" / "multifidelity_formal_approval.json",
        exclusions_path=ROOT / "configs" / "multifidelity_seen_identity_exclusions.json",
        output_dir=tmp_path / "formal-abstain",
        backend="formal",
        device="cuda",
    )
    monkeypatch.setattr(instance, "tiny_mock", True)
    instance.run_phase("prepare")
    monkeypatch.setattr(instance, "tiny_mock", False)
    monkeypatch.setattr(instance, "_verify_implementation_unchanged", lambda: None)
    monkeypatch.setattr(
        generation,
        "build_formal_generation_plan",
        lambda config, forbidden_identities: SimpleNamespace(formal_eligible=True),
    )

    def fail_with_complete_evidence(config, data_root, **kwargs):
        del kwargs
        runner_module._atomic_json(
            Path(data_root) / runner_module.DATASET_MANIFEST_FILENAME,
            {
                "schema_version": "tunnelgeopt.formal_dataset_manifest.v1",
                "run_id": config["run_id"],
                "config_sha256": instance.config_sha256,
                "generation_status": "ABSTAIN",
                "counts": {"planned_cases": 20, "valid_cases": 18, "invalid_cases": 2},
                "solver_mesh_qc": {
                    "passed": False,
                    "partition_section_summary": {"minimum_valid_fraction": 0.9},
                },
            },
        )
        raise generation.FormalGenerationError("2 of 20 injected solves are invalid")

    monkeypatch.setattr(generation, "generate_formal_dataset", fail_with_complete_evidence)
    with pytest.raises(runner_module.FormalAbstain, match="validity gate ABSTAIN"):
        instance.run_phase("generate")
    state = json.loads(instance.paths.state.read_text(encoding="utf-8"))
    assert state["phases"]["generate"]["status"] == "abstained"
    assert set(state["phases"]["generate"]["artifacts"]) == {
        f"data/{runner_module.DATASET_MANIFEST_FILENAME}"
    }
    with pytest.raises(runner_module.FormalRunError, match="requires completed 'generate'"):
        instance.run_phase("train")


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
    access_audit = resumed_manifest["training_access_audit"]
    assert access_audit["schema"] == "tunnelgeopt.formal_training_access_audit.v2"
    assert access_audit["process_isolated"] is True
    assert access_audit["trainer_input_api"] == "redacted_contract_public_train_dev_subprocess"
    assert access_audit["sealed_path_resolution_calls"] == 0
    assert access_audit["sealed_open_calls"] == 0
    assert access_audit["denied_sealed_access_calls"] == 0
    assert access_audit["worker_unexpected_tunnelgeopt_modules"] == []
    assert access_audit["worker_received_contract_keys"] == sorted(
        runner_module.TRAINING_WORKER_ALLOWED_KEYS
    )
    assert set(access_audit["worker_received_paths"]) == {
        "worker_contract",
        "public_inputs",
        "train_dev_labels",
        "checkpoint_output_dir",
    }
    assert Path(access_audit["worker_argv"][1]).name == TRAIN_WORKER.name
    assert access_audit["worker_argv"][2:] == [
        "--contract",
        access_audit["worker_received_paths"]["worker_contract"],
    ]
    assert access_audit["passed"] is True
    events = runner_module._read_access_events(instance.paths.access_log)
    assert not any(
        event["event"] in {"trusted_sealed_path_resolved", "sealed_partition_opened"}
        and event.get("phase") == "train"
        for event in events
    )


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
    opaque_id = runner_module._sha256_value(
        {"run_id": second.config["run_id"], "partition": "locked_iid"}
    )
    manifest["opaque_sealed_stores"][opaque_id] = runner_module._file_sha256(sealed_path)
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


@pytest.mark.parametrize(
    ("completed_phase", "next_phase", "artifact_suffix"),
    [
        ("prepare", "generate", "prepare_manifest.json"),
        ("generate", "train", runner_module.PUBLIC_FILENAME),
        ("train", "evaluate", runner_module.REGISTRY_FILENAME),
        ("evaluate", "analyze", "sealed_metrics.json"),
    ],
)
def test_next_phase_rejects_drifted_predecessor_artifact(
    tmp_path: Path,
    completed_phase: str,
    next_phase: str,
    artifact_suffix: str,
) -> None:
    instance, _, _ = _tiny_runner(tmp_path)
    _run_through(instance, completed_phase)
    state = json.loads(instance.paths.state.read_text(encoding="utf-8"))
    matches = [
        relative
        for relative in state["phases"][completed_phase]["artifacts"]
        if relative.endswith(artifact_suffix)
    ]
    assert len(matches) == 1
    artifact = instance.paths.root / matches[0]
    artifact.write_bytes(artifact.read_bytes() + b"tamper")
    with pytest.raises(runner_module.FormalAbstain, match="artifact drifted"):
        instance.run_phase(next_phase)


def test_tiny_mock_analysis_is_always_abstain(tmp_path: Path) -> None:
    instance, _, _ = _tiny_runner(tmp_path)
    _run_through(instance, "analyze")
    decision = json.loads((instance.paths.analysis / "decision.json").read_text(encoding="utf-8"))
    assert decision["classification"] == "ABSTAIN"
    assert decision["effect_claim_allowed"] is False
