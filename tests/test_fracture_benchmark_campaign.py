from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("scipy")

import tunnelgeopt.fracture_benchmark as fracture_benchmark_module
import tunnelgeopt.fracture_benchmark_mesh as fracture_mesh_module
from scripts import run_fracture_benchmark_campaign as campaign
from tunnelgeopt.fracture import FractureSolverOptions
from tunnelgeopt.fracture_benchmark import ProbeProjectSnapshot, ProbeSourceFile
from tunnelgeopt.fracture_benchmark_validation import (
    load_fracture_sent_sens_config,
    prescribed_displacements,
)


class _FakeProbe:
    def __init__(self, benchmark_id: str, *, protocol_sha256: str = "a" * 64) -> None:
        formal_count = 2000 if benchmark_id == "sent" else 1500
        states = campaign.EXPECTED_THREE_STATE_GRID_MM
        self.payload = {
            "schema": "tunnelgeopt.fracture.sent_sens.intact_probe.v1",
            "status": "DEVELOPMENT_INTACT_FIXED_DAMAGE_PROBE_ONLY",
            "claim_boundary": "not_coupled",
            "benchmark_id": benchmark_id,
            "tier": "coarse",
            "protocol_sha256": protocol_sha256,
            "mesh_plan_sha256": ("b" if benchmark_id == "sent" else "c") * 64,
            "mesh_topology_sha256": ("d" if benchmark_id == "sent" else "e") * 64,
            "bvp_mesh_sha256": ("f" if benchmark_id == "sent" else "0") * 64,
            "mesh_counts": {
                "node_count": 10,
                "element_count": 12,
                "top_node_count": 3,
                "bottom_node_count": 3,
            },
            "material": {
                "young_modulus_kN_per_mm2": 210.0,
                "poisson_ratio": 0.3,
                "fracture_toughness_kN_per_mm": 0.0027,
                "length_scale_mm": 0.015,
                "residual_stiffness": 1.0e-8,
            },
            "prescribed_U_mm": list(states),
            "steps": [
                {
                    "sequence_index": index,
                    "prescribed_U_mm": value,
                    "wall_seconds": 0.1 + index * 0.1,
                    "converged": True,
                    "generalized_load_kN": float(index),
                    "elastic_energy_kN_mm": float(index) * 1.0e-5,
                    "equilibrium_relative_residual": 0.0,
                    "global_force_relative_imbalance": 0.0,
                    "global_moment_relative_imbalance": 0.0,
                    "path_energy_relative_imbalance": 0.0,
                    "damage_component_status": "NOT_APPLICABLE_INTACT_D0_PROBE",
                }
                for index, value in enumerate(states)
            ],
            "median_step_wall_seconds": 0.2,
            "projected_formal_increment_count": formal_count,
            "projected_formal_case_wall_hours": formal_count * 0.2 / 3600.0,
            "projection_interpretation": "intact_fixed_damage_lower_bound_non_authorizing",
            "authorizes_medium_fine_or_formal_run": False,
        }

    def as_dict(self) -> dict:
        return self.payload


def _snapshot(project_root: Path, runner_relative: str) -> ProbeProjectSnapshot:
    return ProbeProjectSnapshot(
        expected_project_head="1" * 40,
        project_head="1" * 40,
        upstream_head="1" * 40,
        upstream_ref="origin/test",
        config_path="configs/fracture_sent_sens_v1.json",
        runner_path=runner_relative,
        source_files=(
            ProbeSourceFile(path=runner_relative, sha256="2" * 64, size_bytes=7),
            ProbeSourceFile(
                path="src/tunnelgeopt/fracture_benchmark.py",
                sha256="3" * 64,
                size_bytes=11,
            ),
        ),
        source_inventory_sha256="4" * 64,
        captured_utc="2026-08-14T00:00:00.000000Z",
        _project_root=project_root.resolve(),
    )


def _write(
    destination: Path,
    snapshot: ProbeProjectSnapshot,
    *,
    probes: dict | None = None,
):
    config = load_fracture_sent_sens_config()
    return campaign.write_paired_campaign_artifact_atomic(
        destination,
        probes=probes or {case: _FakeProbe(case) for case in campaign.CASE_ORDER},
        config=config,
        displacements={case: campaign.EXPECTED_THREE_STATE_GRID_MM for case in campaign.CASE_ORDER},
        formal_increment_counts={
            case: len(prescribed_displacements(config, case)) - 1 for case in campaign.CASE_ORDER
        },
        project_snapshot=snapshot,
        started_utc="2026-08-14T00:00:01.000000Z",
        completed_utc="2026-08-14T00:00:02.000000Z",
        postflight_verified_utc="2026-08-14T00:00:03.000000Z",
        sanitized_command=("python", "scripts/run_fracture_benchmark_campaign.py"),
        solver_options=FractureSolverOptions(),
        runtime_environment={
            "campaign_cpu_policy": {"requested_single_thread": True},
        },
        resource_measurement={
            "status": "AVAILABLE",
            "peak_rss_bytes": 1234,
            "method": "mock",
            "scope": "test",
        },
        paired_wall_seconds=1.0,
    )


def test_main_captures_once_runs_sent_then_sens_six_states_and_postflights_once(
    tmp_path, monkeypatch
) -> None:
    runner = tmp_path / "scripts" / "run_fracture_benchmark_campaign.py"
    runner.parent.mkdir()
    runner.write_text("# mock\n", encoding="utf-8")
    snapshot = _snapshot(tmp_path, "scripts/run_fracture_benchmark_campaign.py")
    output = tmp_path / "evidence" / "paired-run"
    calls: list[str] = []
    solved_states: list[tuple[str, float]] = []

    monkeypatch.setattr(campaign, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(campaign, "RUNNER_PATH", runner)

    def capture(*args, **kwargs):
        calls.append("capture")
        return snapshot

    def reserve(project_root, output_directory):
        calls.append("reserve")
        output.mkdir(parents=True)
        return output, "evidence/paired-run"

    def generate(*, loading, tier):
        calls.append(f"mesh:{loading}:{tier}")
        return SimpleNamespace(case=loading)

    def run_probe(config, mesh, *, benchmark_id, tier, displacements_mm, options):
        calls.append(f"probe:{benchmark_id}:{tier}")
        assert mesh.case == benchmark_id
        solved_states.extend((benchmark_id, value) for value in displacements_mm)
        return _FakeProbe(benchmark_id)

    def postflight(received_snapshot):
        calls.append("postflight")
        assert received_snapshot is snapshot
        assert output.is_dir() and not any(output.iterdir())
        return "2026-08-14T00:00:03.000000Z"

    monkeypatch.setattr(fracture_benchmark_module, "capture_probe_project_preflight", capture)
    monkeypatch.setattr(fracture_benchmark_module, "reserve_probe_output_directory", reserve)
    monkeypatch.setattr(fracture_benchmark_module, "_git_path_is_ignored", lambda *args: False)
    monkeypatch.setattr(fracture_benchmark_module, "probe_runtime_environment", dict)
    monkeypatch.setattr(fracture_benchmark_module, "run_intact_fracture_benchmark_probe", run_probe)
    monkeypatch.setattr(fracture_benchmark_module, "verify_probe_project_postflight", postflight)
    monkeypatch.setattr(fracture_mesh_module, "generate_fracture_benchmark_mesh", generate)
    monkeypatch.setattr(
        campaign,
        "_peak_rss_measurement",
        lambda: {
            "status": "AVAILABLE",
            "peak_rss_bytes": 1234,
            "method": "mock",
            "scope": "test",
        },
    )

    assert (
        campaign.main(
            [
                "--output",
                str(output),
                "--expected-project-head",
                "1" * 40,
                "--run-paired-intact-probe",
                "--approved-development-probe",
            ]
        )
        == 0
    )
    assert calls == [
        "capture",
        "reserve",
        "mesh:sent:coarse",
        "probe:sent:coarse",
        "mesh:sens:coarse",
        "probe:sens:coarse",
        "postflight",
    ]
    assert solved_states == [
        (case, state)
        for case in campaign.CASE_ORDER
        for state in campaign.EXPECTED_THREE_STATE_GRID_MM
    ]
    assert (output / "artifact_manifest.json").is_file()
    assert len(list(output.rglob("result.json"))) == 2


def test_writer_publishes_five_hash_linked_artifacts_and_triage_boundaries(tmp_path) -> None:
    destination = tmp_path / "paired"
    destination.mkdir()
    bundle = _write(destination, _snapshot(tmp_path, "scripts/run.py"))
    manifest_raw = (destination / "artifact_manifest.json").read_bytes()
    manifest = json.loads(manifest_raw)
    result = json.loads((destination / "campaign_result.json").read_bytes())

    assert bundle.artifact_manifest_sha256 == hashlib.sha256(manifest_raw).hexdigest()
    assert manifest["artifact_count"] == 4
    assert manifest["artifact_set_file_count_including_this_manifest"] == 5
    assert [record["path"] for record in manifest["artifacts"]] == [
        "cases/sent/result.json",
        "cases/sens/result.json",
        "implementation_manifest.json",
        "campaign_result.json",
    ]
    for record in manifest["artifacts"]:
        payload = (destination / record["path"]).read_bytes()
        assert record["sha256"] == hashlib.sha256(payload).hexdigest()
        assert record["size_bytes"] == len(payload)
    assert result["case_order"] == ["sent", "sens"]
    assert result["paired_resources"]["wall_seconds"] == 1.0
    assert result["evidence_scope"]["real_paired_probe_completed"] is True
    assert all(
        values["positive_reaction_magnitude_gate_passed"] for values in result["case_qc"].values()
    )
    assert result["authorizes_coupled_fracture_run"] is False
    assert result["paper_effect_evidence"] is False
    assert result["implementation_manifest"]["sha256"] == (bundle.implementation_manifest_sha256)
    assert str(tmp_path).replace("\\", "/") not in manifest_raw.decode("utf-8")


@pytest.mark.parametrize("fault", ("bad_protocol", "bad_qc", "nonfinite"))
def test_any_identity_or_qc_failure_leaves_outer_leaf_without_completion(
    tmp_path, fault: str
) -> None:
    destination = tmp_path / fault
    destination.mkdir()
    probes = {case: _FakeProbe(case) for case in campaign.CASE_ORDER}
    if fault == "bad_protocol":
        probes["sens"] = _FakeProbe("sens", protocol_sha256="9" * 64)
    elif fault == "bad_qc":
        probes["sens"].payload["steps"][1]["equilibrium_relative_residual"] = 1.0
    else:
        probes["sent"].payload["steps"][1]["wall_seconds"] = float("nan")

    with pytest.raises(campaign.FractureBenchmarkCampaignError):
        _write(destination, _snapshot(tmp_path, "scripts/run.py"), probes=probes)
    assert not (destination / "artifact_manifest.json").exists()
    assert not any(destination.iterdir())


@pytest.mark.parametrize("generalized_load", (0.0, -1.0, float("nan"), float("inf")))
def test_positive_displacement_requires_finite_strictly_positive_reaction(
    tmp_path, generalized_load: float
) -> None:
    destination = tmp_path / "bad-reaction"
    destination.mkdir()
    probes = {case: _FakeProbe(case) for case in campaign.CASE_ORDER}
    probes["sent"].payload["steps"][1]["generalized_load_kN"] = generalized_load

    with pytest.raises(campaign.FractureBenchmarkCampaignError, match="reaction magnitude"):
        _write(destination, _snapshot(tmp_path, "scripts/run.py"), probes=probes)
    assert not any(destination.iterdir())


def test_zero_displacement_allows_finite_near_zero_reaction(tmp_path) -> None:
    destination = tmp_path / "zero-state-near-zero"
    destination.mkdir()
    probes = {case: _FakeProbe(case) for case in campaign.CASE_ORDER}
    probes["sent"].payload["steps"][0]["generalized_load_kN"] = -1.0e-14

    _write(destination, _snapshot(tmp_path, "scripts/run.py"), probes=probes)
    assert (destination / "artifact_manifest.json").is_file()


@pytest.mark.parametrize("generalized_load", (-1.0e-6, 1.0e-6))
def test_zero_displacement_rejects_nonzero_baseline_reaction(
    tmp_path, generalized_load: float
) -> None:
    destination = tmp_path / "bad-zero-state-reaction"
    destination.mkdir()
    probes = {case: _FakeProbe(case) for case in campaign.CASE_ORDER}
    probes["sent"].payload["steps"][0]["generalized_load_kN"] = generalized_load

    with pytest.raises(campaign.FractureBenchmarkCampaignError, match="numerical zero tolerance"):
        _write(destination, _snapshot(tmp_path, "scripts/run.py"), probes=probes)
    assert not any(destination.iterdir())


def test_writer_refuses_existing_content_and_concurrent_completion(tmp_path) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "foreign.txt").write_text("occupied", encoding="utf-8")
    with pytest.raises(campaign.FractureBenchmarkCampaignError, match="empty"):
        _write(existing, _snapshot(tmp_path, "scripts/run.py"))
    assert not (existing / "artifact_manifest.json").exists()

    concurrent = tmp_path / "concurrent"
    concurrent.mkdir()
    snapshot = _snapshot(tmp_path, "scripts/run.py")
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(_write, concurrent, snapshot) for _ in range(2)]
    successes = 0
    failures = 0
    for future in futures:
        try:
            future.result()
            successes += 1
        except (FileExistsError, campaign.FractureBenchmarkCampaignError):
            failures += 1
    assert successes == 1
    assert failures == 1
    assert (concurrent / "artifact_manifest.json").is_file()


def test_snapshot_source_closure_requires_this_runner(tmp_path, monkeypatch) -> None:
    runner = tmp_path / "scripts" / "run_fracture_benchmark_campaign.py"
    runner.parent.mkdir()
    runner.write_text("# mock\n", encoding="utf-8")
    monkeypatch.setattr(campaign, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(campaign, "RUNNER_PATH", runner)
    snapshot = _snapshot(tmp_path, "scripts/another_runner.py")

    with pytest.raises(campaign.FractureBenchmarkCampaignError, match="not bound"):
        campaign._validate_snapshot_source_closure(snapshot)


def test_cli_rejects_before_snapshot_or_reservation(monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("preflight must not run for unapproved campaign")

    monkeypatch.setattr(fracture_benchmark_module, "capture_probe_project_preflight", forbidden)
    with pytest.raises(SystemExit, match="approved-development-probe"):
        campaign.main(
            [
                "--output",
                "unused",
                "--expected-project-head",
                "1" * 40,
                "--run-paired-intact-probe",
            ]
        )


@pytest.mark.parametrize("ignored_artifact", campaign.CAMPAIGN_ARTIFACT_RELATIVE_PATHS)
def test_each_ignored_fixed_artifact_rejects_before_leaf_and_solve(
    tmp_path, monkeypatch, ignored_artifact: str
) -> None:
    runner = tmp_path / "scripts" / "run_fracture_benchmark_campaign.py"
    runner.parent.mkdir()
    runner.write_text("# mock\n", encoding="utf-8")
    snapshot = _snapshot(tmp_path, "scripts/run_fracture_benchmark_campaign.py")
    output = tmp_path / "evidence" / "ignored-run"

    monkeypatch.setattr(campaign, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(campaign, "RUNNER_PATH", runner)
    monkeypatch.setattr(
        fracture_benchmark_module,
        "capture_probe_project_preflight",
        lambda *args, **kwargs: snapshot,
    )
    monkeypatch.setattr(
        fracture_benchmark_module,
        "_git_path_is_ignored",
        lambda project_root, relative: relative.endswith(ignored_artifact),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("reservation and solve must not run after ignored-path rejection")

    monkeypatch.setattr(fracture_benchmark_module, "reserve_probe_output_directory", forbidden)
    monkeypatch.setattr(fracture_mesh_module, "generate_fracture_benchmark_mesh", forbidden)

    with pytest.raises(campaign.FractureBenchmarkCampaignError, match="ignored by Git"):
        campaign.main(
            [
                "--output",
                str(output),
                "--expected-project-head",
                "1" * 40,
                "--run-paired-intact-probe",
                "--approved-development-probe",
            ]
        )
    assert not output.exists()
