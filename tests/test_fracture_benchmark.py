from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("scipy")

import tunnelgeopt.fracture_benchmark as fracture_benchmark_module
from scripts import run_fracture_benchmark_probe as probe_runner
from tunnelgeopt.fracture import FractureSolverOptions
from tunnelgeopt.fracture_benchmark import (
    FractureBenchmarkPreflightError,
    ProbeProjectSnapshot,
    ProbeProvenanceError,
    ProbeSourceFile,
    benchmark_material,
    build_prescribed_displacement_states,
    capture_probe_project_preflight,
    lame_to_young_poisson,
    preflight_fracture_benchmark,
    reserve_probe_output_directory,
    run_intact_fracture_benchmark_probe,
    verify_probe_project_postflight,
    write_probe_artifact_atomic,
)
from tunnelgeopt.fracture_benchmark_mesh import benchmark_mesh_plan
from tunnelgeopt.fracture_benchmark_validation import load_fracture_sent_sens_config
from tunnelgeopt.fracture_bvp import prescribed_displacement_mesh_identity


class _FixtureMesh:
    def __init__(self, benchmark_id: str = "sent") -> None:
        nodes = np.asarray([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.5, 0.5]])
        elements = np.asarray([[0, 4, 1], [1, 4, 3], [3, 4, 2], [2, 4, 0]])
        facets = np.asarray(
            [
                [2, 0, 2, 0, 1, 0, 0],
                [3, 1, 4, 4, 3, 4, 4],
            ],
            dtype=np.int64,
        )
        self.mesh = SimpleNamespace(p=nodes.T, t=elements.T, facets=facets)
        self.nodes = nodes
        self.elements = elements
        self.boundary_facets = MappingProxyType(
            {
                "top": np.asarray([0]),
                "bottom": np.asarray([1]),
                "left_upper": np.asarray([2]),
                "left_lower": np.asarray([3]),
                "right": np.asarray([4]),
                "notch_upper": np.asarray([5]),
                "notch_lower": np.asarray([6]),
            }
        )
        self.boundary_nodes = MappingProxyType({"notch_tip": np.asarray([4])})
        self.plan = benchmark_mesh_plan(loading=benchmark_id, tier="coarse")
        self.identity = MappingProxyType(
            {
                "coordinate_order": ("y", "z"),
                "plan_sha256": self.plan.plan_sha256,
                "topology_sha256": "a" * 64,
            }
        )
        self.metadata = MappingProxyType(
            {
                "topology_audit_passed": True,
                "boundary_coverage_audit_passed": True,
                "zero_width_double_face_slit_audit_passed": True,
                "corridor_hmax_audit_passed": True,
            }
        )

    def recompute_topology_sha256(self) -> str:
        return str(self.identity["topology_sha256"])


@pytest.fixture
def config() -> dict:
    return load_fracture_sent_sens_config()


def _fake_project_snapshot(project_root) -> ProbeProjectSnapshot:
    digest = "c" * 64
    return ProbeProjectSnapshot(
        expected_project_head="a" * 40,
        project_head="a" * 40,
        upstream_head="a" * 40,
        upstream_ref="origin/test",
        config_path="configs/fracture_sent_sens_v1.json",
        runner_path="scripts/run_fracture_benchmark_probe.py",
        source_files=(
            ProbeSourceFile(path="src/tunnelgeopt/__init__.py", sha256="b" * 64, size_bytes=1),
        ),
        source_inventory_sha256=digest,
        captured_utc="2026-08-14T00:00:00.000000Z",
        _project_root=project_root.resolve(),
    )


def _write_bundle(probe, destination, snapshot):
    return write_probe_artifact_atomic(
        probe,
        destination,
        project_snapshot=snapshot,
        started_utc="2026-08-14T00:00:01.000000Z",
        completed_utc="2026-08-14T00:00:02.000000Z",
        sanitized_command=(
            "python",
            "scripts/run_fracture_benchmark_probe.py",
            "--expected-project-head",
            "a" * 40,
        ),
        solver_options=FractureSolverOptions(),
        runtime_environment={"threads": {"logical_cpu_count": 1}},
    )


def test_lame_conversion_is_exactly_regressed(config: dict) -> None:
    lam = config["material"]["lame_lambda_kN_per_mm2"]
    mu = config["material"]["shear_modulus_kN_per_mm2"]
    young, poisson = lame_to_young_poisson(lam, mu)
    material = benchmark_material(config)

    assert young == pytest.approx(210.0, rel=5.0e-5)
    assert poisson == pytest.approx(0.3000, rel=5.0e-5)
    assert material.young_modulus == young
    assert material.poisson_ratio == poisson
    assert material.lame_lambda == pytest.approx(lam, rel=2.0e-14)
    assert material.shear_modulus == pytest.approx(mu, rel=2.0e-14)


def test_preflight_rejects_coordinate_rotation_and_target_mismatch(config: dict) -> None:
    rotated = _FixtureMesh()
    rotated.nodes = rotated.nodes[:, ::-1].copy()
    rotated.mesh.p = rotated.nodes.T
    with pytest.raises(FractureBenchmarkPreflightError, match="top label is not y=1"):
        preflight_fracture_benchmark(config, rotated, benchmark_id="sent", tier="coarse")

    wrong_target = _FixtureMesh()
    wrong_target.plan = replace(wrong_target.plan, target_h_mm=0.008)
    with pytest.raises(FractureBenchmarkPreflightError, match="target_h_mm"):
        preflight_fracture_benchmark(config, wrong_target, benchmark_id="sent", tier="coarse")


def test_preflight_rejects_stale_topology_identity(config: dict) -> None:
    mesh = _FixtureMesh()
    mesh.recompute_topology_sha256 = lambda: "b" * 64  # type: ignore[method-assign]
    with pytest.raises(FractureBenchmarkPreflightError, match="topology differs"):
        preflight_fracture_benchmark(config, mesh, benchmark_id="sent", tier="coarse")

    missing_recompute = _FixtureMesh()
    missing_recompute.recompute_topology_sha256 = None  # type: ignore[method-assign]
    with pytest.raises(FractureBenchmarkPreflightError, match="live topology"):
        preflight_fracture_benchmark(config, missing_recompute, benchmark_id="sent", tier="coarse")


@pytest.mark.parametrize(
    ("benchmark_id", "driven_group", "driven_parity"),
    (("sent", "top_u_y", 0), ("sens", "top_u_z", 1)),
)
def test_sent_sens_states_have_exact_node_major_dof_mapping(
    config: dict, benchmark_id: str, driven_group: str, driven_parity: int
) -> None:
    mesh = _FixtureMesh(benchmark_id)
    states = build_prescribed_displacement_states(
        config,
        mesh,
        benchmark_id=benchmark_id,
        tier="coarse",
        displacements_mm=(0.0, 1.0e-8),
    )
    expected_identity = prescribed_displacement_mesh_identity(mesh.mesh)

    assert [state.sequence_index for state in states] == [0, 1]
    assert [state.path_parameter for state in states] == [0.0, 1.0e-8]
    assert all(state.mesh_identity == expected_identity for state in states)
    assert all(state.driven_group == driven_group for state in states)
    assert np.all(states[1].reaction_groups[driven_group] % 2 == driven_parity)
    assert np.all(
        states[1].dirichlet_values[
            np.isin(states[1].dirichlet_dofs, states[1].reaction_groups[driven_group])
        ]
        == 1.0e-8
    )
    other = "top_u_z" if benchmark_id == "sent" else "top_u_y"
    assert np.all(
        states[1].dirichlet_values[
            np.isin(states[1].dirichlet_dofs, states[1].reaction_groups[other])
        ]
        == 0.0
    )


def test_intact_d0_two_state_path_solves_without_formal_trajectory(config: dict) -> None:
    mesh = _FixtureMesh("sent")
    probe = run_intact_fracture_benchmark_probe(
        config,
        mesh,
        benchmark_id="sent",
        tier="coarse",
        displacements_mm=(0.0, 1.0e-8),
        options=FractureSolverOptions(
            max_displacement_iterations=10,
            equilibrium_tolerance=1.0e-8,
        ),
    )

    assert len(probe.steps) == 2
    assert all(step.converged for step in probe.steps)
    assert all(
        step.damage_component_status == "NOT_APPLICABLE_INTACT_D0_PROBE" for step in probe.steps
    )
    assert probe.projected_formal_increment_count == 2000
    assert probe.projection_interpretation == "intact_fixed_damage_lower_bound_non_authorizing"
    assert probe.authorizes_medium_fine_or_formal_run is False


def test_probe_mock_records_each_duration_and_exclusive_bundle_hashes(
    config: dict, tmp_path, monkeypatch
) -> None:
    mesh = _FixtureMesh("sens")
    calls: list[str] = []

    def mock_solver(mesh_like, material, state, *, damage, initial_displacement, options):
        calls.append(state.identity)
        displacement = np.zeros((mesh_like.p.shape[1], 2))
        displacement.ravel()[state.dirichlet_dofs] = state.dirichlet_values
        reaction = np.zeros(2 * mesh_like.p.shape[1])
        reaction[state.reaction_groups[state.driven_group]] = state.path_parameter
        reaction[state.reaction_groups["bottom_u_z"]] = -state.path_parameter
        return SimpleNamespace(
            state=state,
            mesh_identity=state.mesh_identity,
            displacement=displacement,
            reaction=reaction,
            external_force=np.zeros_like(reaction),
            dirichlet_dofs=state.dirichlet_dofs,
            elastic_energy=state.path_parameter**2,
            converged=True,
            generalized_load=float(reaction[state.reaction_groups[state.driven_group]].sum()),
            equilibrium_residual=0.0,
        )

    ticks = iter((0.0, 0.25, 1.0, 1.5, 2.0, 2.75))
    probe = run_intact_fracture_benchmark_probe(
        config,
        mesh,
        benchmark_id="sens",
        tier="coarse",
        displacements_mm=(0.0, 1.0e-8, 2.0e-8),
        step_solver=mock_solver,
        clock=lambda: next(ticks),
    )
    assert len(calls) == 3
    assert [step.wall_seconds for step in probe.steps] == [0.25, 0.5, 0.75]
    assert probe.median_step_wall_seconds == 0.5
    assert probe.projected_formal_increment_count == 1500

    monkeypatch.setattr(
        fracture_benchmark_module,
        "verify_probe_project_postflight",
        lambda snapshot: "2026-08-14T00:00:03.000000Z",
    )
    monkeypatch.setattr(fracture_benchmark_module, "_git_path_is_ignored", lambda *args: False)
    privacy_calls: list[str] = []
    original_privacy_guard = fracture_benchmark_module._reject_host_path_strings

    def record_privacy_guard(value, project_root, path="artifact"):
        if path == "artifact":
            privacy_calls.append(value["schema"])
        return original_privacy_guard(value, project_root, path)

    monkeypatch.setattr(
        fracture_benchmark_module, "_reject_host_path_strings", record_privacy_guard
    )
    destination, _ = reserve_probe_output_directory(tmp_path, tmp_path / "probe-run")
    bundle = _write_bundle(probe, destination, _fake_project_snapshot(tmp_path))
    result_raw = (destination / "result.json").read_bytes()
    manifest_raw = (destination / "artifact_manifest.json").read_bytes()
    result = json.loads(result_raw)
    manifest = json.loads(manifest_raw)
    assert len(bundle.result_sha256) == len(bundle.manifest_sha256) == 64
    assert bundle.result_sha256 == fracture_benchmark_module.hashlib.sha256(result_raw).hexdigest()
    assert (
        bundle.manifest_sha256 == fracture_benchmark_module.hashlib.sha256(manifest_raw).hexdigest()
    )
    assert manifest["artifacts"] == [
        {
            "path": "result.json",
            "sha256": bundle.result_sha256,
            "size_bytes": len(result_raw),
        }
    ]
    assert result["probe"]["authorizes_medium_fine_or_formal_run"] is False
    assert result["evidence_scope"] == {
        "paired_sent_sens_campaign_supported": False,
        "paper_evidence_eligible": False,
        "real_probe_allowed": False,
        "real_probe_definition": "paired_sent_sens_campaign_for_paper_evidence",
        "single_case_only": True,
    }
    assert manifest["real_probe_allowed"] is False
    assert privacy_calls == [
        fracture_benchmark_module.PROBE_RESULT_SCHEMA,
        fracture_benchmark_module.PROBE_MANIFEST_SCHEMA,
    ]
    assert result["execution"]["solver_options"]["equilibrium_tolerance"] == 1.0e-8
    assert "C:\\Users\\" not in result_raw.decode("utf-8")


def test_probe_bundle_refuses_existing_and_concurrent_run_leaf(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(fracture_benchmark_module, "_git_path_is_ignored", lambda *args: False)
    existing = tmp_path / "existing-run"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        reserve_probe_output_directory(tmp_path, existing)

    concurrent = tmp_path / "concurrent-run"
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(reserve_probe_output_directory, tmp_path, concurrent) for _ in range(2)
        ]
    outcomes = []
    for future in futures:
        try:
            outcomes.append(future.result())
        except FileExistsError:
            outcomes.append("EXISTS")
    assert sum(outcome == "EXISTS" for outcome in outcomes) == 1
    assert sum(outcome != "EXISTS" for outcome in outcomes) == 1
    assert not any(concurrent.iterdir())


def test_probe_bundle_recursively_rejects_nonfinite_result(
    config: dict, tmp_path, monkeypatch
) -> None:
    mesh = _FixtureMesh("sent")
    probe = run_intact_fracture_benchmark_probe(
        config,
        mesh,
        benchmark_id="sent",
        tier="coarse",
        displacements_mm=(0.0,),
    )
    bad_step = replace(probe.steps[0], generalized_load_kN=float("nan"))
    bad_probe = replace(probe, steps=(bad_step,))
    monkeypatch.setattr(
        fracture_benchmark_module,
        "verify_probe_project_postflight",
        lambda snapshot: "2026-08-14T00:00:03.000000Z",
    )
    monkeypatch.setattr(fracture_benchmark_module, "_git_path_is_ignored", lambda *args: False)
    destination, _ = reserve_probe_output_directory(tmp_path, tmp_path / "nonfinite-run")
    with pytest.raises(ValueError, match="non-finite"):
        _write_bundle(bad_probe, destination, _fake_project_snapshot(tmp_path))
    assert destination.is_dir() and not any(destination.iterdir())


def test_probe_postflight_rejects_source_inventory_mismatch(tmp_path, monkeypatch) -> None:
    snapshot = _fake_project_snapshot(tmp_path)

    def fake_git_text(project_root, *arguments):
        if arguments[:2] == ("status", "--porcelain=v1"):
            return ""
        return "a" * 40

    monkeypatch.setattr(fracture_benchmark_module, "_git_text", fake_git_text)
    monkeypatch.setattr(
        fracture_benchmark_module,
        "_source_inventory",
        lambda project_root, **paths: (snapshot.source_files, "d" * 64),
    )
    with pytest.raises(ProbeProvenanceError, match="source closure changed"):
        verify_probe_project_postflight(snapshot)


def test_probe_preflight_rejects_expected_head_mismatch(tmp_path, monkeypatch) -> None:
    def fake_git_text(project_root, *arguments):
        if arguments == ("rev-parse", "--show-toplevel"):
            return str(tmp_path)
        if arguments == ("rev-parse", "HEAD"):
            return "b" * 40
        if arguments == ("rev-parse", "@{upstream}"):
            return "a" * 40
        if arguments == ("rev-parse", "--abbrev-ref", "@{upstream}"):
            return "origin/test"
        raise AssertionError(arguments)

    monkeypatch.setattr(fracture_benchmark_module, "_git_text", fake_git_text)
    with pytest.raises(ProbeProvenanceError, match="HEAD == upstream HEAD"):
        capture_probe_project_preflight(
            tmp_path,
            expected_project_head="a" * 40,
            config_path=tmp_path / "config.json",
            runner_path=tmp_path / "runner.py",
        )


def test_unapproved_probe_rejected_before_reservation_or_mesh(tmp_path, monkeypatch) -> None:
    output = tmp_path / "must-not-exist"

    def forbidden_mesh(**kwargs):
        raise AssertionError("mesh generation must not run before approval validation")

    monkeypatch.setattr(probe_runner, "generate_fracture_benchmark_mesh", forbidden_mesh)
    with pytest.raises(SystemExit, match="approved-development-probe"):
        probe_runner.main(
            [
                "--benchmark",
                "sent",
                "--run-intact-probe",
                "--expected-project-head",
                "a" * 40,
                "--u-mm",
                "0",
                "--output",
                str(output),
            ]
        )
    assert not output.exists()


def test_probe_cap_rejected_before_reservation_or_mesh(tmp_path, monkeypatch) -> None:
    output = tmp_path / "must-not-exist"

    def forbidden_mesh(**kwargs):
        raise AssertionError("mesh generation must not run after over-cap request")

    monkeypatch.setattr(probe_runner, "generate_fracture_benchmark_mesh", forbidden_mesh)
    argv = [
        "--benchmark",
        "sent",
        "--run-intact-probe",
        "--approved-development-probe",
        "--expected-project-head",
        "a" * 40,
        "--output",
        str(output),
    ]
    for index in range(13):
        argv.extend(("--u-mm", str(index * 1.0e-8)))
    with pytest.raises(SystemExit, match="capped at 12"):
        probe_runner.main(argv)
    assert not output.exists()


@pytest.mark.parametrize("relative", ("tmp/probe-run", "outputs/probe-run"))
def test_probe_reservation_rejects_git_ignored_paths(relative: str) -> None:
    project_root = Path(fracture_benchmark_module.__file__).resolve().parents[2]
    target = project_root / relative
    assert not target.exists()
    with pytest.raises(ProbeProvenanceError, match="ignored by Git"):
        reserve_probe_output_directory(project_root, target)
    assert not target.exists()


def test_probe_reservation_rejects_ignored_artifact_filename(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        fracture_benchmark_module,
        "_git_path_is_ignored",
        lambda project_root, relative: relative.endswith("result.json"),
    )
    target = tmp_path / "candidate-run"
    with pytest.raises(ProbeProvenanceError, match="artifact files"):
        reserve_probe_output_directory(tmp_path, target)
    assert not target.exists()


def test_probe_bundle_rejects_project_path_in_nested_string(
    config: dict, tmp_path, monkeypatch
) -> None:
    mesh = _FixtureMesh("sent")
    probe = run_intact_fracture_benchmark_probe(
        config,
        mesh,
        benchmark_id="sent",
        tier="coarse",
        displacements_mm=(0.0,),
    )
    monkeypatch.setattr(
        fracture_benchmark_module,
        "verify_probe_project_postflight",
        lambda snapshot: "2026-08-14T00:00:03.000000Z",
    )
    monkeypatch.setattr(fracture_benchmark_module, "_git_path_is_ignored", lambda *args: False)
    destination, _ = reserve_probe_output_directory(tmp_path, tmp_path / "privacy-run")
    with pytest.raises(ProbeProvenanceError, match="local project path"):
        write_probe_artifact_atomic(
            probe,
            destination,
            project_snapshot=_fake_project_snapshot(tmp_path),
            started_utc="2026-08-14T00:00:01.000000Z",
            completed_utc="2026-08-14T00:00:02.000000Z",
            sanitized_command=("python", "runner.py"),
            solver_options=FractureSolverOptions(),
            runtime_environment={
                "nested": {"leak": f"prefix-{str(tmp_path).replace(chr(92), '/')}-suffix"}
            },
        )
    assert not any(destination.iterdir())


def test_probe_cap_prevents_accidental_trajectory(config: dict) -> None:
    mesh = _FixtureMesh()
    with pytest.raises(ValueError, match="capped at 12"):
        run_intact_fracture_benchmark_probe(
            config,
            mesh,
            benchmark_id="sent",
            tier="coarse",
            displacements_mm=np.linspace(0.0, 1.2e-8, 13),
        )
