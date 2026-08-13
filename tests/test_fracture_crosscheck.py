from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tunnelgeopt.elasticity import plane_strain_sigma_xx
from tunnelgeopt.fracture_crosscheck import (
    CrosscheckValidationError,
    MooseCaseResult,
    MooseUnavailableError,
    build_local_tunnel_mesh,
    compare_case,
    load_canonical_mesh,
    load_crosscheck_config,
    parse_gmsh_v22_ascii,
    parse_moose_case_output,
    render_moose_input,
    run_crosscheck,
    solve_local_case,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "fracture_crosscheck_v1.json"


def _config() -> dict:
    return load_crosscheck_config(CONFIG_PATH)


def test_config_and_canonical_mesh_are_strict_and_hash_complete(tmp_path: Path) -> None:
    config = _config()
    assert config["execution"]["moose_executable_linux"].startswith("~/")
    mesh = load_canonical_mesh(config)
    assert mesh.nodes_xy.shape == (8, 2)
    assert mesh.triangles.shape == (8, 3)
    assert mesh.boundary_edges(2).shape == (4, 2)
    assert mesh.boundary_edges(3).shape == (4, 2)
    assert len(mesh.file_sha256) == len(mesh.structure_sha256) == 64
    assert np.all(mesh.triangle_areas > 0.0)

    tampered = tmp_path / "tampered.msh"
    text = mesh.path.read_text(encoding="ascii").replace("1 -2 -2 0", "1 -2.01 -2 0")
    tampered.write_text(text, encoding="ascii")
    parsed = parse_gmsh_v22_ascii(tampered)
    assert parsed.file_sha256 != mesh.file_sha256
    assert parsed.structure_sha256 != mesh.structure_sha256


def test_config_rejects_unknown_key_and_weaker_primary_tolerance(tmp_path: Path) -> None:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["unknown"] = True
    path = tmp_path / "unknown.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CrosscheckValidationError, match="keys mismatch"):
        load_crosscheck_config(path)

    payload.pop("unknown")
    payload["comparison"]["primary_relative_l2_tolerance"] = 1.0e-5
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CrosscheckValidationError, match="may not exceed"):
        load_crosscheck_config(path)


def test_mesh_rejects_foreign_physical_name_and_negative_orientation(tmp_path: Path) -> None:
    text = (ROOT / "moose/fracture_crosscheck/canonical_square_annulus_v1.msh").read_text(
        encoding="ascii"
    )
    wrong_name = tmp_path / "wrong_name.msh"
    wrong_name.write_text(text.replace('1 2 "wall"', '1 2 "cavity"'), encoding="ascii")
    with pytest.raises(CrosscheckValidationError, match="physical names"):
        parse_gmsh_v22_ascii(wrong_name)

    flipped = tmp_path / "flipped.msh"
    flipped.write_text(text.replace("9 2 2 1 1 1 2 6", "9 2 2 1 1 1 6 2"), encoding="ascii")
    with pytest.raises(CrosscheckValidationError, match="positive orientation"):
        parse_gmsh_v22_ascii(flipped)


def test_local_mesh_boundary_partition_and_three_intact_basis_solutions() -> None:
    pytest.importorskip("scipy")
    pytest.importorskip("skfem")
    config = _config()
    mesh = load_canonical_mesh(config)
    tunnel_mesh = build_local_tunnel_mesh(mesh)
    assert tunnel_mesh.metadata["mesh_file_sha256"] == mesh.file_sha256
    results = [solve_local_case(config, mesh, gate="intact", basis_index=i) for i in range(3)]
    assert {result.case_id for result in results} == {
        "intact-basis-0",
        "intact-basis-1",
        "intact-basis-2",
    }
    for result in results:
        assert result.displacement_xy.shape == mesh.nodes_xy.shape
        assert result.stress_inplane.shape == (mesh.triangles.shape[0], 3)
        assert np.isfinite(result.elastic_energy) and result.elastic_energy > 0.0
        assert np.isfinite(result.reaction_xy).all()


def test_fixed_damage_is_nonuniform_strictly_bounded_and_rendered_additively() -> None:
    pytest.importorskip("scipy")
    pytest.importorskip("skfem")
    config = _config()
    mesh = load_canonical_mesh(config)
    result = solve_local_case(config, mesh, gate="fixed_damage", basis_index=0)
    assert 0.0 < float(result.damage.min()) < float(result.damage.max()) < 1.0
    rendered = render_moose_input(config, result)
    assert "expression = '(1.0-c)^2 + eta'" in rendered
    assert "allow_renumbering = false" in rendered
    assert "planar_formulation = PLANE_STRAIN" in rendered
    assert "out_of_plane_direction = z" in rendered
    assert "element_order = THIRD" in rendered
    assert "side_order = THIRD" in rendered
    assert rendered.count("enable_jit = false") == 2
    assert "quadrature_order" not in rendered
    assert "boundary = wall" not in rendered
    assert "prop_names = 'l gc_prop'" in rendered
    assert "@" not in rendered
    undegraded_axis = plane_strain_sigma_xx(
        result.strain_engineering,
        young_modulus=config["material"]["young_modulus_pa"],
        poisson_ratio=config["material"]["poisson_ratio"],
    )
    assert not np.allclose(result.stress_axis, undegraded_axis, rtol=1e-6, atol=1e-6)

    intact = solve_local_case(config, mesh, gate="intact", basis_index=0)
    intact_rendered = render_moose_input(config, intact)
    assert intact_rendered.count("enable_jit = false") == 0


def _as_exact_moose(local) -> MooseCaseResult:
    return MooseCaseResult(
        node_ids=np.arange(local.nodes_xy.shape[0], dtype=np.int64),
        nodes_xy=local.nodes_xy[::-1].copy(),
        displacement_xy=local.displacement_xy[::-1].copy(),
        residual_xy=local.internal_force_xy[::-1].copy(),
        damage=local.damage[::-1].copy() if local.gate == "fixed_damage" else None,
        element_ids=np.arange(local.element_centroids_xy.shape[0], dtype=np.int64),
        element_centroids_xy=local.element_centroids_xy[::-1].copy(),
        strain_engineering=local.strain_engineering[::-1].copy(),
        stress_inplane=local.stress_inplane[::-1].copy(),
        stress_axis=local.stress_axis[::-1].copy(),
        energy_density=local.energy_density[::-1].copy(),
    )


def test_comparison_uses_coordinate_bijections_not_row_or_id_order() -> None:
    pytest.importorskip("scipy")
    pytest.importorskip("skfem")
    config = _config()
    mesh = load_canonical_mesh(config)
    local = solve_local_case(config, mesh, gate="intact", basis_index=0)
    report = compare_case(config, mesh, local, _as_exact_moose(local))
    assert report["pass"] is True
    assert report["node_mapping"]["max_coordinate_error_m"] == 0.0
    assert report["element_mapping"]["max_coordinate_error_m"] == 0.0
    assert all(metric["primary_error"] == 0.0 for metric in report["metrics"].values())


def test_comparison_rejects_ambiguous_coordinates_and_detects_field_tamper() -> None:
    pytest.importorskip("scipy")
    pytest.importorskip("skfem")
    config = _config()
    mesh = load_canonical_mesh(config)
    local = solve_local_case(config, mesh, gate="intact", basis_index=1)
    exact = _as_exact_moose(local)
    duplicate_coordinates = exact.nodes_xy.copy()
    duplicate_coordinates[0] = duplicate_coordinates[1]
    with pytest.raises(CrosscheckValidationError, match="coordinate"):
        compare_case(config, mesh, local, replace(exact, nodes_xy=duplicate_coordinates))

    tampered = exact.stress_inplane.copy()
    tampered[0, 0] += 1000.0
    report = compare_case(config, mesh, local, replace(exact, stress_inplane=tampered))
    assert report["pass"] is False
    assert report["metrics"]["stress_inplane"]["pass"] is False


def test_fixed_damage_energy_uses_exported_moose_nodal_damage() -> None:
    pytest.importorskip("scipy")
    pytest.importorskip("skfem")
    config = _config()
    mesh = load_canonical_mesh(config)
    local = solve_local_case(config, mesh, gate="fixed_damage", basis_index=0)
    exact = _as_exact_moose(local)
    assert exact.damage is not None
    perturbed_damage = exact.damage.copy()
    perturbed_damage[0] += 1.0e-3
    report = compare_case(config, mesh, local, replace(exact, damage=perturbed_damage))
    assert report["metrics"]["nodal_damage"]["pass"] is False
    assert report["metrics"]["energy_density"]["pass"] is False
    assert "raw PF ElasticEnergyAux column excluded" in report["energy_contract"]


def _write_csv(path: Path, fieldnames: list[str], rows: list[list[float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fieldnames)
        writer.writerows(rows)


def test_moose_csv_parser_rejects_nonfinite_and_accepts_tensor_shear_conversion(
    tmp_path: Path,
) -> None:
    node_header = ["id", "x", "y", "z", "disp_x", "disp_y", "resid_x", "resid_y"]
    element_header = [
        "id",
        "x",
        "y",
        "z",
        "strain_xx",
        "strain_yy",
        "strain_xy",
        "stress_xx",
        "stress_yy",
        "stress_xy",
        "stress_zz",
        "energy_density",
    ]
    _write_csv(tmp_path / "moose_nodes_0001.csv", node_header, [[0, 0, 0, 0, 1, 2, 3, 4]])
    _write_csv(
        tmp_path / "moose_elements_0001.csv",
        element_header,
        [[0, 0.2, 0.3, 0, 0.01, 0.02, 0.03, 10, 20, 30, 40, 50]],
    )
    parsed = parse_moose_case_output(tmp_path, expect_damage=False)
    np.testing.assert_array_equal(parsed.strain_engineering, [[0.01, 0.02, 0.06]])

    _write_csv(
        tmp_path / "moose_elements_0001.csv",
        element_header,
        [[0, 0.2, 0.3, 0, 0.01, 0.02, float("nan"), 10, 20, 30, 40, 50]],
    )
    with pytest.raises(CrosscheckValidationError, match="non-finite"):
        parse_moose_case_output(tmp_path, expect_damage=False)


def test_prepare_only_writes_hashed_local_evidence_without_claiming_pass(tmp_path: Path) -> None:
    pytest.importorskip("scipy")
    pytest.importorskip("skfem")
    report = run_crosscheck(CONFIG_PATH, tmp_path / "evidence", run_moose=False)
    assert report["status"] == "prepared_local_only"
    assert report["pass"] is False
    assert report["validated_scope"] == "none; local preparation or failed cross-solver gate"
    manifest = json.loads((tmp_path / "evidence" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "prepared_local_only"
    assert manifest["discretization_contract"]["volume_quadrature"] == "GAUSS THIRD"
    assert "excluded" in manifest["discretization_contract"]["fixed_damage_raw_energy_aux"]
    provenance = manifest["project_source_provenance"]
    assert isinstance(provenance["repository"]["worktree_dirty"], bool)
    assert provenance["repository"]["worktree_dirty"] == (
        provenance["repository"]["status_entry_count"] > 0
    )
    assert len(provenance["repository"]["head_commit"]) == 40
    assert len(provenance["repository"]["status_porcelain_sha256"]) == 64
    implementation = provenance["implementation_files"]
    package_inventory = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "src" / "tunnelgeopt").glob("*.py")
        if path.is_file()
    }
    assert package_inventory <= set(implementation)
    assert "src/tunnelgeopt/fracture_crosscheck.py" in implementation
    assert "src/tunnelgeopt/fracture.py" in implementation
    assert "src/tunnelgeopt/mesh.py" in implementation
    assert "scripts/run_fracture_crosscheck.py" in implementation
    assert "moose/fracture_crosscheck/fixed_damage_same_mesh.i" in implementation
    assert all(len(entry["sha256"]) == 64 for entry in implementation.values())
    assert provenance["runtime"]["distributions"]["scikit-fem"]
    assert len(manifest["cases"]) == 6
    for case in manifest["cases"]:
        assert len(case["local_fields_sha256"]) == 64
        assert case["mesh_file_sha256"] == manifest["mesh_file_sha256"]


def test_artifact_directory_must_be_new_and_empty(tmp_path: Path) -> None:
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "stale_report.json").write_text('{"status":"pass"}', encoding="utf-8")
    with pytest.raises(CrosscheckValidationError, match="new empty directory"):
        run_crosscheck(CONFIG_PATH, occupied, run_moose=False)


def test_generated_parser_cache_is_scoped_recorded_and_purged(tmp_path: Path) -> None:
    from tunnelgeopt.fracture_crosscheck import _purge_ephemeral_parser_cache

    case_dir = tmp_path / "case"
    cache = case_dir / ".jitcache"
    cache.mkdir(parents=True)
    (cache / "d_a").write_bytes(b"abc")
    (cache / "d_b").write_bytes(b"12345")
    metadata = _purge_ephemeral_parser_cache(case_dir)
    assert metadata == {
        "path": ".jitcache",
        "generated": True,
        "file_count": 2,
        "total_bytes": 8,
        "purged": True,
        "included_in_evidence": False,
    }
    assert not cache.exists()


def test_sanitized_log_removes_windows_and_wsl_private_roots() -> None:
    from tunnelgeopt.fracture_crosscheck import MooseEnvironment, _sanitize_log

    environment = MooseEnvironment(
        distribution="Ubuntu",
        linux_home="/users/audit-user",
        executable_linux="/users/audit-user/projects/moose/modules/combined/combined-opt",
        application_version="test",
        executable_sha256="0" * 64,
        source_commit="1" * 40,
        upstream_commit="1" * 40,
        source_tree_clean=True,
    )
    workspace = ROOT.resolve()
    workspace_wsl = "/windows-mount/workspace"
    raw = (
        f"{workspace} {workspace_wsl}/case "
        "/users/audit-user/projects/moose/modules/combined/combined-opt /users/audit-user/x"
    )
    sanitized = _sanitize_log(
        raw,
        repo_root=workspace,
        repo_root_wsl=workspace_wsl,
        environment=environment,
    )
    assert str(workspace) not in sanitized
    assert workspace_wsl not in sanitized
    assert "/users/audit-user" not in sanitized
    assert "<MOOSE_EXECUTABLE>" in sanitized


def test_probe_rejects_executable_version_not_bound_to_source_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import subprocess

    import tunnelgeopt.fracture_crosscheck as crosscheck

    commit = "167ee97da204df5e8643695e18b9b28910f0014b"

    def fake_run_checked(command, *, timeout=120):
        del timeout
        joined = " ".join(command)
        if "--version" in command:
            return subprocess.CompletedProcess(command, 0, "Application Version: deadbeef\n", "")
        if "/bin/sh" in command:
            return subprocess.CompletedProcess(command, 0, "/users/audit-user", "")
        if any(part.endswith("/sha256sum") for part in command):
            return subprocess.CompletedProcess(command, 0, f"{'0' * 64}  combined-opt\n", "")
        if "@{upstream}" in command:
            return subprocess.CompletedProcess(command, 0, f"{commit}\n", "")
        if "status --porcelain" in joined:
            return subprocess.CompletedProcess(command, 0, "", "")
        return subprocess.CompletedProcess(command, 0, "", "")

    def fake_subprocess_run(command, **kwargs):
        del kwargs
        return subprocess.CompletedProcess(command, 0, f"{commit}\n", "")

    monkeypatch.setattr(crosscheck, "_run_checked", fake_run_checked)
    monkeypatch.setattr(crosscheck.subprocess, "run", fake_subprocess_run)
    with pytest.raises(MooseUnavailableError, match="not a prefix"):
        crosscheck.probe_moose(_config())
