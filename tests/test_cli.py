from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("gmsh")
pytest.importorskip("skfem")

from tunnelgeopt import __version__
from tunnelgeopt.cli import build_parser, main
from tunnelgeopt.elastic_schema import load_elastic_record


def _json_output(capsys: pytest.CaptureFixture[str]) -> dict:
    return json.loads(capsys.readouterr().out)


def test_version_and_parser_preserve_A_commands_and_add_B_commands() -> None:
    assert __version__ == "0.2.0"
    parser = build_parser()
    choices = next(action for action in parser._actions if action.dest == "command").choices
    assert set(choices) == {
        "generate",
        "validate",
        "kirsch-check",
        "elastic-solve",
        "elastic-validate",
        "elastic-kirsch",
    }


def test_elastic_solve_and_validate_real_roundtrip(tmp_path, capsys) -> None:
    case_dir = tmp_path / "elastic-case"
    exit_code = main(
        [
            "elastic-solve",
            "--shape",
            "circle",
            "--output",
            str(case_dir),
            "--boundary-points",
            "24",
            "--domain-scale",
            "3",
            "--mesh-size",
            "0.8",
            "--wall-mesh-size",
            "0.25",
            "--farfield-mesh-size",
            "0.8",
            "--young-modulus",
            "100",
            "--poisson-ratio",
            "0.25",
            "--sigma-yy-compression",
            "10",
            "--sigma-zz-compression",
            "5",
            "--tau-yz-compression",
            "1",
        ]
    )
    solved = _json_output(capsys)
    assert exit_code == 0
    assert solved["passed"] is True and solved["saved"] is True
    assert solved["input_stress_convention"] == "compression_positive"
    assert solved["internal_stress_convention"] == "tension_positive"
    assert solved["input_sigma_yz_pa"] == [[10.0, 1.0], [1.0, 5.0]]
    assert solved["internal_sigma_yz_pa"] == [[-10.0, -1.0], [-1.0, -5.0]]

    record = load_elastic_record(case_dir)
    assert np.array_equal(record.sigma_inf, np.asarray([[-10.0, -1.0], [-1.0, -5.0]]))
    assert record.meta["claim_scope"] == "static_homogeneous_linear_elastic_plane_strain_only"

    assert main(["elastic-validate", str(case_dir)]) == 0
    validated = _json_output(capsys)
    assert validated["valid"] is True
    assert validated["nodes"] == solved["nodes"]
    assert validated["elements"] == solved["elements"]
    assert len(validated["arrays_file_sha256"]) == 64
    assert validated["validation_scope"] == "hashes_plus_full_B_elastic_semantic_revalidation"


def test_elastic_kirsch_freezes_multimesh_report_and_fails_threshold(tmp_path, capsys) -> None:
    report_path = tmp_path / "kirsch-failed.json"
    exit_code = main(
        [
            "elastic-kirsch",
            "--output",
            str(report_path),
            "--young-modulus",
            "1e9",
            "--poisson-ratio",
            "0.25",
            "--sigma-yy-compression",
            "1e6",
            "--sigma-zz-compression",
            "0",
            "--max-fine-stress-error",
            "0",
        ]
    )
    summary = _json_output(capsys)
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert exit_code == 2
    assert summary["exit_code"] == 2
    assert report["frozen"] is True and report["passed"] is False
    assert report["status"] == "failed"
    assert [tier["name"] for tier in report["tiers"]] == ["coarse", "medium", "fine"]
    errors = [tier["kirsch_metrics"]["annulus_stress_relative_l2"] for tier in report["tiers"]]
    assert errors[2] < errors[1] < errors[0]
    assert report["checks"]["affine_patch"] is True
    assert report["affine_patch"]["free_dof_residual"] < 1.0e-9
    assert report["checks"]["fine_annulus_stress_relative_l2"] is False
    assert len(report["report_sha256"]) == 64
    assert report["claim_scope"] == "circular_opening_static_linear_elastic_validation_only"
