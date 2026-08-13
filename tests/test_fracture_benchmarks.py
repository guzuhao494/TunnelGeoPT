from __future__ import annotations

import importlib.util
import json
import shutil
import stat
import subprocess
from pathlib import Path
from types import ModuleType

import pytest


def _module() -> ModuleType:
    source = Path(__file__).resolve().parents[1] / "scripts" / "run_fracture_benchmarks.py"
    spec = importlib.util.spec_from_file_location("run_fracture_benchmarks", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(command: list[str], cwd: Path) -> str:
    git = shutil.which("git")
    if git is None:
        pytest.skip("Git executable is required for the provenance fixture")
    return subprocess.run(
        [git, *command], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def _attach_pushed_origin(root: Path, remote: Path, branch: str) -> None:
    remote.mkdir()
    _git(["init", "--bare"], remote)
    _git(["remote", "add", "origin", str(remote)], root)
    _git(["push", "-u", "origin", branch], root)


def _fake_moose(tmp_path: Path) -> tuple[Path, str]:
    module = _module()
    root = tmp_path / "moose"
    for relative in (
        module.REFERENCE_INPUT,
        module.REFERENCE_GOLD,
        module.REFERENCE_TEST_SPEC,
        module.REFERENCE_EXECUTABLE,
    ):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"reference\n")
    runner = root / module.REFERENCE_RUNNER
    runner.parent.mkdir(parents=True, exist_ok=True)
    runner.write_text(
        "import sys\n"
        "assert sys.argv[1:] == ['--re', r'^test:phase_field_fracture\\.crack2d_iso$', "
        "'-j', '1', '--no-color']\n"
        "print('test:phase_field_fracture.crack2d_iso ........ OK')\n",
        encoding="utf-8",
        newline="\n",
    )
    runner.chmod(runner.stat().st_mode | stat.S_IXUSR)
    executable = root / module.REFERENCE_EXECUTABLE
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    _git(["init", "-b", "master"], root)
    _git(["config", "user.email", "test@example.invalid"], root)
    _git(["config", "user.name", "Test"], root)
    _git(["add", "."], root)
    _git(["commit", "-m", "fixture"], root)
    _attach_pushed_origin(root, tmp_path / "moose-origin.git", "master")
    return root, _git(["rev-parse", "HEAD"], root)


def test_validate_plan_hashes_pinned_reference(tmp_path: Path) -> None:
    module = _module()
    root, head = _fake_moose(tmp_path)
    result = module.validate_moose_plan(moose_root=root, expected_moose_head=head)
    assert result["mode"] == "validate_only"
    assert result["moose_git"]["head"] == head
    assert set(result["source_files"]) == {
        "input",
        "gold",
        "test_spec",
        "runner",
        "executable",
    }
    assert all(len(record["sha256"]) == 64 for record in result["source_files"].values())


def test_validate_plan_rejects_dirty_or_wrong_head(tmp_path: Path) -> None:
    module = _module()
    root, head = _fake_moose(tmp_path)
    with pytest.raises(module.BenchmarkContractError, match="does not match"):
        module.validate_moose_plan(moose_root=root, expected_moose_head="0" * 40)
    (root / module.REFERENCE_INPUT).write_text("changed\n", encoding="utf-8")
    with pytest.raises(module.BenchmarkContractError, match="dirty"):
        module.validate_moose_plan(moose_root=root, expected_moose_head=head)


def test_run_archives_exact_pass_and_sanitised_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    root, head = _fake_moose(tmp_path)
    project = tmp_path / "project"
    project.mkdir()
    _git(["init", "-b", "codex/test"], project)
    (project / "tracked.txt").write_text("x\n", encoding="utf-8")
    _git(["config", "user.email", "test@example.invalid"], project)
    _git(["config", "user.name", "Test"], project)
    _git(["add", "."], project)
    _git(["commit", "-m", "fixture"], project)
    _attach_pushed_origin(project, tmp_path / "project-origin.git", "codex/test")
    project_head = _git(["rev-parse", "HEAD"], project)
    monkeypatch.setattr(module, "REPO_ROOT", project)
    output = tmp_path / "artifact"
    result = module.run_moose_reference(
        moose_root=root,
        output_dir=output,
        expected_moose_head=head,
        expected_project_head=project_head,
    )
    assert result["classification"] == "MOOSE_REFERENCE_EXECUTION_CONFIRMED"
    assert result["effect_claim_allowed"] is False
    assert result["execution"]["exact_pass_line_detected"] is True
    manifest = json.loads((output / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["files"]) == {"result.json", "stdout.log", "stderr.log"}
    assert str(root) not in (output / "stdout.log").read_text(encoding="utf-8")
    with pytest.raises(module.BenchmarkContractError, match="already exists"):
        module.run_moose_reference(
            moose_root=root,
            output_dir=output,
            expected_moose_head=head,
            expected_project_head=project_head,
        )


def test_exact_pass_parser_rejects_near_matches() -> None:
    module = _module()
    assert module._contains_exact_pass("test:phase_field_fracture.crack2d_iso .... OK")
    assert not module._contains_exact_pass("test:phase_field_fracture.crack2d_iso .... FAILED")
    assert not module._contains_exact_pass("test:phase_field_fracture.crack2d_iso_wo_time .... OK")
    assert not module._contains_exact_pass("test:phase_field_fracture.crack2d_iso .... NOT OK")
    assert not module._contains_exact_pass(
        "test:phase_field_fracture.crack2d_iso .... FAILED (last known OK)"
    )
    assert not module._contains_exact_pass(
        "prefix test:phase_field_fracture.crack2d_iso .... OK but command was skipped"
    )
