"""Run auditable C-fracture reference benchmarks.

This entry point currently implements only the official MOOSE ``crack2d_iso``
self-test.  Passing it proves that one pinned external reference implementation
can execute its own regression case; it does not validate TunnelGeoPT's local
solver, a tunnel trajectory, or a rockburst claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_INPUT = Path("modules/combined/test/tests/phase_field_fracture/crack2d_iso.i")
REFERENCE_GOLD = Path("modules/combined/test/tests/phase_field_fracture/gold/crack2d_iso_out.e")
REFERENCE_TEST_SPEC = Path("modules/combined/test/tests/phase_field_fracture/tests")
REFERENCE_RUNNER = Path("modules/combined/run_tests")
REFERENCE_EXECUTABLE = Path("modules/combined/combined-opt")
REFERENCE_FILTER = "test:phase_field_fracture.crack2d_iso"
RESULT_SCHEMA = "tunnelgeopt.moose_crack2d_reference.v1"


class BenchmarkContractError(RuntimeError):
    """Raised before execution when the benchmark contract is invalid."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _atomic_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_bytes(value)
    with tempfile.NamedTemporaryFile(
        mode="wb", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(payload).hexdigest()


def _git_value(root: Path, *arguments: str) -> str:
    git = shutil.which("git")
    if git is None:
        raise OSError("Git executable was not found on PATH")
    completed = subprocess.run(
        [git, *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return completed.stdout.strip()


def _require_clean_git(
    root: Path,
    expected_head: str | None,
    *,
    require_pushed_upstream: bool = False,
) -> dict[str, Any]:
    try:
        head = _git_value(root, "rev-parse", "HEAD")
        branch = _git_value(root, "branch", "--show-current")
        status = _git_value(root, "status", "--porcelain", "--untracked-files=no")
        origin_url = _git_value(root, "remote", "get-url", "origin")
        upstream_head = _git_value(root, "rev-parse", "@{upstream}")
    except (OSError, subprocess.SubprocessError) as exc:
        raise BenchmarkContractError(f"could not audit Git repository {root.name}: {exc}") from exc
    if not re.fullmatch(r"[0-9a-f]{40}", head):
        raise BenchmarkContractError(f"invalid Git HEAD for {root.name}")
    if status:
        raise BenchmarkContractError(f"tracked files are dirty in {root.name}")
    if expected_head is not None and head != expected_head:
        raise BenchmarkContractError(
            f"{root.name} HEAD {head} does not match expected {expected_head}"
        )
    if require_pushed_upstream and head != upstream_head:
        raise BenchmarkContractError(f"{root.name} HEAD is not equal to its pushed upstream")
    return {
        "head": head,
        "branch": branch,
        "origin_url": origin_url,
        "upstream_head": upstream_head,
        "tracked_worktree_clean": True,
        "head_equals_upstream": head == upstream_head,
    }


def _conda_moose_packages() -> list[dict[str, str]]:
    prefix_text = os.environ.get("CONDA_PREFIX", "")
    if not prefix_text:
        return []
    metadata_dir = Path(prefix_text) / "conda-meta"
    records: list[dict[str, str]] = []
    for source in sorted(metadata_dir.glob("moose-*.json")):
        try:
            value = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BenchmarkContractError(f"could not read {source.name}: {exc}") from exc
        records.append(
            {
                "name": str(value.get("name", "")),
                "version": str(value.get("version", "")),
                "build": str(value.get("build", "")),
                "channel": str(value.get("channel", "")),
            }
        )
    return records


def _reference_files(moose_root: Path) -> dict[str, Path]:
    files = {
        "input": moose_root / REFERENCE_INPUT,
        "gold": moose_root / REFERENCE_GOLD,
        "test_spec": moose_root / REFERENCE_TEST_SPEC,
        "runner": moose_root / REFERENCE_RUNNER,
        "executable": moose_root / REFERENCE_EXECUTABLE,
    }
    missing = [name for name, path in files.items() if not path.is_file()]
    if missing:
        raise BenchmarkContractError(f"MOOSE reference files are missing: {missing}")
    if not os.access(files["runner"], os.X_OK) or not os.access(files["executable"], os.X_OK):
        raise BenchmarkContractError("MOOSE runner and combined-opt must be executable")
    return files


def _sanitise(text: str, *, moose_root: Path, repo_root: Path) -> str:
    cleaned = text.replace(str(moose_root), "<MOOSE_ROOT>")
    cleaned = cleaned.replace(str(repo_root), "<TUNNELGEOPT_ROOT>")
    home = str(Path.home())
    if home:
        cleaned = cleaned.replace(home, "<HOME>")
    user_tokens = {
        Path.home().name,
        os.environ.get("USER", ""),
        os.environ.get("USERNAME", ""),
        os.environ.get("LOGNAME", ""),
    }
    for token in sorted(user_tokens, key=len, reverse=True):
        if len(token) >= 3:
            cleaned = cleaned.replace(token, "<USER>")
    return cleaned


def _contains_exact_pass(output: str) -> bool:
    exact_test = re.compile(rf"(?<![\w.]){re.escape(REFERENCE_FILTER)}(?=\s)")
    forbidden = re.compile(r"\b(?:FAIL(?:ED)?|NOT\s+OK|SKIP(?:PED)?)\b", re.IGNORECASE)
    matching_lines = []
    for line in output.splitlines():
        if exact_test.search(line) is None:
            continue
        if forbidden.search(line) is not None:
            continue
        if re.search(r"\bOK\s*$", line) is not None:
            matching_lines.append(line)
    return len(matching_lines) == 1


def _json_safe_number(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def validate_moose_plan(
    *, moose_root: Path, expected_moose_head: str | None = None
) -> dict[str, Any]:
    """Validate immutable inputs without running a fracture solve."""

    root = moose_root.resolve()
    files = _reference_files(root)
    git = _require_clean_git(root, expected_moose_head)
    return {
        "schema": RESULT_SCHEMA,
        "mode": "validate_only",
        "reference_case": "official_moose_crack2d_iso",
        "reference_filter": REFERENCE_FILTER,
        "moose_git": git,
        "source_files": {
            name: {
                "relative_path": str(path.relative_to(root)).replace("\\", "/"),
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for name, path in files.items()
        },
        "claim_boundary": {
            "supports": "pinned_moose_reference_self_test_execution",
            "does_not_support": [
                "local_solver_equivalence",
                "tunnel_fracture_trajectory_validity",
                "dynamic_rockburst",
                "field_validity",
            ],
        },
    }


def run_moose_reference(
    *,
    moose_root: Path,
    output_dir: Path,
    expected_moose_head: str | None = None,
    expected_project_head: str | None = None,
    timeout_seconds: float = 900.0,
) -> dict[str, Any]:
    """Run the pinned official test once and write a durable result."""

    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
        raise BenchmarkContractError("timeout_seconds must be finite and positive")
    if expected_moose_head is None or expected_project_head is None:
        raise BenchmarkContractError(
            "expected_moose_head and expected_project_head are required for execution"
        )
    destination = output_dir.resolve()
    if destination.exists():
        raise BenchmarkContractError(f"output directory already exists: {destination.name}")

    plan = validate_moose_plan(
        moose_root=moose_root,
        expected_moose_head=expected_moose_head,
    )
    project_git = _require_clean_git(
        REPO_ROOT,
        expected_project_head,
        require_pushed_upstream=True,
    )
    destination.mkdir(parents=True, exist_ok=False)
    started = datetime.now(UTC)
    files = _reference_files(moose_root.resolve())
    command = [
        sys.executable,
        str(files["runner"]),
        "--re",
        rf"^{re.escape(REFERENCE_FILTER)}$",
        "-j",
        "1",
        "--no-color",
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=files["runner"].parent,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        stdout = _sanitise(completed.stdout, moose_root=moose_root, repo_root=REPO_ROOT)
        stderr = _sanitise(completed.stderr, moose_root=moose_root, repo_root=REPO_ROOT)
        exact_pass = completed.returncode == 0 and _contains_exact_pass(stdout + "\n" + stderr)
        failure_reason = None if exact_pass else "official_test_did_not_report_exact_pass"
        return_code: int | None = completed.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = _sanitise(exc.stdout or "", moose_root=moose_root, repo_root=REPO_ROOT)
        stderr = _sanitise(exc.stderr or "", moose_root=moose_root, repo_root=REPO_ROOT)
        exact_pass = False
        failure_reason = "timeout"
        return_code = None
    except OSError as exc:
        stdout = ""
        stderr = _sanitise(
            f"{type(exc).__name__}: {exc}",
            moose_root=moose_root,
            repo_root=REPO_ROOT,
        )
        exact_pass = False
        failure_reason = "process_launch_failed"
        return_code = None

    stdout_path = destination / "stdout.log"
    stderr_path = destination / "stderr.log"
    stdout_path.write_text(stdout, encoding="utf-8", newline="\n")
    stderr_path.write_text(stderr, encoding="utf-8", newline="\n")
    finished = datetime.now(UTC)
    result: dict[str, Any] = {
        **plan,
        "mode": "executed",
        "classification": (
            "MOOSE_REFERENCE_EXECUTION_CONFIRMED" if exact_pass else "STOP_MOOSE_REFERENCE"
        ),
        "valid": exact_pass,
        "effect_claim_allowed": False,
        "project_git": project_git,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "moose_conda_packages": _conda_moose_packages(),
        },
        "execution": {
            "command": [
                "<PYTHON>",
                "<MOOSE_ROOT>/modules/combined/run_tests",
                "--re",
                rf"^{re.escape(REFERENCE_FILTER)}$",
                "-j",
                "1",
                "--no-color",
            ],
            "started_at_utc": started.isoformat(),
            "finished_at_utc": finished.isoformat(),
            "elapsed_seconds": _json_safe_number((finished - started).total_seconds()),
            "return_code": return_code,
            "exact_pass_line_detected": exact_pass,
            "failure_reason": failure_reason,
            "stdout_sha256": _sha256_file(stdout_path),
            "stderr_sha256": _sha256_file(stderr_path),
        },
    }
    result_sha = _atomic_json(destination / "result.json", result)
    manifest = {
        "schema": "tunnelgeopt.moose_crack2d_artifact_manifest.v1",
        "classification": result["classification"],
        "files": {
            "result.json": result_sha,
            "stdout.log": _sha256_file(stdout_path),
            "stderr.log": _sha256_file(stderr_path),
        },
    }
    _atomic_json(destination / "artifact_manifest.json", manifest)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate-moose", help="validate the reference plan only")
    validate.add_argument("--moose-root", type=Path, required=True)
    validate.add_argument("--expected-moose-head")
    run = subparsers.add_parser("run-moose", help="run and archive the official self-test")
    run.add_argument("--moose-root", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--expected-moose-head", required=True)
    run.add_argument("--expected-project-head", required=True)
    run.add_argument("--timeout-seconds", type=float, default=900.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "validate-moose":
            result = validate_moose_plan(
                moose_root=args.moose_root,
                expected_moose_head=args.expected_moose_head,
            )
        else:
            result = run_moose_reference(
                moose_root=args.moose_root,
                output_dir=args.output,
                expected_moose_head=args.expected_moose_head,
                expected_project_head=args.expected_project_head,
                timeout_seconds=args.timeout_seconds,
            )
    except BenchmarkContractError as exc:
        print(json.dumps({"classification": "ABSTAIN_INVALID", "error": str(exc)}))
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result.get("valid", True) else 3


if __name__ == "__main__":
    raise SystemExit(main())
