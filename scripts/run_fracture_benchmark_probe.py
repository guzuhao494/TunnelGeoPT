#!/usr/bin/env python3
"""Validate a frozen SENT/SENS case or run an explicitly approved tiny probe."""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path

from tunnelgeopt.fracture import FractureSolverOptions
from tunnelgeopt.fracture_benchmark import (
    capture_probe_project_preflight,
    preflight_fracture_benchmark,
    probe_runtime_environment,
    reserve_probe_output_directory,
    run_intact_fracture_benchmark_probe,
    write_probe_artifact_atomic,
)
from tunnelgeopt.fracture_benchmark_mesh import generate_fracture_benchmark_mesh
from tunnelgeopt.fracture_benchmark_validation import (
    default_fracture_sent_sens_config_path,
    load_fracture_sent_sens_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = Path(__file__).resolve()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--benchmark", choices=("sent", "sens"), required=True)
    parser.add_argument("--tier", choices=("coarse", "medium", "fine"), default="coarse")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--validate-only", action="store_true", help="mesh preflight only; never solve"
    )
    mode.add_argument("--run-intact-probe", action="store_true", help="fixed-d=0 development probe")
    parser.add_argument("--approved-development-probe", action="store_true")
    parser.add_argument("--allow-noncoarse-probe", action="store_true")
    parser.add_argument("--u-mm", type=float, action="append", default=[])
    parser.add_argument("--output", type=Path, help="new, unique artifact run directory")
    parser.add_argument(
        "--expected-project-head",
        required=True,
        help="full pushed Git SHA required for both validate-only and probe modes",
    )
    return parser


def _sanitized_probe_command(
    args: argparse.Namespace, *, config_path: str, output: str
) -> list[str]:
    command = [
        "python",
        "scripts/run_fracture_benchmark_probe.py",
        "--config",
        config_path,
        "--benchmark",
        args.benchmark,
        "--tier",
        args.tier,
        "--run-intact-probe",
        "--approved-development-probe",
        "--expected-project-head",
        args.expected_project_head.lower(),
    ]
    if args.allow_noncoarse_probe:
        command.append("--allow-noncoarse-probe")
    for displacement in args.u_mm:
        command.extend(("--u-mm", repr(displacement)))
    command.extend(("--output", output))
    return command


def _validate_probe_arguments(args: argparse.Namespace) -> None:
    """Reject unsafe probe requests before Git checks, reservation, or meshing."""

    if not args.run_intact_probe:
        return
    if not args.approved_development_probe:
        raise SystemExit("probe requires --approved-development-probe")
    if args.tier != "coarse" and not args.allow_noncoarse_probe:
        raise SystemExit("medium/fine probe requires explicit --allow-noncoarse-probe")
    if args.output is None:
        raise SystemExit("probe requires --output for its unique artifact run directory")
    if not args.u_mm:
        raise SystemExit("probe requires explicit repeated --u-mm values, starting with zero")
    if len(args.u_mm) > 12:
        raise SystemExit("development probe is capped at 12 explicit states")
    if not all(math.isfinite(value) for value in args.u_mm):
        raise SystemExit("probe --u-mm values must be finite")
    if args.u_mm[0] != 0.0 or any(
        right <= left for left, right in zip(args.u_mm, args.u_mm[1:], strict=False)
    ):
        raise SystemExit("probe --u-mm values must start at zero and be strictly increasing")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _validate_probe_arguments(args)
    config_path = (
        default_fracture_sent_sens_config_path()
        if args.config is None
        else args.config.resolve(strict=True)
    )
    project_snapshot = capture_probe_project_preflight(
        PROJECT_ROOT,
        expected_project_head=args.expected_project_head,
        config_path=config_path,
        runner_path=RUNNER_PATH,
    )
    output: Path | None = None
    output_relative: str | None = None
    if args.run_intact_probe:
        assert args.output is not None
        output, output_relative = reserve_probe_output_directory(PROJECT_ROOT, args.output)
    config = load_fracture_sent_sens_config(config_path)
    benchmark_mesh = generate_fracture_benchmark_mesh(loading=args.benchmark, tier=args.tier)
    preflight = preflight_fracture_benchmark(
        config, benchmark_mesh, benchmark_id=args.benchmark, tier=args.tier
    )

    # Validate-only is the default and exits before any solver call.
    if not args.run_intact_probe:
        print(
            json.dumps(
                {
                    "status": "VALIDATED_NOT_RUN",
                    "claim_boundary": "mesh_and_adapter_preflight_only",
                    "preflight": preflight.__dict__,
                    "project_head": project_snapshot.project_head,
                    "source_inventory_sha256": project_snapshot.source_inventory_sha256,
                },
                sort_keys=True,
            )
        )
        return 0

    if output is None or output_relative is None:  # pragma: no cover - guarded above
        raise RuntimeError("probe output was not reserved")
    controls = FractureSolverOptions()
    runtime_environment = probe_runtime_environment()
    started_utc = _utc_now()
    probe = run_intact_fracture_benchmark_probe(
        config,
        benchmark_mesh,
        benchmark_id=args.benchmark,
        tier=args.tier,
        displacements_mm=args.u_mm,
        options=controls,
    )
    completed_utc = _utc_now()
    bundle = write_probe_artifact_atomic(
        probe,
        output,
        project_snapshot=project_snapshot,
        started_utc=started_utc,
        completed_utc=completed_utc,
        sanitized_command=_sanitized_probe_command(
            args, config_path=project_snapshot.config_path, output=output_relative
        ),
        solver_options=controls,
        runtime_environment=runtime_environment,
    )
    print(
        json.dumps(
            {
                "status": probe.status,
                "result_sha256": bundle.result_sha256,
                "artifact_manifest_sha256": bundle.manifest_sha256,
                "single_case_only": True,
                "real_probe_allowed": False,
                "paper_evidence_eligible": False,
                "authorizes_medium_fine_or_formal_run": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
