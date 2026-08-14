#!/usr/bin/env python3
"""Validate a frozen SENT/SENS case or run an explicitly approved tiny probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tunnelgeopt.fracture_benchmark import (
    preflight_fracture_benchmark,
    run_intact_fracture_benchmark_probe,
    write_probe_artifact_atomic,
)
from tunnelgeopt.fracture_benchmark_mesh import generate_fracture_benchmark_mesh
from tunnelgeopt.fracture_benchmark_validation import load_fracture_sent_sens_config


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
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_fracture_sent_sens_config(args.config)
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
                },
                sort_keys=True,
            )
        )
        return 0

    if not args.approved_development_probe:
        raise SystemExit("probe requires --approved-development-probe")
    if args.tier != "coarse" and not args.allow_noncoarse_probe:
        raise SystemExit("medium/fine probe requires explicit --allow-noncoarse-probe")
    if not args.u_mm:
        raise SystemExit("probe requires explicit repeated --u-mm values, starting with zero")
    if args.output is None:
        raise SystemExit("probe requires --output for its atomic evidence artifact")
    probe = run_intact_fracture_benchmark_probe(
        config,
        benchmark_mesh,
        benchmark_id=args.benchmark,
        tier=args.tier,
        displacements_mm=args.u_mm,
    )
    digest = write_probe_artifact_atomic(probe, args.output)
    print(
        json.dumps(
            {
                "status": probe.status,
                "artifact_sha256": digest,
                "authorizes_medium_fine_or_formal_run": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
