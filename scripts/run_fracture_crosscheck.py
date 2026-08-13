#!/usr/bin/env python3
"""Prepare or run the frozen local-versus-MOOSE fracture cross-check."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tunnelgeopt.fracture_crosscheck import CrosscheckError, run_crosscheck


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the development-only same-mesh plane-strain cross-check. "
            "MOOSE execution is opt-in and missing outputs fail closed."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/fracture_crosscheck_v1.json"),
        help="strict v1 JSON configuration",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        required=True,
        help="dedicated evidence directory",
    )
    parser.add_argument(
        "--run-moose",
        action="store_true",
        help="execute the configured real WSL MOOSE binary after local preparation",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = run_crosscheck(
            args.config,
            args.artifact_dir,
            run_moose=bool(args.run_moose),
        )
    except CrosscheckError as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(report, sort_keys=True, ensure_ascii=False))
    if args.run_moose and not report["pass"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
