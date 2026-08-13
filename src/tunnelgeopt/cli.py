"""Command-line interface for bounded dataset generation and validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .geometry import make_tunnel_boundary
from .kirsch import kirsch_stress
from .lift import generate_lifted_case
from .schema import load_sample, save_sample


def _generate(args: argparse.Namespace) -> int:
    geometry = make_tunnel_boundary(
        args.shape,
        n_points=args.boundary_points,
        radius=args.radius,
        roughness_amplitude=args.roughness,
        seed=args.seed,
    )
    case = generate_lifted_case(
        geometry,
        n_volume=args.n_volume,
        n_surface=args.n_surface,
        n_prompts=args.n_prompts,
        steps=3,
        domain_scale=args.domain_scale,
        max_step=args.max_step,
        prompt_mode=args.prompt_mode,
        stress_angle_deg=args.stress_angle,
        seed=args.seed,
    )
    metadata = dict(case.meta)
    metadata.update(
        {
            "case_id": args.output.name,
            "num_points": int(case.x.shape[0]),
            "dtype": str(case.x.dtype),
        }
    )
    for trajectory_index, (condition, supervise) in enumerate(
        zip(case.conditions, case.supervises, strict=True)
    ):
        save_sample(
            args.output,
            case.x,
            condition,
            supervise,
            trajectory_index=trajectory_index,
            meta=metadata,
            overwrite=args.overwrite,
        )
    summary = {
        "case_dir": str(args.output.resolve()),
        "shape": args.shape,
        "num_points": int(case.x.shape[0]),
        "num_prompts": len(case.conditions),
        "x_shape": list(case.x.shape),
        "condition_shape": list(case.conditions[0].shape),
        "supervise_shape": list(case.supervises[0].shape),
        "claim_scope": metadata["claim_scope"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _validate(args: argparse.Namespace) -> int:
    sample = load_sample(
        args.case_dir,
        trajectory_index=args.trajectory_index,
        require_meta=args.require_meta,
    )
    result = {
        "valid": True,
        "case_dir": str(args.case_dir.resolve()),
        "trajectory_index": args.trajectory_index,
        "num_points": sample.num_points,
        "dtype": sample.dtype.name,
        "x_shape": list(sample.x.shape),
        "condition_shape": list(sample.condition.shape),
        "supervise_shape": list(sample.supervise.shape),
        "meta_keys": sorted(sample.meta),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _kirsch_check(args: argparse.Namespace) -> int:
    theta = np.linspace(0.0, 2.0 * np.pi, args.points, endpoint=False)
    result = kirsch_stress(
        args.radius * np.cos(theta),
        args.radius * np.sin(theta),
        radius=args.radius,
        sigma_x=args.sigma_x,
        sigma_y=args.sigma_y,
        tau_xy=args.tau_xy,
    )
    report = {
        "points": args.points,
        "max_abs_boundary_radial_stress": float(np.max(np.abs(result["sigma_rr"]))),
        "max_abs_boundary_shear_stress": float(np.max(np.abs(result["tau_rt"]))),
        "max_abs_boundary_hoop_stress": float(np.max(np.abs(result["sigma_tt"]))),
        "sign_convention": "tension_positive",
    }
    print(json.dumps(report, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tunnelgeopt",
        description="Generate and validate TunnelGeoPT synthetic samples.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate one lifted geometry case")
    generate.add_argument(
        "--shape",
        choices=["circle", "horseshoe", "straight_wall_arch"],
        required=True,
    )
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--n-volume", type=int, default=32768)
    generate.add_argument("--n-surface", type=int, default=4096)
    generate.add_argument("--n-prompts", type=int, default=10)
    generate.add_argument("--boundary-points", type=int, default=256)
    generate.add_argument("--radius", type=float, default=1.0)
    generate.add_argument("--roughness", type=float, default=0.0)
    generate.add_argument("--domain-scale", type=float, default=3.0)
    generate.add_argument("--max-step", type=float, default=0.4)
    generate.add_argument(
        "--prompt-mode",
        choices=["random", "stress_aligned"],
        default="random",
    )
    generate.add_argument("--stress-angle", type=float, default=0.0)
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument("--overwrite", action="store_true")
    generate.set_defaults(handler=_generate)

    validate = subparsers.add_parser("validate", help="validate one stored trajectory")
    validate.add_argument("case_dir", type=Path)
    validate.add_argument("--trajectory-index", type=int, default=0)
    validate.add_argument("--require-meta", action="store_true")
    validate.set_defaults(handler=_validate)

    kirsch = subparsers.add_parser("kirsch-check", help="check the traction-free circular boundary")
    kirsch.add_argument("--radius", type=float, default=1.0)
    kirsch.add_argument("--sigma-x", type=float, default=10.0)
    kirsch.add_argument("--sigma-y", type=float, default=4.0)
    kirsch.add_argument("--tau-xy", type=float, default=2.0)
    kirsch.add_argument("--points", type=int, default=360)
    kirsch.set_defaults(handler=_kirsch_check)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
