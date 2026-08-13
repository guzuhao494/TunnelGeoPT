"""Command-line entry points for the A-layer and validated B-elastic layer."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
from collections.abc import Mapping, Sequence
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from .cases import case_group_id, sha256_canonical
from .elastic_schema import load_elastic_record, save_elastic_result
from .elastic_validation import kirsch_metrics, run_affine_patch_test, validate_elastic_result
from .elasticity import solve_plane_strain_excavation
from .geometry import make_tunnel_boundary
from .kirsch import kirsch_stress
from .lift import generate_lifted_case
from .mesh import generate_tunnel_mesh
from .schema import load_sample, save_sample

SHAPES = ("circle", "horseshoe", "straight_wall_arch")
_REPORT_SCHEMA = "tunnelgeopt.elastic_kirsch_report"
_REPORT_VERSION = "0.2.0"


def _print_json(value: Mapping[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


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
    _print_json(
        {
            "case_dir": str(args.output.resolve()),
            "shape": args.shape,
            "num_points": int(case.x.shape[0]),
            "num_prompts": len(case.conditions),
            "x_shape": list(case.x.shape),
            "condition_shape": list(case.conditions[0].shape),
            "supervise_shape": list(case.supervises[0].shape),
            "claim_scope": metadata["claim_scope"],
        }
    )
    return 0


def _validate(args: argparse.Namespace) -> int:
    sample = load_sample(
        args.case_dir,
        trajectory_index=args.trajectory_index,
        require_meta=args.require_meta,
    )
    _print_json(
        {
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
    )
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
    _print_json(
        {
            "points": args.points,
            "max_abs_boundary_radial_stress": float(np.max(np.abs(result["sigma_rr"]))),
            "max_abs_boundary_shear_stress": float(np.max(np.abs(result["tau_rt"]))),
            "max_abs_boundary_hoop_stress": float(np.max(np.abs(result["sigma_tt"]))),
            "sign_convention": "tension_positive",
        }
    )
    return 0


def _package_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "not-installed"


def _environment_snapshot() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": {
            name: _package_version(name) for name in ("numpy", "scipy", "scikit-fem", "gmsh")
        },
    }


def _compression_positive_stress(
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    normal_values = (args.sigma_yy_compression, args.sigma_zz_compression)
    if not all(np.isfinite(value) and value >= 0.0 for value in normal_values):
        raise ValueError("compression-positive normal stresses must be finite and non-negative")
    if not np.isfinite(args.tau_yz_compression):
        raise ValueError("compression-positive shear stress must be finite")
    compression = np.asarray(
        [
            [args.sigma_yy_compression, args.tau_yz_compression],
            [args.tau_yz_compression, args.sigma_zz_compression],
        ],
        dtype=np.float64,
    )
    reference_stress = float(np.linalg.norm(compression, ord="fro"))
    if reference_stress <= np.finfo(float).tiny:
        raise ValueError("the compression-positive stress tensor must be non-zero")
    sigma_xx_argument = getattr(args, "sigma_xx_compression", None)
    sigma_xx_compression = (
        float(sigma_xx_argument)
        if sigma_xx_argument is not None
        else float(args.poisson_ratio) * float(np.trace(compression))
    )
    if not np.isfinite(sigma_xx_compression) or sigma_xx_compression < 0.0:
        raise ValueError("sigma_xx_compression must be finite and non-negative")
    # A change from compression-positive to tension-positive negates the full
    # stress tensor, including shear; doing it here creates one auditable sign
    # boundary for every B-elastic command.
    return compression, -compression, sigma_xx_compression, reference_stress


def _stress_orientation_deg(compression: np.ndarray) -> float:
    return float(
        np.degrees(
            0.5
            * np.arctan2(
                2.0 * compression[0, 1],
                compression[0, 0] - compression[1, 1],
            )
        )
    )


def _elastic_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": _REPORT_VERSION,
        "shape": args.shape,
        "geometry": {
            "radius_m": args.radius,
            "roughness_amplitude_over_radius": args.roughness,
            "boundary_points": args.boundary_points,
            "seed": args.seed,
        },
        "mesh": {
            "domain_scale": args.domain_scale,
            "mesh_size_m": args.mesh_size,
            "wall_mesh_size_m": args.wall_mesh_size,
            "farfield_mesh_size_m": args.farfield_mesh_size,
            "gmsh_algorithm": args.gmsh_algorithm,
        },
        "material_si": {
            "young_modulus_pa": args.young_modulus,
            "poisson_ratio": args.poisson_ratio,
        },
        "stress_input_compression_positive_pa": {
            "sigma_xx": args.sigma_xx_compression,
            "sigma_yy": args.sigma_yy_compression,
            "sigma_zz": args.sigma_zz_compression,
            "tau_yz": args.tau_yz_compression,
        },
        "publication_dtype": args.publication_dtype,
        "quality_control": {
            "max_symmetry_error": args.max_symmetry_error,
            "max_algebraic_residual": args.max_algebraic_residual,
            "max_energy_closure": args.max_energy_closure,
        },
    }


def _physical_case(
    args: argparse.Namespace,
    compression: np.ndarray,
    sigma_xx_compression: float,
    reference_stress: float,
) -> dict[str, Any]:
    stress_mpa = compression / 1.0e6
    return {
        "section_family": args.shape,
        "section_parameters": {
            "radius_m": args.radius,
            "roughness_amplitude_over_radius": args.roughness,
        },
        "material_field_seed": args.seed,
        "joint_network_seed": args.seed,
        "dimensionless_material_parameters": {
            "young_modulus_over_reference_stress": args.young_modulus / reference_stress,
            "poisson_ratio": args.poisson_ratio,
        },
        # The case identity uses MPa to stay inside the manifest's bounded
        # numerical envelope; the solver and persisted arrays remain SI.
        "initial_stress_tensor": [
            [sigma_xx_compression / 1.0e6, 0.0, 0.0],
            [0.0, float(stress_mpa[0, 0]), float(stress_mpa[0, 1])],
            [0.0, float(stress_mpa[1, 0]), float(stress_mpa[1, 1])],
        ],
        "stress_orientation": _stress_orientation_deg(compression),
        "excavation_schedule": [1.0],
        "unloading_schedule": [1.0],
    }


def _elastic_solve(args: argparse.Namespace) -> int:
    compression, tension, sigma_xx_compression, reference_stress = _compression_positive_stress(
        args
    )
    geometry = make_tunnel_boundary(
        args.shape,
        n_points=args.boundary_points,
        radius=args.radius,
        roughness_amplitude=args.roughness,
        seed=args.seed,
    )
    mesh = generate_tunnel_mesh(
        geometry,
        domain_scale=args.domain_scale,
        mesh_size=args.mesh_size,
        wall_mesh_size=args.wall_mesh_size,
        farfield_mesh_size=args.farfield_mesh_size,
        gmsh_algorithm=args.gmsh_algorithm,
    )
    result = solve_plane_strain_excavation(
        mesh,
        young_modulus=args.young_modulus,
        poisson_ratio=args.poisson_ratio,
        sigma_inf=tension,
        sigma_xx_inf=-sigma_xx_compression,
    )
    validation = validate_elastic_result(
        result,
        max_symmetry_error=args.max_symmetry_error,
        max_algebraic_residual=args.max_algebraic_residual,
        max_energy_closure=args.max_energy_closure,
    )
    config = _elastic_config(args)
    config_hash = sha256_canonical(config)
    identity = case_group_id(
        _physical_case(
            args,
            compression,
            sigma_xx_compression,
            reference_stress,
        )
    )
    summary: dict[str, Any] = {
        "passed": bool(validation["passed"]),
        "case_dir": str(args.output.resolve()),
        "shape": args.shape,
        "case_group_id": identity,
        "config_hash": config_hash,
        "nodes": int(result.nodes.shape[0]),
        "elements": int(result.elements.shape[0]),
        "publication_dtype": args.publication_dtype,
        "input_stress_convention": "compression_positive",
        "input_sigma_yz_pa": compression.tolist(),
        "internal_stress_convention": "tension_positive",
        "internal_sigma_yz_pa": tension.tolist(),
        "validation": validation,
        "claim_scope": "static_homogeneous_linear_elastic_plane_strain_only",
    }
    if not validation["passed"]:
        summary["saved"] = False
        _print_json(summary)
        return 2

    paths = save_elastic_result(
        args.output,
        result,
        case_group_id=identity,
        config_hash=config_hash,
        env=_environment_snapshot(),
        meta={
            "shape": args.shape,
            "case_identity_stress_unit": "MPa",
            "input_stress_convention": "compression_positive",
            "internal_stress_convention": "tension_positive",
            "claim_scope": "static_homogeneous_linear_elastic_plane_strain_only",
        },
        publication_dtype=np.dtype(args.publication_dtype),
        overwrite=args.overwrite,
    )
    summary.update(
        {
            "saved": True,
            "arrays": str(paths.arrays.resolve()),
            "metadata": str(paths.meta.resolve()),
        }
    )
    _print_json(summary)
    return 0


def _elastic_validate(args: argparse.Namespace) -> int:
    dtype = np.dtype(args.dtype)
    record = load_elastic_record(args.case_dir, expected_dtype=dtype)
    metadata_path = args.case_dir / "meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    _print_json(
        {
            "valid": True,
            "case_dir": str(args.case_dir.resolve()),
            "dtype": record.dtype.name,
            "nodes": record.num_nodes,
            "elements": record.num_elements,
            "wall_facets": int(record.wall_facets.shape[0]),
            "farfield_facets": int(record.farfield_facets.shape[0]),
            "case_group_id": record.case_group_id,
            "mesh_id": record.mesh_id,
            "config_hash": record.config_hash,
            "arrays_file_sha256": metadata["arrays_file_sha256"],
            "content_sha256": metadata["content_sha256"],
            "diagnostics": dict(record.diagnostics),
            "validation_scope": "hashes_plus_full_B_elastic_semantic_revalidation",
        }
    )
    return 0


def _atomic_write_report(path: Path, report: Mapping[str, Any], *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"frozen report already exists: {path}; pass --overwrite to replace it"
        )
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(
                report,
                stream,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _report_hash(report: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in report.items() if key != "report_sha256"}
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _elastic_kirsch(args: argparse.Namespace) -> int:
    compression, tension, _, _ = _compression_positive_stress(args)
    geometry = make_tunnel_boundary(
        "circle",
        n_points=args.boundary_points,
        radius=args.radius,
        roughness_amplitude=0.0,
        seed=0,
    )
    tier_specs = (
        ("coarse", args.coarse_mesh_size, args.coarse_wall_mesh_size),
        ("medium", args.medium_mesh_size, args.medium_wall_mesh_size),
        ("fine", args.fine_mesh_size, args.fine_wall_mesh_size),
    )
    tiers: list[dict[str, Any]] = []
    for name, mesh_size, wall_mesh_size in tier_specs:
        mesh = generate_tunnel_mesh(
            geometry,
            domain_scale=args.domain_scale,
            mesh_size=mesh_size,
            wall_mesh_size=wall_mesh_size,
            farfield_mesh_size=mesh_size,
            gmsh_algorithm=args.gmsh_algorithm,
        )
        result = solve_plane_strain_excavation(
            mesh,
            young_modulus=args.young_modulus,
            poisson_ratio=args.poisson_ratio,
            sigma_inf=tension,
        )
        generic = validate_elastic_result(
            result,
            max_symmetry_error=args.max_symmetry_error,
            max_algebraic_residual=args.max_algebraic_residual,
            max_energy_closure=args.max_energy_closure,
        )
        metrics = kirsch_metrics(
            result,
            mesh,
            radius=args.radius,
            annulus=(args.annulus_inner, args.annulus_outer),
        )
        tiers.append(
            {
                "name": name,
                "mesh_size_m": mesh_size,
                "wall_mesh_size_m": wall_mesh_size,
                "node_count": int(mesh.nodes.shape[0]),
                "element_count": int(mesh.elements.shape[0]),
                "generic_validation": generic,
                "kirsch_metrics": metrics.as_dict(),
            }
        )

    patch = run_affine_patch_test(
        young_modulus=args.young_modulus,
        poisson_ratio=args.poisson_ratio,
    )
    field_errors = [float(tier["kirsch_metrics"]["annulus_stress_relative_l2"]) for tier in tiers]
    fine_metrics = tiers[-1]["kirsch_metrics"]
    monotonic = all(current <= previous + 1.0e-14 for previous, current in pairwise(field_errors))
    checks = {
        "all_solver_quality_control": all(
            bool(tier["generic_validation"]["passed"]) for tier in tiers
        ),
        "affine_patch": bool(patch["passed"]),
        "fine_annulus_stress_relative_l2": (
            fine_metrics["annulus_stress_relative_l2"] <= args.max_fine_stress_error
        ),
        "fine_peak_hoop_relative_error": (
            fine_metrics["peak_hoop_relative_error"] <= args.max_fine_peak_error
        ),
        "monotonic_annulus_stress_improvement": monotonic,
    }
    passed = all(checks.values())
    report: dict[str, Any] = {
        "schema": _REPORT_SCHEMA,
        "schema_version": _REPORT_VERSION,
        "frozen": True,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "input": {
            "shape": "circle",
            "radius_m": args.radius,
            "boundary_points": args.boundary_points,
            "domain_scale": args.domain_scale,
            "young_modulus_pa": args.young_modulus,
            "poisson_ratio": args.poisson_ratio,
            "stress_convention": "compression_positive",
            "sigma_yz_pa": compression.tolist(),
        },
        "internal": {
            "stress_convention": "tension_positive",
            "sigma_yz_pa": tension.tolist(),
        },
        "comparison_annulus_radius_over_radius": [
            args.annulus_inner,
            args.annulus_outer,
        ],
        "thresholds": {
            "max_fine_annulus_stress_relative_l2": args.max_fine_stress_error,
            "max_fine_peak_hoop_relative_error": args.max_fine_peak_error,
            "require_monotonic_annulus_stress_improvement": True,
            "max_symmetry_error": args.max_symmetry_error,
            "max_algebraic_residual": args.max_algebraic_residual,
            "max_energy_closure": args.max_energy_closure,
        },
        "checks": checks,
        "affine_patch": patch,
        "tiers": tiers,
        "environment": _environment_snapshot(),
        "claim_scope": "circular_opening_static_linear_elastic_validation_only",
    }
    report["report_sha256"] = _report_hash(report)
    _atomic_write_report(args.output, report, overwrite=args.overwrite)
    _print_json(
        {
            "passed": passed,
            "exit_code": 0 if passed else 2,
            "report": str(args.output.resolve()),
            "report_sha256": report["report_sha256"],
            "checks": checks,
            "fine_metrics": fine_metrics,
            "claim_scope": report["claim_scope"],
        }
    )
    return 0 if passed else 2


def _add_elastic_material_and_stress(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--young-modulus", type=float, required=True, help="Young's modulus in Pa")
    parser.add_argument("--poisson-ratio", type=float, required=True)
    parser.add_argument("--sigma-yy-compression", type=float, required=True, help="Pa, >= 0")
    parser.add_argument("--sigma-zz-compression", type=float, required=True, help="Pa, >= 0")
    parser.add_argument(
        "--tau-yz-compression",
        type=float,
        default=0.0,
        help="Pa; negated with the complete tensor at the sign-convention boundary",
    )


def _add_solver_thresholds(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-symmetry-error", type=float, default=1.0e-12)
    parser.add_argument("--max-algebraic-residual", type=float, default=1.0e-9)
    parser.add_argument("--max-energy-closure", type=float, default=1.0e-9)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tunnelgeopt",
        description="Generate and validate TunnelGeoPT A-layer and B-elastic records.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate one lifted A-layer case")
    generate.add_argument("--shape", choices=SHAPES, required=True)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--n-volume", type=int, default=32768)
    generate.add_argument("--n-surface", type=int, default=4096)
    generate.add_argument("--n-prompts", type=int, default=10)
    generate.add_argument("--boundary-points", type=int, default=256)
    generate.add_argument("--radius", type=float, default=1.0)
    generate.add_argument("--roughness", type=float, default=0.0)
    generate.add_argument("--domain-scale", type=float, default=3.0)
    generate.add_argument("--max-step", type=float, default=0.4)
    generate.add_argument("--prompt-mode", choices=["random", "stress_aligned"], default="random")
    generate.add_argument("--stress-angle", type=float, default=0.0)
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument("--overwrite", action="store_true")
    generate.set_defaults(handler=_generate)

    validate = subparsers.add_parser("validate", help="validate one stored A-layer trajectory")
    validate.add_argument("case_dir", type=Path)
    validate.add_argument("--trajectory-index", type=int, default=0)
    validate.add_argument("--require-meta", action="store_true")
    validate.set_defaults(handler=_validate)

    kirsch = subparsers.add_parser("kirsch-check", help="check the analytical circular boundary")
    kirsch.add_argument("--radius", type=float, default=1.0)
    kirsch.add_argument("--sigma-x", type=float, default=10.0)
    kirsch.add_argument("--sigma-y", type=float, default=4.0)
    kirsch.add_argument("--tau-xy", type=float, default=2.0)
    kirsch.add_argument("--points", type=int, default=360)
    kirsch.set_defaults(handler=_kirsch_check)

    elastic_solve = subparsers.add_parser(
        "elastic-solve", help="solve and save one independent B-elastic case"
    )
    elastic_solve.add_argument("--shape", choices=SHAPES, required=True)
    elastic_solve.add_argument("--output", type=Path, required=True)
    elastic_solve.add_argument("--radius", type=float, default=1.0)
    elastic_solve.add_argument("--roughness", type=float, default=0.0)
    elastic_solve.add_argument("--boundary-points", type=int, default=96)
    elastic_solve.add_argument("--seed", type=int, default=0)
    elastic_solve.add_argument("--domain-scale", type=float, default=8.0)
    elastic_solve.add_argument("--mesh-size", type=float, default=0.6)
    elastic_solve.add_argument("--wall-mesh-size", type=float, default=0.125)
    elastic_solve.add_argument("--farfield-mesh-size", type=float, default=0.6)
    elastic_solve.add_argument("--gmsh-algorithm", type=int, default=6)
    _add_elastic_material_and_stress(elastic_solve)
    elastic_solve.add_argument("--sigma-xx-compression", type=float)
    elastic_solve.add_argument(
        "--publication-dtype", choices=["float64", "float32"], default="float64"
    )
    _add_solver_thresholds(elastic_solve)
    elastic_solve.add_argument("--overwrite", action="store_true")
    elastic_solve.set_defaults(handler=_elastic_solve)

    elastic_validate = subparsers.add_parser(
        "elastic-validate", help="load and fully revalidate one B-elastic record"
    )
    elastic_validate.add_argument("case_dir", type=Path)
    elastic_validate.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    elastic_validate.set_defaults(handler=_elastic_validate)

    elastic_kirsch = subparsers.add_parser(
        "elastic-kirsch", help="freeze a thresholded three-mesh Kirsch validation report"
    )
    elastic_kirsch.add_argument("--output", type=Path, required=True)
    elastic_kirsch.add_argument("--radius", type=float, default=1.0)
    elastic_kirsch.add_argument("--boundary-points", type=int, default=96)
    elastic_kirsch.add_argument("--domain-scale", type=float, default=8.0)
    elastic_kirsch.add_argument("--gmsh-algorithm", type=int, default=6)
    elastic_kirsch.add_argument("--coarse-mesh-size", type=float, default=0.8)
    elastic_kirsch.add_argument("--coarse-wall-mesh-size", type=float, default=0.25)
    elastic_kirsch.add_argument("--medium-mesh-size", type=float, default=0.6)
    elastic_kirsch.add_argument("--medium-wall-mesh-size", type=float, default=0.125)
    elastic_kirsch.add_argument("--fine-mesh-size", type=float, default=0.4)
    elastic_kirsch.add_argument("--fine-wall-mesh-size", type=float, default=0.0625)
    elastic_kirsch.add_argument("--annulus-inner", type=float, default=1.25)
    elastic_kirsch.add_argument("--annulus-outer", type=float, default=3.0)
    elastic_kirsch.add_argument("--max-fine-stress-error", type=float, default=0.08)
    elastic_kirsch.add_argument("--max-fine-peak-error", type=float, default=0.10)
    _add_elastic_material_and_stress(elastic_kirsch)
    _add_solver_thresholds(elastic_kirsch)
    elastic_kirsch.add_argument("--overwrite", action="store_true")
    elastic_kirsch.set_defaults(handler=_elastic_kirsch)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
