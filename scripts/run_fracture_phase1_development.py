#!/usr/bin/env python3
"""Run exactly one coarse, development-only Phase-1 fracture trajectory.

The command is dry-run by default.  Execution requires the explicit
``--execute-development-only`` acknowledgement and still cannot produce a
formal 36-case label.  The coarse mesh is for adapter/solver diagnosis; it does
not satisfy the frozen fine-mesh, coupled SENT/SENS, or protocol-scale resource
prerequisites.  The separate fixed-state same-mesh MOOSE gate is already closed.
"""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from tunnelgeopt.fracture import AT2Material
from tunnelgeopt.fracture_loading import compile_phase1_load_schedule
from tunnelgeopt.fracture_trajectory import (
    Phase1TrajectoryIdentity,
    run_phase1_development_trajectory,
    save_and_verify_phase1_development_run,
)
from tunnelgeopt.fracture_validation import load_fracture_phase1_config
from tunnelgeopt.geometry import make_parametric_tunnel_boundary
from tunnelgeopt.mesh import generate_tunnel_mesh

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "development" / "fracture-phase1"
HASHED_SOURCES = (
    ROOT / "configs" / "fracture_phase1_pilot.json",
    ROOT / "src" / "tunnelgeopt" / "fracture.py",
    ROOT / "src" / "tunnelgeopt" / "fracture_loading.py",
    ROOT / "src" / "tunnelgeopt" / "fracture_schema.py",
    ROOT / "src" / "tunnelgeopt" / "fracture_trajectory.py",
    Path(__file__).resolve(),
)


def _source_hash() -> str:
    digest = sha256()
    for path in sorted(HASHED_SOURCES):
        relative = path.relative_to(ROOT).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _mesh_hash(nodes: Any, elements: Any) -> str:
    digest = sha256()
    for value in (nodes, elements):
        contiguous = memoryview(np.ascontiguousarray(value)).cast("B")
        digest.update(len(contiguous).to_bytes(8, "big"))
        digest.update(contiguous)
    return digest.hexdigest()


def _material(config: dict[str, Any], material_id: str, ucs: float) -> AT2Material:
    level = next(item for item in config["materials"]["levels"] if item["id"] == material_id)
    radius = float(config["geometry"]["characteristic_radius_R"])
    return AT2Material(
        young_modulus=float(config["materials"]["fixed"]["youngs_modulus_over_UCS"]) * ucs,
        poisson_ratio=float(config["materials"]["fixed"]["poisson_ratio"]),
        fracture_toughness=float(level["Gc_over_UCS_R"]) * ucs * radius,
        length_scale=float(level["ell_over_R"]) * radius,
        residual_stiffness=float(config["fracture_model"]["residual_stiffness_k"]),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--section", choices=("circle", "horseshoe", "straight_wall_arch"), default="circle"
    )
    parser.add_argument("--material", choices=("m1", "m2", "m3"), default="m1")
    parser.add_argument("--path", choices=("p1", "p2", "p3", "p4"), default="p1")
    parser.add_argument("--ucs", type=float, default=1.0, help="Positive dimensional UCS scale.")
    parser.add_argument("--boundary-points", type=int, default=64)
    parser.add_argument("--mesh-size", type=float, default=0.7)
    parser.add_argument("--wall-mesh-size", type=float, default=0.35)
    parser.add_argument("--farfield-mesh-size", type=float, default=0.8)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--execute-development-only",
        action="store_true",
        help="Acknowledge the expensive coarse diagnostic and execute one identity.",
    )
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    if arguments.ucs <= 0.0:
        raise SystemExit("--ucs must be positive")
    if arguments.boundary_points < 8:
        raise SystemExit("--boundary-points must be at least 8")
    config = load_fracture_phase1_config()
    case_id = f"fp1-{arguments.section}-{arguments.material}-{arguments.path}"
    plan = {
        "case_id": case_id,
        "execution_scope": "one_coarse_development_identity",
        "formal_labels_allowed": False,
        "required_output_count": 41,
        "rollback_strategy": "fresh_complete_accepted_prefix_per_attempt",
        "known_cost": "quadratic_prefix_recomputation_without_restart_state",
        "closed_solver_gate": "relative_potential_energy_increment_at_1e-8",
        "unclosed_prerequisites": [
            "fine_mesh_contract",
            "coupled_SENT_SENS_three_grid",
            "protocol_scale_resource_and_publication_gate",
        ],
    }
    if not arguments.execute_development_only:
        print(json.dumps({"status": "dry_run", **plan}, indent=2, sort_keys=True))
        return 0

    section_config = next(
        item for item in config["geometry"]["sections"] if item["id"] == arguments.section
    )
    radius = float(config["geometry"]["characteristic_radius_R"])
    geometry = make_parametric_tunnel_boundary(
        arguments.section,
        parameters=section_config["parameters"],
        n_points=arguments.boundary_points,
        radius=radius,
        roughness_amplitude=0.0,
        seed=0,
    )
    bounds = config["geometry"]["outer_domain"]["bounds_over_R"]
    outer_bounds = (
        float(bounds["y"][0]) * radius,
        float(bounds["y"][1]) * radius,
        float(bounds["z"][0]) * radius,
        float(bounds["z"][1]) * radius,
    )
    mesh = generate_tunnel_mesh(
        geometry,
        outer_bounds=outer_bounds,
        mesh_size=arguments.mesh_size,
        wall_mesh_size=arguments.wall_mesh_size,
        farfield_mesh_size=arguments.farfield_mesh_size,
    )
    material = _material(config, arguments.material, arguments.ucs)
    schedule = compile_phase1_load_schedule(config, arguments.path, arguments.ucs, mesh)
    mesh_digest = _mesh_hash(mesh.nodes, mesh.elements)
    identity = Phase1TrajectoryIdentity(
        trajectory_id=f"dev-{case_id}-{mesh_digest[:12]}",
        case_id=case_id,
        mesh_id=f"coarse-development-{mesh_digest[:16]}",
        geometry_id=f"{arguments.section}-canonical-development",
        material_id=arguments.material,
        solver_hash=_source_hash(),
        geometry={
            "section_family": arguments.section,
            "characteristic_radius": radius,
            "shape_parameters": section_config["parameters"],
        },
        solver={
            "name": "tunnelgeopt-local-at2-scheduled-development",
            "version": "0.2.0",
            "validation_status": "prerequisites_unclosed",
        },
        meta={
            "mesh_role": "coarse_adapter_diagnostic",
            "boundary_points": arguments.boundary_points,
        },
    )
    force_floor = 1.0e-12 * arguments.ucs * radius
    energy_floor = 1.0e-12 * arguments.ucs * radius**2
    run = run_phase1_development_trajectory(
        mesh,
        material,
        schedule,
        config,
        identity,
        equilibrium_force_normalization_floor=force_floor,
        energy_balance_normalization_floor=energy_floor,
    )
    destination = arguments.output_root.resolve() / case_id
    paths, loaded = save_and_verify_phase1_development_run(
        destination, run, overwrite=arguments.overwrite
    )
    summary = {
        "status": "development_schema_v3_saved_and_reloaded",
        **plan,
        "trajectory_id": loaded.trajectory_id,
        "accepted_internal_steps": loaded.num_steps,
        "rejected_attempts": int(sum(not entry["accepted"] for entry in loaded.attempt_ledger)),
        "arrays": str(paths.arrays),
        "metadata": str(paths.meta),
        "max_global_force_relative_residual": float(run.balance.force_relative_residual.max()),
        "max_global_moment_relative_residual": float(run.balance.moment_relative_residual.max()),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
