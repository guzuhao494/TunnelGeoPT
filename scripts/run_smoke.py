"""Run the bounded data-contract and physics-invariant smoke campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from tunnelgeopt.geometry import make_tunnel_boundary
from tunnelgeopt.kirsch import kirsch_stress
from tunnelgeopt.lift import generate_lifted_case
from tunnelgeopt.schema import load_sample, save_sample


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_case(root: Path, shape: str, prompt_mode: str, seed: int) -> dict[str, object]:
    geometry = make_tunnel_boundary(
        shape,
        n_points=96,
        roughness_amplitude=0.01,
        seed=seed,
    )
    case = generate_lifted_case(
        geometry,
        n_volume=192,
        n_surface=48,
        n_prompts=2,
        steps=3,
        domain_scale=2.5,
        max_step=0.2,
        prompt_mode=prompt_mode,
        stress_angle_deg=30.0,
        seed=seed,
    )
    case_dir = root / f"{shape}_{prompt_mode}"
    meta = dict(case.meta)
    meta.update(
        {
            "case_id": case_dir.name,
            "num_points": int(case.x.shape[0]),
            "dtype": str(case.x.dtype),
        }
    )
    for index, (condition, supervise) in enumerate(
        zip(case.conditions, case.supervises, strict=True)
    ):
        save_sample(
            case_dir,
            case.x,
            condition,
            supervise,
            trajectory_index=index,
            meta=meta,
            overwrite=True,
        )
    loaded = load_sample(case_dir, trajectory_index=1, require_meta=True)
    n_surface = int(meta["n_surface"])
    checks = {
        "shape_contract": loaded.x.shape == (240, 7)
        and loaded.condition.shape == (240, 4)
        and loaded.supervise.shape == (240, 9),
        "finite": bool(
            np.isfinite(loaded.x).all()
            and np.isfinite(loaded.condition).all()
            and np.isfinite(loaded.supervise).all()
        ),
        "surface_step_zero": bool(np.all(loaded.condition[-n_surface:, 3] == np.float16(0.0))),
        "surface_distance_zero": bool(np.all(loaded.x[-n_surface:, 3] == np.float16(0.0))),
        "trajectory_changes": bool(not np.array_equal(case.conditions[0], case.conditions[1])),
    }
    if not all(checks.values()):
        raise AssertionError(f"smoke checks failed for {case_dir.name}: {checks}")
    files = sorted(case_dir.glob("*"))
    return {
        "case_id": case_dir.name,
        "point_count": loaded.num_points,
        "checks": checks,
        "files": {path.name: sha256_file(path) for path in files if path.is_file()},
    }


def kirsch_invariants() -> dict[str, float | bool]:
    theta = np.linspace(0.0, 2.0 * np.pi, 721)
    result = kirsch_stress(
        np.cos(theta),
        np.sin(theta),
        radius=1.0,
        sigma_x=10.0,
        sigma_y=4.0,
        tau_xy=2.0,
    )
    max_radial = float(np.max(np.abs(result["sigma_rr"])))
    max_shear = float(np.max(np.abs(result["tau_rt"])))
    uniaxial = kirsch_stress(
        np.array([0.0]),
        np.array([1.0]),
        radius=1.0,
        sigma_x=10.0,
        sigma_y=0.0,
    )
    concentration = float(uniaxial["sigma_tt"][0] / 10.0)
    return {
        "traction_free_radial_max_abs": max_radial,
        "traction_free_shear_max_abs": max_shear,
        "uniaxial_boundary_concentration_factor": concentration,
        "passed": bool(
            max_radial < 1e-12 and max_shear < 1e-12 and abs(concentration - 3.0) < 1e-12
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("outputs/smoke"))
    parser.add_argument("--report", type=Path, default=Path("validation/smoke_report.json"))
    args = parser.parse_args()
    started = time.perf_counter()
    cases = []
    seed = 100
    for shape in ("circle", "horseshoe", "straight_wall_arch"):
        for prompt_mode in ("random", "stress_aligned"):
            cases.append(run_case(args.output_root, shape, prompt_mode, seed))
            seed += 1
    kirsch = kirsch_invariants()
    if not kirsch["passed"]:
        raise AssertionError(f"Kirsch validation failed: {kirsch}")
    generated_bytes = sum(
        path.stat().st_size for path in args.output_root.rglob("*") if path.is_file()
    )
    pilot_config = Path("configs/pilot.json")
    report = {
        "run_id": "smoke-v0.1.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "claim_type": "computed",
        "claim_scope": "software/data-contract smoke only; no ML or fracture result",
        "command": "python scripts/run_smoke.py --output-root outputs/smoke --report validation/smoke_report.json",
        "pilot_config_sha256": sha256_file(pilot_config) if pilot_config.is_file() else None,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "cases": cases,
        "kirsch": kirsch,
        "resources": {
            "wall_time_seconds": time.perf_counter() - started,
            "generated_bytes": generated_bytes,
            "generated_megabytes": generated_bytes / (1024 * 1024),
        },
        "status": "passed",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "passed", "report": str(args.report), "cases": len(cases)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
