"""Development adapter and bounded timing probe for SENT/SENS coupons.

This module connects the frozen protocol, an audited benchmark mesh, and the
generic prescribed-displacement BVP.  It deliberately does not implement a
formal SENT/SENS trajectory or claim a reproduction of Miehe et al.  The
built-in probe keeps damage fixed at zero; its timing is therefore a labelled,
non-authorizing lower bound for the coupled fracture calculation.

Coordinates and displacement DOFs are frozen as ``[y, z]`` and node-major
``[u_y, u_z]``.  Top/bottom are selected from physical labels, never inferred
from the second coordinate column.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import re
import stat
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .fracture import AT2Material, FractureSolverOptions
from .fracture_benchmark_validation import (
    FROZEN_CANONICAL_SHA256,
    prescribed_displacements,
    validate_fracture_sent_sens_config,
)
from .fracture_bvp import (
    FixedDamageDisplacementBVPResult,
    PrescribedDisplacementState,
    prescribed_displacement_mesh_identity,
    solve_fixed_damage_displacement_bvp,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

PROBE_SCHEMA = "tunnelgeopt.fracture.sent_sens.intact_probe.v1"
PROBE_RESULT_SCHEMA = "tunnelgeopt.fracture.sent_sens.probe_result.v1"
PROBE_MANIFEST_SCHEMA = "tunnelgeopt.fracture.sent_sens.probe_artifact_manifest.v1"
_COORDINATE_TOLERANCE_MM = 2.0e-12
_FORCE_FLOOR_KN = 1.0e-15
_MOMENT_FLOOR_KN_MM = 1.0e-15
_ENERGY_FLOOR_KN_MM = 1.0e-18


class FractureBenchmarkPreflightError(ValueError):
    """Raised before solving when protocol, mesh, or loading identities differ."""


class ProbeProvenanceError(RuntimeError):
    """Raised when the exact clean, pushed project identity cannot be proven."""


@dataclass(frozen=True)
class ProbeSourceFile:
    """One project-relative member of the conservative probe source closure."""

    path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class ProbeProjectSnapshot:
    """Clean pushed Git state and exact source closure captured before a probe."""

    expected_project_head: str
    project_head: str
    upstream_head: str
    upstream_ref: str
    config_path: str
    runner_path: str
    source_files: tuple[ProbeSourceFile, ...]
    source_inventory_sha256: str
    captured_utc: str
    _project_root: Path

    def as_dict(self) -> dict[str, Any]:
        lock_present = any(
            source.path.endswith(".lock") or source.path == "pylock.toml"
            for source in self.source_files
        )
        return {
            "expected_project_head": self.expected_project_head,
            "project_head": self.project_head,
            "upstream_head": self.upstream_head,
            "upstream_ref": self.upstream_ref,
            "config_path": self.config_path,
            "runner_path": self.runner_path,
            "source_closure": {
                "strategy": (
                    "all_sorted_src_tunnelgeopt_top_level_python_plus_runner_actual_config_"
                    "pyproject_git_control_files_and_root_lockfiles"
                ),
                "inventory_sha256": self.source_inventory_sha256,
                "files": [asdict(source) for source in self.source_files],
                "dependency_lock_present": lock_present,
                "environment_exactly_reconstructible_from_lock": lock_present,
            },
            "captured_utc": self.captured_utc,
            "preflight_full_worktree_clean": True,
        }


@dataclass(frozen=True)
class ProbeArtifactBundle:
    """Hashes of both immutable files published in one exclusive run leaf."""

    result_sha256: str
    manifest_sha256: str
    result_size_bytes: int
    manifest_size_bytes: int


@dataclass(frozen=True)
class FractureBenchmarkPreflight:
    benchmark_id: str
    tier: str
    protocol_sha256: str
    mesh_plan_sha256: str
    mesh_topology_sha256: str
    bvp_mesh_sha256: str
    node_count: int
    element_count: int
    top_node_count: int
    bottom_node_count: int


@dataclass(frozen=True)
class ProbeStep:
    sequence_index: int
    prescribed_U_mm: float
    wall_seconds: float
    converged: bool
    generalized_load_kN: float
    elastic_energy_kN_mm: float
    equilibrium_relative_residual: float
    global_force_relative_imbalance: float
    global_moment_relative_imbalance: float
    path_energy_relative_imbalance: float
    damage_component_status: str


@dataclass(frozen=True)
class FractureBenchmarkProbe:
    schema: str
    status: str
    claim_boundary: str
    benchmark_id: str
    tier: str
    protocol_sha256: str
    mesh_plan_sha256: str
    mesh_topology_sha256: str
    bvp_mesh_sha256: str
    node_count: int
    element_count: int
    top_node_count: int
    bottom_node_count: int
    material: Mapping[str, float]
    prescribed_U_mm: tuple[float, ...]
    steps: tuple[ProbeStep, ...]
    median_step_wall_seconds: float
    projected_formal_increment_count: int
    projected_formal_case_wall_hours: float
    projection_interpretation: str
    authorizes_medium_fine_or_formal_run: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "status": self.status,
            "claim_boundary": self.claim_boundary,
            "benchmark_id": self.benchmark_id,
            "tier": self.tier,
            "protocol_sha256": self.protocol_sha256,
            "mesh_plan_sha256": self.mesh_plan_sha256,
            "mesh_topology_sha256": self.mesh_topology_sha256,
            "bvp_mesh_sha256": self.bvp_mesh_sha256,
            "mesh_counts": {
                "node_count": self.node_count,
                "element_count": self.element_count,
                "top_node_count": self.top_node_count,
                "bottom_node_count": self.bottom_node_count,
            },
            "material": dict(self.material),
            "prescribed_U_mm": list(self.prescribed_U_mm),
            "steps": [asdict(step) for step in self.steps],
            "median_step_wall_seconds": self.median_step_wall_seconds,
            "projected_formal_increment_count": self.projected_formal_increment_count,
            "projected_formal_case_wall_hours": self.projected_formal_case_wall_hours,
            "projection_interpretation": self.projection_interpretation,
            "authorizes_medium_fine_or_formal_run": (self.authorizes_medium_fine_or_formal_run),
        }


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _git_text(project_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "unknown git error"
        raise ProbeProvenanceError(f"git {' '.join(arguments)} failed: {detail}")
    return completed.stdout.strip()


def _project_relative_file(project_root: Path, path: str | Path, name: str) -> tuple[Path, str]:
    root = project_root.resolve(strict=True)
    resolved = Path(path).resolve(strict=True)
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ProbeProvenanceError(f"{name} must be inside the project root") from exc
    if not resolved.is_file():
        raise ProbeProvenanceError(f"{name} must be a regular file")
    return resolved, relative.as_posix()


def _probe_source_paths(
    project_root: Path, *, config_path: str | Path, runner_path: str | Path
) -> tuple[tuple[Path, str], ...]:
    root = project_root.resolve(strict=True)
    package_root = root / "src" / "tunnelgeopt"
    package_sources = sorted(package_root.glob("*.py"), key=lambda item: item.name)
    if not package_sources or not (package_root / "__init__.py").is_file():
        raise ProbeProvenanceError("src/tunnelgeopt/*.py closure is missing __init__.py")
    candidates: list[Path] = [
        *package_sources,
        root / "pyproject.toml",
        root / ".gitignore",
        root / ".gitattributes",
    ]
    candidates.extend(
        sorted(
            (
                path
                for path in root.iterdir()
                if path.is_file() and (path.suffix == ".lock" or path.name == "pylock.toml")
            ),
            key=lambda item: item.name,
        )
    )
    candidates.extend((Path(runner_path), Path(config_path)))

    closure: dict[str, Path] = {}
    for candidate in candidates:
        resolved, relative = _project_relative_file(root, candidate, "probe source")
        closure[relative] = resolved
    return tuple((closure[relative], relative) for relative in sorted(closure))


def _source_inventory(
    project_root: Path, *, config_path: str | Path, runner_path: str | Path
) -> tuple[tuple[ProbeSourceFile, ...], str]:
    sources: list[ProbeSourceFile] = []
    for resolved, relative in _probe_source_paths(
        project_root, config_path=config_path, runner_path=runner_path
    ):
        _git_text(project_root, "ls-files", "--error-unmatch", "--", relative)
        payload = resolved.read_bytes()
        sources.append(
            ProbeSourceFile(
                path=relative,
                sha256=hashlib.sha256(payload).hexdigest(),
                size_bytes=len(payload),
            )
        )
    canonical = json.dumps(
        [asdict(source) for source in sources], sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return tuple(sources), hashlib.sha256(canonical).hexdigest()


def capture_probe_project_preflight(
    project_root: str | Path,
    *,
    expected_project_head: str,
    config_path: str | Path,
    runner_path: str | Path,
) -> ProbeProjectSnapshot:
    """Bind a probe to an exact clean commit already present at its upstream."""

    root = Path(project_root).resolve(strict=True)
    expected = expected_project_head.strip().lower()
    if re.fullmatch(r"[0-9a-f]{40}", expected) is None:
        raise ProbeProvenanceError("expected project HEAD must be a full 40-character SHA-1")
    git_root = Path(_git_text(root, "rev-parse", "--show-toplevel")).resolve(strict=True)
    if git_root != root:
        raise ProbeProvenanceError("project_root is not the Git worktree root")
    head = _git_text(root, "rev-parse", "HEAD").lower()
    upstream_head = _git_text(root, "rev-parse", "@{upstream}").lower()
    upstream_ref = _git_text(root, "rev-parse", "--abbrev-ref", "@{upstream}")
    if not (head == upstream_head == expected):
        raise ProbeProvenanceError(
            "probe requires project HEAD == upstream HEAD == --expected-project-head"
        )
    status = _git_text(root, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise ProbeProvenanceError("probe preflight requires a completely clean worktree")
    module_files = {
        "tunnelgeopt.fracture_benchmark": "fracture_benchmark.py",
        "tunnelgeopt.fracture": "fracture.py",
        "tunnelgeopt.fracture_benchmark_mesh": "fracture_benchmark_mesh.py",
        "tunnelgeopt.fracture_benchmark_validation": "fracture_benchmark_validation.py",
        "tunnelgeopt.fracture_bvp": "fracture_bvp.py",
    }
    for module_name, filename in module_files.items():
        module = importlib.import_module(module_name)
        origin = getattr(module, "__file__", None)
        expected_origin = root / "src" / "tunnelgeopt" / filename
        if origin is None or Path(origin).resolve(strict=True) != expected_origin.resolve(
            strict=True
        ):
            raise ProbeProvenanceError(f"imported {module_name} module is outside this worktree")
    _, config_relative = _project_relative_file(root, config_path, "config")
    _, runner_relative = _project_relative_file(root, runner_path, "runner")
    sources, digest = _source_inventory(root, config_path=config_path, runner_path=runner_path)
    return ProbeProjectSnapshot(
        expected_project_head=expected,
        project_head=head,
        upstream_head=upstream_head,
        upstream_ref=upstream_ref,
        config_path=config_relative,
        runner_path=runner_relative,
        source_files=sources,
        source_inventory_sha256=digest,
        captured_utc=_utc_now(),
        _project_root=root,
    )


def verify_probe_project_postflight(snapshot: ProbeProjectSnapshot) -> str:
    """Recheck Git/source identity and cleanliness immediately before result writes."""

    root = snapshot._project_root
    head = _git_text(root, "rev-parse", "HEAD").lower()
    upstream = _git_text(root, "rev-parse", "@{upstream}").lower()
    if not (head == upstream == snapshot.expected_project_head):
        raise ProbeProvenanceError("project or upstream HEAD changed during the probe")
    sources, digest = _source_inventory(
        root,
        config_path=root / snapshot.config_path,
        runner_path=root / snapshot.runner_path,
    )
    if sources != snapshot.source_files or digest != snapshot.source_inventory_sha256:
        raise ProbeProvenanceError("probe source closure changed during the probe")
    status = _git_text(root, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise ProbeProvenanceError("probe postflight found worktree changes created during solving")
    return _utc_now()


def probe_runtime_environment() -> dict[str, Any]:
    """Return a host-path-free runtime/package/thread summary for the artifact."""

    package_versions: dict[str, str | None] = {}
    for distribution in ("tunnelgeopt", "numpy", "scipy", "scikit-fem", "gmsh"):
        try:
            package_versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            package_versions[distribution] = None
    thread_names = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    )
    return {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": package_versions,
        "threads": {
            "logical_cpu_count": os.cpu_count(),
            "environment": {name: os.environ.get(name) for name in thread_names},
        },
    }


def _require_finite_json(value: Any, path: str = "artifact") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string mapping key")
            _require_finite_json(child, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _require_finite_json(child, f"{path}[{index}]")
        return
    raise TypeError(f"{path} contains unsupported JSON value {type(value).__name__}")


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    _require_finite_json(value)
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _reject_host_path_strings(value: Any, project_root: Path, path: str = "artifact") -> None:
    """Reject a local project-root substring before JSON escaping can hide it."""

    forbidden = str(project_root).replace("\\", "/").casefold()
    if isinstance(value, str):
        normalized = value.replace("\\", "/").casefold()
        if forbidden and forbidden in normalized:
            raise ProbeProvenanceError(f"{path} contains the local project path")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            _reject_host_path_strings(key, project_root, f"{path}.<key>")
            _reject_host_path_strings(child, project_root, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_host_path_strings(child, project_root, f"{path}[{index}]")


def _path_is_reparse_point(path: Path) -> bool:
    try:
        attributes = os.lstat(path).st_file_attributes
    except AttributeError:
        return path.is_symlink()
    except FileNotFoundError:
        return False
    return bool(attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def _git_path_is_ignored(project_root: Path, relative_path: str) -> bool:
    completed = subprocess.run(
        ("git", "check-ignore", "--quiet", "--no-index", "--", relative_path),
        cwd=project_root,
        check=False,
        capture_output=True,
    )
    if completed.returncode == 0:
        return True
    if completed.returncode == 1:
        return False
    detail = completed.stderr.decode("utf-8", errors="replace").strip() or "unknown git error"
    raise ProbeProvenanceError(f"git check-ignore failed: {detail}")


def _write_exclusive_file(path: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary_name, path)
        os.unlink(temporary_name)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def reserve_probe_output_directory(
    project_root: str | Path, output_directory: str | Path
) -> tuple[Path, str]:
    """Exclusively reserve an empty in-project run leaf before mesh generation."""

    root = Path(project_root).resolve(strict=True)
    candidate = Path(output_directory)
    if not candidate.is_absolute():
        candidate = root / candidate
    target = Path(os.path.abspath(candidate))
    try:
        relative = target.relative_to(root).as_posix()
    except ValueError as exc:
        raise ProbeProvenanceError(
            "probe output directory must be inside the project root"
        ) from exc
    if target == root:
        raise ProbeProvenanceError("probe output directory must be a unique child run leaf")
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"probe output directory already exists: {target}")
    relative_parts = Path(relative).parts
    if any(part.casefold() == ".git" for part in relative_parts):
        raise ProbeProvenanceError("probe output directory must not be inside .git")
    parent = root
    for part in relative_parts[:-1]:
        parent = parent / part
        if (parent.exists() or parent.is_symlink()) and _path_is_reparse_point(parent):
            raise ProbeProvenanceError(
                "probe output directory must not traverse a symlink, junction, or reparse point"
            )
    ignored_candidates = (
        relative,
        f"{relative}/result.json",
        f"{relative}/artifact_manifest.json",
    )
    if any(_git_path_is_ignored(root, candidate) for candidate in ignored_candidates):
        raise ProbeProvenanceError(
            "probe output directory or artifact files must not be ignored by Git"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(exist_ok=False)
    return target, relative


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FractureBenchmarkPreflightError(f"{name} must be a mapping")
    return value


def _benchmark_entry(config: Mapping[str, Any], benchmark_id: str) -> Mapping[str, Any]:
    entries = config["loading"]["benchmarks"]
    try:
        return next(entry for entry in entries if entry["id"] == benchmark_id)
    except StopIteration as exc:
        raise FractureBenchmarkPreflightError(f"unknown benchmark {benchmark_id!r}") from exc


def _tier_entry(config: Mapping[str, Any], tier: str) -> Mapping[str, Any]:
    try:
        return next(entry for entry in config["mesh"]["tiers"] if entry["id"] == tier)
    except StopIteration as exc:
        raise FractureBenchmarkPreflightError(f"unknown mesh tier {tier!r}") from exc


def lame_to_young_poisson(lame_lambda: float, shear_modulus: float) -> tuple[float, float]:
    """Convert 3-D isotropic Lame parameters to ``(E, nu)`` and regress back."""

    lam = float(lame_lambda)
    mu = float(shear_modulus)
    if not math.isfinite(lam) or not math.isfinite(mu) or mu <= 0.0 or 3.0 * lam + 2.0 * mu <= 0.0:
        raise ValueError("Lame lambda and shear modulus must define a stable isotropic material")
    denominator = lam + mu
    if denominator == 0.0:
        raise ValueError("lambda + mu must be nonzero")
    young = mu * (3.0 * lam + 2.0 * mu) / denominator
    poisson = lam / (2.0 * denominator)
    material = AT2Material(young, poisson, 1.0, 1.0)
    if not math.isclose(material.lame_lambda, lam, rel_tol=2.0e-14, abs_tol=1.0e-14):
        raise RuntimeError("lambda -> (E, nu) -> lambda regression failed")
    if not math.isclose(material.shear_modulus, mu, rel_tol=2.0e-14, abs_tol=1.0e-14):
        raise RuntimeError("mu -> (E, nu) -> mu regression failed")
    return young, poisson


def benchmark_material(config: Mapping[str, Any]) -> AT2Material:
    """Build the exact AT2 material encoded by the frozen protocol."""

    validate_fracture_sent_sens_config(config)
    values = config["material"]
    young, poisson = lame_to_young_poisson(
        values["lame_lambda_kN_per_mm2"], values["shear_modulus_kN_per_mm2"]
    )
    return AT2Material(
        young_modulus=young,
        poisson_ratio=poisson,
        fracture_toughness=float(values["critical_fracture_energy_kN_per_mm"]),
        length_scale=float(values["regularization_length_ell_mm"]),
        residual_stiffness=float(config["fracture_model"]["residual_stiffness_k"]),
    )


def _facet_nodes(benchmark_mesh: Any, label: str) -> IntArray:
    facets_by_label = _require_mapping(benchmark_mesh.boundary_facets, "mesh.boundary_facets")
    if label not in facets_by_label:
        raise FractureBenchmarkPreflightError(f"mesh is missing facet label {label!r}")
    facets = np.asarray(facets_by_label[label], dtype=np.int64)
    mesh_facets = np.asarray(benchmark_mesh.mesh.facets, dtype=np.int64)
    if (
        facets.ndim != 1
        or facets.size == 0
        or np.any(facets < 0)
        or np.any(facets >= mesh_facets.shape[1])
    ):
        raise FractureBenchmarkPreflightError(f"facet label {label!r} is empty or invalid")
    return np.unique(mesh_facets[:, facets])


def _close(actual: Any, expected: Any, name: str) -> None:
    if not np.allclose(
        np.asarray(actual, dtype=float), np.asarray(expected, dtype=float), rtol=0.0, atol=1.0e-14
    ):
        raise FractureBenchmarkPreflightError(f"{name} differs from frozen config")


def preflight_fracture_benchmark(
    config: Mapping[str, Any], benchmark_mesh: Any, *, benchmark_id: str, tier: str
) -> FractureBenchmarkPreflight:
    """Validate config/mesh/loading identity without invoking either BVP solver."""

    validate_fracture_sent_sens_config(config)
    benchmark = _benchmark_entry(config, benchmark_id)
    mesh_tier = _tier_entry(config, tier)
    plan = benchmark_mesh.plan
    if plan.loading != benchmark_id or plan.tier != tier:
        raise FractureBenchmarkPreflightError("mesh loading/tier differs from requested case")
    _close(plan.target_h_mm, mesh_tier["h_target_mm"], "mesh target_h_mm")
    _close(plan.farfield_h_mm, mesh_tier["bulk_h_target_mm"], "mesh farfield_h_mm")
    corridor = benchmark["refined_corridor"]
    notch_line = config["geometry"]["notch"]["line_mm"]
    expected_notch = (
        (float(notch_line["y"]), float(notch_line["z"][0])),
        (float(notch_line["y"]), float(notch_line["z"][1])),
    )
    _close(plan.notch_polyline_yz_mm, expected_notch, "mesh notch polyline")
    _close(
        plan.notch_band_half_width_mm,
        corridor["notch_face_and_tip_refinement_distance_mm"],
        "mesh notch refinement distance",
    )
    _close(
        plan.propagation_corridor_polyline_yz_mm,
        corridor["centerline_yz_mm"],
        "mesh propagation corridor",
    )
    _close(
        plan.propagation_corridor_half_width_mm,
        corridor["half_width_mm"],
        "mesh propagation corridor half width",
    )

    identity = _require_mapping(benchmark_mesh.identity, "mesh.identity")
    metadata = _require_mapping(benchmark_mesh.metadata, "mesh.metadata")
    if tuple(identity.get("coordinate_order", ())) != ("y", "z"):
        raise FractureBenchmarkPreflightError("mesh coordinate order must be [y,z]")
    if identity.get("plan_sha256") != plan.plan_sha256:
        raise FractureBenchmarkPreflightError("mesh contains stale plan identity")
    topology_sha = identity.get("topology_sha256")
    if not isinstance(topology_sha, str) or len(topology_sha) != 64:
        raise FractureBenchmarkPreflightError("mesh topology identity is missing")
    recompute_topology = getattr(benchmark_mesh, "recompute_topology_sha256", None)
    if not callable(recompute_topology):
        raise FractureBenchmarkPreflightError("mesh must expose live topology-hash recomputation")
    if recompute_topology() != topology_sha:
        raise FractureBenchmarkPreflightError("mesh topology differs from stored identity")
    for audit in (
        "topology_audit_passed",
        "boundary_coverage_audit_passed",
        "zero_width_double_face_slit_audit_passed",
        "corridor_hmax_audit_passed",
    ):
        if metadata.get(audit) is not True:
            raise FractureBenchmarkPreflightError(f"mesh audit {audit!r} did not pass")

    expected_labels = tuple(config["geometry"]["boundary_labels"])
    point_labels = {"notch_tip"}
    facet_labels = set(expected_labels) - point_labels
    if set(benchmark_mesh.boundary_facets) != facet_labels:
        raise FractureBenchmarkPreflightError("mesh facet labels differ from frozen config")
    boundary_nodes = _require_mapping(
        getattr(benchmark_mesh, "boundary_nodes", None), "mesh.boundary_nodes"
    )
    if set(boundary_nodes) != point_labels:
        raise FractureBenchmarkPreflightError("mesh point labels differ from frozen config")
    tip_nodes = np.asarray(boundary_nodes["notch_tip"], dtype=np.int64)
    if tip_nodes.shape != (1,):
        raise FractureBenchmarkPreflightError("notch_tip must identify exactly one node")

    nodes = np.asarray(benchmark_mesh.nodes, dtype=np.float64)
    if nodes.ndim != 2 or nodes.shape[1] != 2 or not np.isfinite(nodes).all():
        raise FractureBenchmarkPreflightError("mesh nodes must be finite [N,2] [y,z]")
    _close(nodes[tip_nodes[0]], expected_notch[-1], "notch tip coordinate")
    top_nodes = _facet_nodes(benchmark_mesh, "top")
    bottom_nodes = _facet_nodes(benchmark_mesh, "bottom")
    if not np.allclose(nodes[top_nodes, 0], 1.0, rtol=0.0, atol=_COORDINATE_TOLERANCE_MM):
        raise FractureBenchmarkPreflightError("top label is not y=1; possible [z,y] swap")
    if not np.allclose(nodes[bottom_nodes, 0], 0.0, rtol=0.0, atol=_COORDINATE_TOLERANCE_MM):
        raise FractureBenchmarkPreflightError("bottom label is not y=0; possible [z,y] swap")
    if np.intersect1d(top_nodes, bottom_nodes).size:
        raise FractureBenchmarkPreflightError("top and bottom node sets overlap")

    current_bvp_sha = prescribed_displacement_mesh_identity(benchmark_mesh.mesh)
    return FractureBenchmarkPreflight(
        benchmark_id=benchmark_id,
        tier=tier,
        protocol_sha256=FROZEN_CANONICAL_SHA256,
        mesh_plan_sha256=plan.plan_sha256,
        mesh_topology_sha256=topology_sha,
        bvp_mesh_sha256=current_bvp_sha,
        node_count=int(nodes.shape[0]),
        element_count=int(np.asarray(benchmark_mesh.elements).shape[0]),
        top_node_count=int(top_nodes.size),
        bottom_node_count=int(bottom_nodes.size),
    )


def build_prescribed_displacement_states(
    config: Mapping[str, Any],
    benchmark_mesh: Any,
    *,
    benchmark_id: str,
    tier: str,
    displacements_mm: Sequence[float],
) -> tuple[PrescribedDisplacementState, ...]:
    """Create an exact, immutable node-major path after strict preflight."""

    preflight = preflight_fracture_benchmark(
        config, benchmark_mesh, benchmark_id=benchmark_id, tier=tier
    )
    values = np.asarray(displacements_mm, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("displacements_mm must be a finite non-empty vector")
    if values[0] != 0.0 or (values.size > 1 and np.any(np.diff(values) <= 0.0)):
        raise ValueError("probe path must start at U=0 and then be strictly increasing")
    formal_grid = prescribed_displacements(config, benchmark_id)
    if values[-1] > formal_grid[-1]:
        raise ValueError("probe displacement exceeds the frozen formal endpoint")

    nodes = np.asarray(benchmark_mesh.nodes)
    top_nodes = _facet_nodes(benchmark_mesh, "top")
    bottom_nodes = _facet_nodes(benchmark_mesh, "bottom")
    bottom_y = np.sort(2 * bottom_nodes)
    bottom_z = np.sort(2 * bottom_nodes + 1)
    top_y = np.sort(2 * top_nodes)
    top_z = np.sort(2 * top_nodes + 1)
    constrained = np.unique(np.concatenate((bottom_y, bottom_z, top_y, top_z)))
    groups = {"bottom_u_y": bottom_y, "bottom_u_z": bottom_z, "top_u_y": top_y, "top_u_z": top_z}
    driven = "top_u_y" if benchmark_id == "sent" else "top_u_z"
    states: list[PrescribedDisplacementState] = []
    for index, displacement in enumerate(values):
        prescribed = np.zeros(constrained.size, dtype=np.float64)
        prescribed[np.isin(constrained, groups[driven])] = displacement
        payload = {
            "schema": "tunnelgeopt.fracture.benchmark.state.v1",
            "protocol_sha256": preflight.protocol_sha256,
            "mesh_sha256": preflight.bvp_mesh_sha256,
            "benchmark_id": benchmark_id,
            "tier": tier,
            "sequence_index": index,
            "U_mm": float(displacement),
            "dirichlet_dofs": constrained.tolist(),
            "dirichlet_values": prescribed.tolist(),
            "driven_group": driven,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        states.append(
            PrescribedDisplacementState(
                identity=f"fss1-{benchmark_id}-{tier}-{index:04d}-{digest[:16]}",
                mesh_identity=preflight.bvp_mesh_sha256,
                sequence_index=index,
                path_parameter=float(displacement),
                dirichlet_dofs=constrained,
                dirichlet_values=prescribed,
                external_force=np.zeros(2 * nodes.shape[0]),
                reaction_groups=groups,
                driven_group=driven,
            )
        )
    return tuple(states)


def _relative_global_balances(
    result: FixedDamageDisplacementBVPResult, nodes: FloatArray
) -> tuple[float, float]:
    # Only constrained residual entries are physical support reactions.  Free
    # residuals belong to the independent equilibrium-residual gate.
    support = np.zeros(2 * nodes.shape[0], dtype=np.float64)
    support[result.dirichlet_dofs] = np.asarray(result.reaction)[result.dirichlet_dofs]
    support = support.reshape((-1, 2))
    applied = np.asarray(result.external_force, dtype=np.float64).reshape((-1, 2))
    resultant = support + applied
    force_numerator = float(np.linalg.norm(resultant.sum(axis=0)))
    force_denominator = max(
        float(np.linalg.norm(support, axis=1).sum() + np.linalg.norm(applied, axis=1).sum()),
        _FORCE_FLOOR_KN,
    )
    # Out-of-plane moment about frozen origin (y,z)=(0,0): M_x = y*F_z-z*F_y.
    support_moment = nodes[:, 0] * support[:, 1] - nodes[:, 1] * support[:, 0]
    applied_moment = nodes[:, 0] * applied[:, 1] - nodes[:, 1] * applied[:, 0]
    moment_numerator = abs(float((support_moment + applied_moment).sum()))
    moment_denominator = max(
        float(np.abs(support_moment).sum() + np.abs(applied_moment).sum()),
        _MOMENT_FLOOR_KN_MM,
    )
    return force_numerator / force_denominator, moment_numerator / moment_denominator


def run_intact_fracture_benchmark_probe(
    config: Mapping[str, Any],
    benchmark_mesh: Any,
    *,
    benchmark_id: str,
    tier: str,
    displacements_mm: Sequence[float],
    options: FractureSolverOptions | None = None,
    step_solver: Callable[
        ..., FixedDamageDisplacementBVPResult
    ] = solve_fixed_damage_displacement_bvp,
    clock: Callable[[], float] = time.perf_counter,
) -> FractureBenchmarkProbe:
    """Run a bounded fixed-``d=0`` probe; never authorize a formal computation."""

    if len(displacements_mm) > 12:
        raise ValueError("development probe is capped at 12 explicit states")
    before = preflight_fracture_benchmark(
        config, benchmark_mesh, benchmark_id=benchmark_id, tier=tier
    )
    states = build_prescribed_displacement_states(
        config,
        benchmark_mesh,
        benchmark_id=benchmark_id,
        tier=tier,
        displacements_mm=displacements_mm,
    )
    material = benchmark_material(config)
    nodes = np.asarray(benchmark_mesh.nodes, dtype=np.float64)
    damage = np.zeros(nodes.shape[0], dtype=np.float64)
    controls = options or FractureSolverOptions()
    steps: list[ProbeStep] = []
    prior_reaction: FloatArray | None = None
    prior_u: FloatArray | None = None
    initial_energy: float | None = None
    path_work = 0.0
    initial_displacement: FloatArray | None = None
    for state in states:
        start = clock()
        result = step_solver(
            benchmark_mesh.mesh,
            material,
            state,
            damage=damage,
            initial_displacement=initial_displacement,
            options=controls,
        )
        elapsed = float(clock() - start)
        if not math.isfinite(elapsed) or elapsed < 0.0:
            raise RuntimeError("probe clock produced an invalid duration")
        if (
            result.state.identity != state.identity
            or result.mesh_identity != before.bvp_mesh_sha256
        ):
            raise RuntimeError("probe solver returned a mismatched state or mesh identity")
        if initial_energy is None:
            initial_energy = float(result.elastic_energy)
        current_u = np.asarray(result.displacement).ravel()[state.dirichlet_dofs]
        current_reaction = np.asarray(result.reaction)[state.dirichlet_dofs]
        if prior_reaction is not None and prior_u is not None:
            path_work += float(0.5 * (prior_reaction + current_reaction) @ (current_u - prior_u))
        energy_change = float(result.elastic_energy) - initial_energy
        energy_numerator = abs(energy_change - path_work)
        energy_denominator = max(abs(energy_change), abs(path_work), _ENERGY_FLOOR_KN_MM)
        force_balance, moment_balance = _relative_global_balances(result, nodes)
        steps.append(
            ProbeStep(
                sequence_index=state.sequence_index,
                prescribed_U_mm=state.path_parameter,
                wall_seconds=elapsed,
                converged=bool(result.converged),
                generalized_load_kN=float(result.generalized_load),
                elastic_energy_kN_mm=float(result.elastic_energy),
                equilibrium_relative_residual=float(result.equilibrium_residual),
                global_force_relative_imbalance=force_balance,
                global_moment_relative_imbalance=moment_balance,
                path_energy_relative_imbalance=energy_numerator / energy_denominator,
                damage_component_status="NOT_APPLICABLE_INTACT_D0_PROBE",
            )
        )
        prior_reaction = current_reaction.copy()
        prior_u = current_u.copy()
        initial_displacement = np.asarray(result.displacement).copy()

    after = preflight_fracture_benchmark(
        config, benchmark_mesh, benchmark_id=benchmark_id, tier=tier
    )
    if after != before:
        raise RuntimeError("mesh identity or metadata changed during the probe")
    durations = np.asarray([step.wall_seconds for step in steps], dtype=np.float64)
    median = float(np.median(durations))
    formal_increment_count = len(prescribed_displacements(config, benchmark_id)) - 1
    projection = formal_increment_count * median / 3600.0
    return FractureBenchmarkProbe(
        schema=PROBE_SCHEMA,
        status="DEVELOPMENT_INTACT_FIXED_DAMAGE_PROBE_ONLY",
        claim_boundary="not_Miehe_reproduction_not_coupled_timing_not_Phase1_ready",
        benchmark_id=benchmark_id,
        tier=tier,
        protocol_sha256=before.protocol_sha256,
        mesh_plan_sha256=before.mesh_plan_sha256,
        mesh_topology_sha256=before.mesh_topology_sha256,
        bvp_mesh_sha256=before.bvp_mesh_sha256,
        node_count=before.node_count,
        element_count=before.element_count,
        top_node_count=before.top_node_count,
        bottom_node_count=before.bottom_node_count,
        material=MappingProxyType(
            {
                "young_modulus_kN_per_mm2": material.young_modulus,
                "poisson_ratio": material.poisson_ratio,
                "fracture_toughness_kN_per_mm": material.fracture_toughness,
                "length_scale_mm": material.length_scale,
                "residual_stiffness": material.residual_stiffness,
            }
        ),
        prescribed_U_mm=tuple(float(value) for value in displacements_mm),
        steps=tuple(steps),
        median_step_wall_seconds=median,
        projected_formal_increment_count=formal_increment_count,
        projected_formal_case_wall_hours=projection,
        projection_interpretation="intact_fixed_damage_lower_bound_non_authorizing",
        authorizes_medium_fine_or_formal_run=False,
    )


def write_probe_artifact_atomic(
    probe: FractureBenchmarkProbe,
    output_directory: str | Path,
    *,
    project_snapshot: ProbeProjectSnapshot,
    started_utc: str,
    completed_utc: str,
    sanitized_command: Sequence[str],
    solver_options: FractureSolverOptions,
    runtime_environment: Mapping[str, Any],
) -> ProbeArtifactBundle:
    """Publish ``result.json`` and its manifest in a never-reused run leaf.

    The leaf is reserved with ``mkdir(exist_ok=False)`` and each file is opened
    exclusively.  An interrupted write intentionally leaves an unusable leaf;
    evidence is never retried into, replaced, or completed by a later writer.
    """

    root = project_snapshot._project_root
    target = Path(output_directory).resolve(strict=False)
    try:
        output_relative = target.relative_to(root).as_posix()
    except ValueError as exc:
        raise ProbeProvenanceError(
            "probe output directory must be inside the project root"
        ) from exc
    if target == root:
        raise ProbeProvenanceError("probe output directory must be a unique child run leaf")
    if not target.is_dir() or any(target.iterdir()):
        raise ProbeProvenanceError("probe output leaf was not exclusively reserved and empty")
    postflight_utc = verify_probe_project_postflight(project_snapshot)
    command = list(sanitized_command)
    if not command or not all(isinstance(argument, str) and argument for argument in command):
        raise ValueError("sanitized_command must contain non-empty strings")
    result = {
        "schema": PROBE_RESULT_SCHEMA,
        "status": probe.status,
        "claim_boundary": probe.claim_boundary,
        "timing": {
            "started_utc": started_utc,
            "completed_utc": completed_utc,
            "postflight_verified_utc": postflight_utc,
        },
        "execution": {
            "sanitized_command": command,
            "solver_options": asdict(solver_options),
            "runtime_environment": dict(runtime_environment),
        },
        "project_provenance": project_snapshot.as_dict(),
        "postflight": {
            "head_equals_upstream_equals_expected": True,
            "source_inventory_unchanged": True,
            "full_worktree_clean_rechecked_with_reserved_empty_leaf": True,
        },
        "evidence_scope": {
            "single_case_only": True,
            "paired_sent_sens_campaign_supported": False,
            "real_probe_allowed": False,
            "real_probe_definition": "paired_sent_sens_campaign_for_paper_evidence",
            "paper_evidence_eligible": False,
        },
        "probe": probe.as_dict(),
    }
    _reject_host_path_strings(result, root)
    result_payload = _canonical_json_bytes(result)
    result_sha256 = hashlib.sha256(result_payload).hexdigest()
    manifest = {
        "schema": PROBE_MANIFEST_SCHEMA,
        "status": "IMMUTABLE_EXCLUSIVE_PROBE_ARTIFACT_SET",
        "output_directory": output_relative,
        "project_head": project_snapshot.project_head,
        "source_inventory_sha256": project_snapshot.source_inventory_sha256,
        "artifacts": [
            {
                "path": "result.json",
                "sha256": result_sha256,
                "size_bytes": len(result_payload),
            }
        ],
        "manifest_sha256_reporting": "returned_by_writer_and_printed_by_cli_not_self_embedded",
        "authorizes_medium_fine_or_formal_run": False,
        "single_case_only": True,
        "real_probe_allowed": False,
        "paper_evidence_eligible": False,
    }
    _reject_host_path_strings(manifest, root)
    manifest_payload = _canonical_json_bytes(manifest)
    manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()

    _write_exclusive_file(target / "result.json", result_payload)
    # Completion marker is deliberately linked last. A leaf without this file
    # is interrupted evidence and must never be resumed.
    _write_exclusive_file(target / "artifact_manifest.json", manifest_payload)
    return ProbeArtifactBundle(
        result_sha256=result_sha256,
        manifest_sha256=manifest_sha256,
        result_size_bytes=len(result_payload),
        manifest_size_bytes=len(manifest_payload),
    )


__all__ = [
    "PROBE_MANIFEST_SCHEMA",
    "PROBE_RESULT_SCHEMA",
    "PROBE_SCHEMA",
    "FractureBenchmarkPreflight",
    "FractureBenchmarkPreflightError",
    "FractureBenchmarkProbe",
    "ProbeArtifactBundle",
    "ProbeProjectSnapshot",
    "ProbeProvenanceError",
    "ProbeSourceFile",
    "ProbeStep",
    "benchmark_material",
    "build_prescribed_displacement_states",
    "capture_probe_project_preflight",
    "lame_to_young_poisson",
    "preflight_fracture_benchmark",
    "probe_runtime_environment",
    "reserve_probe_output_directory",
    "run_intact_fracture_benchmark_probe",
    "verify_probe_project_postflight",
    "write_probe_artifact_atomic",
]
