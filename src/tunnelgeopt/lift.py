"""GeoPT-compatible geometric lifting for tunnel cross-sections.

This module intentionally implements only the cheap geometry-boundary prior.
It does not model momentum balance, damage, fracture, contact, or rockburst.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .geometry import (
    TunnelGeometry,
    embed_yz,
    nearest_boundary_vectors,
    sample_rock_points,
    surface_points_and_normals,
)

FloatArray = NDArray[np.floating]


@dataclass(frozen=True)
class LiftedCase:
    """In-memory representation of one geometry and several lifted prompts."""

    x: FloatArray
    conditions: tuple[FloatArray, ...]
    supervises: tuple[FloatArray, ...]
    meta: dict[str, Any]


def _cross_2d(a: FloatArray, b: FloatArray) -> FloatArray:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _advance_until_first_hit(
    positions: FloatArray,
    displacement: FloatArray,
    boundary: FloatArray,
) -> FloatArray:
    """Advance line segments, stopping at 99% of the first wall intersection."""

    q = boundary
    s = np.roll(boundary, -1, axis=0) - boundary
    p = positions[:, None, :]
    r = displacement[:, None, :]
    q_minus_p = q[None, :, :] - p
    denominator = _cross_2d(r, s[None, :, :])
    non_parallel = np.abs(denominator) > 1e-12
    safe_denominator = np.where(non_parallel, denominator, 1.0)
    t = _cross_2d(q_minus_p, s[None, :, :]) / safe_denominator
    u = _cross_2d(q_minus_p, r) / safe_denominator
    valid = non_parallel & (t >= 0.0) & (t <= 1.0) & (u >= 0.0) & (u <= 1.0)
    first_t = np.min(np.where(valid, t, np.inf), axis=1)
    factor = np.where(np.isfinite(first_t), 0.99 * first_t, 1.0)
    return positions + factor[:, None] * displacement


def _sample_directions(
    rng: np.random.Generator,
    n_volume: int,
    n_surface: int,
    *,
    prompt_mode: str,
    stress_angle_deg: float,
    wall_directions: FloatArray,
) -> FloatArray:
    angles = rng.uniform(0.0, 2.0 * np.pi, n_volume + n_surface)
    directions = np.column_stack([np.sin(angles), np.cos(angles)])
    if prompt_mode == "stress_aligned":
        use_principal = rng.random(n_volume) < 0.6
        sign = rng.choice(np.array([-1.0, 1.0]), size=n_volume)
        jitter = rng.normal(0.0, np.deg2rad(15.0), size=n_volume)
        principal_angle = np.deg2rad(stress_angle_deg) + jitter
        principal = sign[:, None] * np.column_stack(
            [np.sin(principal_angle), np.cos(principal_angle)]
        )
        radial_sign = rng.choice(np.array([-1.0, 1.0]), size=n_volume)
        radial = radial_sign[:, None] * wall_directions
        specialized = np.where(use_principal[:, None], principal, radial)
        directions[:n_volume] = specialized
    elif prompt_mode != "random":
        raise ValueError("prompt_mode must be 'random' or 'stress_aligned'")
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
    return directions


def generate_lifted_case(
    geometry: TunnelGeometry,
    *,
    n_volume: int = 32768,
    n_surface: int = 4096,
    n_prompts: int = 10,
    steps: int = 3,
    domain_scale: float = 3.0,
    min_step: float = 0.0,
    max_step: float = 0.4,
    prompt_mode: str = "random",
    stress_angle_deg: float = 0.0,
    seed: int = 0,
) -> LiftedCase:
    """Generate one lightweight lifted-geometry case.

    The serialized feature dimensions match GeoPT: ``x[:,7]``, a four-channel
    direction/step condition, and ``3 * steps`` vector-distance targets.  The
    official nine-channel target therefore requires ``steps=3``.
    """

    if n_prompts <= 0 or steps <= 0:
        raise ValueError("n_prompts and steps must be positive")
    if min_step < 0.0 or max_step <= min_step:
        raise ValueError("step range must satisfy 0 <= min_step < max_step")
    rng = np.random.default_rng(seed)
    volume = sample_rock_points(geometry, n_volume, domain_scale=domain_scale, seed=seed)
    surface, surface_normals = surface_points_and_normals(geometry, n_surface)
    volume_distance, volume_direction, _ = nearest_boundary_vectors(volume, geometry.boundary_yz)
    # Match GeoPT's released input convention: a volume-point geometry
    # direction points from the query toward its closest surface. Supervision
    # below uses the opposite sign (position - closest).
    x_volume = np.column_stack([embed_yz(volume), volume_distance, embed_yz(-volume_direction)])
    x_surface = np.column_stack([embed_yz(surface), np.zeros(n_surface), embed_yz(surface_normals)])
    x = np.vstack([x_volume, x_surface]).astype(np.float16)

    conditions: list[FloatArray] = []
    supervises: list[FloatArray] = []
    for _ in range(n_prompts):
        directions = _sample_directions(
            rng,
            n_volume,
            n_surface,
            prompt_mode=prompt_mode,
            stress_angle_deg=stress_angle_deg,
            wall_directions=volume_direction,
        )
        step_lengths = rng.uniform(min_step, max_step, n_volume + n_surface)
        step_lengths[n_volume:] = 0.0
        condition = np.column_stack([embed_yz(directions), step_lengths]).astype(np.float16)

        positions = np.vstack([volume, surface]).copy()
        targets: list[FloatArray] = []
        for step_index in range(steps):
            _, _, nearest = nearest_boundary_vectors(positions, geometry.boundary_yz)
            targets.append(embed_yz(positions - nearest))
            if step_index == steps - 1:
                break
            displacement = directions[:n_volume] * step_lengths[:n_volume, None]
            positions[:n_volume] = _advance_until_first_hit(
                positions[:n_volume], displacement, geometry.boundary_yz
            )
            positions[n_volume:] = surface
        conditions.append(condition)
        supervises.append(np.concatenate(targets, axis=1).astype(np.float16))

    meta: dict[str, Any] = {
        "schema_version": "0.1.0",
        "claim_scope": "geometry_boundary_prior_only",
        "shape": geometry.shape,
        "coordinate_convention": "x=tunnel_axis,y=vertical,z=horizontal",
        "characteristic_radius": geometry.characteristic_radius,
        "normalization": "coordinates divided by characteristic_radius",
        "roughness_amplitude": geometry.roughness_amplitude,
        "geometry_seed": geometry.seed,
        "generation_seed": int(seed),
        "n_volume": int(n_volume),
        "n_surface": int(n_surface),
        "n_prompts": int(n_prompts),
        "steps": int(steps),
        "prompt_mode": prompt_mode,
        "stress_angle_deg": float(stress_angle_deg),
        "domain_scale": float(domain_scale),
        "step_range": [float(min_step), float(max_step)],
    }
    return LiftedCase(
        x=x,
        conditions=tuple(conditions),
        supervises=tuple(supervises),
        meta=meta,
    )
