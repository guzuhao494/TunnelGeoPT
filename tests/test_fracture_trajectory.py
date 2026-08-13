from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("scipy")
skfem = pytest.importorskip("skfem")

from tunnelgeopt.elasticity import compute_element_strain
from tunnelgeopt.fracture import (
    AT2Material,
    FractureSolverOptions,
    ScheduledAT2Result,
    ScheduledAT2StepResult,
    miehe_spectral_response,
)
from tunnelgeopt.fracture_loading import Phase1LoadSchedule, compile_phase1_load_schedule
from tunnelgeopt.fracture_trajectory import (
    FractureTrajectoryAdapterError,
    FractureTrajectoryRunFailed,
    Phase1TrajectoryIdentity,
    damage_graph_connectivity,
    run_phase1_development_trajectory,
    save_and_verify_phase1_development_run,
)
from tunnelgeopt.fracture_validation import load_fracture_phase1_config
from tunnelgeopt.mesh import FARFIELD, ROCK, WALL, TunnelMesh


def _annulus_mesh() -> TunnelMesh:
    nodes = np.asarray(
        [
            [-2.0, -2.0],
            [2.0, -2.0],
            [2.0, 2.0],
            [-2.0, 2.0],
            [-1.0, -1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
        ],
        dtype=np.float64,
    )
    elements = np.asarray(
        [
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=np.int64,
    )
    mesh = skfem.MeshTri(nodes.T, elements.T, validate=True, sort_t=False)
    boundary = np.asarray(mesh.boundary_facets(), dtype=np.int64)
    boundary_nodes = np.asarray(mesh.facets[:, boundary], dtype=np.int64)
    wall = boundary[np.all(np.isin(boundary_nodes, [4, 5, 6, 7]), axis=0)]
    farfield = boundary[np.all(np.isin(boundary_nodes, [0, 1, 2, 3]), axis=0)]
    mesh = mesh.with_boundaries({WALL: wall, FARFIELD: farfield})
    return TunnelMesh(
        mesh=mesh,
        nodes=np.asarray(mesh.p.T, dtype=np.float64),
        elements=np.asarray(mesh.t.T, dtype=np.int64),
        boundary_facets={WALL: wall, FARFIELD: farfield},
        facet_markers=np.zeros(mesh.facets.shape[1], dtype=np.int64),
        cell_markers=np.ones(mesh.t.shape[1], dtype=np.int64),
        physical_tags={ROCK: 1, WALL: 1, FARFIELD: 2},
        outer_bounds=(-2.0, 2.0, -2.0, 2.0),
        metadata={"test_fixture": "square_annulus"},
    )


def _material() -> AT2Material:
    return AT2Material(
        young_modulus=500.0,
        poisson_ratio=0.25,
        fracture_toughness=8.0e-6,
        length_scale=0.04,
        residual_stiffness=1.0e-8,
    )


def _identity() -> Phase1TrajectoryIdentity:
    return Phase1TrajectoryIdentity(
        trajectory_id="development-circle-m1-p1",
        case_id="fp1-circle-m1-p1",
        mesh_id="test-annulus-v1",
        geometry_id="circle-test-v1",
        material_id="m1",
        solver_hash="d" * 64,
        geometry={"section_family": "circle", "characteristic_radius": 1.0},
        solver={"name": "deterministic-test-prefix-solver", "version": "1"},
        env={"fixture": "unit-test"},
        meta={"purpose": "adapter-regression"},
    )


def _affine_displacement(
    nodes: np.ndarray, material: AT2Material, schedule: Phase1LoadSchedule, s: float
) -> np.ndarray:
    stress = schedule.state_at(s).farfield_stress_tension_positive_yz
    lame_lambda = material.lame_lambda
    mu = material.shear_modulus
    normal = np.asarray(
        [[lame_lambda + 2.0 * mu, lame_lambda], [lame_lambda, lame_lambda + 2.0 * mu]]
    )
    normal_strain = np.linalg.solve(normal, np.diag(stress))
    strain = np.asarray([normal_strain[0], normal_strain[1], stress[0, 1] / mu])
    tensor = np.asarray([[strain[0], 0.5 * strain[2]], [0.5 * strain[2], strain[1]]])
    return nodes @ tensor.T


def _internal_force(
    nodes: np.ndarray, elements: np.ndarray, area: np.ndarray, stress: np.ndarray
) -> np.ndarray:
    triangles = nodes[elements]
    determinant = (triangles[:, 1, 0] - triangles[:, 0, 0]) * (
        triangles[:, 2, 1] - triangles[:, 0, 1]
    ) - (triangles[:, 2, 0] - triangles[:, 0, 0]) * (triangles[:, 1, 1] - triangles[:, 0, 1])
    gradients = np.empty((elements.shape[0], 3, 2), dtype=np.float64)
    gradients[:, 0, 0] = (triangles[:, 1, 1] - triangles[:, 2, 1]) / determinant
    gradients[:, 0, 1] = (triangles[:, 2, 0] - triangles[:, 1, 0]) / determinant
    gradients[:, 1, 0] = (triangles[:, 2, 1] - triangles[:, 0, 1]) / determinant
    gradients[:, 1, 1] = (triangles[:, 0, 0] - triangles[:, 2, 0]) / determinant
    gradients[:, 2, 0] = (triangles[:, 0, 1] - triangles[:, 1, 1]) / determinant
    gradients[:, 2, 1] = (triangles[:, 1, 0] - triangles[:, 0, 0]) / determinant
    matrices = np.zeros((elements.shape[0], 3, 6), dtype=np.float64)
    matrices[:, 0, 0::2] = gradients[:, :, 0]
    matrices[:, 1, 1::2] = gradients[:, :, 1]
    matrices[:, 2, 0::2] = gradients[:, :, 1]
    matrices[:, 2, 1::2] = gradients[:, :, 0]
    local = area[:, None] * np.einsum("mij,mi->mj", matrices, stress)
    assembled = np.zeros(2 * nodes.shape[0], dtype=np.float64)
    dofs = np.empty((elements.shape[0], 6), dtype=np.int64)
    dofs[:, 0::2] = 2 * elements
    dofs[:, 1::2] = 2 * elements + 1
    np.add.at(assembled, dofs.ravel(), local.ravel())
    return assembled


def _synthetic_step(
    mesh: TunnelMesh,
    material: AT2Material,
    schedule: Phase1LoadSchedule,
    s: float,
    *,
    converged: bool,
    rejected_damage: float = 0.0,
    energy_change: float = 0.0,
) -> ScheduledAT2StepResult:
    state = schedule.state_at(s)
    displacement = _affine_displacement(mesh.nodes, material, schedule, s)
    strain, area = compute_element_strain(mesh.nodes, mesh.elements, displacement)
    element_damage = np.full(mesh.elements.shape[0], rejected_damage, dtype=np.float64)
    response = miehe_spectral_response(strain, material, damage=element_damage)
    stress = np.column_stack(
        (response.stress[:, 1, 1], response.stress[:, 2, 2], response.stress[:, 1, 2])
    )
    internal = _internal_force(mesh.nodes, mesh.elements, area, stress)
    farfield_nodes = np.unique(mesh.mesh.facets[:, mesh.boundary_facets[FARFIELD]])
    dirichlet = np.column_stack((2 * farfield_nodes, 2 * farfield_nodes + 1)).ravel()
    free = np.setdiff1d(np.arange(2 * mesh.nodes.shape[0]), dirichlet)
    wall_force = np.zeros_like(internal)
    wall_force[free] = internal[free]
    damage = np.full(mesh.nodes.shape[0], rejected_damage, dtype=np.float64)
    elastic_energy = float(
        np.sum(
            area
            * (
                ((1.0 - rejected_damage) ** 2 + material.residual_stiffness) * response.psi_positive
                + response.psi_negative
            )
        )
    )
    fracture_energy = float(
        material.fracture_toughness
        * np.sum(area)
        * rejected_damage**2
        / (2.0 * material.length_scale)
    )
    neumann = float(np.dot(wall_force, displacement.ravel()))
    return ScheduledAT2StepResult(
        load_parameter=float(s),
        displacement=displacement,
        correction_displacement=np.zeros_like(displacement),
        damage=damage,
        strain=strain,
        stress=stress,
        psi_positive=np.asarray(response.psi_positive),
        psi_negative=np.asarray(response.psi_negative),
        history=np.asarray(response.psi_positive),
        elastic_energy=elastic_energy,
        fracture_energy=fracture_energy,
        external_work=neumann,
        internal_force=internal,
        wall_nodal_force=wall_force,
        dirichlet_dofs=dirichlet,
        farfield_prescribed_displacement=displacement.ravel()[dirichlet],
        total_potential_energy=elastic_energy + fracture_energy - neumann,
        equilibrium_residual=0.0,
        kkt_residual=0.0,
        complementarity_residual=0.0,
        irreversibility_violation=0.0,
        range_violation=0.0,
        displacement_change=0.0,
        damage_change=0.0,
        energy_change=energy_change,
        staggered_iterations=1,
        displacement_iterations=1,
        damage_iterations=1,
        converged=converged,
        load_state=state,
    )


def _prefix_solver(
    *,
    reject_direct_first_interval: bool = False,
    reject_every_candidate: bool = False,
    tamper_candidate_load_state: bool = False,
    mutate_recomputed_origin: bool = False,
    energy_change: float = 0.0,
):
    calls: list[tuple[float, ...]] = []

    def solve(
        mesh: TunnelMesh,
        material: AT2Material,
        schedule: Phase1LoadSchedule,
        *,
        load_path: Any,
        options: Any,
    ) -> ScheduledAT2Result:
        parameters = tuple(load_path.load_parameters)
        calls.append(parameters)
        steps: list[ScheduledAT2StepResult] = []
        for index, parameter in enumerate(parameters):
            is_candidate = index == len(parameters) - 1 and parameter != 0.0
            direct_first = (
                reject_direct_first_interval
                and is_candidate
                and parameter == 0.025
                and 0.0125 not in parameters
            )
            rejected = is_candidate and (reject_every_candidate or direct_first)
            step = _synthetic_step(
                mesh,
                material,
                schedule,
                parameter,
                converged=not rejected,
                rejected_damage=0.2 if rejected else 0.0,
                energy_change=energy_change,
            )
            if tamper_candidate_load_state and is_candidate:
                state = step.load_state
                step = replace(
                    step,
                    load_state=replace(
                        state,
                        wall_facet_ids=state.wall_facet_ids[::-1],
                        wall_zone_weights=state.wall_zone_weights[::-1],
                        wall_release=state.wall_release[::-1],
                    ),
                )
            if mutate_recomputed_origin and len(calls) > 1 and index == 0:
                changed = step.displacement.copy()
                changed[0, 0] += 1.0e-6
                step = replace(step, displacement=changed)
            steps.append(step)
        assert options.raise_on_nonconvergence is False
        return ScheduledAT2Result(
            nodes=mesh.nodes.copy(),
            elements=mesh.elements.copy(),
            steps=tuple(steps),
            material=material,
            load_path=load_path,
            options=options,
            load_schedule=schedule,
        )

    return solve, calls


def _run(solve_prefix: Any, *, options: FractureSolverOptions | None = None):
    mesh = _annulus_mesh()
    config = load_fracture_phase1_config()
    schedule = compile_phase1_load_schedule(config, "p1", 1.0, mesh)
    return run_phase1_development_trajectory(
        mesh,
        _material(),
        schedule,
        config,
        _identity(),
        equilibrium_force_normalization_floor=1.0e-12,
        energy_balance_normalization_floor=1.0e-12,
        options=options,
        solve_prefix=solve_prefix,
    )


def test_adaptive_retry_rolls_back_and_maps_all_41_required_outputs(tmp_path: Path) -> None:
    solver, calls = _prefix_solver(reject_direct_first_interval=True)
    run = _run(solver)

    assert run.formal_labels_allowed is False
    assert run.solver_energy_increment_residual_available is True
    assert run.trajectory.meta["solver_energy_increment_residual_available"] is True
    assert run.trajectory.meta["solver_energy_increment_tolerance"] == 1.0e-8
    assert np.allclose(run.trajectory.meta["solver_energy_increment_residual"], 0.0)
    assert np.allclose(run.trajectory.staggered_potential_energy_change, 0.0)
    assert run.trajectory.solver["relative_energy_increment_tolerance"] == 1.0e-8
    assert run.trajectory.num_steps == 42
    frozen_required = np.asarray(
        load_fracture_phase1_config()["time_discretization"]["required_output_s"]
    )
    assert np.array_equal(run.required_output_s, frozen_required)
    assert run.required_output_indices[1] == 2
    assert run.trajectory.load_parameter[1] == 0.0125
    assert np.all(np.diff(run.trajectory.load_parameter) > 0.0)
    assert np.all(run.trajectory.damage == 0.0)
    assert np.all(run.trajectory.history == run.trajectory.psi_plus)
    assert np.allclose(run.trajectory.cumulative_external_work, 0.0)
    assert len(calls) == 43

    rejected = [entry for entry in run.trajectory.attempt_ledger if not entry["accepted"]]
    assert len(rejected) == 1
    assert rejected[0]["load_parameter_target"] == 0.025
    assert rejected[0]["load_state_sha256"] is None
    assert rejected[0]["staggered_potential_energy_change"] is None
    assert rejected[0]["wall_work_increment"] is None
    assert all(
        entry["staggered_potential_energy_change"] == 0.0
        for entry in run.trajectory.attempt_ledger
        if entry["accepted"]
    )
    assert run.trajectory.retry_count[1] == 1
    assert run.trajectory.step_halvings[1] == 1
    assert np.all(run.balance.force_relative_residual == 0.0)
    assert np.all(run.balance.moment_relative_residual == 0.0)

    paths, loaded = save_and_verify_phase1_development_run(tmp_path / "development", run)
    assert paths.arrays.is_file() and paths.meta.is_file()
    assert np.array_equal(loaded.load_parameter, run.trajectory.load_parameter)


def test_retry_exhaustion_fails_closed_without_rejected_work_or_hash() -> None:
    solver, _ = _prefix_solver(reject_every_candidate=True)
    with pytest.raises(FractureTrajectoryRunFailed, match="retry budget exhausted") as captured:
        _run(solver)
    failure = captured.value
    assert failure.accepted_load_parameters == (0.0,)
    rejected = [entry for entry in failure.attempt_ledger if not entry["accepted"]]
    assert len(rejected) == 7
    assert [entry["attempt_index"] for entry in rejected] == list(range(7))
    assert all(entry["load_state_sha256"] is None for entry in rejected)
    assert all(entry["staggered_potential_energy_change"] is None for entry in rejected)
    assert all(entry["cumulative_external_work"] is None for entry in rejected)


def test_facet_order_tamper_is_rejected_before_schema_construction() -> None:
    solver, _ = _prefix_solver(tamper_candidate_load_state=True)
    with pytest.raises(FractureTrajectoryAdapterError, match="facet row order"):
        _run(solver)


def test_nondeterministic_fresh_prefix_is_rejected() -> None:
    solver, _ = _prefix_solver(mutate_recomputed_origin=True)
    with pytest.raises(FractureTrajectoryAdapterError, match="accepted prefix field displacement"):
        _run(solver)


def test_damage_connectivity_is_derived_by_widest_path() -> None:
    elements = np.asarray([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
    wall = np.asarray([[0, 1]], dtype=np.int64)
    farfield = np.asarray([[2, 3]], dtype=np.int64)
    damage = np.asarray([0.9, 0.8, 0.6, 0.7], dtype=np.float64)
    assert damage_graph_connectivity(elements, wall, farfield, damage) == pytest.approx(0.7)
    assert damage_graph_connectivity(elements, wall, farfield, np.zeros(4)) == 0.0


def test_frozen_energy_tolerance_is_enforced_independently() -> None:
    solver, _ = _prefix_solver(energy_change=2.0e-8)
    with pytest.raises(FractureTrajectoryRunFailed, match="initial s=0 state failed") as captured:
        _run(solver)
    assert captured.value.attempt_ledger[0]["failure_code"] == ("ENERGY_INCREMENT_NOT_CONVERGED")

    permissive = FractureSolverOptions(
        staggered_tolerance=1.0e-8,
        energy_tolerance=1.0e-7,
        raise_on_nonconvergence=False,
    )
    accepted_solver, _ = _prefix_solver()
    with pytest.raises(ValueError, match="looser energy tolerance"):
        _run(accepted_solver, options=permissive)
