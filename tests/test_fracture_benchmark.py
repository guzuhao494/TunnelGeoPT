from __future__ import annotations

import json
from dataclasses import replace
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("scipy")

from tunnelgeopt.fracture import FractureSolverOptions
from tunnelgeopt.fracture_benchmark import (
    FractureBenchmarkPreflightError,
    benchmark_material,
    build_prescribed_displacement_states,
    lame_to_young_poisson,
    preflight_fracture_benchmark,
    run_intact_fracture_benchmark_probe,
    write_probe_artifact_atomic,
)
from tunnelgeopt.fracture_benchmark_mesh import benchmark_mesh_plan
from tunnelgeopt.fracture_benchmark_validation import load_fracture_sent_sens_config
from tunnelgeopt.fracture_bvp import prescribed_displacement_mesh_identity


class _FixtureMesh:
    def __init__(self, benchmark_id: str = "sent") -> None:
        nodes = np.asarray([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.5, 0.5]])
        elements = np.asarray([[0, 4, 1], [1, 4, 3], [3, 4, 2], [2, 4, 0]])
        facets = np.asarray(
            [
                [2, 0, 2, 0, 1, 0, 0],
                [3, 1, 4, 4, 3, 4, 4],
            ],
            dtype=np.int64,
        )
        self.mesh = SimpleNamespace(p=nodes.T, t=elements.T, facets=facets)
        self.nodes = nodes
        self.elements = elements
        self.boundary_facets = MappingProxyType(
            {
                "top": np.asarray([0]),
                "bottom": np.asarray([1]),
                "left_upper": np.asarray([2]),
                "left_lower": np.asarray([3]),
                "right": np.asarray([4]),
                "notch_upper": np.asarray([5]),
                "notch_lower": np.asarray([6]),
            }
        )
        self.boundary_nodes = MappingProxyType({"notch_tip": np.asarray([4])})
        self.plan = benchmark_mesh_plan(loading=benchmark_id, tier="coarse")
        self.identity = MappingProxyType(
            {
                "coordinate_order": ("y", "z"),
                "plan_sha256": self.plan.plan_sha256,
                "topology_sha256": "a" * 64,
            }
        )
        self.metadata = MappingProxyType(
            {
                "topology_audit_passed": True,
                "boundary_coverage_audit_passed": True,
                "zero_width_double_face_slit_audit_passed": True,
                "corridor_hmax_audit_passed": True,
            }
        )

    def recompute_topology_sha256(self) -> str:
        return str(self.identity["topology_sha256"])


@pytest.fixture
def config() -> dict:
    return load_fracture_sent_sens_config()


def test_lame_conversion_is_exactly_regressed(config: dict) -> None:
    lam = config["material"]["lame_lambda_kN_per_mm2"]
    mu = config["material"]["shear_modulus_kN_per_mm2"]
    young, poisson = lame_to_young_poisson(lam, mu)
    material = benchmark_material(config)

    assert young == pytest.approx(210.0, rel=5.0e-5)
    assert poisson == pytest.approx(0.3000, rel=5.0e-5)
    assert material.young_modulus == young
    assert material.poisson_ratio == poisson
    assert material.lame_lambda == pytest.approx(lam, rel=2.0e-14)
    assert material.shear_modulus == pytest.approx(mu, rel=2.0e-14)


def test_preflight_rejects_coordinate_rotation_and_target_mismatch(config: dict) -> None:
    rotated = _FixtureMesh()
    rotated.nodes = rotated.nodes[:, ::-1].copy()
    rotated.mesh.p = rotated.nodes.T
    with pytest.raises(FractureBenchmarkPreflightError, match="top label is not y=1"):
        preflight_fracture_benchmark(config, rotated, benchmark_id="sent", tier="coarse")

    wrong_target = _FixtureMesh()
    wrong_target.plan = replace(wrong_target.plan, target_h_mm=0.008)
    with pytest.raises(FractureBenchmarkPreflightError, match="target_h_mm"):
        preflight_fracture_benchmark(config, wrong_target, benchmark_id="sent", tier="coarse")


def test_preflight_rejects_stale_topology_identity(config: dict) -> None:
    mesh = _FixtureMesh()
    mesh.recompute_topology_sha256 = lambda: "b" * 64  # type: ignore[method-assign]
    with pytest.raises(FractureBenchmarkPreflightError, match="topology differs"):
        preflight_fracture_benchmark(config, mesh, benchmark_id="sent", tier="coarse")

    missing_recompute = _FixtureMesh()
    missing_recompute.recompute_topology_sha256 = None  # type: ignore[method-assign]
    with pytest.raises(FractureBenchmarkPreflightError, match="live topology"):
        preflight_fracture_benchmark(config, missing_recompute, benchmark_id="sent", tier="coarse")


@pytest.mark.parametrize(
    ("benchmark_id", "driven_group", "driven_parity"),
    (("sent", "top_u_y", 0), ("sens", "top_u_z", 1)),
)
def test_sent_sens_states_have_exact_node_major_dof_mapping(
    config: dict, benchmark_id: str, driven_group: str, driven_parity: int
) -> None:
    mesh = _FixtureMesh(benchmark_id)
    states = build_prescribed_displacement_states(
        config,
        mesh,
        benchmark_id=benchmark_id,
        tier="coarse",
        displacements_mm=(0.0, 1.0e-8),
    )
    expected_identity = prescribed_displacement_mesh_identity(mesh.mesh)

    assert [state.sequence_index for state in states] == [0, 1]
    assert [state.path_parameter for state in states] == [0.0, 1.0e-8]
    assert all(state.mesh_identity == expected_identity for state in states)
    assert all(state.driven_group == driven_group for state in states)
    assert np.all(states[1].reaction_groups[driven_group] % 2 == driven_parity)
    assert np.all(
        states[1].dirichlet_values[
            np.isin(states[1].dirichlet_dofs, states[1].reaction_groups[driven_group])
        ]
        == 1.0e-8
    )
    other = "top_u_z" if benchmark_id == "sent" else "top_u_y"
    assert np.all(
        states[1].dirichlet_values[
            np.isin(states[1].dirichlet_dofs, states[1].reaction_groups[other])
        ]
        == 0.0
    )


def test_intact_d0_two_state_path_solves_without_formal_trajectory(config: dict) -> None:
    mesh = _FixtureMesh("sent")
    probe = run_intact_fracture_benchmark_probe(
        config,
        mesh,
        benchmark_id="sent",
        tier="coarse",
        displacements_mm=(0.0, 1.0e-8),
        options=FractureSolverOptions(
            max_displacement_iterations=10,
            equilibrium_tolerance=1.0e-8,
        ),
    )

    assert len(probe.steps) == 2
    assert all(step.converged for step in probe.steps)
    assert all(
        step.damage_component_status == "NOT_APPLICABLE_INTACT_D0_PROBE" for step in probe.steps
    )
    assert probe.projected_formal_increment_count == 2000
    assert probe.projection_interpretation == "intact_fixed_damage_lower_bound_non_authorizing"
    assert probe.authorizes_medium_fine_or_formal_run is False


def test_probe_mock_records_each_duration_and_atomic_hash(config: dict, tmp_path) -> None:
    mesh = _FixtureMesh("sens")
    calls: list[str] = []

    def mock_solver(mesh_like, material, state, *, damage, initial_displacement, options):
        calls.append(state.identity)
        displacement = np.zeros((mesh_like.p.shape[1], 2))
        displacement.ravel()[state.dirichlet_dofs] = state.dirichlet_values
        reaction = np.zeros(2 * mesh_like.p.shape[1])
        reaction[state.reaction_groups[state.driven_group]] = state.path_parameter
        reaction[state.reaction_groups["bottom_u_z"]] = -state.path_parameter
        return SimpleNamespace(
            state=state,
            mesh_identity=state.mesh_identity,
            displacement=displacement,
            reaction=reaction,
            external_force=np.zeros_like(reaction),
            dirichlet_dofs=state.dirichlet_dofs,
            elastic_energy=state.path_parameter**2,
            converged=True,
            generalized_load=float(reaction[state.reaction_groups[state.driven_group]].sum()),
            equilibrium_residual=0.0,
        )

    ticks = iter((0.0, 0.25, 1.0, 1.5, 2.0, 2.75))
    probe = run_intact_fracture_benchmark_probe(
        config,
        mesh,
        benchmark_id="sens",
        tier="coarse",
        displacements_mm=(0.0, 1.0e-8, 2.0e-8),
        step_solver=mock_solver,
        clock=lambda: next(ticks),
    )
    assert len(calls) == 3
    assert [step.wall_seconds for step in probe.steps] == [0.25, 0.5, 0.75]
    assert probe.median_step_wall_seconds == 0.5
    assert probe.projected_formal_increment_count == 1500

    destination = tmp_path / "probe.json"
    digest = write_probe_artifact_atomic(probe, destination)
    raw = destination.read_bytes()
    assert len(digest) == 64
    assert json.loads(raw)["authorizes_medium_fine_or_formal_run"] is False
    assert "C:\\Users\\" not in raw.decode("utf-8")


def test_probe_cap_prevents_accidental_trajectory(config: dict) -> None:
    mesh = _FixtureMesh()
    with pytest.raises(ValueError, match="capped at 12"):
        run_intact_fracture_benchmark_probe(
            config,
            mesh,
            benchmark_id="sent",
            tier="coarse",
            displacements_mm=np.linspace(0.0, 1.2e-8, 13),
        )
