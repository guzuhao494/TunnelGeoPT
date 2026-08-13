import numpy as np

from tunnelgeopt.geometry import make_tunnel_boundary
from tunnelgeopt.lift import generate_lifted_case


def test_lifted_case_matches_geopt_feature_dimensions() -> None:
    geometry = make_tunnel_boundary("horseshoe", n_points=64, seed=3)
    case = generate_lifted_case(
        geometry,
        n_volume=96,
        n_surface=32,
        n_prompts=2,
        steps=3,
        max_step=0.2,
        seed=5,
    )
    assert case.x.shape == (128, 7)
    assert len(case.conditions) == 2
    assert len(case.supervises) == 2
    assert case.conditions[0].shape == (128, 4)
    assert case.supervises[0].shape == (128, 9)
    assert case.x.dtype == np.float16
    assert np.isfinite(case.x).all()
    assert np.all(case.conditions[0][-32:, -1] == 0.0)
    expected_t0 = -case.x[:96, 3:4].astype(np.float32) * case.x[:96, 4:7].astype(np.float32)
    assert np.allclose(case.supervises[0][:96, :3].astype(np.float32), expected_t0, atol=2e-3)


def test_stress_aligned_prompt_is_deterministic_for_seed() -> None:
    geometry = make_tunnel_boundary("circle", n_points=64)
    kwargs = {
        "n_volume": 48,
        "n_surface": 16,
        "n_prompts": 1,
        "prompt_mode": "stress_aligned",
        "stress_angle_deg": 30.0,
        "seed": 9,
    }
    first = generate_lifted_case(geometry, **kwargs)
    second = generate_lifted_case(geometry, **kwargs)
    assert np.array_equal(first.x, second.x)
    assert np.array_equal(first.conditions[0], second.conditions[0])
    assert np.array_equal(first.supervises[0], second.supervises[0])
