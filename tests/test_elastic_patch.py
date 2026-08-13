from __future__ import annotations

import pytest

pytest.importorskip("skfem")

from tunnelgeopt.elastic_validation import run_affine_patch_test


@pytest.mark.parametrize("young_modulus", [1.0e-6, 100.0, 1.0e9])
def test_full_assembled_affine_patch_recovers_uniform_stress(
    young_modulus: float,
) -> None:
    report = run_affine_patch_test(young_modulus=young_modulus)
    assert report["passed"] is True
    assert report["stress_relative_l2"] < 1e-12
    assert report["free_dof_residual"] < 1e-12
