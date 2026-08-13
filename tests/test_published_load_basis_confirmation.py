from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts" / "confirmation" / "linear-load-basis-v0.5.0" / "confirmation.json"
EXPECTED_SHA256 = "b86efe9e283f4ee1f00cb2d8cb01d754475fb9a8ddf1e34338d01ecd5c89e736"
EXPECTED_IMPLEMENTATION_HEAD = "44d244e344a0e40dbf33fdaa21cc823b8f46a85a"


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _artifact() -> tuple[bytes, dict[str, Any]]:
    payload = ARTIFACT.read_bytes()
    value = json.loads(payload, object_pairs_hook=_reject_duplicate_keys)
    assert isinstance(value, dict)
    return payload, value


def _walk_floats(value: Any) -> list[float]:
    if isinstance(value, dict):
        return [number for child in value.values() for number in _walk_floats(child)]
    if isinstance(value, list):
        return [number for child in value for number in _walk_floats(child)]
    return [value] if isinstance(value, float) else []


def test_published_confirmation_is_canonical_authenticated_and_private_path_free() -> None:
    payload, result = _artifact()
    assert hashlib.sha256(payload).hexdigest() == EXPECTED_SHA256
    canonical = (
        json.dumps(
            result,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    assert payload == canonical
    assert all(math.isfinite(number) for number in _walk_floats(result))
    text = payload.decode("utf-8")
    assert "C:\\Users\\" not in text
    assert "C:/Users/" not in text
    assert "17196" not in text


def test_published_confirmation_ledger_and_raw_metrics_reproduce_decision() -> None:
    _, result = _artifact()
    evaluation = result["evaluation"]
    summary = evaluation["execution_summary"]
    assert summary == {
        "attempted_direct_fem_solve_count": 24,
        "completed_validated_direct_fem_solve_count": 24,
        "failed_direct_fem_solve_count": 0,
        "not_attempted_direct_fem_solve_count": 0,
        "planned_direct_fem_solve_count": 24,
        "solver_returned_count": 24,
    }
    geometries = evaluation["geometry_results"]
    assert [row["section_family"] for row in geometries] == [
        "circle",
        "horseshoe",
        "straight_wall_arch",
    ]
    ledger = [record for row in geometries for record in row["solver_records"]]
    assert len(ledger) == 24
    assert [record["planned_solve_index"] for record in ledger] == list(range(24))
    assert len({record["case_group_id"] for record in ledger}) == 24
    assert len({record["load_group_id"] for record in ledger}) == 8
    assert all(record["status"] == "completed" for record in ledger)
    for row in geometries:
        records = row["solver_records"]
        assert [record["load_role"] for record in records] == ["basis"] * 3 + ["heldout"] * 5
        assert len({record["mesh_identity_sha256"] for record in records}) == 1
        assert len({record["query_hash"] for record in records}) == 1
        assert len({record["boundary_float64_sha256"] for record in records}) == 1

    errors = [
        error
        for row in geometries
        for error in row["response_analyses"]["query_total_in_plane_stress"]["heldout_relative_l2"]
    ]
    assert len(errors) == 15
    assert statistics.median(errors) == pytest.approx(4.885724690966474e-15, rel=0.0, abs=1e-30)
    assert max(errors) == pytest.approx(5.882054174085674e-15, rel=0.0, abs=1e-30)
    assert all(result["gate_checks"].values())
    assert len(result["gate_checks"]) == 17
    assert result["classification"] == "LINEAR_ELASTIC_LOAD_AXIS_FACTORIZATION_CONFIRMED"
    assert result["all_gates_passed"] is True


def test_published_confirmation_retains_narrow_provenance_and_claim_scope() -> None:
    _, result = _artifact()
    preflight = result["execution_preflight"]
    assert preflight["git_head"] == EXPECTED_IMPLEMENTATION_HEAD
    assert preflight["git_upstream_commit"] == EXPECTED_IMPLEMENTATION_HEAD
    assert preflight["head_equals_upstream"] is True
    assert result["plan"]["identity_audit"]["passed"] is True
    assert all(
        not intersection
        for intersection in result["plan"]["identity_audit"]["intersections"].values()
    )
    confirmed = set(result["claim_scope"]["confirmed"])
    excluded = set(result["claim_scope"]["excluded"])
    assert "linear_factorization_of_in_plane_farfield_load_axis" in confirmed
    assert {
        "geometry_generalization",
        "mesh_generalization",
        "material_generalization",
        "fracture",
        "damage",
        "rockburst",
        "field_prediction",
        "engineering_truth",
    }.issubset(excluded)
