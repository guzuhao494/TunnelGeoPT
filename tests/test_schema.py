from __future__ import annotations

import json

import numpy as np
import pytest

from tunnelgeopt.schema import (
    SchemaValidationError,
    load_sample,
    save_sample,
    validate_arrays,
    validate_meta,
)


def make_arrays(n: int = 8, dtype=np.float16):
    x = np.arange(n * 7, dtype=np.float32).reshape(n, 7).astype(dtype)
    condition = np.zeros((n, 4), dtype=dtype)
    condition[:, 0] = 1
    supervise = np.zeros((n, 9), dtype=dtype)
    return x, condition, supervise


def test_save_load_round_trip_with_meta(tmp_path):
    x, condition, supervise = make_arrays()
    meta = {
        "case_id": "tunnel-001",
        "num_points": len(x),
        "dtype": "float16",
        "normalization": {"characteristic_length": 5.0},
    }

    paths = save_sample(
        tmp_path / "case_001",
        x,
        condition,
        supervise,
        trajectory_index=3,
        meta=meta,
    )
    sample = load_sample(tmp_path / "case_001", trajectory_index=3, require_meta=True)

    assert paths.condition.name == "condition_3.npy"
    assert paths.supervise.name == "supervise_3.npy"
    np.testing.assert_array_equal(sample.x, x)
    np.testing.assert_array_equal(sample.condition, condition)
    np.testing.assert_array_equal(sample.supervise, supervise)
    assert sample.meta == meta
    assert sample.num_points == 8
    assert sample.dtype == np.dtype(np.float16)


def test_second_trajectory_reuses_matching_geometry_and_meta(tmp_path):
    x, condition, supervise = make_arrays()
    meta = {"case_id": "shared"}
    case_dir = tmp_path / "case"
    save_sample(case_dir, x, condition, supervise, trajectory_index=0, meta=meta)
    save_sample(case_dir, x, condition + np.float16(0.5), supervise, trajectory_index=1, meta=meta)

    sample = load_sample(case_dir, trajectory_index=1, require_meta=True)
    assert sample.meta == meta
    np.testing.assert_array_equal(sample.condition, condition + np.float16(0.5))


@pytest.mark.parametrize(
    ("which", "shape", "expected_text"),
    [
        ("x", (4, 6), "x must have shape [N,7]"),
        ("condition", (4, 5), "condition must have shape [N,4]"),
        ("supervise", (4, 8), "supervise must have shape [N,9]"),
    ],
)
def test_rejects_wrong_width(which, shape, expected_text):
    arrays = dict(zip(("x", "condition", "supervise"), make_arrays(4)))
    arrays[which] = np.zeros(shape, dtype=np.float16)

    with pytest.raises(SchemaValidationError, match=r"must have shape") as exc_info:
        validate_arrays(**arrays)

    assert expected_text in str(exc_info.value)


def test_rejects_point_count_mismatch():
    x, condition, supervise = make_arrays(5)

    with pytest.raises(SchemaValidationError, match="Point-count mismatch"):
        validate_arrays(x, condition[:-1], supervise)


@pytest.mark.parametrize(
    ("which", "value"),
    [("x", np.nan), ("condition", np.inf), ("supervise", -np.inf)],
)
def test_rejects_non_finite_values(which, value):
    arrays = dict(zip(("x", "condition", "supervise"), make_arrays(5)))
    arrays[which][0, 0] = value

    with pytest.raises(SchemaValidationError, match="non-finite"):
        validate_arrays(**arrays)


def test_default_dtype_is_strict_float16():
    x, condition, supervise = make_arrays(dtype=np.float32)

    with pytest.raises(SchemaValidationError, match="requires dtype float16"):
        validate_arrays(x, condition, supervise)

    assert validate_arrays(x, condition, supervise, expected_dtype=np.float32) == len(x)


def test_rejects_mixed_dtypes_even_without_exact_expected_dtype():
    x, condition, supervise = make_arrays()
    condition = condition.astype(np.float32)

    with pytest.raises(SchemaValidationError, match="Dtype mismatch"):
        validate_arrays(x, condition, supervise, expected_dtype=None, require_same_dtype=True)


def test_save_casts_to_requested_dtype(tmp_path):
    x, condition, supervise = make_arrays(dtype=np.float32)
    save_sample(tmp_path, x, condition, supervise)

    sample = load_sample(tmp_path)
    assert sample.dtype == np.dtype(np.float16)
    assert sample.condition.dtype == np.dtype(np.float16)
    assert sample.supervise.dtype == np.dtype(np.float16)


def test_existing_trajectory_is_protected(tmp_path):
    x, condition, supervise = make_arrays()
    save_sample(tmp_path, x, condition, supervise)

    with pytest.raises(FileExistsError, match="Trajectory file already exists"):
        save_sample(tmp_path, x, condition, supervise)


def test_metadata_conflict_is_detected_before_new_trajectory_write(tmp_path):
    x, condition, supervise = make_arrays()
    save_sample(
        tmp_path,
        x,
        condition,
        supervise,
        trajectory_index=0,
        meta={"case_id": "original"},
    )

    with pytest.raises(FileExistsError, match="metadata.*different content"):
        save_sample(
            tmp_path,
            x,
            condition,
            supervise,
            trajectory_index=1,
            meta={"case_id": "different"},
        )

    assert not (tmp_path / "condition_1.npy").exists()
    assert not (tmp_path / "supervise_1.npy").exists()


def test_load_reports_missing_required_array(tmp_path):
    x, _, _ = make_arrays()
    np.save(tmp_path / "x.npy", x, allow_pickle=False)

    with pytest.raises(FileNotFoundError, match="condition_0.npy"):
        load_sample(tmp_path)


def test_meta_must_be_json_serializable_and_consistent():
    with pytest.raises(SchemaValidationError, match="JSON-serializable"):
        validate_meta({"bad": {1, 2, 3}})
    with pytest.raises(SchemaValidationError, match="num_points.*does not match"):
        validate_meta({"num_points": 7}, num_points=8)
    with pytest.raises(SchemaValidationError, match="dtype.*does not match"):
        validate_meta({"dtype": "float32"}, dtype=np.float16)


def test_invalid_meta_file_is_rejected(tmp_path):
    x, condition, supervise = make_arrays()
    save_sample(tmp_path, x, condition, supervise)
    (tmp_path / "meta.json").write_text("{not-json", encoding="utf-8")

    with pytest.raises(SchemaValidationError, match="Could not load metadata"):
        load_sample(tmp_path)


def test_meta_file_is_utf8_json(tmp_path):
    x, condition, supervise = make_arrays()
    save_sample(
        tmp_path,
        x,
        condition,
        supervise,
        meta={"description": "硬岩隧洞"},
    )

    meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    assert meta["description"] == "硬岩隧洞"
