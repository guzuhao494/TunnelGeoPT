"""Stable physical-case identities and leakage-safe frozen dataset splits.

The parent case identity intentionally contains physics only.  Meshes, solver
fidelity, checkpoints, and restart attempts are derived records so that every
representation of the same physical problem remains in one dataset split.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from hashlib import sha256
from itertools import pairwise
from numbers import Integral, Real
from pathlib import Path
from typing import Any


class CaseValidationError(ValueError):
    """Raised when a case, split, or manifest violates the frozen contract."""


SECTION_FAMILIES = ("circle", "horseshoe", "straight_wall_arch")
SPLIT_NAMES = ("train", "dev", "locked_test")
SPLIT_METHOD = "sha256_sort_within_section_then_largest_remainder"
DEFAULT_SPLIT_RATIOS: dict[str, float] = {
    "train": 0.70,
    "dev": 0.15,
    "locked_test": 0.15,
}
CASE_IDENTITY_FIELDS = (
    "section_family",
    "section_parameters",
    "material_field_seed",
    "joint_network_seed",
    "dimensionless_material_parameters",
    "initial_stress_tensor",
    "stress_orientation",
    "excavation_schedule",
    "unloading_schedule",
)

_BOOKKEEPING_FIELDS = {"case_group_id", "split", "content_hash"}
_DERIVED_BOOKKEEPING_FIELDS = {"derived_record_id", "split", "content_hash"}
_HEX_DIGITS = frozenset("0123456789abcdef")
_MAX_ABS_VALUE = 1.0e6
_MAX_SEED = 2**32 - 1


def _normalise_json(value: Any, *, path: str = "$.") -> Any:
    """Return a JSON-native value with deterministic numeric normalisation."""

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number):
            raise CaseValidationError(f"{path} contains a non-finite number")
        if number == 0.0:
            return 0
        if number.is_integer() and abs(number) <= 2**53:
            return int(number)
        return number
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise CaseValidationError(f"{path} object keys must be non-empty strings")
            result[key] = _normalise_json(item, path=f"{path}{key}.")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalise_json(item, path=f"{path}[{index}].") for index, item in enumerate(value)]
    raise CaseValidationError(
        f"{path} contains unsupported type {type(value).__name__}; use JSON-native values"
    )


def canonical_json(value: Any) -> str:
    """Serialize *value* as compact, sorted, finite canonical JSON."""

    normalised = _normalise_json(value)
    return json.dumps(
        normalised,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_canonical(value: Any) -> str:
    """Return the lowercase SHA-256 digest of canonical UTF-8 JSON."""

    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CaseValidationError(f"{name} must be a mapping")
    return value


def _require_number(
    value: Any,
    name: str,
    *,
    lower: float = -_MAX_ABS_VALUE,
    upper: float = _MAX_ABS_VALUE,
    lower_open: bool = False,
    upper_open: bool = False,
) -> int | float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise CaseValidationError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise CaseValidationError(f"{name} must be finite")
    lower_bad = result <= lower if lower_open else result < lower
    upper_bad = result >= upper if upper_open else result > upper
    if lower_bad or upper_bad:
        left = "(" if lower_open else "["
        right = ")" if upper_open else "]"
        raise CaseValidationError(f"{name} must lie in {left}{lower}, {upper}{right}")
    return _normalise_json(value, path=f"$.{name}.")


def _validate_seed(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise CaseValidationError(f"{name} must be an integer")
    result = int(value)
    if not 0 <= result <= _MAX_SEED:
        raise CaseValidationError(f"{name} must lie in [0, {_MAX_SEED}]")
    return result


def _validate_numeric_mapping(value: Any, name: str, *, section: bool = False) -> dict[str, Any]:
    mapping = _require_mapping(value, name)
    if not mapping:
        raise CaseValidationError(f"{name} must not be empty")
    result: dict[str, Any] = {}
    for key, raw in mapping.items():
        if not isinstance(key, str) or not key:
            raise CaseValidationError(f"{name} keys must be non-empty strings")
        field = f"{name}.{key}"
        key_lower = key.lower()
        if key_lower.startswith("roughness_amplitude"):
            result[key] = _require_number(raw, field, lower=0.0, upper=0.08)
        elif key_lower == "poisson_ratio":
            result[key] = _require_number(
                raw, field, lower=-1.0, upper=0.5, lower_open=True, upper_open=True
            )
        elif key_lower in {"friction_angle_deg", "dilation_angle_deg"}:
            result[key] = _require_number(raw, field, lower=0.0, upper=90.0, upper_open=True)
        elif (
            section
            and any(
                token in key_lower
                for token in ("radius", "diameter", "width", "height", "span", "scale", "ratio")
            )
        ) or any(
            token in key_lower for token in ("modulus", "density", "strength", "cohesion", "energy")
        ):
            result[key] = _require_number(raw, field, lower=0.0, lower_open=True)
        else:
            result[key] = _require_number(raw, field)
    return dict(sorted(result.items()))


def _validate_stress_tensor(value: Any) -> list[list[int | float]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise CaseValidationError("initial_stress_tensor must be a 3x3 matrix or six Voigt values")
    rows = list(value)
    if len(rows) == 6 and all(
        isinstance(item, Real) and not isinstance(item, bool) for item in rows
    ):
        xx, yy, zz, xy, yz, xz = (
            _require_number(item, f"initial_stress_tensor[{index}]")
            for index, item in enumerate(rows)
        )
        return [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]
    if len(rows) != 3 or any(
        not isinstance(row, Sequence) or isinstance(row, (str, bytes, bytearray)) or len(row) != 3
        for row in rows
    ):
        raise CaseValidationError("initial_stress_tensor must be a 3x3 matrix or six Voigt values")
    matrix = [
        [_require_number(item, f"initial_stress_tensor[{i}][{j}]") for j, item in enumerate(row)]
        for i, row in enumerate(rows)
    ]
    for i in range(3):
        for j in range(i + 1, 3):
            if not math.isclose(
                float(matrix[i][j]), float(matrix[j][i]), rel_tol=1e-12, abs_tol=1e-12
            ):
                raise CaseValidationError("initial_stress_tensor must be symmetric")
    return matrix


def _validate_orientation(value: Any) -> Any:
    if isinstance(value, Real) and not isinstance(value, bool):
        angle = float(
            _require_number(
                value,
                "stress_orientation",
                lower=-180.0,
                upper=180.0,
                upper_open=True,
            )
        )
        return _normalise_json(angle % 180.0)
    mapping = _require_mapping(value, "stress_orientation")
    if set(mapping) != {"azimuth_deg", "dip_deg"}:
        raise CaseValidationError(
            "stress_orientation mapping must contain exactly azimuth_deg and dip_deg"
        )
    return {
        "azimuth_deg": _require_number(
            mapping["azimuth_deg"],
            "stress_orientation.azimuth_deg",
            lower=0.0,
            upper=360.0,
            upper_open=True,
        ),
        "dip_deg": _require_number(
            mapping["dip_deg"], "stress_orientation.dip_deg", lower=-90.0, upper=90.0
        ),
    }


def _validate_schedule(value: Any, name: str) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise CaseValidationError(f"{name} must be a non-empty sequence")
    items = list(value)
    if not items:
        raise CaseValidationError(f"{name} must not be empty")
    if all(isinstance(item, Real) and not isinstance(item, bool) for item in items):
        return [
            _require_number(item, f"{name}[{index}]", lower=0.0, upper=1.0)
            for index, item in enumerate(items)
        ]
    pairs: list[list[int | float]] = []
    for index, item in enumerate(items):
        if (
            not isinstance(item, Sequence)
            or isinstance(item, (str, bytes, bytearray))
            or len(item) != 2
        ):
            raise CaseValidationError(
                f"{name} entries must all be fractions or [time, fraction] pairs"
            )
        time_value = _require_number(item[0], f"{name}[{index}].time", lower=0.0)
        fraction = _require_number(item[1], f"{name}[{index}].fraction", lower=0.0, upper=1.0)
        pairs.append([time_value, fraction])
    times = [float(pair[0]) for pair in pairs]
    if any(current <= previous for previous, current in pairwise(times)):
        raise CaseValidationError(f"{name} times must be strictly increasing")
    return pairs


def case_identity_payload(case: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return the physics-only canonical parent-case payload."""

    case = _require_mapping(case, "case")
    missing = [field for field in CASE_IDENTITY_FIELDS if field not in case]
    if missing:
        raise CaseValidationError(f"case is missing identity fields: {', '.join(missing)}")
    family = case["section_family"]
    if family not in SECTION_FAMILIES:
        raise CaseValidationError(f"section_family must be one of {', '.join(SECTION_FAMILIES)}")
    payload = {
        "section_family": family,
        "section_parameters": _validate_numeric_mapping(
            case["section_parameters"], "section_parameters", section=True
        ),
        "material_field_seed": _validate_seed(case["material_field_seed"], "material_field_seed"),
        "joint_network_seed": _validate_seed(case["joint_network_seed"], "joint_network_seed"),
        "dimensionless_material_parameters": _validate_numeric_mapping(
            case["dimensionless_material_parameters"], "dimensionless_material_parameters"
        ),
        "initial_stress_tensor": _validate_stress_tensor(case["initial_stress_tensor"]),
        "stress_orientation": _validate_orientation(case["stress_orientation"]),
        "excavation_schedule": _validate_schedule(
            case["excavation_schedule"], "excavation_schedule"
        ),
        "unloading_schedule": _validate_schedule(case["unloading_schedule"], "unloading_schedule"),
    }
    return _normalise_json(payload)


def case_group_id(case: Mapping[str, Any]) -> str:
    """Hash the physical identity; derived solver details are deliberately ignored."""

    return sha256_canonical(case_identity_payload(case))


def _validate_hash(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX_DIGITS for character in value)
    ):
        raise CaseValidationError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _normalise_ratios(ratios: Mapping[str, float]) -> dict[str, float]:
    ratios = _require_mapping(ratios, "ratios")
    if set(ratios) != set(SPLIT_NAMES):
        raise CaseValidationError(f"ratios must use exactly these keys: {SPLIT_NAMES}")
    result = {
        name: float(_require_number(ratios[name], f"ratios.{name}", lower=0.0, upper=1.0))
        for name in SPLIT_NAMES
    }
    if not math.isclose(sum(result.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise CaseValidationError("split ratios must sum to 1")
    return result


def largest_remainder_counts(
    total: int, ratios: Mapping[str, float] = DEFAULT_SPLIT_RATIOS
) -> dict[str, int]:
    """Allocate integer split sizes using Hamilton's largest-remainder rule."""

    if isinstance(total, bool) or not isinstance(total, Integral) or total < 0:
        raise CaseValidationError("total must be a non-negative integer")
    normalised = _normalise_ratios(ratios)
    quotas = {name: int(total) * normalised[name] for name in SPLIT_NAMES}
    counts = {name: math.floor(quotas[name]) for name in SPLIT_NAMES}
    remaining = int(total) - sum(counts.values())
    priority = sorted(
        range(len(SPLIT_NAMES)),
        key=lambda index: (-(quotas[SPLIT_NAMES[index]] - counts[SPLIT_NAMES[index]]), index),
    )
    for index in priority[:remaining]:
        counts[SPLIT_NAMES[index]] += 1
    return counts


def _content_hash(record: Mapping[str, Any]) -> str:
    return sha256_canonical({key: value for key, value in record.items() if key != "content_hash"})


def freeze_case_splits(
    cases: Sequence[Mapping[str, Any]],
    ratios: Mapping[str, float] = DEFAULT_SPLIT_RATIOS,
) -> list[dict[str, Any]]:
    """Assign deterministic, section-stratified train/dev/locked-test splits."""

    normalised_ratios = _normalise_ratios(ratios)
    by_family: dict[str, list[dict[str, Any]]] = {family: [] for family in SECTION_FAMILIES}
    seen: set[str] = set()
    for index, case in enumerate(cases):
        payload = case_identity_payload(case)
        identity = sha256_canonical(payload)
        if identity in seen:
            raise CaseValidationError(
                f"duplicate parent case_group_id at input index {index}: {identity}"
            )
        seen.add(identity)
        if "case_group_id" in case and case["case_group_id"] != identity:
            raise CaseValidationError(f"case_group_id mismatch at input index {index}")
        by_family[payload["section_family"]].append({**payload, "case_group_id": identity})

    frozen: list[dict[str, Any]] = []
    for family in SECTION_FAMILIES:
        records = sorted(by_family[family], key=lambda item: item["case_group_id"])
        counts = largest_remainder_counts(len(records), normalised_ratios)
        cursor = 0
        for split in SPLIT_NAMES:
            for record in records[cursor : cursor + counts[split]]:
                complete = {**record, "split": split}
                complete["content_hash"] = _content_hash(complete)
                frozen.append(complete)
            cursor += counts[split]
    return sorted(frozen, key=lambda item: (item["section_family"], item["case_group_id"]))


def _parent_split_map(parents_or_manifest: Any) -> dict[str, str]:
    parents = (
        parents_or_manifest.get("cases")
        if isinstance(parents_or_manifest, Mapping)
        else parents_or_manifest
    )
    if not isinstance(parents, Sequence) or isinstance(parents, (str, bytes, bytearray)):
        raise CaseValidationError("parents must be frozen case records or a manifest")
    result: dict[str, str] = {}
    for index, parent in enumerate(parents):
        parent = _require_mapping(parent, f"parents[{index}]")
        identity = _validate_hash(parent.get("case_group_id"), f"parents[{index}].case_group_id")
        split = parent.get("split")
        if split not in SPLIT_NAMES:
            raise CaseValidationError(f"parents[{index}].split must be one of {SPLIT_NAMES}")
        if identity in result:
            raise CaseValidationError(f"duplicate parent case_group_id: {identity}")
        result[identity] = split
    return result


def derived_record_id(record: Mapping[str, Any]) -> str:
    """Hash one solver/mesh/fidelity child record without inherited bookkeeping."""

    record = _require_mapping(record, "derived record")
    parent = _validate_hash(record.get("case_group_id"), "derived record.case_group_id")
    payload = {
        key: value for key, value in record.items() if key not in _DERIVED_BOOKKEEPING_FIELDS
    }
    payload["case_group_id"] = parent
    if len(payload) == 1:
        raise CaseValidationError(
            "derived record must contain solver, mesh, fidelity, or run metadata"
        )
    return sha256_canonical(payload)


def inherit_derived_splits(
    derived_records: Sequence[Mapping[str, Any]], parents_or_manifest: Any
) -> list[dict[str, Any]]:
    """Attach each child's parent split and reject orphaned or duplicate children."""

    parent_splits = _parent_split_map(parents_or_manifest)
    inherited: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw_record in enumerate(derived_records):
        record = dict(_normalise_json(_require_mapping(raw_record, f"derived_records[{index}]")))
        parent = _validate_hash(
            record.get("case_group_id"), f"derived_records[{index}].case_group_id"
        )
        if parent not in parent_splits:
            raise CaseValidationError(f"derived record references unknown parent: {parent}")
        expected_split = parent_splits[parent]
        if "split" in record and record["split"] != expected_split:
            raise CaseValidationError(
                f"derived record split {record['split']!r} conflicts with parent split {expected_split!r}"
            )
        identity = derived_record_id(record)
        if identity in seen:
            raise CaseValidationError(
                f"duplicate derived_record_id at input index {index}: {identity}"
            )
        seen.add(identity)
        if "derived_record_id" in record and record["derived_record_id"] != identity:
            raise CaseValidationError(f"derived_record_id mismatch at input index {index}")
        clean = {
            key: value for key, value in record.items() if key not in _DERIVED_BOOKKEEPING_FIELDS
        }
        complete = {
            **clean,
            "derived_record_id": identity,
            "split": expected_split,
        }
        complete["content_hash"] = _content_hash(complete)
        inherited.append(complete)
    return sorted(inherited, key=lambda item: (item["case_group_id"], item["derived_record_id"]))


def _split_summary(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_section: dict[str, dict[str, int]] = {}
    overall = {split: 0 for split in SPLIT_NAMES}
    for family in SECTION_FAMILIES:
        counts = {split: 0 for split in SPLIT_NAMES}
        for case in cases:
            if case["section_family"] == family:
                counts[case["split"]] += 1
                overall[case["split"]] += 1
        by_section[family] = counts
    return {"overall": overall, "by_section": by_section}


def compute_manifest_content_hash(manifest: Mapping[str, Any]) -> str:
    """Hash semantic manifest content, excluding both top-level hash fields."""

    manifest = _require_mapping(manifest, "manifest")
    return sha256_canonical(
        {
            key: value
            for key, value in manifest.items()
            if key not in {"content_hash", "manifest_hash"}
        }
    )


def compute_manifest_hash(manifest: Mapping[str, Any]) -> str:
    """Hash the complete manifest envelope except ``manifest_hash`` itself."""

    manifest = _require_mapping(manifest, "manifest")
    return sha256_canonical(
        {key: value for key, value in manifest.items() if key != "manifest_hash"}
    )


def build_case_manifest(
    cases: Sequence[Mapping[str, Any]],
    *,
    derived_records: Sequence[Mapping[str, Any]] = (),
    ratios: Mapping[str, float] = DEFAULT_SPLIT_RATIOS,
    metadata: Mapping[str, Any] | None = None,
    manifest_version: str = "b-elastic-cases-v1",
) -> dict[str, Any]:
    """Build a deterministic manifest with record, content, and envelope hashes."""

    if not isinstance(manifest_version, str) or not manifest_version:
        raise CaseValidationError("manifest_version must be a non-empty string")
    normalised_ratios = _normalise_ratios(ratios)
    frozen = freeze_case_splits(cases, normalised_ratios)
    children = inherit_derived_splits(derived_records, frozen)
    manifest: dict[str, Any] = {
        "manifest_version": manifest_version,
        "split_policy": {
            "method": SPLIT_METHOD,
            "ratios": normalised_ratios,
            "order": list(SPLIT_NAMES),
            "test_is_locked": True,
        },
        "split_counts": _split_summary(frozen),
        "cases": frozen,
        "derived_records": children,
        "metadata": _normalise_json(metadata or {}),
    }
    manifest["content_hash"] = compute_manifest_content_hash(manifest)
    manifest["manifest_hash"] = compute_manifest_hash(manifest)
    return manifest


def verify_case_manifest(manifest: Mapping[str, Any]) -> None:
    """Recompute all identities, assignments, inheritance, counts, and hashes."""

    manifest = _require_mapping(manifest, "manifest")
    required = {
        "manifest_version",
        "split_policy",
        "split_counts",
        "cases",
        "derived_records",
        "metadata",
        "content_hash",
        "manifest_hash",
    }
    if set(manifest) != required:
        missing = sorted(required - set(manifest))
        extra = sorted(set(manifest) - required)
        raise CaseValidationError(f"manifest key mismatch; missing={missing}, extra={extra}")
    policy = _require_mapping(manifest["split_policy"], "split_policy")
    if policy.get("method") != SPLIT_METHOD:
        raise CaseValidationError("unsupported split policy method")
    if policy.get("order") != list(SPLIT_NAMES) or policy.get("test_is_locked") is not True:
        raise CaseValidationError("split policy order or locked-test marker changed")
    ratios = _normalise_ratios(_require_mapping(policy.get("ratios"), "split_policy.ratios"))
    supplied_cases = manifest["cases"]
    if not isinstance(supplied_cases, Sequence) or isinstance(
        supplied_cases, (str, bytes, bytearray)
    ):
        raise CaseValidationError("manifest.cases must be a sequence")
    expected_cases = freeze_case_splits(supplied_cases, ratios)
    if canonical_json(supplied_cases) != canonical_json(expected_cases):
        raise CaseValidationError(
            "case records, ordering, split, identity, or content hash changed"
        )
    supplied_children = manifest["derived_records"]
    if not isinstance(supplied_children, Sequence) or isinstance(
        supplied_children, (str, bytes, bytearray)
    ):
        raise CaseValidationError("manifest.derived_records must be a sequence")
    expected_children = inherit_derived_splits(supplied_children, expected_cases)
    if canonical_json(supplied_children) != canonical_json(expected_children):
        raise CaseValidationError("derived records, ordering, inheritance, or content hash changed")
    if canonical_json(manifest["split_counts"]) != canonical_json(_split_summary(expected_cases)):
        raise CaseValidationError("split_counts does not match case records")
    _normalise_json(manifest["metadata"])
    expected_content = compute_manifest_content_hash(manifest)
    if manifest["content_hash"] != expected_content:
        raise CaseValidationError("manifest content_hash mismatch")
    expected_manifest = compute_manifest_hash(manifest)
    if manifest["manifest_hash"] != expected_manifest:
        raise CaseValidationError("manifest_hash mismatch")


def write_case_manifest(
    path: str | Path, manifest: Mapping[str, Any], *, overwrite: bool = False
) -> Path:
    """Verify and atomically write a canonical UTF-8 manifest."""

    verify_case_manifest(manifest)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"manifest already exists: {destination}")
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination


def load_case_manifest(path: str | Path) -> dict[str, Any]:
    """Load and fully verify a manifest from disk."""

    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CaseValidationError(f"could not load case manifest: {exc}") from exc
    if not isinstance(value, dict):
        raise CaseValidationError("case manifest root must be an object")
    verify_case_manifest(value)
    return value


# Short aliases keep call sites readable without weakening the explicit API.
freeze_splits = freeze_case_splits
build_manifest = build_case_manifest
verify_manifest = verify_case_manifest
