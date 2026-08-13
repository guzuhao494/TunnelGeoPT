"""GeoPT-compatible array schema utilities.

The compatibility layer deliberately covers only the three arrays emitted by
GeoPT's lifted-geometry generator.  Domain-specific, high-fidelity rock
mechanics fields belong in a separate downstream schema rather than being
silently packed into these fixed-width arrays.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

X_WIDTH = 7
CONDITION_WIDTH = 4
SUPERVISE_WIDTH = 9

GEOPT_DTYPE = np.dtype(np.float16)
SUPPORTED_DTYPES = (np.dtype(np.float16), np.dtype(np.float32))

X_FILENAME = "x.npy"
CONDITION_FILENAME = "condition_{trajectory_index}.npy"
SUPERVISE_FILENAME = "supervise_{trajectory_index}.npy"
META_FILENAME = "meta.json"


class SchemaValidationError(ValueError):
    """Raised when arrays or metadata violate the GeoPT compatibility schema."""


@dataclass(frozen=True)
class GeoPTSamplePaths:
    """Resolved paths for one trajectory within a geometry case directory."""

    case_dir: Path
    x: Path
    condition: Path
    supervise: Path
    meta: Path


@dataclass(frozen=True)
class GeoPTSample:
    """One GeoPT-compatible geometry/trajectory sample."""

    x: np.ndarray
    condition: np.ndarray
    supervise: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)
    trajectory_index: int = 0

    @property
    def num_points(self) -> int:
        return int(self.x.shape[0])

    @property
    def dtype(self) -> np.dtype:
        return self.x.dtype

    def validate(
        self,
        *,
        expected_dtype: Any | None = GEOPT_DTYPE,
        require_same_dtype: bool = True,
    ) -> None:
        """Validate this sample in place without modifying its arrays."""

        validate_arrays(
            self.x,
            self.condition,
            self.supervise,
            expected_dtype=expected_dtype,
            require_same_dtype=require_same_dtype,
        )
        validate_meta(self.meta, num_points=self.num_points, dtype=self.dtype)


def _validate_trajectory_index(trajectory_index: int) -> int:
    if isinstance(trajectory_index, bool) or not isinstance(trajectory_index, int):
        raise TypeError(
            f"trajectory_index must be a non-negative integer; got {trajectory_index!r}."
        )
    if trajectory_index < 0:
        raise ValueError(
            f"trajectory_index must be a non-negative integer; got {trajectory_index}."
        )
    return trajectory_index


def sample_paths(case_dir: str | os.PathLike[str], trajectory_index: int = 0) -> GeoPTSamplePaths:
    """Return the official-compatible filenames for one trajectory."""

    index = _validate_trajectory_index(trajectory_index)
    root = Path(case_dir)
    return GeoPTSamplePaths(
        case_dir=root,
        x=root / X_FILENAME,
        condition=root / CONDITION_FILENAME.format(trajectory_index=index),
        supervise=root / SUPERVISE_FILENAME.format(trajectory_index=index),
        meta=root / META_FILENAME,
    )


def _require_array(name: str, value: Any, width: int) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise SchemaValidationError(f"{name} must be a numpy.ndarray; got {type(value).__name__}.")
    if value.ndim != 2:
        raise SchemaValidationError(
            f"{name} must be a rank-2 array with shape [N,{width}]; got shape {value.shape}."
        )
    if value.shape[1] != width:
        raise SchemaValidationError(f"{name} must have shape [N,{width}]; got shape {value.shape}.")
    if value.shape[0] <= 0:
        raise SchemaValidationError(
            f"{name} must contain at least one point; got shape {value.shape}."
        )
    if value.dtype not in SUPPORTED_DTYPES:
        supported = ", ".join(dtype.name for dtype in SUPPORTED_DTYPES)
        raise SchemaValidationError(
            f"{name} must use a supported floating dtype ({supported}); got {value.dtype}."
        )
    if not np.isfinite(value).all():
        invalid_count = int(value.size - np.count_nonzero(np.isfinite(value)))
        raise SchemaValidationError(
            f"{name} contains {invalid_count} non-finite value(s); "
            "NaN and infinity are not allowed."
        )
    return value


def validate_arrays(
    x: np.ndarray,
    condition: np.ndarray,
    supervise: np.ndarray,
    *,
    expected_dtype: Any | None = GEOPT_DTYPE,
    require_same_dtype: bool = True,
) -> int:
    """Validate the three official-compatible arrays and return ``N``.

    Args:
        x: Geometry features with shape ``[N, 7]``.
        condition: Point-wise lifted dynamics with shape ``[N, 4]``.
        supervise: Three vector-distance targets with shape ``[N, 9]``.
        expected_dtype: Exact required dtype.  The compatibility default is
            ``numpy.float16``.  Pass ``numpy.float32`` for an explicitly
            selected higher-precision variant, or ``None`` to accept either
            supported dtype.
        require_same_dtype: Require all three arrays to have one dtype.  This
            should normally remain enabled.
    """

    arrays = {
        "x": _require_array("x", x, X_WIDTH),
        "condition": _require_array("condition", condition, CONDITION_WIDTH),
        "supervise": _require_array("supervise", supervise, SUPERVISE_WIDTH),
    }

    point_counts = {name: value.shape[0] for name, value in arrays.items()}
    if len(set(point_counts.values())) != 1:
        details = ", ".join(f"{name}={count}" for name, count in point_counts.items())
        raise SchemaValidationError(
            f"Point-count mismatch across GeoPT arrays; expected a shared N, got {details}."
        )

    if require_same_dtype:
        dtypes = {value.dtype for value in arrays.values()}
        if len(dtypes) != 1:
            details = ", ".join(f"{name}={value.dtype.name}" for name, value in arrays.items())
            raise SchemaValidationError(
                f"Dtype mismatch across GeoPT arrays; expected one shared dtype, got {details}."
            )

    if expected_dtype is not None:
        required = np.dtype(expected_dtype)
        if required not in SUPPORTED_DTYPES:
            supported = ", ".join(dtype.name for dtype in SUPPORTED_DTYPES)
            raise ValueError(f"expected_dtype must be one of {supported} or None; got {required}.")
        mismatches = [
            f"{name}={value.dtype.name}"
            for name, value in arrays.items()
            if value.dtype != required
        ]
        if mismatches:
            raise SchemaValidationError(
                f"GeoPT compatibility requires dtype {required.name}; got "
                + ", ".join(mismatches)
                + "."
            )

    return int(x.shape[0])


def validate_meta(
    meta: Mapping[str, Any] | None,
    *,
    num_points: int | None = None,
    dtype: Any | None = None,
) -> dict[str, Any]:
    """Validate and return a JSON-safe copy of optional case metadata."""

    if meta is None:
        return {}
    if not isinstance(meta, Mapping):
        raise SchemaValidationError(
            f"meta must be a mapping suitable for meta.json; got {type(meta).__name__}."
        )
    if any(not isinstance(key, str) for key in meta):
        raise SchemaValidationError("meta.json keys must all be strings.")

    try:
        encoded = json.dumps(dict(meta), ensure_ascii=False, allow_nan=False, sort_keys=True)
        normalized = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise SchemaValidationError(
            f"meta must contain only finite, JSON-serializable values: {exc}"
        ) from exc

    if "num_points" in normalized:
        declared = normalized["num_points"]
        if isinstance(declared, bool) or not isinstance(declared, int) or declared <= 0:
            raise SchemaValidationError(
                "meta['num_points'] must be a positive integer when present."
            )
        if num_points is not None and declared != num_points:
            raise SchemaValidationError(
                "meta['num_points'] does not match the arrays: "
                f"meta={declared}, arrays={num_points}."
            )

    if "dtype" in normalized:
        declared_dtype = normalized["dtype"]
        if not isinstance(declared_dtype, str):
            raise SchemaValidationError(
                "meta['dtype'] must be a NumPy dtype name string when present."
            )
        try:
            parsed_dtype = np.dtype(declared_dtype)
        except TypeError as exc:
            raise SchemaValidationError(
                f"meta['dtype'] is not a valid NumPy dtype: {declared_dtype!r}."
            ) from exc
        if dtype is not None and parsed_dtype != np.dtype(dtype):
            raise SchemaValidationError(
                "meta['dtype'] does not match the arrays: "
                f"meta={parsed_dtype.name}, arrays={np.dtype(dtype).name}."
            )

    return normalized


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            temp_path = Path(stream.name)
            np.save(stream, array, allow_pickle=False)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _atomic_save_json(path: Path, meta: Mapping[str, Any]) -> None:
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as stream:
            temp_path = Path(stream.name)
            json.dump(
                dict(meta),
                stream,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _load_npy(path: Path, *, mmap_mode: str | None = None) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required GeoPT array: {path}")
    try:
        return np.load(path, allow_pickle=False, mmap_mode=mmap_mode)
    except (OSError, ValueError) as exc:
        raise SchemaValidationError(f"Could not load NumPy array {path}: {exc}") from exc


def _load_meta(path: Path, *, required: bool) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing required GeoPT metadata: {path}")
        return {}
    if not path.is_file():
        raise SchemaValidationError(f"GeoPT metadata path is not a file: {path}")
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SchemaValidationError(f"Could not load metadata {path}: {exc}") from exc
    return validate_meta(value)


def save_sample(
    case_dir: str | os.PathLike[str],
    x: np.ndarray,
    condition: np.ndarray,
    supervise: np.ndarray,
    *,
    trajectory_index: int = 0,
    meta: Mapping[str, Any] | None = None,
    dtype: Any = GEOPT_DTYPE,
    overwrite: bool = False,
) -> GeoPTSamplePaths:
    """Validate and atomically save one GeoPT-compatible trajectory.

    ``x.npy`` and ``meta.json`` are shared by all trajectories in a case.  A
    later call may append a new trajectory when its ``x`` and metadata exactly
    match the existing shared files.  Existing trajectory files are protected
    unless ``overwrite=True``.
    """

    target_dtype = np.dtype(dtype)
    if target_dtype not in SUPPORTED_DTYPES:
        supported = ", ".join(item.name for item in SUPPORTED_DTYPES)
        raise ValueError(f"dtype must be one of {supported}; got {target_dtype}.")

    arrays = []
    for name, value in (
        ("x", x),
        ("condition", condition),
        ("supervise", supervise),
    ):
        try:
            converted = np.asarray(value, dtype=target_dtype, order="C")
        except (TypeError, ValueError, OverflowError) as exc:
            raise SchemaValidationError(
                f"{name} cannot be converted to {target_dtype.name}: {exc}"
            ) from exc
        arrays.append(converted)
    x_array, condition_array, supervise_array = arrays

    num_points = validate_arrays(
        x_array,
        condition_array,
        supervise_array,
        expected_dtype=target_dtype,
    )
    normalized_meta = validate_meta(meta, num_points=num_points, dtype=target_dtype)
    paths = sample_paths(case_dir, trajectory_index)
    paths.case_dir.mkdir(parents=True, exist_ok=True)

    write_x = True
    if paths.x.exists() and not overwrite:
        existing_x = _load_npy(paths.x)
        if existing_x.dtype != x_array.dtype or not np.array_equal(existing_x, x_array):
            raise FileExistsError(
                f"Shared geometry file already exists with different content: {paths.x}"
            )
        write_x = False

    for target in (paths.condition, paths.supervise):
        if target.exists() and not overwrite:
            raise FileExistsError(
                f"Trajectory file already exists: {target}. Pass overwrite=True to replace it."
            )

    write_meta = meta is not None
    if meta is not None and paths.meta.exists() and not overwrite:
        existing_meta = _load_meta(paths.meta, required=True)
        if existing_meta != normalized_meta:
            raise FileExistsError(
                f"Shared metadata already exists with different content: {paths.meta}"
            )
        write_meta = False

    # All predictable conflicts have been checked before the first write.
    if write_x:
        _atomic_save_npy(paths.x, x_array)
    _atomic_save_npy(paths.condition, condition_array)
    _atomic_save_npy(paths.supervise, supervise_array)
    if write_meta:
        _atomic_save_json(paths.meta, normalized_meta)

    return paths


def load_sample(
    case_dir: str | os.PathLike[str],
    *,
    trajectory_index: int = 0,
    expected_dtype: Any | None = GEOPT_DTYPE,
    require_same_dtype: bool = True,
    require_meta: bool = False,
    mmap_mode: str | None = None,
) -> GeoPTSample:
    """Load and strictly validate one GeoPT-compatible trajectory."""

    paths = sample_paths(case_dir, trajectory_index)
    x = _load_npy(paths.x, mmap_mode=mmap_mode)
    condition = _load_npy(paths.condition, mmap_mode=mmap_mode)
    supervise = _load_npy(paths.supervise, mmap_mode=mmap_mode)
    num_points = validate_arrays(
        x,
        condition,
        supervise,
        expected_dtype=expected_dtype,
        require_same_dtype=require_same_dtype,
    )
    meta = _load_meta(paths.meta, required=require_meta)
    meta = validate_meta(meta, num_points=num_points, dtype=x.dtype)
    return GeoPTSample(
        x=x,
        condition=condition,
        supervise=supervise,
        meta=meta,
        trajectory_index=trajectory_index,
    )


__all__ = [
    "CONDITION_WIDTH",
    "GEOPT_DTYPE",
    "SUPERVISE_WIDTH",
    "SUPPORTED_DTYPES",
    "X_WIDTH",
    "GeoPTSample",
    "GeoPTSamplePaths",
    "SchemaValidationError",
    "load_sample",
    "sample_paths",
    "save_sample",
    "validate_arrays",
    "validate_meta",
]
