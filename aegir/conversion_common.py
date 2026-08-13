from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import tempfile
from typing import Iterator

import numpy as np
from PIL import Image


OPENCV_TO_OPENGL = np.diag([1.0, -1.0, -1.0])


@contextmanager
def atomic_output_dir(
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> Iterator[tuple[Path, Path]]:
    """Yield a staging directory and publish it only after successful conversion."""

    output = Path(output_dir).resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f"output already exists (pass --overwrite to replace it): {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        yield staging, output
        if output.exists():
            shutil.rmtree(output)
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def resolve_suffix(path: Path, suffix: str) -> Path:
    """Resolve metadata paths that intentionally omit their file extension."""

    return path if path.suffix else path.with_suffix(suffix)


def read_exr(path: str | Path, *, rgb: bool) -> np.ndarray:
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required to read float EXR data") from exc
    value = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if value is None:
        raise ValueError(f"OpenCV could not read EXR image: {path}")
    if rgb:
        if value.ndim == 2:
            value = np.repeat(value[..., None], 3, axis=-1)
        if value.ndim != 3 or value.shape[-1] not in (3, 4):
            raise ValueError(f"expected HxWx3/4 EXR at {path}, got {value.shape}")
        value = value[..., :3][..., ::-1]
    elif value.ndim == 3:
        value = value[..., 0]
    return value.astype(np.float32, copy=True)


def tone_map_hdr(rgb: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Apply the official HyperSim 90th-percentile tone-mapping strategy."""

    if rgb.ndim != 3 or rgb.shape[-1] != 3 or valid_mask.shape != rgb.shape[:2]:
        raise ValueError("tone-map RGB and mask shapes do not match")
    valid = valid_mask & np.isfinite(rgb).all(axis=-1)
    if np.any(valid):
        brightness = 0.3 * rgb[..., 0] + 0.59 * rgb[..., 1] + 0.11 * rgb[..., 2]
        percentile = float(np.percentile(brightness[valid], 90.0))
        scale = 0.0 if percentile < 1.0e-4 else 0.8 ** 2.2 / percentile
    else:
        scale = 1.0
    mapped = np.power(np.maximum(scale * np.where(np.isfinite(rgb), rgb, 0.0), 0.0), 1.0 / 2.2)
    mapped[~valid_mask] = 0.0
    return np.rint(np.clip(mapped, 0.0, 1.0) * 255.0).astype(np.uint8)


def encode_normal_rgb(normal: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    if normal.shape != valid_mask.shape + (3,):
        raise ValueError("normal and mask shapes do not match")
    length = np.linalg.norm(normal, axis=-1, keepdims=True)
    normalized = normal / np.maximum(length, 1.0e-12)
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    encoded = np.rint(np.clip(normalized * 0.5 + 0.5, 0.0, 1.0) * 255.0).astype(np.uint8)
    encoded[~valid_mask] = 0
    return encoded


def ray_distance_to_z(
    distance: np.ndarray,
    *,
    fl_x: float,
    fl_y: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    height, width = distance.shape
    yy, xx = np.mgrid[:height, :width]
    norm = np.sqrt(
        ((xx - cx) / fl_x) ** 2 + ((yy - cy) / fl_y) ** 2 + 1.0
    )
    return (distance / norm).astype(np.float32)


def validate_c2w(transform: np.ndarray, *, label: str) -> None:
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError(f"invalid 4x4 c2w matrix for {label}")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-6):
        raise ValueError(f"invalid homogeneous row for {label}")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-5):
        raise ValueError(f"non-orthonormal c2w rotation for {label}")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-5):
        raise ValueError(f"left-handed c2w rotation for {label}")


def save_rgb(path: Path, value: np.ndarray) -> None:
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError(f"expected uint8 RGB for {path}, got {value.dtype} {value.shape}")
    Image.fromarray(value, mode="RGB").save(path)


def save_mask(path: Path, valid: np.ndarray) -> None:
    Image.fromarray((valid.astype(np.uint8) * 255), mode="L").save(path)
