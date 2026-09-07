#!/usr/bin/env python3
"""Correct legacy Blender captures whose fy ignored the render aspect ratio.

The source capture is never modified.  Immutable rendered products are linked
into a new dataset, while z-depth is materialized again because the original
ray-distance-to-z conversion used the incorrect focal length.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np


TRANSFORM_NAMES = (
    "transforms.json",
    "transforms_train.json",
    "transforms_val.json",
    "transforms_test.json",
)
RENDER_DIRECTORIES = (
    "images",
    "radiance_hdr",
    "normals_world",
    "albedo",
    "roughness",
    "metallic",
    "masks",
    "emitter_masks",
    "object_ids",
)


def frame_intrinsics(frame: dict, metadata: dict) -> tuple[float, float, float, float, int, int]:
    return (
        float(frame.get("fl_x", metadata["fl_x"])),
        float(frame.get("fl_y", metadata["fl_y"])),
        float(frame.get("cx", metadata["cx"])),
        float(frame.get("cy", metadata["cy"])),
        int(frame.get("w", metadata["w"])),
        int(frame.get("h", metadata["h"])),
    )


def corrected_focal(fx: float, fy: float, focal_axis: str) -> float:
    return fx if focal_axis == "x" else fy


def correct_z_depth(
    depth_z: np.ndarray,
    old_fx: float,
    old_fy: float,
    new_fx: float,
    new_fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """Reconstruct ray distance with old K, then convert it using corrected K."""
    if depth_z.ndim != 2:
        raise ValueError(f"expected HxW z-depth, got {depth_z.shape}")
    height, width = depth_z.shape
    xs_old = (np.arange(width, dtype=np.float64) + 0.5 - cx) / old_fx
    ys_old = (np.arange(height, dtype=np.float64) + 0.5 - cy) / old_fy
    xs_new = (np.arange(width, dtype=np.float64) + 0.5 - cx) / new_fx
    ys_new = (np.arange(height, dtype=np.float64) + 0.5 - cy) / new_fy
    old_factor = np.sqrt(1.0 + ys_old[:, None] ** 2 + xs_old[None, :] ** 2)
    new_factor = np.sqrt(1.0 + ys_new[:, None] ** 2 + xs_new[None, :] ** 2)
    result = depth_z.astype(np.float64, copy=False) * old_factor / new_factor
    result[~np.isfinite(depth_z) | (depth_z <= 0.0)] = 0.0
    return result.astype(np.float32)


def update_metadata(metadata: dict, focal_axis: str, source: Path) -> dict:
    result = json.loads(json.dumps(metadata))
    frames = result.get("frames", [])
    for frame in frames:
        fx, fy, cx, cy, width, height = frame_intrinsics(frame, result)
        focal = corrected_focal(fx, fy, focal_axis)
        frame.update({"fl_x": focal, "fl_y": focal, "cx": cx, "cy": cy, "w": width, "h": height})
    root_focal = corrected_focal(float(result["fl_x"]), float(result["fl_y"]), focal_axis)
    result["fl_x"] = root_focal
    result["fl_y"] = root_focal
    result["aegir_intrinsics_correction"] = {
        "format": "aegir-intrinsics-correction-v1",
        "source_dataset": str(source),
        "method": "preserve Blender render-fit focal axis; enforce square-pixel fx=fy",
        "preserved_focal_axis": focal_axis,
        "depth_correction": "recover ray distance with legacy K, then reconvert to z-depth with corrected K",
    }
    # This is generated only after fixed-pose triangulation has completed.
    result.pop("ply_file_path", None)
    return result


def correct_dataset(source: Path, output: Path, focal_axis: str = "x") -> dict:
    source = source.resolve()
    output = output.resolve()
    if not (source / "transforms.json").is_file():
        raise FileNotFoundError(source / "transforms.json")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        main_metadata = json.loads((source / "transforms.json").read_text(encoding="utf-8"))
        frames = main_metadata.get("frames", [])
        if not frames:
            raise ValueError("transforms.json has no frames")

        for directory in RENDER_DIRECTORIES:
            source_directory = source / directory
            if source_directory.exists():
                relative_target = os.path.relpath(source_directory, stage)
                (stage / directory).symlink_to(relative_target, target_is_directory=True)

        (stage / "depths").mkdir()
        ratios: list[float] = []
        old_intrinsics: list[tuple[float, float]] = []
        new_intrinsics: list[tuple[float, float]] = []
        seen_depths: set[str] = set()
        for frame in frames:
            old_fx, old_fy, cx, cy, width, height = frame_intrinsics(frame, main_metadata)
            focal = corrected_focal(old_fx, old_fy, focal_axis)
            depth_relative = frame["depth_file_path"]
            if depth_relative in seen_depths:
                continue
            seen_depths.add(depth_relative)
            depth = np.load(source / depth_relative, allow_pickle=False)
            if depth.shape != (height, width):
                raise ValueError(
                    f"{depth_relative}: metadata is {width}x{height}, depth is {depth.shape}"
                )
            corrected = correct_z_depth(depth, old_fx, old_fy, focal, focal, cx, cy)
            destination = stage / depth_relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.save(destination, corrected, allow_pickle=False)
            valid = depth > 0.0
            ratios.extend((corrected[valid] / depth[valid]).astype(np.float64).tolist())
            old_intrinsics.append((old_fx, old_fy))
            new_intrinsics.append((focal, focal))

        for name in TRANSFORM_NAMES:
            path = source / name
            if path.is_file():
                metadata = json.loads(path.read_text(encoding="utf-8"))
                corrected_metadata = update_metadata(metadata, focal_axis, source)
                (stage / name).write_text(
                    json.dumps(corrected_metadata, indent=2) + "\n", encoding="utf-8"
                )

        if (source / "emitters.json").is_file():
            shutil.copy2(source / "emitters.json", stage / "emitters.json")

        ratio_array = np.asarray(ratios, dtype=np.float64)
        report = {
            "status": "complete",
            "source_dataset": str(source),
            "output_dataset": str(output),
            "frames": len(frames),
            "corrected_depth_files": len(seen_depths),
            "preserved_focal_axis": focal_axis,
            "old_intrinsics_unique": sorted({tuple(round(v, 9) for v in pair) for pair in old_intrinsics}),
            "new_intrinsics_unique": sorted({tuple(round(v, 9) for v in pair) for pair in new_intrinsics}),
            "valid_depth_new_over_old": {
                "min": float(ratio_array.min()),
                "median": float(np.median(ratio_array)),
                "max": float(ratio_array.max()),
            },
            "excluded_derived_products": ["colmap", "mono_depth", "mono_depth_zoe_raw", "sfm_depths"],
        }
        (stage / "intrinsics_correction.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        stage.replace(output)
        return report
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--preserve-focal-axis",
        choices=("x", "y"),
        default="x",
        help="legacy focal value known to match Blender's fitted render frustum (default: x)",
    )
    args = parser.parse_args()
    report = correct_dataset(args.source, args.output, args.preserve_focal_axis)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
