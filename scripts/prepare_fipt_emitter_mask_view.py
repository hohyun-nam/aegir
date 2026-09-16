#!/usr/bin/env python3
"""Attach FIPT segmentation-derived emitter masks to an AEGIR dataset view."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np
from PIL import Image


def ensure_relative_symlink(source: Path, output: Path) -> None:
    if not source.exists():
        return
    expected = source.resolve()
    if output.is_symlink():
        if output.resolve() != expected:
            raise RuntimeError(f"{output} points to the wrong target")
        return
    if output.exists():
        raise RuntimeError(f"refusing to replace existing {output}")
    output.symlink_to(os.path.relpath(source, start=output.parent), target_is_directory=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--segmentation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--segment-id", type=int, action="append", required=True)
    args = parser.parse_args()

    source = args.source.resolve()
    segmentation_dir = args.segmentation_dir.resolve()
    output = args.output.resolve()
    if source == output:
        raise ValueError("--source and --output must differ")
    metadata = json.loads((source / "transforms.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    emitter_dir = output / "emitter_masks"
    emitter_dir.mkdir(exist_ok=True)

    segment_ids = np.asarray(sorted(set(args.segment_id)), dtype=np.int64)
    emitter_pixels = 0
    visible_frames = 0
    for order, frame in enumerate(metadata["frames"]):
        source_index = int(frame.get("source_frame_index", order))
        segmentation_path = segmentation_dir / f"{source_index:03d}.exr"
        segmentation = cv2.imread(str(segmentation_path), cv2.IMREAD_UNCHANGED)
        if segmentation is None:
            raise FileNotFoundError(segmentation_path)
        if segmentation.ndim == 3:
            segmentation = segmentation[..., 0]
        segmentation = np.rint(segmentation).astype(np.int64)
        emitter = np.isin(segmentation, segment_ids)
        mask_name = f"frame_{order:04d}.png"
        Image.fromarray(emitter.astype(np.uint8) * 255, mode="L").save(
            emitter_dir / mask_name
        )
        frame["emitter_mask_path"] = f"emitter_masks/{mask_name}"
        count = int(emitter.sum())
        emitter_pixels += count
        visible_frames += int(count > 0)

    # Resolve paths individually. Do not link whole directories containing GT answers.
    for frame in metadata["frames"]:
        for key in ("file_path", "mask_path", "radiance_hdr_file_path", "radiance_exr_file_path",
                    "depth_file_path", "normal_world_file_path", "normal_camera_file_path",
                    "normal_file_path", "albedo_gt_file_path", "roughness_gt_file_path", "metallic_gt_file_path"):
            if frame.get(key):
                frame[key] = str((source / frame[key]).resolve())
    for key in ("train_filenames", "val_filenames", "test_filenames"):
        if key in metadata:
            metadata[key] = [str((source / value).resolve()) for value in metadata[key]]
    (output / "transforms.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"frames: {len(metadata['frames'])}")
    print(f"emitter-visible frames: {visible_frames}")
    print(f"emitter pixels: {emitter_pixels}")
    print(f"segment ids: {segment_ids.tolist()}")
    print(f"wrote: {output / 'transforms.json'}")


if __name__ == "__main__":
    main()
