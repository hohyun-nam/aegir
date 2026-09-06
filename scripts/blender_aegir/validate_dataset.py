"""Validate an AEGIR Blender capture before COLMAP/training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--min-train-frames", type=int, default=100)
    parser.add_argument("--min-test-frames", type=int, default=20)
    parser.add_argument("--min-emitter-visible-frames", type=int, default=5)
    args = parser.parse_args()

    root = args.data.resolve()
    metadata = json.loads((root / "transforms.json").read_text(encoding="utf-8"))
    frames = metadata.get("frames", [])
    if not frames:
        raise ValueError("dataset has no frames")
    width, height = int(metadata["w"]), int(metadata["h"])

    split_sets = {
        split: set(metadata.get(f"{split}_filenames", []))
        for split in ("train", "val", "test")
    }
    if len(split_sets["train"]) < args.min_train_frames:
        raise ValueError(f"too few training frames: {len(split_sets['train'])}")
    if len(split_sets["test"]) < args.min_test_frames:
        raise ValueError(f"too few test frames: {len(split_sets['test'])}")
    if split_sets["train"] & split_sets["val"] or split_sets["train"] & split_sets["test"]:
        raise ValueError("held-out frames overlap training frames")

    emitters_payload = json.loads((root / "emitters.json").read_text(encoding="utf-8"))
    emitter_pass_indices = {
        int(item["object_pass_index"]): item["name"]
        for item in emitters_payload["emitters"]
    }
    emitter_visible = {name: 0 for name in emitter_pass_indices.values()}
    hdr_maxima = []
    valid_fractions = []

    for index, frame in enumerate(frames):
        expected = (height, width)
        image = np.asarray(Image.open(root / frame["file_path"]).convert("RGB"))
        hdr = np.load(root / frame["radiance_hdr_file_path"], allow_pickle=False)
        depth = np.load(root / frame["depth_file_path"], allow_pickle=False)
        normal = np.load(root / frame["normal_world_file_path"], allow_pickle=False)
        ids = np.load(root / frame["object_id_file_path"], allow_pickle=False)
        mask = np.asarray(Image.open(root / frame["mask_path"]).convert("L")) > 0
        if image.shape != (*expected, 3) or hdr.shape != (*expected, 3):
            raise ValueError(f"RGB/HDR shape mismatch at frame {index}")
        if depth.shape != expected or normal.shape != (*expected, 3) or ids.shape != expected:
            raise ValueError(f"geometry pass shape mismatch at frame {index}")
        if mask.shape != expected or not np.any(mask):
            raise ValueError(f"invalid mask at frame {index}")
        if not np.all(np.isfinite(hdr)) or np.any(hdr < 0.0):
            raise ValueError(f"invalid linear HDR at frame {index}")
        if not np.all(np.isfinite(depth)) or np.any(depth < 0.0):
            raise ValueError(f"invalid metric depth at frame {index}")
        transform = np.asarray(frame["transform_matrix"], dtype=np.float64)
        if transform.shape != (4, 4) or not np.allclose(transform[3], [0, 0, 0, 1]):
            raise ValueError(f"invalid OpenGL c2w at frame {index}")
        hdr_maxima.append(float(hdr.max()))
        valid_fractions.append(float(mask.mean()))
        for pass_index, name in emitter_pass_indices.items():
            if np.any(ids == pass_index):
                emitter_visible[name] += 1

    missing = {
        name: count
        for name, count in emitter_visible.items()
        if count < args.min_emitter_visible_frames
    }
    if missing:
        raise ValueError(
            f"emitters are not visible in enough frames (minimum "
            f"{args.min_emitter_visible_frames}): {missing}"
        )
    summary = {
        "status": "valid",
        "frames": len(frames),
        "train_frames": len(split_sets["train"]),
        "val_frames": len(split_sets["val"]),
        "test_frames": len(split_sets["test"]),
        "resolution": [width, height],
        "hdr_max": max(hdr_maxima),
        "median_valid_fraction": float(np.median(valid_fractions)),
        "emitter_visible_frames": emitter_visible,
    }
    (root / "validation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
