#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

TOP_LEVEL_SUPERVISION_KEYS = {
    "aegir_depth_convention",
    "aegir_depth_mode",
    "depth_unit_scale_factor",
    "normal_camera_convention",
    "normal_encoding",
    "normal_space",
    "source_depth_mode",
    "source_normal_space",
}

FRAME_SUPERVISION_KEYS = {
    "albedo_gt_file_path",
    "depth_file_path",
    "normal_camera_file_path",
    "normal_file_path",
    "radiance_hdr_file_path",
    "roughness_gt_file_path",
    "emitter_mask_path",
}

EMITTER_EXEMPT_VALUE = 64


def encode_training_mask(
    valid: np.ndarray,
    emitter: np.ndarray,
    *,
    encode_emitter_exemption: bool,
) -> np.ndarray:
    """Encode valid, invalid, and loss-exempt emitter pixels in one PNG mask."""
    encoded = np.zeros(valid.shape, dtype=np.uint8)
    encoded[valid & ~emitter] = 255
    if encode_emitter_exemption:
        encoded[valid & emitter] = EMITTER_EXEMPT_VALUE
    return encoded


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def ensure_relative_symlink(source_item: Path, output_dir: Path, name: str) -> None:
    if not source_item.exists():
        return
    destination = output_dir / name
    expected = source_item.resolve()
    if destination.is_symlink():
        if destination.resolve() != expected:
            raise RuntimeError(f"{destination} points to the wrong target")
        return
    if destination.exists():
        raise RuntimeError(f"refusing to replace existing {destination}")
    relative_target = os.path.relpath(source_item, start=output_dir)
    destination.symlink_to(relative_target, target_is_directory=source_item.is_dir())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a Nerfstudio view containing RGB, fixed poses, masks, and COLMAP seed only."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mono-depth-source",
        type=Path,
        default=None,
        help="Optional RGB-derived mono-depth directory to expose as output/mono_depth.",
    )
    parser.add_argument(
        "--exclude-emitter-masks",
        action="store_true",
        help="Write valid masks with emitter pixels removed for native 2DGS training.",
    )
    parser.add_argument(
        "--emitter-mask-dilation",
        type=int,
        default=2,
        help="Emitter exclusion margin in source-image pixels (default: 2).",
    )
    parser.add_argument(
        "--encode-emitter-exemption",
        action="store_true",
        help=(
            "Encode valid emitter pixels as gray (64): excluded from RGB/depth "
            "supervision but exempt from invalid-opacity removal."
        ),
    )
    parser.add_argument(
        "--point-cloud-source",
        type=Path,
        default=None,
        help="Optional PLY seed to expose as output/point_cloud.ply instead of source/colmap.",
    )
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    if source == output:
        raise ValueError("--source and --output must be different")
    if args.emitter_mask_dilation < 0:
        raise ValueError("--emitter-mask-dilation must be non-negative")
    if args.encode_emitter_exemption and not args.exclude_emitter_masks:
        raise ValueError(
            "--encode-emitter-exemption requires --exclude-emitter-masks"
        )
    transforms_path = source / "transforms.json"
    if not transforms_path.is_file():
        raise FileNotFoundError(transforms_path)
    point_cloud_source = (
        args.point_cloud_source.resolve()
        if args.point_cloud_source is not None
        else None
    )
    if point_cloud_source is None and not (source / "colmap" / "sparse_pc.ply").is_file():
        raise FileNotFoundError(source / "colmap" / "sparse_pc.ply")
    if point_cloud_source is not None and not point_cloud_source.is_file():
        raise FileNotFoundError(point_cloud_source)

    with transforms_path.open("r", encoding="utf-8") as stream:
        transforms = json.load(stream)
    for key in TOP_LEVEL_SUPERVISION_KEYS:
        transforms.pop(key, None)
    source_frames = {
        frame["file_path"]: deepcopy(frame) for frame in transforms["frames"]
    }
    for frame in transforms["frames"]:
        for key in FRAME_SUPERVISION_KEYS:
            frame.pop(key, None)

    # Removing radiance turns this into an LDR-only dataset. Preserve an exact
    # generated-camera mapping only when every frame carries its exposure;
    # otherwise make the unknown-camera fallback explicit.
    source_color = transforms.get("aegir_color", {})
    source_target = source_color.get("target_png", {})
    has_frame_exposure = all("ldr_exposure_scale" in frame for frame in transforms["frames"])
    if has_frame_exposure and all(key in source_target for key in ("tone_mapper", "transfer_function")):
        target_png = {
            "tone_mapper": source_target["tone_mapper"],
            "transfer_function": source_target["transfer_function"],
            "exposure_policy": "fixed_per_frame",
            "exposure_scale": 1.0,
            "source": source_target.get("source", "preserved converter metadata"),
        }
    else:
        target_png = {
            "tone_mapper": "reinhard",
            "transfer_function": "srgb",
            "exposure_policy": "learned_scene",
            "exposure_scale": 1.0,
            "source": "explicit fallback: source PNG camera response is unavailable",
        }
    transforms["aegir_color"] = {
        "format": "aegir-color-v1",
        "observation_domain": "ldr",
        "target_png": target_png,
    }
    if point_cloud_source is not None:
        transforms["ply_file_path"] = "point_cloud.ply"

    forbidden = TOP_LEVEL_SUPERVISION_KEYS.intersection(transforms)
    forbidden.update(
        key for frame in transforms["frames"] for key in FRAME_SUPERVISION_KEYS.intersection(frame)
    )
    if forbidden:
        raise RuntimeError(f"supervision metadata survived sanitization: {sorted(forbidden)}")

    output.mkdir(parents=True, exist_ok=True)
    ensure_relative_symlink(source / "images", output, "images")
    if args.exclude_emitter_masks:
        masks_output = output / "masks"
        if masks_output.is_symlink():
            raise RuntimeError(
                f"refusing to write exclusion masks through existing symlink {masks_output}"
            )
        masks_output.mkdir(parents=True, exist_ok=True)
        kernel_size = 2 * args.emitter_mask_dilation + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        excluded_pixels = 0
        for index, frame in enumerate(transforms["frames"]):
            source_frame = source_frames[frame["file_path"]]
            image_path = source / source_frame["file_path"]
            with Image.open(image_path) as image:
                width, height = image.size
            valid_path_value = source_frame.get("mask_path")
            if valid_path_value is None:
                valid = np.ones((height, width), dtype=bool)
            else:
                valid = np.asarray(
                    Image.open(resolve(source, valid_path_value)).convert("L"),
                    dtype=np.uint8,
                )
                if valid.shape != (height, width):
                    valid = cv2.resize(valid, (width, height), interpolation=cv2.INTER_NEAREST)
                valid = valid > 0
            emitter_path_value = source_frame.get("emitter_mask_path")
            if emitter_path_value is None:
                raise KeyError(f"frame {frame['file_path']} has no emitter_mask_path")
            emitter = np.asarray(
                Image.open(resolve(source, emitter_path_value)).convert("L"),
                dtype=np.uint8,
            )
            if emitter.shape != (height, width):
                emitter = cv2.resize(emitter, (width, height), interpolation=cv2.INTER_NEAREST)
            emitter = emitter > 0
            if args.emitter_mask_dilation > 0:
                emitter = cv2.dilate(emitter.astype(np.uint8), kernel, iterations=1).astype(bool)
            excluded_pixels += int((valid & emitter).sum())
            mask_name = f"{index:06d}_{Path(frame['file_path']).stem}.png"
            encoded_mask = encode_training_mask(
                valid,
                emitter,
                encode_emitter_exemption=args.encode_emitter_exemption,
            )
            Image.fromarray(encoded_mask, mode="L").save(masks_output / mask_name)
            frame["mask_path"] = f"masks/{mask_name}"
        print(f"emitter-excluded valid pixels: {excluded_pixels}")
        if args.encode_emitter_exemption:
            print(
                "mask classes: invalid=0, emitter-exempt=64, "
                "valid-supervision=255"
            )
    else:
        ensure_relative_symlink(source / "masks", output, "masks")
    if point_cloud_source is None:
        ensure_relative_symlink(source / "colmap", output, "colmap")
    else:
        ensure_relative_symlink(point_cloud_source, output, "point_cloud.ply")
    if args.mono_depth_source is not None:
        mono_depth_source = args.mono_depth_source.resolve()
        ensure_relative_symlink(mono_depth_source, output, "mono_depth")
    with (output / "transforms.json").open("w", encoding="utf-8") as stream:
        json.dump(transforms, stream, indent=2)
        stream.write("\n")
    print(f"frames: {len(transforms['frames'])}")
    print(f"wrote: {output / 'transforms.json'}")


if __name__ == "__main__":
    main()
