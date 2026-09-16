#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from PIL import Image

from aegir.data import select_frame_indices
from aegir.provenance import (
    artifact_request,
    dataset_signature,
    publish_artifact,
    validate_predicted_depth,
    verify_cache,
)


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_depth(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        depth = np.load(path, allow_pickle=False)
    else:
        depth = np.asarray(Image.open(path), dtype=np.float32)
    return np.squeeze(depth).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Back-project certified RGB-model metre Z-depth and fixed poses into a seed PLY."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--update-transforms",
        type=Path,
        default=None,
        help="Set ply_file_path in this transforms.json after writing the PLY.",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--pixel-stride", type=int, default=4)
    parser.add_argument("--voxel-size", type=float, default=0.02)
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=15.0)
    parser.add_argument("--statistical-nb-neighbors", type=int, default=20)
    parser.add_argument("--statistical-std-ratio", type=float, default=2.0)
    parser.add_argument(
        "--exclude-emitter-masks",
        action="store_true",
        help="Remove emitter-mask pixels before back-projection.",
    )
    parser.add_argument(
        "--emitter-mask-dilation",
        type=int,
        default=2,
        help="Emitter exclusion margin in source-image pixels (default: 2).",
    )
    parser.add_argument("--allow-gt-depth", action="store_true", help="Explicit diagnostic opt-in for uncertified/GT depth")
    args = parser.parse_args()

    data = args.data.resolve()
    output = args.output.resolve()
    if args.pixel_stride <= 0:
        raise ValueError("--pixel-stride must be positive")
    if args.voxel_size <= 0:
        raise ValueError("--voxel-size must be positive")
    if not (0 <= args.min_depth < args.max_depth):
        raise ValueError("depth limits must satisfy 0 <= min < max")
    if args.emitter_mask_dilation < 0:
        raise ValueError("--emitter-mask-dilation must be non-negative")

    transforms_path = data / "transforms.json"
    with transforms_path.open("r", encoding="utf-8") as stream:
        meta = json.load(stream)

    if not args.allow_gt_depth:
        validate_predicted_depth(data, meta)
    request = artifact_request("depth_seed", {key: value for key, value in vars(args).items() if key not in {"data", "output", "update_transforms"}}, {"source": dataset_signature(data)})
    manifest_path = output.with_suffix(output.suffix + ".provenance.json")
    if verify_cache(manifest_path, request, output):
        print(f"Verified seed cache: {output}")
        return
    selected = set(select_frame_indices(meta, meta["frames"], args.split or "all"))

    points_chunks: list[np.ndarray] = []
    colors_chunks: list[np.ndarray] = []
    used_frames = 0
    sampled_pixels = 0

    for index, frame in enumerate(meta["frames"]):
        if index not in selected:
            continue

        image_path = resolve(data, frame["file_path"])
        stem = image_path.stem
        depth_path = resolve(
            data, frame.get("depth_file_path", f"mono_depth/{stem}.npy")
        )
        if not image_path.is_file() or not depth_path.is_file():
            raise FileNotFoundError(f"missing RGB/depth pair: {image_path}, {depth_path}")

        rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        depth = load_depth(depth_path)
        height = int(frame.get("h", meta.get("h", rgb.shape[0])))
        width = int(frame.get("w", meta.get("w", rgb.shape[1])))
        if rgb.shape[:2] != (height, width):
            rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
        if depth.shape != (height, width):
            depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)

        mask_path_value = frame.get("mask_path")
        if mask_path_value is None:
            mask = np.ones((height, width), dtype=bool)
        else:
            mask_image = np.asarray(
                Image.open(resolve(data, mask_path_value)).convert("L"), dtype=np.uint8
            )
            if mask_image.shape != (height, width):
                mask_image = cv2.resize(
                    mask_image, (width, height), interpolation=cv2.INTER_NEAREST
                )
            mask = mask_image > 0

        if args.exclude_emitter_masks:
            emitter_path_value = frame.get("emitter_mask_path")
            if emitter_path_value is None:
                raise KeyError(f"frame {frame['file_path']} has no emitter_mask_path")
            emitter_mask = np.asarray(
                Image.open(resolve(data, emitter_path_value)).convert("L"), dtype=np.uint8
            )
            if emitter_mask.shape != (height, width):
                emitter_mask = cv2.resize(
                    emitter_mask, (width, height), interpolation=cv2.INTER_NEAREST
                )
            emitter_mask = emitter_mask > 0
            if args.emitter_mask_dilation > 0:
                kernel_size = 2 * args.emitter_mask_dilation + 1
                kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
                emitter_mask = cv2.dilate(
                    emitter_mask.astype(np.uint8), kernel, iterations=1
                ).astype(bool)
            mask &= ~emitter_mask

        rows = np.arange(0, height, args.pixel_stride, dtype=np.int32)
        cols = np.arange(0, width, args.pixel_stride, dtype=np.int32)
        uu, vv = np.meshgrid(cols, rows)
        z = depth[vv, uu]
        valid = (
            np.isfinite(z)
            & (z >= args.min_depth)
            & (z <= args.max_depth)
            & mask[vv, uu]
        )
        if not np.any(valid):
            continue

        u = uu[valid].astype(np.float64)
        v = vv[valid].astype(np.float64)
        z = z[valid].astype(np.float64)
        fx = float(frame.get("fl_x", meta["fl_x"]))
        fy = float(frame.get("fl_y", meta.get("fl_y", fx)))
        cx = float(frame.get("cx", meta.get("cx", width * 0.5)))
        cy = float(frame.get("cy", meta.get("cy", height * 0.5)))

        # Stored depth is OpenCV camera Z-depth, while transforms.json uses an
        # OpenGL camera-to-world pose (right, up, backward). Convert camera axes
        # before applying c2w.
        camera_gl = np.column_stack(
            ((u - cx) * z / fx, -(v - cy) * z / fy, -z)
        )
        c2w = np.asarray(frame["transform_matrix"], dtype=np.float64)
        if c2w.shape == (3, 4):
            c2w = np.vstack((c2w, [0.0, 0.0, 0.0, 1.0]))
        world = camera_gl @ c2w[:3, :3].T + c2w[:3, 3]

        points_chunks.append(world)
        colors_chunks.append(rgb[vv[valid], uu[valid]].astype(np.float64) / 255.0)
        used_frames += 1
        sampled_pixels += len(world)

    if not points_chunks:
        raise RuntimeError(f"no valid {args.split!r} depth samples found in {data}")

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.concatenate(points_chunks, axis=0))
    cloud.colors = o3d.utility.Vector3dVector(np.concatenate(colors_chunks, axis=0))
    before_voxel = len(cloud.points)
    cloud = cloud.voxel_down_sample(args.voxel_size)
    after_voxel = len(cloud.points)

    if args.statistical_nb_neighbors > 0 and len(cloud.points) > args.statistical_nb_neighbors:
        cloud, _ = cloud.remove_statistical_outlier(
            nb_neighbors=args.statistical_nb_neighbors,
            std_ratio=args.statistical_std_ratio,
        )
    final_count = len(cloud.points)
    if final_count == 0:
        raise RuntimeError("outlier filtering removed every point")

    output.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(output), cloud, write_ascii=False, compressed=False):
        raise RuntimeError(f"failed to write {output}")

    publish_artifact(manifest_path, request, output)
    xyz = np.asarray(cloud.points)
    print(f"frames: {used_frames}")
    print(f"sampled valid pixels: {sampled_pixels}")
    print(f"points: {before_voxel} -> voxel {after_voxel} -> filtered {final_count}")
    print(f"bounds min: {xyz.min(axis=0).tolist()}")
    print(f"bounds max: {xyz.max(axis=0).tolist()}")
    print(f"wrote: {output}")

    if args.update_transforms is not None:
        publish_path = args.update_transforms.resolve()
        with publish_path.open("r", encoding="utf-8") as stream:
            publish_meta = json.load(stream)
        publish_meta["ply_file_path"] = Path(
            os.path.relpath(output, start=publish_path.parent)
        ).as_posix()
        temporary = publish_path.with_suffix(publish_path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(publish_meta, stream, indent=2)
            stream.write("\n")
        temporary.replace(publish_path)
        print(f"updated: {publish_path} -> {publish_meta['ply_file_path']}")


if __name__ == "__main__":
    main()
