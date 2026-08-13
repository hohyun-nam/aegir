from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
from PIL import Image

from aegir.conversion_common import (
    atomic_output_dir,
    encode_normal_rgb,
    ray_distance_to_z,
    read_exr,
    save_mask,
    save_rgb,
    tone_map_hdr,
    validate_c2w,
)

# FIPT's official loader traces camera rays as (-x, -y, +z): the numeric
# transform matrices therefore use left/up/forward camera axes. Nerfstudio
# stores camera-to-world matrices with right/up/back (OpenGL) camera axes.
# Both changes are a proper 180-degree rotation about camera Y.
FIPT_TO_NERFSTUDIO_OPENGL = np.diag([-1.0, 1.0, -1.0, 1.0])


def fipt_pose_to_nerfstudio(source_c2w: np.ndarray) -> np.ndarray:
    """Convert FIPT left/up/forward c2w to Nerfstudio OpenGL c2w."""

    source_c2w = np.asarray(source_c2w, dtype=np.float64)
    validate_c2w(source_c2w, label="FIPT source pose")
    converted = source_c2w @ FIPT_TO_NERFSTUDIO_OPENGL
    validate_c2w(converted, label="FIPT Nerfstudio pose")
    return converted


def fipt_world_normal_to_nerfstudio_camera(
    normal_world: np.ndarray,
    nerfstudio_c2w: np.ndarray,
) -> np.ndarray:
    """Rotate Blender's world-space Normal pass into OpenGL camera space."""

    if normal_world.ndim != 3 or normal_world.shape[-1] != 3:
        raise ValueError(f"expected HxWx3 FIPT world normals, got {normal_world.shape}")
    rotation_c2w = np.asarray(nerfstudio_c2w, dtype=np.float64)[:3, :3]
    return np.einsum("...i,ij->...j", normal_world, rotation_c2w).astype(
        np.float32, copy=False
    )


def _load_split(scene_dir: Path, split: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = scene_dir / split / "transforms.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    frames = metadata.get("frames", [])
    if not frames:
        raise ValueError(f"no frames in {path}")
    return metadata, frames


def _material_png(value: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if value.ndim == 2:
        value = value[..., None]
    if value.shape[-1] == 1:
        value = np.repeat(value, 3, axis=-1)
    value = np.where(np.isfinite(value), value, 0.0)
    result = np.rint(np.clip(value[..., :3], 0.0, 1.0) * 255.0).astype(np.uint8)
    result[~valid] = 0
    return result


def convert_fipt_synthetic_scene(
    scene_dir: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
    max_frames: int | None = None,
) -> dict[str, Any]:
    scene_dir = Path(scene_dir).resolve()
    if not scene_dir.is_dir():
        raise FileNotFoundError(scene_dir)

    split_payloads = [_load_split(scene_dir, split) for split in ("train", "val")]
    camera_angles = [float(metadata["camera_angle_x"]) for metadata, _ in split_payloads]
    if not np.allclose(camera_angles, camera_angles[0], atol=1.0e-8):
        raise ValueError(f"train/val camera angle mismatch in {scene_dir}")
    source_frames: list[tuple[str, dict[str, Any]]] = []
    for split, (_, frames) in zip(("train", "val"), split_payloads):
        source_frames.extend((split, frame) for frame in frames)
    if max_frames is not None:
        source_frames = source_frames[: max(0, int(max_frames))]
    if not source_frames:
        raise ValueError("max_frames must select at least one frame")

    first_path = scene_dir / source_frames[0][1]["file_path"]
    first_hdr = read_exr(first_path.with_suffix(".exr"), rgb=True)
    height, width = first_hdr.shape[:2]
    fl_x = 0.5 * width / np.tan(0.5 * camera_angles[0])
    fl_y = fl_x
    cx = 0.5 * width
    cy = 0.5 * height

    with atomic_output_dir(output_dir, overwrite=overwrite) as (staging, output):
        for name in (
            "images",
            "radiance_hdr",
            "depths",
            "normals_camera",
            "masks",
            "albedo_gt",
            "roughness_gt",
        ):
            (staging / name).mkdir()

        frames: list[dict[str, Any]] = []
        train_filenames: list[str] = []
        eval_filenames: list[str] = []
        generated_ldr = 0
        for output_index, (split, source_frame) in enumerate(source_frames):
            source_image_base = scene_dir / source_frame["file_path"]
            source_stem = source_image_base.name
            source_root = scene_dir / split
            sources = {
                "hdr": source_image_base.with_suffix(".exr"),
                "depth": source_root / "Depth" / f"{source_stem}.exr",
                "normal": source_root / "Normal" / f"{source_stem}.exr",
                "albedo": source_root / "DiffCol" / f"{source_stem}.exr",
                "roughness": source_root / "Roughness" / f"{source_stem}.exr",
            }
            for path in sources.values():
                if not path.is_file():
                    raise FileNotFoundError(path)

            hdr = read_exr(sources["hdr"], rgb=True)
            ray_depth = read_exr(sources["depth"], rgb=False)
            normal_world = read_exr(sources["normal"], rgb=True)
            albedo = read_exr(sources["albedo"], rgb=True)
            roughness = read_exr(sources["roughness"], rgb=False)
            expected_rgb = (height, width, 3)
            expected_scalar = (height, width)
            if (
                hdr.shape != expected_rgb
                or normal_world.shape != expected_rgb
                or albedo.shape != expected_rgb
                or ray_depth.shape != expected_scalar
                or roughness.shape != expected_scalar
            ):
                raise ValueError(f"FIPT modality shape mismatch in {split}/{source_stem}")
            source_transform = np.asarray(source_frame["transform_matrix"], dtype=np.float64)
            transform = fipt_pose_to_nerfstudio(source_transform)
            normal_camera = fipt_world_normal_to_nerfstudio_camera(normal_world, transform)

            valid = np.isfinite(ray_depth) & (ray_depth > 0.0)
            valid &= np.isfinite(normal_camera).all(axis=-1)
            valid &= np.linalg.norm(normal_camera, axis=-1) > 0.5
            if not np.any(valid):
                raise ValueError(f"FIPT frame has no valid geometry: {sources['depth']}")
            depth = ray_distance_to_z(
                np.where(np.isfinite(ray_depth), ray_depth, 0.0),
                fl_x=fl_x,
                fl_y=fl_y,
                cx=cx,
                cy=cy,
            )
            depth[~valid] = 0.0
            hdr = np.where(np.isfinite(hdr), hdr, 0.0).astype(np.float32)
            hdr[~valid] = 0.0

            name = f"frame_{output_index:04d}"
            image_output = staging / "images" / f"{name}.png"
            hdr_output = staging / "radiance_hdr" / f"{name}.npy"
            depth_output = staging / "depths" / f"{name}.npy"
            normal_output = staging / "normals_camera" / f"{name}.png"
            mask_output = staging / "masks" / f"{name}.png"
            albedo_output = staging / "albedo_gt" / f"{name}.png"
            roughness_output = staging / "roughness_gt" / f"{name}.png"

            source_png = source_image_base.with_suffix(".png")
            if source_png.is_file():
                with Image.open(source_png) as image:
                    if image.size != (width, height):
                        raise ValueError(f"FIPT PNG size mismatch in {source_png}: {image.size}")
                shutil.copy2(source_png, image_output)
            else:
                save_rgb(image_output, tone_map_hdr(hdr, valid))
                generated_ldr += 1
            np.save(hdr_output, hdr, allow_pickle=False)
            np.save(depth_output, depth.astype(np.float32), allow_pickle=False)
            save_rgb(normal_output, encode_normal_rgb(normal_camera, valid))
            save_mask(mask_output, valid)
            save_rgb(albedo_output, _material_png(albedo, valid))
            save_rgb(roughness_output, _material_png(roughness, valid))

            image_rel = image_output.relative_to(staging).as_posix()
            frames.append(
                {
                    "file_path": image_rel,
                    "radiance_hdr_file_path": hdr_output.relative_to(staging).as_posix(),
                    "depth_file_path": depth_output.relative_to(staging).as_posix(),
                    "normal_camera_file_path": normal_output.relative_to(staging).as_posix(),
                    "mask_path": mask_output.relative_to(staging).as_posix(),
                    "albedo_gt_file_path": albedo_output.relative_to(staging).as_posix(),
                    "roughness_gt_file_path": roughness_output.relative_to(staging).as_posix(),
                    "transform_matrix": transform.tolist(),
                    "fl_x": float(fl_x),
                    "fl_y": float(fl_y),
                    "cx": float(cx),
                    "cy": float(cy),
                    "w": width,
                    "h": height,
                    "source_split": split,
                    "source_frame": source_stem,
                }
            )
            if split == "train":
                train_filenames.append(image_rel)
            else:
                eval_filenames.append(image_rel)
            if (output_index + 1) % 25 == 0 or output_index + 1 == len(source_frames):
                print(f"[{scene_dir.name}] converted {output_index + 1}/{len(source_frames)} frames", flush=True)

        metadata: dict[str, Any] = {
            "camera_model": "OPENCV",
            "fl_x": float(fl_x),
            "fl_y": float(fl_y),
            "cx": float(cx),
            "cy": float(cy),
            "w": width,
            "h": height,
            "k1": 0.0,
            "k2": 0.0,
            "p1": 0.0,
            "p2": 0.0,
            "aegir_note": "FIPT-Synthetic left-up-forward c2w converted to Nerfstudio OpenGL c2w; Blender world normals converted to OpenGL camera space; ray distance converted to camera-z depth",
            "aegir_source_camera_convention": "fipt_left_up_forward_c2w",
            "aegir_camera_convention": "opengl_c2w",
            "source_depth_mode": "ray",
            "aegir_depth_convention": "opencv",
            "aegir_depth_mode": "z",
            "depth_unit_scale_factor": 1.0,
            "source_normal_space": "world",
            "normal_space": "camera",
            "normal_camera_convention": "opengl",
            "normal_encoding": "RGB uint8 maps XYZ from [0,255] to [-1,1]",
            "source_scene": str(scene_dir),
            "train_filenames": train_filenames,
            "val_filenames": eval_filenames,
            "test_filenames": eval_filenames,
            "frames": frames,
        }
        (staging / "transforms.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )

    return {
        "scene": scene_dir.name,
        "output_dir": str(output),
        "frames": len(frames),
        "train_frames": len(train_filenames),
        "eval_frames": len(eval_filenames),
        "generated_ldr_frames": generated_ldr,
        "width": width,
        "height": height,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a FIPT-Synthetic scene to AEGIR/Nerfstudio format."
    )
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-frames", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = convert_fipt_synthetic_scene(
        args.scene_dir,
        args.output_dir,
        overwrite=args.overwrite,
        max_frames=args.max_frames,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
