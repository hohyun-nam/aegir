from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

OPENCV_TO_OPENGL = np.diag([1.0, -1.0, -1.0])


def _load_counted_matrices(path: Path, rows: int, columns: int) -> np.ndarray:
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        raise ValueError(f"empty camera file: {path}")
    count = int(lines[0])
    values = np.asarray(
        [[float(value) for value in line.split()] for line in lines[1:]],
        dtype=np.float64,
    )
    expected_shape = (count * rows, columns)
    if values.shape != expected_shape:
        raise ValueError(f"expected {expected_shape} values in {path}, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite camera value in {path}")
    return values.reshape(count, rows, columns)


def _orthonormalize(rotation: np.ndarray) -> np.ndarray:
    left, _, right = np.linalg.svd(rotation)
    result = left @ right
    if np.linalg.det(result) < 0.0:
        left[:, -1] *= -1.0
        result = left @ right
    return result


def _validate_camera_files(rt: np.ndarray, intrinsics: np.ndarray, cam: np.ndarray) -> np.ndarray:
    count = len(rt)
    if len(intrinsics) != count or len(cam) != count:
        raise ValueError(f"camera count mismatch: Rt={count}, K_list={len(intrinsics)}, cam={len(cam)}")
    if not np.allclose(rt[:, 3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1.0e-6):
        raise ValueError("Rt.txt matrices must have a homogeneous [0, 0, 0, 1] row")
    if not np.allclose(intrinsics[:, 2], np.array([0.0, 0.0, 1.0]), atol=1.0e-6):
        raise ValueError("K_list.txt matrices must have a homogeneous [0, 0, 1] row")
    if np.any(intrinsics[:, 0, 0] <= 0.0) or np.any(intrinsics[:, 1, 1] <= 0.0):
        raise ValueError("camera focal lengths must be positive")

    rotations = rt[:, :3, :3]
    origins = rt[:, :3, 3]
    look_directions = cam[:, 1] - cam[:, 0]
    look_directions /= np.linalg.norm(look_directions, axis=1, keepdims=True)
    up_directions = cam[:, 2] / np.linalg.norm(cam[:, 2], axis=1, keepdims=True)
    origin_error = float(np.max(np.abs(origins - cam[:, 0])))
    forward_error = float(np.max(np.abs(rotations[:, :, 2] - look_directions)))
    up_error = float(np.max(np.abs(-rotations[:, :, 1] - up_directions)))
    if origin_error > 2.0e-3 or forward_error > 5.0e-3 or up_error > 5.0e-3:
        raise ValueError(
            "Rt.txt is not an OpenCV camera-to-world matrix according to cam.txt: "
            f"origin_error={origin_error:.6g}, forward_error={forward_error:.6g}, "
            f"up_error={up_error:.6g}"
        )
    return np.stack([_orthonormalize(rotation) for rotation in rotations])


def _read_exr_rgb(path: Path) -> np.ndarray:
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required to preserve float EXR radiance") from exc
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"OpenCV could not read EXR image: {path}")
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    if image.ndim != 3 or image.shape[-1] not in (3, 4):
        raise ValueError(f"expected HxWx3/4 EXR image at {path}, got {image.shape}")
    image = image[..., :3][..., ::-1].astype(np.float32, copy=True)
    if not np.isfinite(image).all():
        raise ValueError(f"non-finite HDR radiance in {path}")
    return image


def _camera_points(depth: np.ndarray, intrinsic: np.ndarray, mode: str) -> np.ndarray:
    height, width = depth.shape
    yy, xx = np.mgrid[:height, :width]
    x = (xx - intrinsic[0, 2]) / intrinsic[0, 0]
    y = (yy - intrinsic[1, 2]) / intrinsic[1, 1]
    directions = np.stack((x, y, np.ones_like(x)), axis=-1)
    if mode == "ray":
        directions /= np.maximum(np.linalg.norm(directions, axis=-1, keepdims=True), 1.0e-12)
    elif mode != "z":
        raise ValueError(f"unsupported depth mode: {mode}")
    return directions * depth[..., None]


def _geometry_normal_score(
    scene_dir: Path,
    rotations: np.ndarray,
    intrinsics: np.ndarray,
    frame_indices: list[int],
    depth_mode: str,
) -> float:
    scores: list[np.ndarray] = []
    for index in frame_indices:
        stem = f"{index:03d}_0001"
        depth = np.load(scene_dir / "MiDepth" / f"{stem}.npy", allow_pickle=False).astype(np.float64)
        normal = np.asarray(Image.open(scene_dir / "MiNormalGlobal" / f"{stem}.png"))[..., :3].astype(
            np.float64
        )
        normal = normal / 127.5 - 1.0
        mask = np.asarray(Image.open(scene_dir / "ImMask" / f"{stem}.png"))
        if mask.ndim == 3:
            mask = mask[..., 0]
        if depth.ndim != 2 or normal.shape != depth.shape + (3,) or mask.shape != depth.shape:
            raise ValueError(f"modality shape mismatch for frame {index} in {scene_dir}")

        points = _camera_points(depth, intrinsics[index], depth_mode)
        delta_x = points[1:-1, 2:] - points[1:-1, :-2]
        delta_y = points[2:, 1:-1] - points[:-2, 1:-1]
        camera_normal = np.cross(delta_y, delta_x)
        camera_normal_length = np.linalg.norm(camera_normal, axis=-1, keepdims=True)
        camera_normal /= np.maximum(camera_normal_length, 1.0e-12)
        world_prediction = camera_normal @ rotations[index].T

        world_reference = normal[1:-1, 1:-1]
        reference_length = np.linalg.norm(world_reference, axis=-1, keepdims=True)
        world_reference /= np.maximum(reference_length, 1.0e-12)
        valid = (
            (depth[1:-1, 1:-1] > 0.0)
            & (depth[1:-1, 2:] > 0.0)
            & (depth[1:-1, :-2] > 0.0)
            & (depth[2:, 1:-1] > 0.0)
            & (depth[:-2, 1:-1] > 0.0)
            & (mask[1:-1, 1:-1] > 0)
            & (camera_normal_length[..., 0] > 1.0e-8)
            & (reference_length[..., 0] > 0.5)
        )
        dots = np.sum(world_prediction * world_reference, axis=-1)[valid]
        if dots.size:
            scores.append(dots[::4])
    if not scores:
        raise ValueError(f"no valid depth/normal pixels found in {scene_dir}")
    return float(np.median(np.concatenate(scores)))


def validate_source_geometry(
    scene_dir: Path,
    rotations: np.ndarray,
    intrinsics: np.ndarray,
) -> dict[str, float]:
    last = len(rotations) - 1
    frame_indices = sorted({0, len(rotations) // 2, last})
    z_score = _geometry_normal_score(scene_dir, rotations, intrinsics, frame_indices, "z")
    ray_score = _geometry_normal_score(scene_dir, rotations, intrinsics, frame_indices, "ray")
    if z_score < 0.9:
        raise ValueError(
            f"Rt/depth/world-normal projection validation failed: median z-depth normal dot={z_score:.6f}"
        )
    return {"z_depth_normal_dot": z_score, "ray_depth_normal_dot": ray_score}


def _required_frame_paths(scene_dir: Path, index: int) -> dict[str, Path]:
    stem = f"{index:03d}_0001"
    return {
        "image": scene_dir / "Image" / f"{stem}.png",
        "hdr": scene_dir / "Image" / f"{stem}.exr",
        "depth": scene_dir / "MiDepth" / f"{stem}.npy",
        "normal": scene_dir / "MiNormalGlobal" / f"{stem}.png",
        "mask": scene_dir / "ImMask" / f"{stem}.png",
    }


def _validate_export(output_dir: Path, expected_frames: int) -> None:
    metadata = json.loads((output_dir / "transforms.json").read_text(encoding="utf-8"))
    frames = metadata.get("frames", [])
    if len(frames) != expected_frames:
        raise ValueError(f"expected {expected_frames} exported frames, got {len(frames)}")
    for index in sorted({0, len(frames) // 2, len(frames) - 1}):
        frame = frames[index]
        for key in (
            "file_path",
            "radiance_hdr_file_path",
            "depth_file_path",
            "normal_world_file_path",
            "mask_path",
        ):
            path = output_dir / frame[key]
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"missing or empty exported file: {path}")
        image = Image.open(output_dir / frame["file_path"])
        depth = np.load(output_dir / frame["depth_file_path"], allow_pickle=False)
        hdr = np.load(output_dir / frame["radiance_hdr_file_path"], allow_pickle=False)
        if depth.shape != (image.height, image.width):
            raise ValueError(f"depth/image shape mismatch in exported frame {index}")
        if hdr.shape != (image.height, image.width, 3):
            raise ValueError(f"HDR/image shape mismatch in exported frame {index}")
        transform = np.asarray(frame["transform_matrix"], dtype=np.float64)
        if transform.shape != (4, 4) or not np.allclose(
            transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1.0e-5
        ):
            raise ValueError(f"invalid exported c2w rotation in frame {index}")


def convert_fipt_real_scene(
    scene_dir: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
    max_frames: int | None = None,
    validate_geometry: bool = True,
) -> dict[str, Any]:
    scene_dir = Path(scene_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if not scene_dir.is_dir():
        raise FileNotFoundError(scene_dir)
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"output already exists (pass --overwrite to replace it): {output_dir}")

    rt = _load_counted_matrices(scene_dir / "Rt.txt", 4, 4)
    intrinsics = _load_counted_matrices(scene_dir / "K_list.txt", 3, 3)
    cam = _load_counted_matrices(scene_dir / "cam.txt", 3, 3)
    rotations = _validate_camera_files(rt, intrinsics, cam)
    source_count = len(rt)
    frame_count = source_count if max_frames is None else min(source_count, int(max_frames))
    if frame_count <= 0:
        raise ValueError("max_frames must select at least one frame")

    first_paths = _required_frame_paths(scene_dir, 0)
    for path in first_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    with Image.open(first_paths["image"]) as first_image:
        width, height = first_image.size

    geometry_metrics: dict[str, float] = {}
    if validate_geometry and frame_count == source_count:
        geometry_metrics = validate_source_geometry(scene_dir, rotations, intrinsics)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=str(output_dir.parent)))
    try:
        for name in ("images", "radiance_hdr", "depths", "normals_world", "masks"):
            (staging_dir / name).mkdir()

        frames: list[dict[str, Any]] = []
        for index in range(frame_count):
            source = _required_frame_paths(scene_dir, index)
            for path in source.values():
                if not path.is_file():
                    raise FileNotFoundError(path)
            frame_name = f"frame_{index:04d}"
            image_path = staging_dir / "images" / f"{frame_name}.png"
            hdr_path = staging_dir / "radiance_hdr" / f"{frame_name}.npy"
            depth_path = staging_dir / "depths" / f"{frame_name}.npy"
            normal_path = staging_dir / "normals_world" / f"{frame_name}.png"
            mask_path = staging_dir / "masks" / f"{frame_name}.png"

            with Image.open(source["image"]) as image:
                if image.size != (width, height):
                    raise ValueError(f"image size mismatch in {source['image']}: {image.size}")
            with Image.open(source["normal"]) as normal_image:
                if normal_image.size != (width, height):
                    raise ValueError(f"normal/image size mismatch in {source['normal']}: {normal_image.size}")
            with Image.open(source["mask"]) as mask_image:
                if mask_image.size != (width, height):
                    raise ValueError(f"mask/image size mismatch in {source['mask']}: {mask_image.size}")
            shutil.copy2(source["image"], image_path)
            shutil.copy2(source["normal"], normal_path)
            shutil.copy2(source["mask"], mask_path)

            hdr = _read_exr_rgb(source["hdr"])
            depth = np.load(source["depth"], allow_pickle=False).astype(np.float32)
            if hdr.shape != (height, width, 3) or depth.shape != (height, width):
                raise ValueError(
                    f"modality shape mismatch in source frame {index}: "
                    f"image={(height, width)}, hdr={hdr.shape}, depth={depth.shape}"
                )
            if not np.isfinite(depth).all() or np.any(depth < 0.0):
                raise ValueError(f"invalid depth values in {source['depth']}")
            np.save(hdr_path, hdr, allow_pickle=False)
            np.save(depth_path, depth, allow_pickle=False)

            c2w = np.eye(4, dtype=np.float64)
            c2w[:3, :3] = rotations[index] @ OPENCV_TO_OPENGL
            c2w[:3, 3] = rt[index, :3, 3]
            intrinsic = intrinsics[index]
            frames.append(
                {
                    "file_path": image_path.relative_to(staging_dir).as_posix(),
                    "radiance_hdr_file_path": hdr_path.relative_to(staging_dir).as_posix(),
                    "depth_file_path": depth_path.relative_to(staging_dir).as_posix(),
                    "normal_world_file_path": normal_path.relative_to(staging_dir).as_posix(),
                    "mask_path": mask_path.relative_to(staging_dir).as_posix(),
                    "transform_matrix": c2w.tolist(),
                    "fl_x": float(intrinsic[0, 0]),
                    "fl_y": float(intrinsic[1, 1]),
                    "cx": float(intrinsic[0, 2]),
                    "cy": float(intrinsic[1, 2]),
                    "w": width,
                    "h": height,
                    "source_frame_index": index,
                }
            )
            if (index + 1) % 25 == 0 or index + 1 == frame_count:
                print(f"[{scene_dir.name}] converted {index + 1}/{frame_count} frames", flush=True)

        metadata: dict[str, Any] = {
            "camera_model": "OPENCV",
            "fl_x": float(intrinsics[0, 0, 0]),
            "fl_y": float(intrinsics[0, 1, 1]),
            "cx": float(intrinsics[0, 0, 2]),
            "cy": float(intrinsics[0, 1, 2]),
            "w": width,
            "h": height,
            "k1": 0.0,
            "k2": 0.0,
            "p1": 0.0,
            "p2": 0.0,
            "aegir_note": "FIPT-Real converted from Rt.txt OpenCV c2w to Nerfstudio OpenGL c2w",
            "aegir_source_camera_convention": "opencv_c2w",
            "aegir_camera_convention": "opengl_c2w",
            "aegir_depth_convention": "opencv",
            "aegir_depth_mode": "z",
            "depth_unit_scale_factor": 1.0,
            "aegir_color": {
                "format": "aegir-color-v1",
                "observation_domain": "linear_hdr",
                "hdr_encoding": "scene_linear_rgb",
                "target_png": {"source": "FIPT-Real source PNG camera response unknown"},
            },
            "normal_space": "world",
            "normal_world_encoding": "RGB uint8 maps XYZ from [0,255] to [-1,1]",
            "source_scene": str(scene_dir),
            "source_frame_count": source_count,
            "geometry_validation": geometry_metrics,
            "frames": frames,
        }
        (staging_dir / "transforms.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        _validate_export(staging_dir, frame_count)

        if output_dir.exists():
            shutil.rmtree(output_dir)
        staging_dir.rename(output_dir)
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    return {
        "scene": scene_dir.name,
        "output_dir": str(output_dir),
        "frames": frame_count,
        "width": width,
        "height": height,
        **geometry_metrics,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert an exported FIPT-Real scene to AEGIR/Nerfstudio format."
    )
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--skip-geometry-validation",
        action="store_true",
        help="skip depth-derived versus world-normal validation (not recommended)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = convert_fipt_real_scene(
        args.scene_dir,
        args.output_dir,
        overwrite=args.overwrite,
        max_frames=args.max_frames,
        validate_geometry=not args.skip_geometry_validation,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
