from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.linalg import rq

from aegir.conversion_common import (
    OPENCV_TO_OPENGL,
    atomic_output_dir,
    encode_normal_rgb,
    independent_val_test_split,
    ray_distance_to_z,
    save_mask,
    save_rgb,
    tone_map_hdr_with_exposure,
    validate_c2w,
)

KNOWN_STANDARD_PROJECTION_SCENES = {
    "ai_001_004",
    "ai_004_002",
    "ai_010_006",
    "ai_032_001",
    "ai_034_002",
}

# Official apple/ml-hypersim metadata_camera_parameters.csv projection. This scene
# uses a small physical-camera lens shift, so the canonical 60-degree matrix is not exact.
PROJECTION_OVERRIDES = {
    "ai_015_003": np.array(
        [
            [1.7320207448018146, 0.0, 0.0, 0.0],
            [0.0, 2.309400998024209, 0.0, 0.0],
            [0.0, 0.005897784853020349, -1.0019846452565866, -2.002002002002002],
            [0.0, 0.005886001067100229, -0.9999826779333966, 0.0],
        ],
        dtype=np.float64,
    )
}


def _read_hdf5(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        if "dataset" not in handle:
            raise KeyError(f"missing HDF5 dataset in {path}")
        return handle["dataset"][:]


def _meters_per_asset_unit(scene_dir: Path) -> float:
    path = scene_dir / "_detail" / "metadata_scene.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    matches = [row for row in rows if row["parameter_name"] == "meters_per_asset_unit"]
    if len(matches) != 1:
        raise ValueError(f"expected one meters_per_asset_unit in {path}")
    value = float(matches[0]["parameter_value"])
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"invalid meters_per_asset_unit in {path}: {value}")
    return value


def _camera_names(scene_dir: Path) -> list[str]:
    path = scene_dir / "_detail" / "metadata_cameras.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        names = [row["camera_name"] for row in csv.DictReader(handle)]
    if not names:
        raise ValueError(f"no cameras in {path}")
    return names


def _standard_projection(width: int, height: int) -> np.ndarray:
    fov_x = np.pi / 3.0
    fov_y = 2.0 * np.arctan(height * np.tan(fov_x / 2.0) / width)
    near = 1.0
    far = 1000.0
    projection = np.zeros((4, 4), dtype=np.float64)
    projection[0, 0] = 1.0 / np.tan(fov_x / 2.0)
    projection[1, 1] = 1.0 / np.tan(fov_y / 2.0)
    projection[2, 2] = -(far + near) / (far - near)
    projection[2, 3] = -(2.0 * far * near) / (far - near)
    projection[3, 2] = -1.0
    return projection


def _projection_from_csv(path: Path, scene_name: str) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as handle:
        matches = [row for row in csv.DictReader(handle) if row["scene_name"] == scene_name]
    if len(matches) != 1:
        raise ValueError(f"expected one camera-parameter row for {scene_name} in {path}")
    row = matches[0]
    return np.array(
        [[float(row[f"M_proj_{i}{j}"]) for j in range(4)] for i in range(4)],
        dtype=np.float64,
    )


def hypersim_projection(
    scene_name: str,
    width: int,
    height: int,
    camera_parameters_csv: Path | None = None,
) -> np.ndarray:
    if camera_parameters_csv is not None:
        return _projection_from_csv(camera_parameters_csv, scene_name)
    if scene_name in PROJECTION_OVERRIDES:
        return PROJECTION_OVERRIDES[scene_name].copy()
    if scene_name in KNOWN_STANDARD_PROJECTION_SCENES:
        return _standard_projection(width, height)
    raise ValueError(
        f"no bundled HyperSim projection for {scene_name}; pass --camera-parameters-csv "
        "from apple/ml-hypersim/contrib/mikeroberts3000"
    )


def _pixel_projection(
    projection: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    return np.stack(
        (
            0.5 * (width - 1) * (projection[0, :3] + projection[3, :3]),
            0.5 * (height - 1) * (-projection[1, :3] + projection[3, :3]),
            projection[3, :3],
        )
    )


def _factor_pinhole(pixel_camera: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    intrinsic, rotation = rq(pixel_camera)
    signs = np.sign(np.diag(intrinsic))
    signs[signs == 0.0] = 1.0
    sign_matrix = np.diag(signs)
    intrinsic = intrinsic @ sign_matrix
    rotation = sign_matrix @ rotation
    if np.linalg.det(rotation) < 0.0:
        intrinsic = -intrinsic
        rotation = -rotation
    if not np.allclose(intrinsic @ rotation, pixel_camera, atol=1.0e-5):
        raise ValueError("could not factor HyperSim projection matrix")
    intrinsic /= intrinsic[2, 2]
    skew_tolerance = 1.0e-5 * max(abs(intrinsic[0, 0]), abs(intrinsic[1, 1]))
    if abs(intrinsic[0, 1]) > skew_tolerance or not np.allclose(
        rotation.T @ rotation, np.eye(3), atol=1.0e-6
    ):
        raise ValueError("HyperSim projection is not representable as a zero-skew pinhole camera")
    if intrinsic[0, 0] <= 0.0 or intrinsic[1, 1] <= 0.0:
        raise ValueError("invalid factored HyperSim focal length")
    intrinsic[0, 1] = 0.0
    return intrinsic, rotation


def rectify_projection(
    projection: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Factor a camera-space HyperSim projection into OpenCV K and a rotation."""

    return _factor_pinhole(_pixel_projection(projection, width, height))


def rectify_world_camera(
    projection: np.ndarray,
    width: int,
    height: int,
    source_camera_to_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Factor the full HyperSim world-to-pixel map into K and world-to-OpenCV R.

    A few source scenes contain a non-rigid camera orientation matrix. Factoring
    projection @ inverse(orientation) keeps that local scale in K and produces
    a proper rotation for the exported Nerfstudio pose.
    """

    if source_camera_to_world.shape != (3, 3) or not np.isfinite(source_camera_to_world).all():
        raise ValueError("invalid HyperSim camera-to-world orientation")
    pixel_world_camera = _pixel_projection(projection, width, height) @ np.linalg.inv(source_camera_to_world)
    return _factor_pinhole(pixel_world_camera)


def _frame_ids(final_dir: Path) -> list[int]:
    pattern = re.compile(r"frame\.(\d{4})\.color\.hdf5$")
    ids = []
    for path in final_dir.glob("frame.*.color.hdf5"):
        match = pattern.match(path.name)
        if match:
            ids.append(int(match.group(1)))
    return sorted(ids)


def _material_png(value: np.ndarray, valid: np.ndarray) -> np.ndarray:
    value = np.where(np.isfinite(value), value, 0.0)
    result = np.rint(np.clip(value, 0.0, 1.0) * 255.0).astype(np.uint8)
    result[~valid] = 0
    return result


def convert_hypersim_scene(
    scene_dir: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
    max_frames: int | None = None,
    camera_parameters_csv: str | Path | None = None,
) -> dict[str, Any]:
    scene_dir = Path(scene_dir).resolve()
    if not scene_dir.is_dir():
        raise FileNotFoundError(scene_dir)
    camera_csv = Path(camera_parameters_csv).resolve() if camera_parameters_csv else None
    scene_name = scene_dir.name
    meters_scale = _meters_per_asset_unit(scene_dir)
    camera_names = _camera_names(scene_dir)

    records: list[dict[str, Any]] = []
    camera_payloads: dict[str, dict[str, Any]] = {}
    for camera_name in camera_names:
        camera_dir = scene_dir / "_detail" / camera_name
        final_dir = scene_dir / "images" / f"scene_{camera_name}_final_hdf5"
        geometry_dir = scene_dir / "images" / f"scene_{camera_name}_geometry_hdf5"
        if not final_dir.is_dir() or not geometry_dir.is_dir():
            raise FileNotFoundError(f"missing HyperSim image directories for {camera_name}")
        frame_indices = _read_hdf5(camera_dir / "camera_keyframe_frame_indices.hdf5").astype(int)
        positions = _read_hdf5(camera_dir / "camera_keyframe_positions.hdf5").astype(np.float64)
        orientations = _read_hdf5(camera_dir / "camera_keyframe_orientations.hdf5").astype(np.float64)
        if len(frame_indices) != len(positions) or len(positions) != len(orientations):
            raise ValueError(f"HyperSim camera array length mismatch for {camera_name}")
        keyframe_by_frame = {int(frame): index for index, frame in enumerate(frame_indices)}
        ids = _frame_ids(final_dir)
        if not ids:
            raise ValueError(f"no public color frames in {final_dir}")
        camera_payloads[camera_name] = {
            "positions": positions,
            "orientations": orientations,
            "keyframe_by_frame": keyframe_by_frame,
            "final_dir": final_dir,
            "geometry_dir": geometry_dir,
            "frame_ids": ids,
        }
        for local_index, frame_id in enumerate(ids):
            if frame_id not in keyframe_by_frame:
                raise ValueError(f"frame {frame_id} missing from {camera_name} keyframes")
            records.append(
                {
                    "camera_name": camera_name,
                    "frame_id": frame_id,
                    "local_index": local_index,
                }
            )
    if max_frames is not None:
        records = records[: max(0, int(max_frames))]
    if not records:
        raise ValueError("max_frames must select at least one HyperSim frame")

    first_record = records[0]
    first_payload = camera_payloads[first_record["camera_name"]]
    first_color = _read_hdf5(first_payload["final_dir"] / f"frame.{first_record['frame_id']:04d}.color.hdf5")
    if first_color.ndim != 3 or first_color.shape[-1] != 3:
        raise ValueError(f"unexpected HyperSim color shape: {first_color.shape}")
    height, width = first_color.shape[:2]
    projection = hypersim_projection(scene_name, width, height, camera_csv)

    validation_indices: set[int] = set()
    for camera_name in camera_names:
        camera_record_indices = [
            index for index, record in enumerate(records) if record["camera_name"] == camera_name
        ]
        if camera_record_indices:
            validation_indices.update(
                {
                    camera_record_indices[0],
                    camera_record_indices[len(camera_record_indices) // 2],
                    camera_record_indices[-1],
                }
            )
    position_errors: list[np.ndarray] = []
    normal_dots: list[np.ndarray] = []

    with atomic_output_dir(output_dir, overwrite=overwrite) as (staging, output):
        for name in (
            "images",
            "radiance_hdr",
            "depths",
            "normals_camera",
            "masks",
            "albedo_gt",
        ):
            (staging / name).mkdir()

        frames: list[dict[str, Any]] = []
        train_filenames: list[str] = []
        eval_filenames: list[str] = []
        for output_index, record in enumerate(records):
            camera_name = record["camera_name"]
            frame_id = int(record["frame_id"])
            payload = camera_payloads[camera_name]
            final_dir: Path = payload["final_dir"]
            geometry_dir: Path = payload["geometry_dir"]
            root = f"frame.{frame_id:04d}"
            paths = {
                "color": final_dir / f"{root}.color.hdf5",
                "albedo": final_dir / f"{root}.diffuse_reflectance.hdf5",
                "depth": geometry_dir / f"{root}.depth_meters.hdf5",
                "normal_camera": geometry_dir / f"{root}.normal_cam.hdf5",
                "normal_world": geometry_dir / f"{root}.normal_world.hdf5",
                "position": geometry_dir / f"{root}.position.hdf5",
                "entity": geometry_dir / f"{root}.render_entity_id.hdf5",
            }
            for path in paths.values():
                if not path.is_file():
                    raise FileNotFoundError(path)

            hdr = _read_hdf5(paths["color"]).astype(np.float32)
            albedo = _read_hdf5(paths["albedo"]).astype(np.float32)
            ray_depth = _read_hdf5(paths["depth"]).astype(np.float32)
            normal_source = _read_hdf5(paths["normal_camera"]).astype(np.float32)
            entity = _read_hdf5(paths["entity"])
            if (
                hdr.shape != (height, width, 3)
                or albedo.shape != (height, width, 3)
                or ray_depth.shape != (height, width)
                or normal_source.shape != (height, width, 3)
                or entity.shape != (height, width)
            ):
                raise ValueError(f"HyperSim modality shape mismatch in {camera_name}/{root}")
            valid = entity != -1
            valid &= np.isfinite(ray_depth) & (ray_depth > 0.0)
            valid &= np.isfinite(normal_source).all(axis=-1)
            valid &= np.linalg.norm(normal_source, axis=-1) > 0.5
            if not np.any(valid):
                raise ValueError(f"HyperSim frame has no valid geometry: {paths['depth']}")

            keyframe_index = payload["keyframe_by_frame"][frame_id]
            source_orientation = payload["orientations"][keyframe_index]
            source_position = payload["positions"][keyframe_index]
            frame_intrinsic, world_to_opencv = rectify_world_camera(
                projection,
                width,
                height,
                source_orientation,
            )
            frame_fl_x = float(frame_intrinsic[0, 0])
            frame_fl_y = float(frame_intrinsic[1, 1])
            frame_cx = float(frame_intrinsic[0, 2])
            frame_cy = float(frame_intrinsic[1, 2])
            depth = ray_distance_to_z(
                np.where(np.isfinite(ray_depth), ray_depth, 0.0),
                fl_x=frame_fl_x,
                fl_y=frame_fl_y,
                cx=frame_cx,
                cy=frame_cy,
            )
            depth[~valid] = 0.0
            hdr = np.where(np.isfinite(hdr), hdr, 0.0).astype(np.float32)
            hdr[~valid] = 0.0

            transform = np.eye(4, dtype=np.float64)
            transform[:3, :3] = world_to_opencv.T @ OPENCV_TO_OPENGL
            transform[:3, 3] = source_position * meters_scale
            validate_c2w(transform, label=f"HyperSim {camera_name}/{root}")
            normal_world_from_source = normal_source @ source_orientation.T
            normal_camera = normal_world_from_source @ transform[:3, :3]

            if output_index in validation_indices:
                position_reference = _read_hdf5(paths["position"]).astype(np.float64)
                normal_world_reference = _read_hdf5(paths["normal_world"]).astype(np.float64)
                yy, xx = np.mgrid[:height, :width]
                points_cv = np.stack(
                    (
                        (xx - frame_cx) / frame_fl_x * depth,
                        (yy - frame_cy) / frame_fl_y * depth,
                        depth,
                    ),
                    axis=-1,
                )
                points_gl = points_cv @ OPENCV_TO_OPENGL
                points_world = points_gl @ transform[:3, :3].T + transform[:3, 3]
                reference_world = position_reference * meters_scale
                sample_valid = valid & np.isfinite(reference_world).all(axis=-1)
                errors = np.linalg.norm(points_world - reference_world, axis=-1)[sample_valid]
                if errors.size:
                    position_errors.append(errors[::32])

                predicted_world_normal = normal_camera @ transform[:3, :3].T
                predicted_length = np.linalg.norm(predicted_world_normal, axis=-1, keepdims=True)
                reference_length = np.linalg.norm(normal_world_reference, axis=-1, keepdims=True)
                normal_valid = sample_valid & np.isfinite(normal_world_reference).all(axis=-1)
                normal_valid &= (reference_length[..., 0] > 0.5) & (predicted_length[..., 0] > 0.5)
                predicted_normalized = predicted_world_normal / np.maximum(predicted_length, 1.0e-12)
                reference_normalized = normal_world_reference / np.maximum(reference_length, 1.0e-12)
                dots = np.sum(predicted_normalized * reference_normalized, axis=-1)[normal_valid]
                if dots.size:
                    normal_dots.append(np.clip(dots[::32], -1.0, 1.0))

            name = f"frame_{output_index:04d}"
            image_output = staging / "images" / f"{name}.png"
            hdr_output = staging / "radiance_hdr" / f"{name}.npy"
            depth_output = staging / "depths" / f"{name}.npy"
            normal_output = staging / "normals_camera" / f"{name}.png"
            mask_output = staging / "masks" / f"{name}.png"
            albedo_output = staging / "albedo_gt" / f"{name}.png"
            mapped, ldr_exposure = tone_map_hdr_with_exposure(hdr, valid)
            save_rgb(image_output, mapped)
            np.save(hdr_output, hdr, allow_pickle=False)
            np.save(depth_output, depth.astype(np.float32), allow_pickle=False)
            save_rgb(normal_output, encode_normal_rgb(normal_camera, valid))
            save_mask(mask_output, valid)
            save_rgb(albedo_output, _material_png(albedo, valid))

            image_rel = image_output.relative_to(staging).as_posix()
            frames.append(
                {
                    "file_path": image_rel,
                    "radiance_hdr_file_path": hdr_output.relative_to(staging).as_posix(),
                    "depth_file_path": depth_output.relative_to(staging).as_posix(),
                    "normal_camera_file_path": normal_output.relative_to(staging).as_posix(),
                    "mask_path": mask_output.relative_to(staging).as_posix(),
                    "albedo_gt_file_path": albedo_output.relative_to(staging).as_posix(),
                    "transform_matrix": transform.tolist(),
                    "fl_x": frame_fl_x,
                    "fl_y": frame_fl_y,
                    "cx": frame_cx,
                    "cy": frame_cy,
                    "w": width,
                    "h": height,
                    "source_camera": camera_name,
                    "source_frame_index": frame_id,
                    "ldr_exposure_scale": ldr_exposure,
                    "ldr_observation": "generated_hypersim_percentile_gamma22",
                }
            )
            if int(record["local_index"]) % 8 == 7:
                eval_filenames.append(image_rel)
            else:
                train_filenames.append(image_rel)
            if (output_index + 1) % 25 == 0 or output_index + 1 == len(records):
                print(f"[{scene_name}] converted {output_index + 1}/{len(records)} frames", flush=True)

        if not position_errors or not normal_dots:
            raise ValueError(f"could not validate HyperSim position/normal geometry for {scene_name}")
        position_error = float(np.median(np.concatenate(position_errors)))
        normal_dot = float(np.median(np.concatenate(normal_dots)))
        if position_error > 0.02 or normal_dot < 0.99:
            raise ValueError(
                f"HyperSim geometry validation failed: position median={position_error:.6g} m, "
                f"normal median dot={normal_dot:.6g}"
            )

        val_filenames, test_filenames, split_metadata = independent_val_test_split(
            train_filenames, eval_filenames
        )
        metadata: dict[str, Any] = {
            "camera_model": "OPENCV",
            "w": width,
            "h": height,
            "k1": 0.0,
            "k2": 0.0,
            "p1": 0.0,
            "p2": 0.0,
            "aegir_note": "HyperSim projective camera rectified to Nerfstudio pinhole/OpenGL c2w in meters",
            "aegir_source_camera_convention": "hypersim_opengl_c2w_asset_units",
            "aegir_camera_convention": "opengl_c2w_meters",
            "source_depth_mode": "ray",
            "aegir_depth_convention": "opencv",
            "aegir_depth_mode": "z",
            "depth_unit_scale_factor": 1.0,
            "aegir_color": {
                "format": "aegir-color-v1",
                "observation_domain": "linear_hdr",
                "hdr_encoding": "scene_linear_rgb",
                "target_png": {
                    "exposure_policy": "fixed_per_frame",
                    "tone_mapper": "clip",
                    "transfer_function": "gamma22",
                    "source": "HyperSim 90th-percentile display mapping",
                },
            },
            "normal_space": "camera",
            "normal_camera_convention": "opengl",
            "normal_encoding": "RGB uint8 maps XYZ from [0,255] to [-1,1]",
            "meters_per_asset_unit": meters_scale,
            "source_scene": str(scene_dir),
            "source_cameras": camera_names,
            "geometry_validation": {
                "position_median_error_meters": position_error,
                "normal_world_median_dot": normal_dot,
            },
            "train_filenames": train_filenames,
            "val_filenames": val_filenames,
            "test_filenames": test_filenames,
            "aegir_evaluation_split": split_metadata,
            "frames": frames,
        }
        (staging / "transforms.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    return {
        "scene": scene_name,
        "output_dir": str(output),
        "frames": len(frames),
        "train_frames": len(train_filenames),
        "eval_frames": len(eval_filenames),
        "cameras": camera_names,
        "width": width,
        "height": height,
        "position_median_error_meters": position_error,
        "normal_world_median_dot": normal_dot,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a raw HyperSim HDF5 scene to AEGIR/Nerfstudio format."
    )
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--camera-parameters-csv",
        type=Path,
        help="official apple/ml-hypersim metadata_camera_parameters.csv (optional for bundled scenes)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = convert_hypersim_scene(
        args.scene_dir,
        args.output_dir,
        overwrite=args.overwrite,
        max_frames=args.max_frames,
        camera_parameters_csv=args.camera_parameters_csv,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
