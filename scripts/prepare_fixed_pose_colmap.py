#!/usr/bin/env python3
"""Triangulate a COLMAP sparse cloud while keeping Nerfstudio poses fixed.

The input ``transforms.json`` is expected to contain OpenGL camera-to-world
matrices.  COLMAP receives the equivalent OpenCV world-to-camera matrices as
an already registered model.  Feature extraction and matching are run first,
then ``point_triangulator`` creates points without estimating camera poses.

The script deliberately verifies the output poses against the supplied poses
before publishing the PLY path in ``transforms.json``.  This makes an accidental
pose-refinement configuration a hard failure instead of silently misaligning
RGB, depth, normals, and material supervision.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from nerfstudio.data.utils.colmap_parsing_utils import (
    qvec2rotmat,
    read_images_binary,
    read_points3D_binary,
    rotmat2qvec,
)

OPENGL_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])
IDENTITY_APPLIED_TRANSFORM = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
]
COLMAP_OPENCV_MODEL_ID = 4
COLMAP_OPENCV_NUM_PARAMS = 8


@dataclass(frozen=True)
class FrameCamera:
    frame: dict[str, Any]
    image_name: str
    width: int
    height: int
    params: tuple[float, ...]
    w2c: np.ndarray


def _value(frame: dict[str, Any], meta: dict[str, Any], key: str, default: Any = None) -> Any:
    if key in frame:
        return frame[key]
    return meta.get(key, default)


def _run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _load_frames(scene: Path) -> tuple[dict[str, Any], list[FrameCamera]]:
    transforms_path = scene / "transforms.json"
    if not transforms_path.is_file():
        raise FileNotFoundError(f"missing {transforms_path}")

    meta = json.loads(transforms_path.read_text(encoding="utf-8"))
    raw_frames = meta.get("frames")
    if not isinstance(raw_frames, list) or len(raw_frames) < 2:
        raise ValueError("transforms.json must contain at least two frames")
    if meta.get("camera_model", "OPENCV") != "OPENCV":
        raise ValueError("this fixed-pose pipeline currently requires camera_model=OPENCV")

    images_dir = (scene / "images").resolve()
    if not images_dir.is_dir():
        raise FileNotFoundError(f"missing image directory: {images_dir}")

    result: list[FrameCamera] = []
    seen_names: set[str] = set()
    for index, frame in enumerate(raw_frames):
        if "transform_matrix" not in frame:
            raise ValueError(
                f"frame {index} has no transform_matrix; fixed-pose triangulation "
                "cannot be used when poses are absent"
            )
        image_path = (scene / frame["file_path"]).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"missing image: {image_path}")
        try:
            image_name = image_path.relative_to(images_dir).as_posix()
        except ValueError as exc:
            raise ValueError(f"image must be below {images_dir}: {image_path}") from exc
        if image_name in seen_names:
            raise ValueError(f"duplicate image path in transforms.json: {image_name}")
        seen_names.add(image_name)

        width = int(_value(frame, meta, "w"))
        height = int(_value(frame, meta, "h"))
        params = tuple(
            float(_value(frame, meta, key, 0.0))
            for key in ("fl_x", "fl_y", "cx", "cy", "k1", "k2", "p1", "p2")
        )
        if width <= 0 or height <= 0 or params[0] <= 0 or params[1] <= 0:
            raise ValueError(f"invalid intrinsics for {image_name}: {width}x{height}, {params}")

        c2w_gl = np.asarray(frame["transform_matrix"], dtype=np.float64)
        if c2w_gl.shape != (4, 4) or not np.isfinite(c2w_gl).all():
            raise ValueError(f"invalid transform_matrix for {image_name}")
        if not np.allclose(c2w_gl[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
            raise ValueError(f"non-rigid homogeneous row for {image_name}")
        rotation = c2w_gl[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-5):
            raise ValueError(f"non-orthonormal pose rotation for {image_name}")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=2e-5):
            raise ValueError(f"pose rotation determinant is not +1 for {image_name}")

        # Nerfstudio: OpenGL c2w (right, up, back).
        # COLMAP: OpenCV w2c (right, down, forward).
        c2w_cv = c2w_gl @ OPENGL_TO_OPENCV
        w2c = np.linalg.inv(c2w_cv)
        result.append(FrameCamera(frame, image_name, width, height, params, w2c))

    return meta, result


def _prepare_colmap_masks(scene: Path, work_dir: Path, frames: list[FrameCamera]) -> Path | None:
    """Create COLMAP's ``image.ext.png`` mask layout as symlinks."""
    if not all("mask_path" in item.frame for item in frames):
        print("No complete mask set found; extracting features without masks.", flush=True)
        return None

    mask_root = work_dir / "feature_masks"
    mask_root.mkdir(parents=True, exist_ok=True)
    for item in frames:
        source = (scene / item.frame["mask_path"]).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"missing mask: {source}")
        # COLMAP appends '.png' to the full image name, e.g. foo.jpg.png.
        target = mask_root / f"{item.image_name}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink() or target.exists():
            target.unlink()
        target.symlink_to(os.path.relpath(source, target.parent))
    return mask_root


def _patch_database_and_write_input_model(
    database_path: Path,
    input_model: Path,
    frames: list[FrameCamera],
) -> dict[str, FrameCamera]:
    by_name = {item.image_name: item for item in frames}
    input_model.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(database_path) as connection:
        image_rows = connection.execute(
            "SELECT image_id, name, camera_id FROM images ORDER BY image_id"
        ).fetchall()
        database_names = {str(row[1]) for row in image_rows}
        if database_names != set(by_name):
            missing = sorted(set(by_name) - database_names)
            extra = sorted(database_names - set(by_name))
            raise RuntimeError(f"COLMAP database image mismatch; missing={missing[:5]}, extra={extra[:5]}")

        camera_ids = [int(row[2]) for row in image_rows]
        if len(set(camera_ids)) != len(camera_ids):
            raise RuntimeError("feature extraction did not create one camera per image")

        cameras_lines = [
            "# Camera list with one line of data per camera:",
            "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
            f"# Number of cameras: {len(image_rows)}",
        ]
        images_lines = [
            "# Image list with two lines of data per image:",
            "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
            "#   POINTS2D[] as (X, Y, POINT3D_ID)",
            f"# Number of images: {len(image_rows)}, mean observations per image: 0",
        ]

        for image_id, name, camera_id in image_rows:
            item = by_name[str(name)]
            params_blob = struct.pack("<" + "d" * COLMAP_OPENCV_NUM_PARAMS, *item.params)
            connection.execute(
                "UPDATE cameras SET model=?, width=?, height=?, params=?, prior_focal_length=1 "
                "WHERE camera_id=?",
                (
                    COLMAP_OPENCV_MODEL_ID,
                    item.width,
                    item.height,
                    params_blob,
                    int(camera_id),
                ),
            )

            params_text = " ".join(f"{value:.17g}" for value in item.params)
            cameras_lines.append(
                f"{int(camera_id)} OPENCV {item.width} {item.height} {params_text}"
            )

            qvec = rotmat2qvec(item.w2c[:3, :3])
            tvec = item.w2c[:3, 3]
            pose_text = " ".join(f"{float(value):.17g}" for value in (*qvec, *tvec))
            images_lines.extend(
                [f"{int(image_id)} {pose_text} {int(camera_id)} {name}", ""]
            )

        connection.commit()

        camera_count = connection.execute("SELECT COUNT(*) FROM cameras").fetchone()[0]
        parameter_sizes = connection.execute("SELECT length(params) FROM cameras").fetchall()
        if camera_count != len(image_rows) or any(row[0] != 8 * COLMAP_OPENCV_NUM_PARAMS for row in parameter_sizes):
            raise RuntimeError("failed to publish exact per-image camera intrinsics to COLMAP database")

    (input_model / "cameras.txt").write_text("\n".join(cameras_lines) + "\n", encoding="utf-8")
    (input_model / "images.txt").write_text("\n".join(images_lines) + "\n", encoding="utf-8")
    (input_model / "points3D.txt").write_text(
        "# 3D point list with one line of data per point:\n"
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n"
        "# Number of points: 0, mean track length: 0\n",
        encoding="utf-8",
    )
    return by_name


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def _pair_image_ids(pair_id: int) -> tuple[int, int]:
    max_image_id = 2_147_483_647
    image_id2 = pair_id % max_image_id
    image_id1 = (pair_id - image_id2) // max_image_id
    return image_id1, image_id2


def _replace_two_view_geometry_with_fixed_poses(
    database_path: Path,
    expected_by_name: dict[str, FrameCamera],
    max_error: float,
) -> tuple[int, int]:
    """Verify raw descriptor matches with the supplied camera geometry.

    COLMAP's default geometric verifier estimates a relative pose for every
    pair. That can reject correct matches when views are far apart or contain
    repeated synthetic textures. In a fixed-pose pipeline the supplied poses
    are the stronger source of truth, so use their fundamental matrices for a
    Sampson-error check and put those inlier matches back into COLMAP's
    ``two_view_geometries`` table. COLMAP still owns feature extraction,
    descriptor matching, track construction, and point triangulation.
    """
    if max_error <= 0:
        raise ValueError("fixed geometry max error must be positive")

    with sqlite3.connect(database_path) as connection:
        image_rows = connection.execute("SELECT image_id, name FROM images").fetchall()
        by_id = {int(image_id): expected_by_name[str(name)] for image_id, name in image_rows}
        keypoints: dict[int, np.ndarray] = {}
        empty_keypoint_images = 0
        for image_id, rows, cols, blob in connection.execute(
            "SELECT image_id, rows, cols, data FROM keypoints"
        ):
            row_count = int(rows)
            column_count = int(cols)
            if row_count == 0 or blob is None:
                # COLMAP stores a NULL blob for a successfully processed image
                # with zero detected features. This is expected for occasional
                # textureless synthetic views and must not abort triangulation
                # for all other fixed-pose cameras.
                keypoints[int(image_id)] = np.empty((0, 2), dtype=np.float64)
                empty_keypoint_images += 1
                continue
            if column_count < 2:
                raise RuntimeError(
                    f"invalid keypoint column count for image {image_id}: {column_count}"
                )
            points = np.frombuffer(blob, dtype=np.float32)
            expected_values = row_count * column_count
            if points.size != expected_values:
                raise RuntimeError(
                    f"keypoint blob size mismatch for image {image_id}: "
                    f"expected {expected_values}, got {points.size}"
                )
            keypoints[int(image_id)] = points.reshape(row_count, column_count)[:, :2].astype(
                np.float64
            )

        pair_count = 0
        inlier_count = 0
        match_rows = connection.execute(
            "SELECT pair_id, rows, cols, data FROM matches WHERE rows > 0 AND data IS NOT NULL"
        ).fetchall()
        # Clear geometries first so pairs rejected by the fixed poses cannot be
        # consumed accidentally by point_triangulator.
        connection.execute("DELETE FROM two_view_geometries")

        insert = (
            "INSERT INTO two_view_geometries "
            "(pair_id, rows, cols, data, config, F, E, H, qvec, tvec) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )
        for pair_id, rows, cols, blob in match_rows:
            image_id1, image_id2 = _pair_image_ids(int(pair_id))
            camera1 = by_id[image_id1]
            camera2 = by_id[image_id2]
            if len(keypoints.get(image_id1, ())) == 0 or len(keypoints.get(image_id2, ())) == 0:
                continue
            matches = np.frombuffer(blob, dtype=np.uint32).reshape(int(rows), int(cols))[:, :2]
            points1 = keypoints[image_id1][matches[:, 0]]
            points2 = keypoints[image_id2][matches[:, 1]]

            relative = camera2.w2c @ np.linalg.inv(camera1.w2c)
            rotation = relative[:3, :3]
            translation = relative[:3, 3]
            baseline = float(np.linalg.norm(translation))
            if baseline <= 1e-8:
                continue

            k1 = np.asarray(
                [
                    [camera1.params[0], 0.0, camera1.params[2]],
                    [0.0, camera1.params[1], camera1.params[3]],
                    [0.0, 0.0, 1.0],
                ]
            )
            k2 = np.asarray(
                [
                    [camera2.params[0], 0.0, camera2.params[2]],
                    [0.0, camera2.params[1], camera2.params[3]],
                    [0.0, 0.0, 1.0],
                ]
            )
            essential = _skew(translation / baseline) @ rotation
            fundamental = np.linalg.inv(k2).T @ essential @ np.linalg.inv(k1)

            homogeneous1 = np.column_stack([points1, np.ones(len(points1))])
            homogeneous2 = np.column_stack([points2, np.ones(len(points2))])
            lines2 = (fundamental @ homogeneous1.T).T
            lines1 = (fundamental.T @ homogeneous2.T).T
            residual = np.sum(homogeneous2 * lines2, axis=1)
            denominator = (
                lines1[:, 0] ** 2
                + lines1[:, 1] ** 2
                + lines2[:, 0] ** 2
                + lines2[:, 1] ** 2
            )
            sampson_error = residual**2 / np.maximum(denominator, 1e-12)
            keep = np.isfinite(sampson_error) & (sampson_error <= max_error**2)
            inliers = np.ascontiguousarray(matches[keep], dtype=np.uint32)
            if len(inliers) == 0:
                continue

            qvec = np.ascontiguousarray(rotmat2qvec(rotation), dtype=np.float64)
            tvec = np.ascontiguousarray(translation / baseline, dtype=np.float64)
            connection.execute(
                insert,
                (
                    int(pair_id),
                    len(inliers),
                    2,
                    inliers.tobytes(),
                    2,  # CALIBRATED
                    np.ascontiguousarray(fundamental, dtype=np.float64).tobytes(),
                    np.ascontiguousarray(essential, dtype=np.float64).tobytes(),
                    None,
                    qvec.tobytes(),
                    tvec.tobytes(),
                ),
            )
            pair_count += 1
            inlier_count += len(inliers)

        connection.commit()

    if pair_count == 0 or inlier_count == 0:
        raise RuntimeError("no descriptor matches are consistent with the supplied camera poses")
    print(
        f"Fixed-pose geometric verification: {inlier_count} matches in {pair_count} image pairs; "
        f"{empty_keypoint_images} image(s) had zero detected keypoints",
        flush=True,
    )
    return pair_count, inlier_count


def _validate_output_poses(
    sparse_dir: Path,
    expected_by_name: dict[str, FrameCamera],
    tolerance: float,
) -> tuple[float, float]:
    images = read_images_binary(sparse_dir / "images.bin")
    output_by_name = {image.name: image for image in images.values()}
    if set(output_by_name) != set(expected_by_name):
        missing = sorted(set(expected_by_name) - set(output_by_name))
        extra = sorted(set(output_by_name) - set(expected_by_name))
        raise RuntimeError(f"triangulated model image mismatch; missing={missing[:5]}, extra={extra[:5]}")

    max_rotation_error = 0.0
    max_translation_error = 0.0
    for name, expected in expected_by_name.items():
        actual = output_by_name[name]
        actual_rotation = qvec2rotmat(actual.qvec)
        # COLMAP stores rotations as normalized quaternions. Compare against
        # the exact rotation serialized into the input model, rather than the
        # source matrix directly: converted datasets can contain harmless
        # sub-micro numerical non-orthogonality that makes the trace/acos
        # angle formula report a much larger false error.
        expected_rotation = qvec2rotmat(rotmat2qvec(expected.w2c[:3, :3]))
        # Rotation angle is a clearer invariant than raw quaternion signs.
        delta = actual_rotation @ expected_rotation.T
        cosine = float(np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0))
        rotation_error = float(np.arccos(cosine))
        translation_error = float(np.linalg.norm(np.asarray(actual.tvec) - expected.w2c[:3, 3]))
        max_rotation_error = max(max_rotation_error, rotation_error)
        max_translation_error = max(max_translation_error, translation_error)

    if max_rotation_error > tolerance or max_translation_error > tolerance:
        raise RuntimeError(
            "COLMAP changed supplied camera poses: "
            f"max rotation={max_rotation_error:.3e} rad, "
            f"max translation={max_translation_error:.3e} m, tolerance={tolerance:.3e}"
        )
    return max_rotation_error, max_translation_error


def _publish_transforms(scene: Path, meta: dict[str, Any], ply_relative_path: str) -> None:
    transforms_path = scene / "transforms.json"
    updated = dict(meta)
    updated["ply_file_path"] = ply_relative_path
    # The PLY and saved OpenGL poses already share the same world frame. This
    # explicit identity prevents Nerfstudio from assuming its legacy COLMAP
    # axis swap merely because colmap/sparse/0 now exists.
    updated["applied_transform"] = IDENTITY_APPLIED_TRANSFORM
    temporary_path = transforms_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(updated, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_path, transforms_path)


def prepare(args: argparse.Namespace) -> None:
    scene = args.data.resolve()
    if not scene.is_dir():
        raise FileNotFoundError(scene)
    colmap = args.colmap.resolve()
    if not colmap.is_file():
        raise FileNotFoundError(f"COLMAP executable not found: {colmap}")

    work_dir = scene / "colmap"
    database_path = work_dir / "database.db"
    input_model = work_dir / "sparse_fixed_input"
    sparse_dir = work_dir / "sparse" / "0"
    ply_path = work_dir / "sparse_pc.ply"
    marker_path = work_dir / "fixed_pose_complete.json"
    ply_relative_path = ply_path.relative_to(scene).as_posix()

    meta, frames = _load_frames(scene)
    if (
        not args.overwrite
        and marker_path.is_file()
        and ply_path.is_file()
        and meta.get("ply_file_path") == ply_relative_path
    ):
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if meta.get("applied_transform") != IDENTITY_APPLIED_TRANSFORM:
            _publish_transforms(scene, meta, ply_relative_path)
        print(
            f"Already complete: {scene.name}, {marker.get('points3D', '?')} points, "
            f"poses fixed and verified.",
            flush=True,
        )
        return
    if work_dir.exists():
        if not args.overwrite:
            raise RuntimeError(
                f"incomplete COLMAP directory already exists: {work_dir}; "
                "inspect it, then rerun with --overwrite to rebuild this exact scene"
            )
        shutil.rmtree(work_dir)

    work_dir.mkdir(parents=True)
    sparse_dir.mkdir(parents=True)
    mask_root = _prepare_colmap_masks(scene, work_dir, frames)

    feature_command = [
        str(colmap),
        "feature_extractor",
        "--database_path",
        str(database_path),
        "--image_path",
        str(scene / "images"),
        "--ImageReader.camera_model",
        "OPENCV",
        "--ImageReader.single_camera_per_image",
        "1",
        "--FeatureExtraction.use_gpu",
        "1" if args.gpu else "0",
        "--FeatureExtraction.gpu_index",
        str(args.gpu_index),
        "--SiftExtraction.max_num_features",
        str(args.max_num_features),
    ]
    if mask_root is not None:
        feature_command.extend(["--ImageReader.mask_path", str(mask_root)])
    _run(feature_command)

    expected_by_name = _patch_database_and_write_input_model(database_path, input_model, frames)

    _run(
        [
            str(colmap),
            "exhaustive_matcher",
            "--database_path",
            str(database_path),
            "--FeatureMatching.use_gpu",
            "1" if args.gpu else "0",
            "--FeatureMatching.gpu_index",
            str(args.gpu_index),
            "--FeatureMatching.guided_matching",
            "0",
            "--SiftMatching.max_ratio",
            str(args.sift_max_ratio),
            "--SiftMatching.max_distance",
            str(args.sift_max_distance),
        ]
    )

    verified_pairs, verified_matches = _replace_two_view_geometry_with_fixed_poses(
        database_path,
        expected_by_name,
        args.fixed_geometry_max_error,
    )

    _run(
        [
            str(colmap),
            "point_triangulator",
            "--database_path",
            str(database_path),
            "--image_path",
            str(scene / "images"),
            "--input_path",
            str(input_model),
            "--output_path",
            str(sparse_dir),
            "--clear_points",
            "1",
            "--refine_intrinsics",
            "0",
            "--Mapper.fix_existing_frames",
            "1",
            "--Mapper.min_num_matches",
            "2",
            "--Mapper.ba_refine_focal_length",
            "0",
            "--Mapper.ba_refine_principal_point",
            "0",
            "--Mapper.ba_refine_extra_params",
            "0",
            "--Mapper.ba_refine_sensor_from_rig",
            "0",
            "--Mapper.tri_ignore_two_view_tracks",
            "0",
            "--Mapper.tri_min_angle",
            "0.5",
            "--Mapper.extract_colors",
            "1",
        ]
    )

    max_rotation_error, max_translation_error = _validate_output_poses(
        sparse_dir, expected_by_name, args.pose_tolerance
    )
    points = read_points3D_binary(sparse_dir / "points3D.bin")
    if len(points) < args.min_points:
        raise RuntimeError(
            f"only {len(points)} sparse points were triangulated; minimum is {args.min_points}"
        )

    _run(
        [
            str(colmap),
            "model_converter",
            "--input_path",
            str(sparse_dir),
            "--output_path",
            str(ply_path),
            "--output_type",
            "PLY",
        ]
    )
    if not ply_path.is_file() or ply_path.stat().st_size == 0:
        raise RuntimeError(f"COLMAP did not create a valid PLY: {ply_path}")

    marker = {
        "scene": scene.name,
        "pose_source": "transforms.json (fixed)",
        "point_source": "COLMAP feature matching + point_triangulator",
        "frames": len(frames),
        "fixed_pose_verified_pairs": verified_pairs,
        "fixed_pose_verified_matches": verified_matches,
        "points3D": len(points),
        "max_rotation_error_rad": max_rotation_error,
        "max_translation_error_m": max_translation_error,
        "ply_file_path": ply_relative_path,
    }
    marker_path.write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    _publish_transforms(scene, meta, ply_relative_path)
    print(json.dumps(marker, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Nerfstudio scene directory")
    parser.add_argument(
        "--colmap",
        type=Path,
        default=Path(__file__).resolve().parents[3]
        / ".conda"
        / "envs"
        / "aegir_colmap"
        / "bin"
        / "colmap",
        help="isolated COLMAP executable",
    )
    parser.add_argument("--gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gpu-index", default="0")
    parser.add_argument("--max-num-features", type=int, default=8192)
    parser.add_argument("--sift-max-ratio", type=float, default=0.9)
    parser.add_argument("--sift-max-distance", type=float, default=0.8)
    parser.add_argument("--fixed-geometry-max-error", type=float, default=4.0)
    parser.add_argument("--min-points", type=int, default=100)
    parser.add_argument("--pose-tolerance", type=float, default=1e-6)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        prepare(parse_args())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc
