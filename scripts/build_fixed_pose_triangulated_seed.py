#!/usr/bin/env python3
"""Filter a fixed-pose COLMAP triangulation into a metric 2DGS seed.

Only RGB feature tracks and the supplied camera poses are used.  Monocular or
ground-truth depth is never read.  Points with weak tracks, large COLMAP
reprojection error, low parallax, or implausible camera distance are removed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from nerfstudio.data.utils.colmap_parsing_utils import (
    qvec2rotmat,
    read_images_binary,
    read_points3D_binary,
)

from aegir.provenance import artifact_request, publish_artifact, sha256_file, verify_cache


def maximum_triangulation_angle_deg(point: np.ndarray, centers: np.ndarray) -> float:
    rays = centers - point[None, :]
    lengths = np.linalg.norm(rays, axis=1)
    if len(rays) < 2 or np.any(lengths <= 1.0e-8):
        return 0.0
    rays /= lengths[:, None]
    minimum_cosine = float(np.min(rays @ rays.T))
    return float(np.degrees(np.arccos(np.clip(minimum_cosine, -1.0, 1.0))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-track-length", type=int, default=2)
    parser.add_argument("--max-reprojection-error", type=float, default=2.0)
    parser.add_argument("--min-triangulation-angle-deg", type=float, default=2.0)
    parser.add_argument("--min-camera-distance", type=float, default=0.1)
    parser.add_argument("--max-camera-distance", type=float, default=8.0)
    parser.add_argument("--statistical-nb-neighbors", type=int, default=20)
    parser.add_argument("--statistical-std-ratio", type=float, default=1.0)
    args = parser.parse_args()

    data = args.data.resolve()
    output = args.output.resolve()
    model = data / "colmap" / "sparse" / "0"
    marker_path = data / "colmap" / "fixed_pose_complete.json"
    transforms_path = data / "transforms.json"
    required = [
        marker_path,
        transforms_path,
        model / "cameras.bin",
        model / "images.bin",
        model / "points3D.bin",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.min_track_length < 2:
        raise ValueError("--min-track-length must be at least 2")
    if args.max_reprojection_error <= 0 or args.min_triangulation_angle_deg <= 0:
        raise ValueError("reprojection and triangulation-angle thresholds must be positive")
    if not 0 <= args.min_camera_distance < args.max_camera_distance:
        raise ValueError("camera-distance limits must satisfy 0 <= min < max")
    if args.statistical_nb_neighbors < 0 or args.statistical_std_ratio <= 0:
        raise ValueError("statistical filtering requires neighbors >= 0 and std ratio > 0")

    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if (
        marker.get("pose_source") != "transforms.json (fixed)"
        or marker.get("point_source") != "COLMAP feature matching + point_triangulator"
        or float(marker.get("max_translation_error_m", np.inf)) > 1.0e-5
        or float(marker.get("max_rotation_error_rad", np.inf)) > 1.0e-5
    ):
        raise ValueError("COLMAP model is not certified fixed-pose triangulation")

    settings = {
        "poses_fixed": True,
        "uses_gt_depth": False,
        "exclude_emitter_masks": False,
        "min_track_length": args.min_track_length,
        "max_reprojection_error": args.max_reprojection_error,
        "min_triangulation_angle_deg": args.min_triangulation_angle_deg,
        "min_camera_distance": args.min_camera_distance,
        "max_camera_distance": args.max_camera_distance,
        "statistical_nb_neighbors": args.statistical_nb_neighbors,
        "statistical_std_ratio": args.statistical_std_ratio,
    }
    inputs = {str(path.relative_to(data)): sha256_file(path) for path in required}
    request = artifact_request("fixed_pose_triangulated_seed", settings, inputs)
    manifest_path = output.with_suffix(output.suffix + ".provenance.json")
    if verify_cache(manifest_path, request, output):
        print(f"Verified triangulated seed cache: {output}")
        return

    images = read_images_binary(model / "images.bin")
    points = read_points3D_binary(model / "points3D.bin")
    centers = {
        image_id: -qvec2rotmat(image.qvec).T @ np.asarray(image.tvec, dtype=np.float64)
        for image_id, image in images.items()
    }

    xyz: list[np.ndarray] = []
    rgb: list[np.ndarray] = []
    rejected = {"track": 0, "error": 0, "camera": 0, "distance": 0, "angle": 0}
    angles: list[float] = []
    for point_id in sorted(points):
        point = points[point_id]
        position = np.asarray(point.xyz, dtype=np.float64)
        if len(point.image_ids) < args.min_track_length:
            rejected["track"] += 1
            continue
        if not np.isfinite(point.error) or point.error > args.max_reprojection_error:
            rejected["error"] += 1
            continue
        try:
            observed_centers = np.stack([centers[int(image_id)] for image_id in point.image_ids])
        except KeyError:
            rejected["camera"] += 1
            continue
        distances = np.linalg.norm(observed_centers - position[None, :], axis=1)
        nearest = float(np.min(distances))
        if (
            not np.isfinite(position).all()
            or not np.isfinite(distances).all()
            or nearest < args.min_camera_distance
            or nearest > args.max_camera_distance
        ):
            rejected["distance"] += 1
            continue
        angle = maximum_triangulation_angle_deg(position, observed_centers)
        if angle < args.min_triangulation_angle_deg:
            rejected["angle"] += 1
            continue
        xyz.append(position)
        rgb.append(np.asarray(point.rgb, dtype=np.float64) / 255.0)
        angles.append(angle)

    if len(xyz) < 100:
        raise RuntimeError(f"only {len(xyz)} verified triangulated points survived")
    xyz_array = np.stack(xyz)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(xyz_array)
    cloud.colors = o3d.utility.Vector3dVector(np.stack(rgb))
    before_statistical = len(cloud.points)
    if (
        args.statistical_nb_neighbors > 0
        and before_statistical > args.statistical_nb_neighbors
    ):
        cloud, _ = cloud.remove_statistical_outlier(
            nb_neighbors=args.statistical_nb_neighbors,
            std_ratio=args.statistical_std_ratio,
        )
    xyz_array = np.asarray(cloud.points)
    if len(xyz_array) < 100:
        raise RuntimeError(
            f"only {len(xyz_array)} verified points survived statistical filtering"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(output), cloud, write_ascii=False, compressed=False):
        raise RuntimeError(f"failed to write {output}")
    publish_artifact(manifest_path, request, output)

    print(f"COLMAP points: {len(points)}")
    print(f"verified points: {before_statistical} -> statistical {len(xyz_array)}")
    print(f"rejected: {json.dumps(rejected, sort_keys=True)}")
    print(f"triangulation angle median: {np.median(angles):.3f} deg")
    print(f"bounds min: {xyz_array.min(axis=0).tolist()}")
    print(f"bounds max: {xyz_array.max(axis=0).tolist()}")
    print(f"wrote: {output}")


if __name__ == "__main__":
    main()
