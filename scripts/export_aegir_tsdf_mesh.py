"""Extract a diagnostic triangle mesh from an AEGIR shared-Gaussian checkpoint.

The learned 2D Gaussians are rasterized into depth/albedo maps from the dataset
cameras. Open3D then fuses those maps into a TSDF volume. Both the unfiltered
mesh (useful for spotting floaters) and a connected-component-filtered mesh are
written so geometry failures are not hidden by post-processing.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import open3d as o3d
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aegir.checkpoint import load_checkpoint
from aegir.config import config_from_dict
from aegir.inspect import load_camera_path


def uniformly_spaced_indices(length: int, count: int | None) -> list[int]:
    if length <= 0:
        raise ValueError("camera path has no frames")
    if count is None or count >= length:
        return list(range(length))
    if count <= 0:
        raise ValueError("--max-views must be positive")
    if count == 1:
        return [0]
    return [round(i * (length - 1) / (count - 1)) for i in range(count)]


def camera_to_open3d(camera: dict) -> tuple[o3d.camera.PinholeCameraIntrinsic, np.ndarray]:
    width = int(camera["width"])
    height = int(camera["height"])
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width,
        height,
        float(camera["fx"]),
        float(camera["fy"]),
        float(camera["cx"]),
        float(camera["cy"]),
    )
    c2w_opengl = camera["c2w"].detach().cpu().numpy().astype(np.float64)
    # AEGIR cameras use OpenGL (+x right, +y up, -z forward), whereas Open3D
    # expects OpenCV (+x right, +y down, +z forward).
    cv_to_gl = np.diag([1.0, -1.0, -1.0, 1.0])
    c2w_opencv = c2w_opengl @ cv_to_gl
    return intrinsic, np.linalg.inv(c2w_opencv)


def automatic_depth_truncation(
    cameras: list[dict], bounds_min: np.ndarray, bounds_max: np.ndarray
) -> float:
    corners = np.asarray(list(product(*zip(bounds_min, bounds_max))), dtype=np.float64)
    farthest = 0.0
    for camera in cameras:
        center = camera["c2w"][:3, 3].detach().cpu().numpy()
        farthest = max(farthest, float(np.linalg.norm(corners - center, axis=1).max()))
    return max(farthest * 1.05, 1.0e-3)


def filter_components(
    mesh: o3d.geometry.TriangleMesh,
    keep_components: int,
    min_component_triangles: int,
) -> o3d.geometry.TriangleMesh:
    result = copy.deepcopy(mesh)
    if len(result.triangles) == 0:
        return result
    labels, counts, _ = result.cluster_connected_triangles()
    labels = np.asarray(labels)
    counts = np.asarray(counts)
    order = np.argsort(counts)[::-1]
    component_ids = [
        int(component)
        for component in order[:keep_components]
        if counts[component] >= min_component_triangles
    ]
    if not component_ids:
        component_ids = [int(order[0])]
    result.remove_triangles_by_mask(~np.isin(labels, component_ids))
    result.remove_unreferenced_vertices()
    result.remove_degenerate_triangles()
    result.remove_duplicated_triangles()
    result.remove_duplicated_vertices()
    result.remove_non_manifold_edges()
    result.compute_vertex_normals()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rasterize AEGIR depths and fuse them into diagnostic raw/clean TSDF meshes."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--transforms", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="filtered output .ply path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--max-views",
        type=int,
        default=None,
        help="uniformly sample this many cameras; default uses every camera",
    )
    parser.add_argument(
        "--mesh-resolution",
        type=int,
        default=512,
        help="scene-bounds diagonal divided by this value gives the automatic voxel size",
    )
    parser.add_argument("--voxel-size", type=float, default=None)
    parser.add_argument("--sdf-trunc", type=float, default=None)
    parser.add_argument("--depth-trunc", type=float, default=None)
    parser.add_argument(
        "--min-alpha",
        type=float,
        default=0.5,
        help="discard uncertain rasterized depths below this accumulated opacity",
    )
    parser.add_argument("--keep-components", type=int, default=20)
    parser.add_argument("--min-component-triangles", type=int, default=100)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.mesh_resolution <= 0:
        raise ValueError("--mesh-resolution must be positive")
    if not 0.0 <= args.min_alpha <= 1.0:
        raise ValueError("--min-alpha must be in [0, 1]")
    if args.keep_components <= 0 or args.min_component_triangles < 0:
        raise ValueError("component filter arguments must be non-negative")

    scene, _, payload = load_checkpoint(args.checkpoint, args.device)
    config = config_from_dict(payload["config"])
    all_cameras = load_camera_path(
        args.transforms,
        payload["source_dnsplatter_checkpoint"],
        config.raster.image_size,
    )
    selected_indices = uniformly_spaced_indices(len(all_cameras), args.max_views)
    cameras = [all_cameras[index] for index in selected_indices]

    bounds = scene.bounds(config.raster.bounds_quantile)
    bounds_min = bounds[0].detach().cpu().numpy().astype(np.float64)
    bounds_max = bounds[1].detach().cpu().numpy().astype(np.float64)
    scene_extent = float(np.linalg.norm(bounds_max - bounds_min))
    voxel_size = args.voxel_size or scene_extent / args.mesh_resolution
    sdf_trunc = args.sdf_trunc or 5.0 * voxel_size
    depth_trunc = args.depth_trunc or automatic_depth_truncation(
        cameras, bounds_min, bounds_max
    )
    if min(voxel_size, sdf_trunc, depth_trunc) <= 0.0:
        raise ValueError("voxel size and truncation distances must be positive")

    print(
        f"TSDF views={len(cameras)}/{len(all_cameras)} image_size={config.raster.image_size} "
        f"voxel={voxel_size:.6g} sdf_trunc={sdf_trunc:.6g} depth_trunc={depth_trunc:.6g}",
        flush=True,
    )
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    for position, (frame_index, camera) in enumerate(zip(selected_indices, cameras), start=1):
        camera_device = {
            key: value.to(args.device) if torch.is_tensor(value) else value
            for key, value in camera.items()
        }
        gbuffer = scene.rasterize(
            camera_device,
            alpha_threshold=max(config.raster.alpha_threshold, args.min_alpha),
            rasterize_mode=config.raster.rasterize_mode,
        )
        depth = gbuffer["depth"][..., 0].detach().float().cpu().numpy()
        alpha = gbuffer["alpha"][..., 0].detach().float().cpu().numpy()
        valid = np.isfinite(depth) & (depth > 0.0) & (alpha >= args.min_alpha)
        depth = np.where(valid, depth, 0.0).astype(np.float32)
        albedo = gbuffer["albedo"].detach().float().cpu().numpy()
        albedo = np.nan_to_num(albedo, nan=0.0, posinf=1.0, neginf=0.0)
        color = np.asarray(np.clip(albedo, 0.0, 1.0) * 255.0 + 0.5, dtype=np.uint8)
        intrinsic, extrinsic = camera_to_open3d(camera)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(color)),
            o3d.geometry.Image(np.ascontiguousarray(depth)),
            depth_scale=1.0,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False,
        )
        volume.integrate(rgbd, intrinsic, extrinsic)
        print(
            f"[{position:03d}/{len(cameras):03d}] frame={frame_index:04d} "
            f"valid_depth={valid.mean():.1%}",
            flush=True,
        )

    raw_mesh = volume.extract_triangle_mesh()
    raw_mesh.compute_vertex_normals()
    if len(raw_mesh.vertices) == 0 or len(raw_mesh.triangles) == 0:
        raise RuntimeError(
            "TSDF produced an empty mesh; lower --min-alpha or adjust voxel/depth truncation"
        )
    clean_mesh = filter_components(
        raw_mesh,
        args.keep_components,
        args.min_component_triangles,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw_output = args.output.with_name(f"{args.output.stem}_raw{args.output.suffix}")
    if not o3d.io.write_triangle_mesh(str(raw_output), raw_mesh, write_ascii=False):
        raise RuntimeError(f"failed to write {raw_output}")
    if not o3d.io.write_triangle_mesh(str(args.output), clean_mesh, write_ascii=False):
        raise RuntimeError(f"failed to write {args.output}")

    metadata = {
        "checkpoint": str(args.checkpoint.resolve()),
        "transforms": str(args.transforms.resolve()),
        "raw_mesh": str(raw_output.resolve()),
        "filtered_mesh": str(args.output.resolve()),
        "total_views": len(all_cameras),
        "selected_frame_indices": selected_indices,
        "image_size": config.raster.image_size,
        "min_alpha": args.min_alpha,
        "bounds_min": bounds_min.tolist(),
        "bounds_max": bounds_max.tolist(),
        "voxel_size": voxel_size,
        "sdf_trunc": sdf_trunc,
        "depth_trunc": depth_trunc,
        "raw_vertices": len(raw_mesh.vertices),
        "raw_triangles": len(raw_mesh.triangles),
        "filtered_vertices": len(clean_mesh.vertices),
        "filtered_triangles": len(clean_mesh.triangles),
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"raw mesh:      {raw_output}")
    print(f"filtered mesh: {args.output}")
    print(f"metadata:      {metadata_path}")


if __name__ == "__main__":
    main()
