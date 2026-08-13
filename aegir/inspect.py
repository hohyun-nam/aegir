from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from .checkpoint import emitter_state_dict, load_checkpoint
from .config import config_from_dict
from .data import (
    MultiViewSceneDataset,
    infer_dataparser_transform_path,
    load_dataparser_transform,
    saved_pose_to_model_transform,
    transform_camera,
)
from .emitter_inspection import create_emitter_evolution_outputs
from .gaussian_scene import GaussianMaterialBackend
from .irgs_tracer import IRGSVisibilityTracer
from .render import DeferredRenderer
from .trainer import initialize_emitters
from .visibility import scene_scale_from_bounds


def _camera_to(camera: dict, device: str) -> dict:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in camera.items()}


def _select_flat(gbuffer: dict[str, Tensor], ids: Tensor) -> dict[str, Tensor]:
    selected = {}
    height, width = gbuffer["points"].shape[:2]
    for key, value in gbuffer.items():
        if value.ndim >= 2 and value.shape[:2] == (height, width):
            selected[key] = value.reshape(-1, value.shape[-1])[ids]
    return selected


@torch.inference_mode()
def render_full_frame(
    scene,
    emitters,
    camera,
    config,
    chunk_size: int = 256,
    visibility_tracer: IRGSVisibilityTracer | None = None,
) -> dict[str, Tensor]:
    gbuffer = scene.rasterize(
        camera,
        alpha_threshold=config.raster.alpha_threshold,
        rasterize_mode=config.raster.rasterize_mode,
    )
    height, width = gbuffer["points"].shape[:2]
    scene_scale = scene_scale_from_bounds(scene.bounds(config.raster.bounds_quantile))
    if visibility_tracer is None and config.render.point_visibility:
        visibility_tracer = IRGSVisibilityTracer(
            scene, config.render, scene_scale=scene_scale
        )
    elif visibility_tracer is not None:
        scene_scale = visibility_tracer.scene_scale
    renderer = DeferredRenderer(
        config.render,
        visibility_fn=visibility_tracer.visibility if visibility_tracer is not None else None,
        scene_scale=scene_scale,
    )
    outputs: dict[str, list[Tensor]] = {}
    proxy_points, proxy_radiance = scene.bounce_proxy(config.render.bounce_proxy_points)
    for start in range(0, height * width, chunk_size):
        ids = torch.arange(start, min(start + chunk_size, height * width), device=scene.means.device)
        chunk = _select_flat(gbuffer, ids)
        if config.render.indirect_bounce:
            chunk["bounce_points"] = proxy_points
            chunk["bounce_radiance"] = proxy_radiance
            chunk["bounce_radius"] = scene_scale * config.render.bounce_radius_fraction
            chunk["bounce_max_distance"] = scene_scale * config.render.bounce_max_distance_fraction
        rendered = renderer.render(chunk, emitters)
        for key, value in rendered.items():
            outputs.setdefault(key, []).append(value.reshape(-1, value.shape[-1]).cpu())
    return {key: torch.cat(chunks, dim=0).reshape(height, width, -1) for key, chunks in outputs.items()} | {
        "depth": gbuffer["depth"].cpu(),
        "alpha": gbuffer["alpha"].cpu(),
    }


def _to_u8(value: Tensor) -> np.ndarray:
    value = value.detach().float().cpu().clamp(0.0, 1.0)
    if value.shape[-1] == 1:
        value = value.expand(*value.shape[:-1], 3)
    return (value.numpy() * 255.0 + 0.5).astype(np.uint8)


def _normal_u8(value: Tensor) -> np.ndarray:
    return _to_u8(value * 0.5 + 0.5)


def _depth_u8(value: Tensor) -> np.ndarray:
    depth = value[..., 0].float()
    valid = torch.isfinite(depth) & (depth > 0)
    result = torch.zeros_like(depth)
    if bool(valid.any()):
        lo = torch.quantile(depth[valid], 0.02)
        hi = torch.quantile(depth[valid], 0.98)
        result[valid] = ((depth[valid] - lo) / (hi - lo).clamp_min(1.0e-6)).clamp(0.0, 1.0)
    return _to_u8(result[..., None])


def inspect_checkpoint(
    checkpoint: str | Path,
    nerfstudio_dir: str | Path,
    prior_dir: str | Path | None,
    output_dir: str | Path,
    device: str,
    max_views: int | None = None,
    render_samples: int = 16,
    split: str = "test",
) -> None:
    scene, emitters, payload = load_checkpoint(checkpoint, device)
    config = config_from_dict(payload["config"])
    config.render.num_brdf_samples = render_samples
    config.render.num_light_samples = render_samples
    dataset = MultiViewSceneDataset(
        nerfstudio_dir,
        payload["source_dnsplatter_checkpoint"],
        prior_dir,
        image_size=config.raster.image_size,
        max_views=max_views,
        split=split,
    )
    output_dir = Path(output_dir)
    compare_dir = output_dir / "target_render_compare_frames"
    compare_dir.mkdir(parents=True, exist_ok=True)
    metrics = []
    scene_scale = scene_scale_from_bounds(scene.bounds(config.raster.bounds_quantile))
    visibility_tracer = (
        IRGSVisibilityTracer(scene, config.render, scene_scale=scene_scale)
        if config.render.point_visibility
        else None
    )
    for frame in dataset.frames:
        rendered = render_full_frame(
            scene,
            emitters,
            _camera_to(frame.camera, device),
            config,
            visibility_tracer=visibility_tracer,
        )
        target = frame.target
        squared_error = (rendered["rgb"] - target).square()
        valid = frame.mask.to(dtype=squared_error.dtype)
        valid_values = valid.sum() * squared_error.shape[-1]
        if not bool(valid_values > 0):
            raise ValueError(f"frame {frame.index} has no valid pixels")
        mse = (squared_error * valid).sum().div(valid_values).clamp_min(1.0e-12)
        psnr = float((-10.0 * torch.log10(mse)).cpu())
        metrics.append({"frame": frame.index, "psnr": psnr})
        target_image = _to_u8(target)
        render_image = _to_u8(rendered["rgb"])
        comparison = np.concatenate((target_image, render_image), axis=1)
        Image.fromarray(comparison).save(compare_dir / f"frame_{frame.index:04d}.png")
        if len(metrics) == 1:
            Image.fromarray(render_image).save(output_dir / "render_rgb.png")
            Image.fromarray(target_image).save(output_dir / "target_rgb.png")
            Image.fromarray(_to_u8(rendered["albedo"])).save(output_dir / "albedo.png")
            Image.fromarray(_to_u8(rendered["roughness"])).save(output_dir / "roughness.png")
            Image.fromarray(_to_u8(rendered["metallic"])).save(output_dir / "metallic.png")
            Image.fromarray(_normal_u8(rendered["normals"])).save(output_dir / "normals.png")
            Image.fromarray(_depth_u8(rendered["depth"])).save(output_dir / "depth.png")
    summary = {
        "checkpoint": str(Path(checkpoint).resolve()),
        "views": len(metrics),
        "split": split,
        "mean_psnr": sum(item["psnr"] for item in metrics) / len(metrics),
        "metrics": metrics,
    }
    initial_emitter_state = payload.get("initial_emitter_state")
    if initial_emitter_state is not None:
        create_emitter_evolution_outputs(
            initial_emitter_state,
            emitter_state_dict(emitters),
            dataset.frames,
            scene.means,
            scene.bounds(config.raster.bounds_quantile),
            output_dir,
        )
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "metrics"}, indent=2))


@torch.no_grad()
def inspect_emitter_evolution(
    checkpoint: str | Path,
    nerfstudio_dir: str | Path,
    prior_dir: str | Path | None,
    output_dir: str | Path,
    device: str,
    max_views: int | None = None,
    split: str = "all",
) -> None:
    scene, emitters, payload = load_checkpoint(checkpoint, device)
    config = config_from_dict(payload["config"])
    dataset = MultiViewSceneDataset(
        nerfstudio_dir,
        payload["source_dnsplatter_checkpoint"],
        prior_dir,
        image_size=config.raster.image_size,
        max_views=max_views,
        split=split,
    )
    initial_state = payload.get("initial_emitter_state")
    reconstructed = initial_state is None
    if reconstructed:
        initial_scene = GaussianMaterialBackend(
            payload["source_dnsplatter_checkpoint"],
            opacity_threshold=config.raster.opacity_threshold,
        ).to(device)
        bounds = tuple(value.to(device) for value in initial_scene.bounds(config.raster.bounds_quantile))
        initial_emitters = initialize_emitters(initial_scene, dataset, bounds, config, device)
        initial_scene.sanitize_parameters(bounds)
        initial_emitters.sanitize_parameters(bounds)
        initial_state = emitter_state_dict(initial_emitters)
    output_dir = Path(output_dir)
    create_emitter_evolution_outputs(
        initial_state,
        emitter_state_dict(emitters),
        dataset.frames,
        scene.means,
        scene.bounds(config.raster.bounds_quantile),
        output_dir,
    )
    result = {
        "checkpoint": str(Path(checkpoint).resolve()),
        "initial_state": "reconstructed" if reconstructed else "stored in checkpoint",
        "initial_emitters": int(initial_state["centers"].shape[0]),
        "final_emitters": emitters.num_emitters,
        "output_dir": str(output_dir.resolve()),
    }
    print(json.dumps(result, indent=2))


def load_camera_path(
    transforms_path: str | Path,
    dnsplatter_checkpoint: str | Path,
    image_size: int,
) -> list[dict]:
    transforms_path = Path(transforms_path)
    metadata = json.loads(transforms_path.read_text(encoding="utf-8"))
    frames = metadata.get("frames", [])
    if not frames:
        raise ValueError(f"{transforms_path} has no frames")
    dataparser = load_dataparser_transform(infer_dataparser_transform_path(Path(dnsplatter_checkpoint)))
    saved_to_model = saved_pose_to_model_transform(dataparser, metadata)
    source_width = float(metadata.get("w", image_size))
    source_height = float(metadata.get("h", image_size))
    cameras = []
    for frame in frames:
        fx = float(frame.get("fl_x", metadata.get("fl_x")))
        fy = float(frame.get("fl_y", metadata.get("fl_y", fx)))
        cx = float(frame.get("cx", metadata.get("cx", source_width * 0.5)))
        cy = float(frame.get("cy", metadata.get("cy", source_height * 0.5)))
        camera = {
            "c2w": torch.as_tensor(frame["transform_matrix"], dtype=torch.float32),
            "fx": fx * image_size / source_width,
            "fy": fy * image_size / source_height,
            "cx": cx * image_size / source_width,
            "cy": cy * image_size / source_height,
            "width": image_size,
            "height": image_size,
            "camera_convention": "opengl",
        }
        cameras.append(transform_camera(camera, saved_to_model))
    return cameras


def render_camera_path(
    checkpoint: str | Path,
    transforms_path: str | Path,
    output_dir: str | Path,
    device: str,
    max_frames: int | None = None,
    render_samples: int = 16,
    light_scale: float = 1.0,
) -> None:
    scene, emitters, payload = load_checkpoint(checkpoint, device)
    config = config_from_dict(payload["config"])
    config.render.num_brdf_samples = render_samples
    config.render.num_light_samples = render_samples
    cameras = load_camera_path(
        transforms_path,
        payload["source_dnsplatter_checkpoint"],
        config.raster.image_size,
    )
    if max_frames is not None:
        cameras = cameras[:max_frames]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        emitters.log_colors.add_(float(np.log(max(light_scale, 1.0e-8))))
    scene_scale = scene_scale_from_bounds(scene.bounds(config.raster.bounds_quantile))
    visibility_tracer = (
        IRGSVisibilityTracer(scene, config.render, scene_scale=scene_scale)
        if config.render.point_visibility
        else None
    )
    for index, camera in enumerate(cameras):
        rendered = render_full_frame(
            scene,
            emitters,
            _camera_to(camera, device),
            config,
            visibility_tracer=visibility_tracer,
        )
        Image.fromarray(_to_u8(rendered["rgb"])).save(output_dir / f"frame_{index:04d}.png")
    print(f"rendered {len(cameras)} novel/path views to {output_dir}")
