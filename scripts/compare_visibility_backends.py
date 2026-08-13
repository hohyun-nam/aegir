from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aegir.checkpoint import load_checkpoint
from aegir.config import config_from_dict
from aegir.irgs_tracer import IRGSVisibilityTracer
from aegir.render import DeferredRenderer
from aegir.visibility import scene_scale_from_bounds


def _project_points(points: torch.Tensor, panel_size: int) -> np.ndarray:
    points = points.detach().float().cpu()
    centered = points - points.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    xy = centered @ vh[:2].T
    low = torch.quantile(xy, 0.01, dim=0)
    high = torch.quantile(xy, 0.99, dim=0)
    xy = ((xy - low) / (high - low).clamp_min(1.0e-6)).clamp(0.0, 1.0)
    margin = 14
    xy = xy * (panel_size - 2 * margin) + margin
    xy[:, 1] = panel_size - xy[:, 1]
    return xy.numpy()


def _visibility_color(value: float) -> tuple[int, int, int]:
    value = float(np.clip(value, 0.0, 1.0))
    return int(235 * (1.0 - value)), int(205 * value), int(255 * value)


def _difference_color(value: float) -> tuple[int, int, int]:
    value = float(np.clip(value, 0.0, 1.0))
    return int(255 * value), int(55 * (1.0 - value)), int(130 * (1.0 - value))


def _save_surface_map(
    path: Path,
    points: torch.Tensor,
    legacy: torch.Tensor,
    irgs: torch.Tensor,
) -> None:
    panel_size = 380
    header = 44
    gap = 10
    image = Image.new("RGB", (panel_size * 3 + gap * 2, panel_size + header), "white")
    draw = ImageDraw.Draw(image)
    xy = _project_points(points, panel_size)
    values = (
        ("Legacy point proxy", legacy, _visibility_color),
        ("IRGS OptiX", irgs, _visibility_color),
        ("Absolute difference", (legacy - irgs).abs(), _difference_color),
    )
    for panel, (title, panel_values, color_fn) in enumerate(values):
        left = panel * (panel_size + gap)
        draw.text((left + 8, 8), title, fill="black")
        draw.text((left + 8, 24), f"mean={float(panel_values.mean()):.4f}", fill="black")
        for (x, y), value in zip(xy, panel_values.detach().cpu().numpy(), strict=True):
            x = float(x) + left
            y = float(y) + header
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color_fn(float(value)))
    image.save(path)


@torch.inference_mode()
def compare(args: argparse.Namespace) -> dict[str, float | int | str]:
    scene, emitters, payload = load_checkpoint(args.checkpoint, args.device)
    config = config_from_dict(payload["config"])
    config.render.visibility_occluders = args.legacy_occluders
    points = scene.visibility_points(args.num_surface_points)
    generator = torch.Generator(device=points.device).manual_seed(args.seed)
    _, directions, distances, _ = emitters.sample_surface(points, 1, generator=generator)

    scene_scale = scene_scale_from_bounds(scene.bounds(config.raster.bounds_quantile))
    tracer = IRGSVisibilityTracer(scene, config.render, scene_scale=scene_scale)
    irgs = tracer.visibility(points[:, None].expand_as(directions), directions, distances)
    legacy_renderer = DeferredRenderer(config.render, scene_scale=scene_scale)
    legacy_points = scene.visibility_points(args.legacy_occluders)
    legacy = legacy_renderer._pointcloud_visibility(points, directions, distances, legacy_points)

    legacy_per_point = legacy.mean(dim=1).squeeze(-1)
    irgs_per_point = irgs.mean(dim=1).squeeze(-1)
    difference = (legacy - irgs).abs()
    summary: dict[str, float | int | str] = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "gaussians": scene.gaussian_count,
        "emitters": emitters.num_emitters,
        "surface_points": int(points.shape[0]),
        "shadow_rays": int(directions.numel() // 3),
        "legacy_occluders": int(legacy_points.shape[0]),
        "legacy_visibility_mean": float(legacy.mean()),
        "irgs_visibility_mean": float(irgs.mean()),
        "mean_absolute_difference": float(difference.mean()),
        "p95_absolute_difference": float(torch.quantile(difference, 0.95)),
        "fraction_difference_over_0_1": float((difference > 0.1).float().mean()),
        "binary_shadow_disagreement": float(((legacy < 0.5) != (irgs < 0.5)).float().mean()),
        "irgs_soft_visibility_fraction": float(((irgs > 0.05) & (irgs < 0.95)).float().mean()),
    }

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(
        output / "visibility_samples.npz",
        points=points.detach().cpu().numpy(),
        directions=directions.detach().cpu().numpy(),
        distances=distances.detach().cpu().numpy(),
        legacy_visibility=legacy.detach().cpu().numpy(),
        irgs_visibility=irgs.detach().cpu().numpy(),
    )
    _save_surface_map(
        output / "visibility_surface_map.png",
        points,
        legacy_per_point,
        irgs_per_point,
    )
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare legacy point shadows against IRGS OptiX")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-surface-points", type=int, default=1024)
    parser.add_argument("--legacy-occluders", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    compare(parser.parse_args())


if __name__ == "__main__":
    main()
