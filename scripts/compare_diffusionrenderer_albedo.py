#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from aegir.checkpoint import load_checkpoint
from aegir.data import resize_map, resize_mask
from aegir.inspect import load_camera_path
from aegir.math_utils import linear_to_srgb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare DiffusionRenderer albedo priors with two AEGIR checkpoints."
    )
    parser.add_argument("--checkpoint-a", type=Path, required=True)
    parser.add_argument("--checkpoint-b", type=Path, required=True)
    parser.add_argument("--label-a", default="MODEL A")
    parser.add_argument("--label-b", default="MODEL B")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--prior-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def to_u8_srgb(value: torch.Tensor) -> np.ndarray:
    value = linear_to_srgb(value.detach().float().cpu())
    return (value.clamp(0, 1).numpy() * 255.0 + 0.5).astype(np.uint8)


def load_rgb(path: Path, image_size: int) -> torch.Tensor:
    value = torch.from_numpy(np.asarray(Image.open(path).convert("RGB")).copy()).float() / 255.0
    return resize_map(value, (image_size, image_size))


def load_mask(path: Path, image_size: int) -> torch.Tensor:
    value = torch.from_numpy(np.asarray(Image.open(path).convert("L")).copy()) > 127
    return resize_mask(value, (image_size, image_size))[..., 0]


def make_panel(images: list[np.ndarray], labels: list[str], frame_index: int) -> Image.Image:
    height, width = images[0].shape[:2]
    header = 44
    canvas = Image.new("RGB", (width * len(images), height + header), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for column, (array, label) in enumerate(zip(images, labels)):
        canvas.paste(Image.fromarray(array), (column * width, header))
        draw.text((column * width + 8, 8), label, fill="black", font=font)
    draw.text((8, 25), f"frame {frame_index:04d}", fill=(80, 80, 80), font=font)
    return canvas


def make_contact_sheet(paths: list[Path], output: Path, columns: int = 3, thumb_width: int = 640) -> None:
    if not paths:
        return
    thumbs = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        thumb_height = round(image.height * thumb_width / image.width)
        thumbs.append(image.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS))
    rows = (len(thumbs) + columns - 1) // columns
    sheet = Image.new("RGB", (thumb_width * columns, thumbs[0].height * rows), "white")
    for index, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((index % columns) * thumb_width, (index // columns) * thumb.height))
    sheet.save(output, quality=92)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frames_dir = args.output / "frames"
    frames_dir.mkdir(exist_ok=True)

    transforms_path = args.dataset / "transforms.json"
    metadata = json.loads(transforms_path.read_text(encoding="utf-8"))
    frames = metadata["frames"]
    scene_a, _, payload_a = load_checkpoint(args.checkpoint_a, args.device)
    scene_b, _, payload_b = load_checkpoint(args.checkpoint_b, args.device)
    cameras_a = load_camera_path(
        transforms_path, payload_a["source_dnsplatter_checkpoint"], args.image_size
    )
    cameras_b = load_camera_path(
        transforms_path, payload_b["source_dnsplatter_checkpoint"], args.image_size
    )
    if not (len(frames) == len(cameras_a) == len(cameras_b)):
        raise ValueError("frame and camera counts do not match")

    rows: list[dict[str, float | int]] = []
    panel_paths: list[Path] = []
    with torch.inference_mode():
        for index, (frame, camera_a, camera_b) in enumerate(zip(frames, cameras_a, cameras_b)):
            camera_a = {
                key: value.to(args.device) if torch.is_tensor(value) else value
                for key, value in camera_a.items()
            }
            camera_b = {
                key: value.to(args.device) if torch.is_tensor(value) else value
                for key, value in camera_b.items()
            }
            gbuffer_a = scene_a.rasterize(camera_a)
            gbuffer_b = scene_b.rasterize(camera_b)
            albedo_a = gbuffer_a["albedo"].detach().float().cpu()
            albedo_b = gbuffer_b["albedo"].detach().float().cpu()
            alpha_a = gbuffer_a["alpha"][..., 0].detach().float().cpu()
            alpha_b = gbuffer_b["alpha"][..., 0].detach().float().cpu()

            rgb_path = args.dataset / frame["file_path"]
            mask_path = args.dataset / frame["mask_path"]
            rgb = load_rgb(rgb_path, args.image_size)
            mask = load_mask(mask_path, args.image_size)
            prior_path = args.prior_dir / f"frame_{index:04d}.pt"
            prior_payload = torch.load(prior_path, map_location="cpu", weights_only=False)
            prior = resize_map(prior_payload["prior_albedo"].float(), (args.image_size, args.image_size))

            valid = mask & (alpha_a > 1.0e-3) & (alpha_b > 1.0e-3)
            valid3 = valid[..., None]
            if valid.any():
                mae_a = (albedo_a - prior).abs()[valid3.expand_as(albedo_a)].mean().item()
                mae_b = (albedo_b - prior).abs()[valid3.expand_as(albedo_b)].mean().item()
                mae_ab = (albedo_a - albedo_b).abs()[valid3.expand_as(albedo_a)].mean().item()
            else:
                mae_a = mae_b = mae_ab = float("nan")
            rows.append(
                {
                    "frame": index,
                    "secondary_vs_prior_mae_linear": mae_a,
                    "sh_vs_prior_mae_linear": mae_b,
                    "secondary_vs_sh_mae_linear": mae_ab,
                    "valid_fraction": valid.float().mean().item(),
                }
            )

            black = torch.zeros_like(prior)
            prior_display = torch.where(mask[..., None], prior, black)
            a_display = torch.where(valid3, albedo_a, black)
            b_display = torch.where(valid3, albedo_b, black)
            rgb_display = torch.where(mask[..., None], rgb, torch.zeros_like(rgb))
            images = [
                (rgb_display.clamp(0, 1).numpy() * 255.0 + 0.5).astype(np.uint8),
                to_u8_srgb(prior_display),
                to_u8_srgb(a_display),
                to_u8_srgb(b_display),
            ]
            panel = make_panel(
                images,
                ["INPUT RGB", "DIFFUSIONRENDERER PRIOR", args.label_a, args.label_b],
                index,
            )
            panel_path = frames_dir / f"frame_{index:04d}.jpg"
            panel.save(panel_path, quality=94)
            panel_paths.append(panel_path)
            print(f"[{index + 1:03d}/{len(frames):03d}] {panel_path.name}", flush=True)

    with (args.output / "per_frame_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    finite = lambda key: np.asarray([row[key] for row in rows], dtype=np.float64)
    summary = {
        "frame_count": len(rows),
        "checkpoint_a": str(args.checkpoint_a),
        "checkpoint_b": str(args.checkpoint_b),
        "prior_dir": str(args.prior_dir),
        "secondary_vs_prior_mean_mae_linear": float(np.nanmean(finite("secondary_vs_prior_mae_linear"))),
        "sh_vs_prior_mean_mae_linear": float(np.nanmean(finite("sh_vs_prior_mae_linear"))),
        "secondary_vs_sh_mean_mae_linear": float(np.nanmean(finite("secondary_vs_sh_mae_linear"))),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    make_contact_sheet(panel_paths, args.output / "all191_contact_sheet.jpg")
    ranked = sorted(
        range(len(rows)),
        key=lambda index: max(
            rows[index]["secondary_vs_prior_mae_linear"], rows[index]["sh_vs_prior_mae_linear"]
        ),
        reverse=True,
    )[:30]
    make_contact_sheet(
        [panel_paths[index] for index in ranked],
        args.output / "top30_prior_difference_contact_sheet.jpg",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
