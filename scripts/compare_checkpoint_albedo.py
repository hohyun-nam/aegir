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
from aegir.config import config_from_dict
from aegir.data import MultiViewSceneDataset
from aegir.math_utils import linear_to_srgb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render GT RGB, DiffusionRenderer albedo, and one checkpoint's learned albedo."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--prior-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="LEARNED ALBEDO")
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="all")
    parser.add_argument("--max-views", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def to_u8(value: torch.Tensor) -> np.ndarray:
    return (value.detach().float().cpu().clamp(0, 1).numpy() * 255.0 + 0.5).astype(np.uint8)


def linear_to_u8_srgb(value: torch.Tensor) -> np.ndarray:
    return to_u8(linear_to_srgb(value.detach().float().cpu()))


def make_panel(images: list[np.ndarray], labels: list[str], frame_index: int) -> Image.Image:
    height, width = images[0].shape[:2]
    header = 44
    canvas = Image.new("RGB", (width * len(images), height + header), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for column, (array, label) in enumerate(zip(images, labels, strict=True)):
        canvas.paste(Image.fromarray(array), (column * width, header))
        draw.text((column * width + 8, 8), label, fill="black", font=font)
    draw.text((8, 25), f"frame {frame_index:04d}", fill=(80, 80, 80), font=font)
    return canvas


def make_contact_sheet(paths: list[Path], output: Path, columns: int = 3) -> None:
    if not paths:
        return
    width = 768
    thumbs = []
    for path in paths:
        panel = Image.open(path).convert("RGB")
        height = round(panel.height * width / panel.width)
        thumbs.append(panel.resize((width, height), Image.Resampling.LANCZOS))
    rows = (len(thumbs) + columns - 1) // columns
    sheet = Image.new("RGB", (width * columns, thumbs[0].height * rows), "white")
    for index, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((index % columns) * width, (index // columns) * thumb.height))
    sheet.save(output, quality=92)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frames_dir = args.output / "frames"
    frames_dir.mkdir(exist_ok=True)

    scene, _, payload = load_checkpoint(args.checkpoint, args.device)
    config = config_from_dict(payload["config"])
    dataset = MultiViewSceneDataset(
        args.dataset,
        payload["source_dnsplatter_checkpoint"],
        args.prior_dir,
        image_size=config.raster.image_size,
        max_views=args.max_views,
        split=args.split,
    )

    rows: list[dict[str, float | int]] = []
    panels: list[Path] = []
    with torch.inference_mode():
        for ordinal, frame in enumerate(dataset.frames, start=1):
            camera = {
                key: value.to(args.device) if torch.is_tensor(value) else value
                for key, value in frame.camera.items()
            }
            gbuffer = scene.rasterize(
                camera,
                alpha_threshold=config.raster.alpha_threshold,
                rasterize_mode=config.raster.rasterize_mode,
            )
            learned = gbuffer["albedo"].detach().float().cpu()
            alpha = gbuffer["alpha"].detach().float().cpu()[..., 0]
            prior = frame.priors["albedo"].detach().float().cpu()
            mask = frame.mask[..., 0].cpu() & (alpha > config.raster.alpha_threshold)
            valid3 = mask[..., None]
            black = torch.zeros_like(learned)

            mae = (
                float((learned - prior).abs()[valid3.expand_as(learned)].mean())
                if bool(mask.any())
                else float("nan")
            )
            rows.append(
                {
                    "frame": frame.index,
                    "learned_vs_prior_mae_linear": mae,
                    "valid_fraction": float(mask.float().mean()),
                }
            )
            panel = make_panel(
                [
                    to_u8(torch.where(frame.mask, frame.target, torch.zeros_like(frame.target))),
                    linear_to_u8_srgb(torch.where(valid3, prior, black)),
                    linear_to_u8_srgb(torch.where(valid3, learned, black)),
                ],
                ["GT IMAGE", "DIFFUSIONRENDERER ALBEDO", args.label],
                frame.index,
            )
            panel_path = frames_dir / f"frame_{frame.index:04d}.jpg"
            panel.save(panel_path, quality=94)
            panels.append(panel_path)
            print(f"[{ordinal:03d}/{len(dataset.frames):03d}] {panel_path.name}", flush=True)

    with (args.output / "per_frame_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    errors = np.asarray([row["learned_vs_prior_mae_linear"] for row in rows], dtype=np.float64)
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "frame_count": len(rows),
        "split": args.split,
        "learned_vs_prior_mean_mae_linear": float(np.nanmean(errors)),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    make_contact_sheet(panels, args.output / "all_frames_contact_sheet.jpg")
    ranked = np.argsort(np.nan_to_num(errors, nan=-1.0))[::-1][: min(30, len(panels))]
    make_contact_sheet(
        [panels[int(index)] for index in ranked],
        args.output / "top30_prior_difference_contact_sheet.jpg",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
