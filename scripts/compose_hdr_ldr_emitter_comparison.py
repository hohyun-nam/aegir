#!/usr/bin/env python3
"""Compare emitter candidates from tone-mapped LDR and linear HDR observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_erosion


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def luma(value: np.ndarray) -> np.ndarray:
    return value[..., 0] * 0.299 + value[..., 1] * 0.587 + value[..., 2] * 0.114


def false_color(value: np.ndarray, upper: float, log: bool = False) -> np.ndarray:
    work = np.log1p(np.maximum(value, 0.0)) if log else np.maximum(value, 0.0)
    limit = np.log1p(upper) if log else upper
    normalized = np.clip(work / max(float(limit), 1.0e-8), 0.0, 1.0)
    return (colormaps["inferno"](normalized)[..., :3] * 255.0 + 0.5).astype(np.uint8)


def overlay(rgb: np.ndarray, candidate: np.ndarray, gt: np.ndarray) -> np.ndarray:
    result = rgb.astype(np.float32).copy()
    result[candidate] = result[candidate] * 0.30 + np.array([255, 45, 32]) * 0.70
    boundary = gt & ~binary_erosion(gt, iterations=3)
    result[boundary] = np.array([38, 255, 130])
    return np.clip(result, 0, 255).astype(np.uint8)


def metric(candidate: np.ndarray, gt: np.ndarray) -> tuple[float, float, int, int]:
    tp = int(np.logical_and(candidate, gt).sum())
    fp = int(np.logical_and(candidate, ~gt).sum())
    fn = int(np.logical_and(~candidate, gt).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return precision, recall, tp, fp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=110)
    parser.add_argument("--quantile", type=float, default=0.995)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.data.resolve()
    metadata = json.loads((root / "transforms.json").read_text(encoding="utf-8"))
    frames = metadata["frames"]
    train_files = set(metadata.get("train_filenames", [f["file_path"] for f in frames]))
    ldr_samples: list[np.ndarray] = []
    hdr_samples: list[np.ndarray] = []
    for frame in frames:
        if frame["file_path"] not in train_files:
            continue
        rgb = np.asarray(Image.open(root / frame["file_path"]).convert("RGB"), dtype=np.float32) / 255.0
        hdr = np.load(root / frame["radiance_hdr_file_path"], allow_pickle=False).astype(np.float32)
        ldr_samples.append(luma(rgb[::4, ::4]).reshape(-1))
        hdr_samples.append(luma(hdr[::4, ::4]).reshape(-1))
    ldr_values = np.concatenate(ldr_samples)
    hdr_values = np.concatenate(hdr_samples)
    ldr_threshold = float(np.quantile(ldr_values, args.quantile))
    hdr_threshold = float(np.quantile(hdr_values, args.quantile))
    hdr_display_upper = float(np.quantile(hdr_values, 0.999))

    frame = frames[args.frame]
    rgb_u8 = np.asarray(Image.open(root / frame["file_path"]).convert("RGB"))
    rgb = rgb_u8.astype(np.float32) / 255.0
    hdr = np.load(root / frame["radiance_hdr_file_path"], allow_pickle=False).astype(np.float32)
    gt = np.asarray(Image.open(root / frame["emitter_mask_path"]).convert("L")) > 0
    ldr_luma = luma(rgb)
    hdr_luma = luma(hdr)
    ldr_candidate = ldr_luma >= ldr_threshold
    hdr_candidate = hdr_luma >= hdr_threshold
    ldr_metric = metric(ldr_candidate, gt)
    hdr_metric = metric(hdr_candidate, gt)
    ldr_summary = (
        f"Precision {ldr_metric[0]*100:.1f}%  ·  Recall {ldr_metric[1]*100:.1f}%"
        if np.any(ldr_candidate)
        else f"No candidates  ·  Recall {ldr_metric[1]*100:.1f}%"
    )
    hdr_summary = (
        f"Precision {hdr_metric[0]*100:.1f}%  ·  Recall {hdr_metric[1]*100:.1f}%"
        if np.any(hdr_candidate)
        else f"No candidates  ·  Recall {hdr_metric[1]*100:.1f}%"
    )

    gt_visual = np.zeros((*gt.shape, 3), dtype=np.uint8)
    gt_visual[gt] = (245, 245, 245)
    entries = (
        ("Tone-mapped LDR", rgb_u8, "Absolute radiance is exposure-dependent"),
        ("LDR luminance", false_color(ldr_luma, 1.0), f"Global P99.5 threshold = {ldr_threshold:.3f}"),
        ("LDR candidates + GT outline", overlay(rgb_u8, ldr_candidate, gt), ldr_summary),
        ("Linear HDR luminance", false_color(hdr_luma, hdr_display_upper, log=True),
         f"Scene-linear range · max {float(hdr_luma.max()):.2f}"),
        ("HDR candidates + GT outline", overlay(rgb_u8, hdr_candidate, gt), hdr_summary),
        ("Blender GT emitter mask", gt_visual, f"Emitter pixels = {int(gt.sum()):,}"),
    )

    tile = 512
    columns, rows = 3, 2
    margin, gap = 28, 18
    title_h, label_h, footer_h = 105, 82, 104
    width = margin * 2 + columns * tile + (columns - 1) * gap
    height = title_h + rows * (tile + label_h) + gap + footer_h
    canvas = Image.new("RGB", (width, height), "#171A1F")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 20), "Why Linear HDR Matters for Emitter Initialization", font=font(35, True), fill="#F5F7FA")
    draw.text(
        (margin, 64),
        f"Real Blender example · frame {args.frame:04d} · identical global top-{(1-args.quantile)*100:.1f}% selection rule",
        font=font(20),
        fill="#AEB8C2",
    )
    for index, (label, value, subtitle) in enumerate(entries):
        column, row = index % columns, index // columns
        left = margin + column * (tile + gap)
        top = title_h + row * (tile + label_h + gap)
        panel = Image.fromarray(value).resize((tile, tile), Image.Resampling.LANCZOS)
        canvas.paste(panel, (left, top))
        label_bounds = draw.textbbox((0, 0), label, font=font(24, True))
        label_width = label_bounds[2] - label_bounds[0]
        draw.text((left + (tile - label_width) / 2, top + tile + 10), label, font=font(24, True), fill="#F0F2F5")
        sub_bounds = draw.textbbox((0, 0), subtitle, font=font(16))
        sub_width = sub_bounds[2] - sub_bounds[0]
        draw.text((left + (tile - sub_width) / 2, top + tile + 47), subtitle, font=font(16), fill="#AEB8C2")

    footer_y = height - footer_h + 16
    draw.rounded_rectangle((margin, footer_y, width - margin, height - 22), radius=18, fill="#232831", outline="#4B5563", width=2)
    footer = (
        "LDR misses the visible stand emitter because tone mapping removes cross-view radiometric scale.\n"
        "Linear HDR preserves emission strength; the later 3D pipeline additionally applies validity, persistence, DBSCAN, and PCA."
    )
    draw.multiline_text((margin + 22, footer_y + 14), footer, font=font(17), fill="#E2E7EC", spacing=8)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(args.output.resolve())
    print(f"LDR threshold={ldr_threshold:.6f} precision={ldr_metric[0]:.6f} recall={ldr_metric[1]:.6f}")
    print(f"HDR threshold={hdr_threshold:.6f} precision={hdr_metric[0]:.6f} recall={hdr_metric[1]:.6f}")


if __name__ == "__main__":
    main()
