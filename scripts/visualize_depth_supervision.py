"""Visualize the RGB, monocular depth supervision, and available metric depth."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


def load_depth(path: Path) -> np.ndarray:
    depth = np.asarray(np.load(path), dtype=np.float32).squeeze()
    if depth.ndim != 2:
        raise ValueError(f"expected a 2D depth map, got {depth.shape}: {path}")
    return depth


def colorize(depth: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0.0)
    value = np.zeros(depth.shape, dtype=np.uint8)
    if valid.any():
        lo, hi = np.quantile(depth[valid], [0.01, 0.99])
        if hi <= lo:
            hi = lo + 1.0
        normalized = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
        value[valid] = np.asarray(normalized[valid] * 255.0 + 0.5, dtype=np.uint8)
    colored = cv2.applyColorMap(value, cv2.COLORMAP_TURBO)[..., ::-1]
    colored[~valid] = 0
    return colored


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    valid = np.isfinite(a) & np.isfinite(b) & (a > 0.0) & (b > 0.0)
    if int(valid.sum()) < 2:
        return float("nan")
    av = a[valid].astype(np.float64)
    bv = b[valid].astype(np.float64)
    if av.std() <= 0.0 or bv.std() <= 0.0:
        return float("nan")
    return float(np.corrcoef(av, bv)[0, 1])


def caption(image: np.ndarray, text: str) -> Image.Image:
    panel = Image.fromarray(image).convert("RGB")
    result = Image.new("RGB", (panel.width, panel.height + 32), "white")
    result.paste(panel, (0, 32))
    ImageDraw.Draw(result).text((8, 8), text, fill="black")
    return result


def make_sheet(paths: list[Path], output: Path, columns: int = 3) -> None:
    images = [Image.open(path).convert("RGB") for path in paths]
    if not images:
        return
    thumb_width = 640
    thumbs = []
    for image in images:
        height = round(image.height * thumb_width / image.width)
        thumbs.append(image.resize((thumb_width, height), Image.Resampling.LANCZOS))
    rows = (len(thumbs) + columns - 1) // columns
    cell_height = max(image.height for image in thumbs)
    sheet = Image.new("RGB", (columns * thumb_width, rows * cell_height), "white")
    for index, image in enumerate(thumbs):
        sheet.paste(image, ((index % columns) * thumb_width, (index // columns) * cell_height))
    sheet.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worst-count", type=int, default=15)
    args = parser.parse_args()

    metadata = json.loads((args.data / "transforms.json").read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    comparisons = args.output / "rgb_zoe_isaac_depth"
    comparisons.mkdir(exist_ok=True)
    rows = []
    paths = []

    for index, frame in enumerate(metadata["frames"]):
        stem = Path(frame["file_path"]).stem
        rgb = np.asarray(Image.open(args.data / frame["file_path"]).convert("RGB"))
        zoe = load_depth(args.data / "mono_depth" / f"{stem}.npy")
        metric_rel = frame.get("depth_file_path", f"depths/{stem}.npy")
        metric = load_depth(args.data / metric_rel)
        if rgb.shape[:2] != zoe.shape:
            rgb = cv2.resize(rgb, (zoe.shape[1], zoe.shape[0]), interpolation=cv2.INTER_AREA)
        if metric.shape != zoe.shape:
            metric = cv2.resize(metric, (zoe.shape[1], zoe.shape[0]), interpolation=cv2.INTER_NEAREST)
        correlation = pearson(metric[::4, ::4], zoe[::4, ::4])
        panels = [
            caption(rgb, f"frame {index:04d} RGB"),
            caption(colorize(zoe), "ZoeDepth used by DN (per-frame 1-99%)"),
            caption(colorize(metric), f"Isaac metric depth (per-frame 1-99%); Pearson={correlation:.3f}"),
        ]
        comparison = Image.new("RGB", (sum(panel.width for panel in panels), max(panel.height for panel in panels)))
        x = 0
        for panel in panels:
            comparison.paste(panel, (x, 0))
            x += panel.width
        output_path = comparisons / f"frame_{index:04d}.png"
        comparison.save(output_path)
        paths.append(output_path)
        valid_metric = metric[np.isfinite(metric) & (metric > 0.0)]
        valid_zoe = zoe[np.isfinite(zoe) & (zoe > 0.0)]
        rows.append(
            {
                "frame": index,
                "pearson_zoe_vs_isaac": correlation,
                "isaac_depth_median": float(np.median(valid_metric)),
                "zoe_stored_median": float(np.median(valid_zoe)),
                "comparison": str(output_path),
            }
        )

    with (args.output / "frame_stats.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    worst = sorted(rows, key=lambda row: np.nan_to_num(row["pearson_zoe_vs_isaac"], nan=-1.0))[
        : args.worst_count
    ]
    make_sheet([Path(row["comparison"]) for row in worst], args.output / "worst_zoe_contact_sheet.png")
    print(f"comparisons: {comparisons}")
    print(f"worst contact sheet: {args.output / 'worst_zoe_contact_sheet.png'}")
    print(f"statistics: {args.output / 'frame_stats.csv'}")


if __name__ == "__main__":
    main()
