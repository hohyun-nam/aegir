#!/usr/bin/env python3
"""Compose matched GT and checkpoint renders from AEGIR inspect directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def parse_inspect(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("inspect must be LABEL=PATH")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("inspect must be LABEL=PATH")
    return label, Path(path)


def to_u8(value: np.ndarray) -> np.ndarray:
    return (np.clip(value, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect", action="append", type=parse_inspect, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.inspect) < 2:
        raise ValueError("at least two --inspect LABEL=PATH inputs are required")

    records = []
    common_frames: set[int] | None = None
    for label, directory in args.inspect:
        metrics_path = directory / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        frames = {int(item["frame"]) for item in metrics["metrics"]}
        common_frames = frames if common_frames is None else common_frames & frames
        records.append((label, directory, metrics))
    frame_ids = sorted(common_frames or ())
    if not frame_ids:
        raise ValueError("inspect directories have no common evaluated frames")

    reference_metrics = records[0][2]
    for label, _, metrics in records[1:]:
        if metrics.get("split") != reference_metrics.get("split"):
            raise ValueError(f"{label} disagrees on split")
        for key in (
            "seed",
            "frame_seed_rule",
            "num_brdf_samples",
            "num_light_samples",
            "num_bounce_samples",
            "metric_domain",
        ):
            if metrics["evaluation"].get(key) != reference_metrics["evaluation"].get(key):
                raise ValueError(f"{label} disagrees on evaluation.{key}")

    frame_output = args.output / "frames"
    frame_output.mkdir(parents=True, exist_ok=True)
    contact_tiles = []
    for frame_id in frame_ids:
        loaded = []
        target = None
        reference_frame = None
        for label, directory, metrics in records:
            metric = next(item for item in metrics["metrics"] if int(item["frame"]) == frame_id)
            identity = (metric.get("source_file"), metric.get("seed"))
            if reference_frame is None:
                reference_frame = identity
            elif identity != reference_frame:
                raise ValueError(f"frame {frame_id} source/seed differs between inspect runs")
            raw_path = directory / "raw_metric_frames" / f"frame_{frame_id:04d}.npz"
            with np.load(raw_path) as raw:
                target_rgb = raw["target_rgb"]
                if target is None:
                    target = target_rgb
                elif not np.array_equal(target, target_rgb):
                    raise ValueError(f"frame {frame_id} targets differ between inspect runs")
                loaded.append((label, raw["render_rgb"]))
        assert target is not None
        images = [("GT", target), *loaded]
        height, width = target.shape[:2]
        header = 28
        canvas = Image.new("RGB", (width * len(images), height + header), "black")
        draw = ImageDraw.Draw(canvas)
        for column, (label, value) in enumerate(images):
            canvas.paste(Image.fromarray(to_u8(value)), (column * width, header))
            draw.text((column * width + 6, 7), label, fill="white")
        canvas.save(frame_output / f"frame_{frame_id:04d}.png")
        if len(contact_tiles) < 10:
            contact_tiles.append(canvas)

    if contact_tiles:
        width, height = contact_tiles[0].size
        contact = Image.new("RGB", (width, height * len(contact_tiles)), "black")
        for row, tile in enumerate(contact_tiles):
            contact.paste(tile, (0, row * height))
        contact.save(args.output / "contact_10frames.png")

    summary = {
        "frames": frame_ids,
        "runs": [
            {
                "label": label,
                "inspect_dir": str(directory.resolve()),
                "checkpoint": metrics["checkpoint"],
                "mean_psnr": metrics["mean_psnr"],
                "mean_ssim": metrics["mean_ssim"],
                "mean_lpips": metrics["mean_lpips"],
            }
            for label, directory, metrics in records
        ],
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
