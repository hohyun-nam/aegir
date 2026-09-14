#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

CONFIGS = (
    ("128 / 1", "render_128x1", 128, 1),
    ("128 / 8", "render_128x8", 128, 8),
    ("512 / 8", "render_512x8", 512, 8),
)
METRIC_KEYS = ("psnr", "ssim", "lpips", "log_hdr_l1")


def load_metrics(path: Path) -> dict:
    return json.loads((path / "metrics.json").read_text(encoding="utf-8"))


def by_frame(payload: dict) -> dict[int, dict]:
    return {int(item["frame"]): item for item in payload["metrics"]}


def to_image(array: np.ndarray) -> Image.Image:
    pixels = (np.clip(array.astype(np.float32), 0.0, 1.0) * 255.0 + 0.5).astype(
        np.uint8
    )
    return Image.fromarray(pixels)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare matched 128/1, 128/8, and 512/8 inspect renders."
    )
    parser.add_argument("--inspect-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contact-columns", type=int, default=2)
    args = parser.parse_args()

    runs = []
    for label, directory, primary_samples, bounce_samples in CONFIGS:
        path = args.inspect_root / directory / "inspect_secondary_bounce"
        metrics = load_metrics(path)
        evaluation = metrics["evaluation"]
        if evaluation["training_sh_proxy_used"]:
            raise ValueError(f"{label} unexpectedly uses the training SH proxy")
        if evaluation["visualization_denoiser"] != "none":
            raise ValueError(f"{label} unexpectedly uses a visualization denoiser")
        if int(evaluation["num_brdf_samples"]) != primary_samples:
            raise ValueError(f"{label} has the wrong BRDF/light sample count")
        if int(evaluation["num_bounce_samples"]) != bounce_samples:
            raise ValueError(f"{label} has the wrong secondary sample count")
        runs.append((label, path, metrics, by_frame(metrics)))

    reference_metrics = runs[0][2]
    reference_frames = runs[0][3]
    for label, _, metrics, frames in runs[1:]:
        if (metrics["views"], metrics["split"]) != (
            reference_metrics["views"],
            reference_metrics["split"],
        ):
            raise ValueError(f"{label} uses a different view selection")
        if frames.keys() != reference_frames.keys():
            raise ValueError(f"{label} contains different frame IDs")

    frames_dir = args.output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    panels: list[Image.Image] = []
    rows = []
    for frame in sorted(reference_frames):
        raw_frames = []
        for label, path, _, frames in runs:
            if frames[frame]["source_file"] != reference_frames[frame]["source_file"]:
                raise ValueError(f"{label} frame {frame} uses a different source pose")
            raw_path = path / "raw_metric_frames" / f"frame_{frame:04d}.npz"
            with np.load(raw_path) as raw:
                raw_frames.append(
                    {
                        "target": raw["target_rgb"].copy(),
                        "render": raw["render_rgb"].copy(),
                        "valid": raw["valid_mask"].copy(),
                    }
                )
        for index, raw in enumerate(raw_frames[1:], start=1):
            if not np.array_equal(raw_frames[0]["target"], raw["target"]):
                raise ValueError(f"frame {frame} targets differ for {runs[index][0]}")
            if not np.array_equal(raw_frames[0]["valid"], raw["valid"]):
                raise ValueError(f"frame {frame} valid masks differ for {runs[index][0]}")

        images = [to_image(raw_frames[0]["target"])] + [
            to_image(raw["render"]) for raw in raw_frames
        ]
        width, height = images[0].size
        header = 28
        labels = ("GT",) + tuple(run[0] for run in runs)
        panel = Image.new("RGB", (width * len(images), height + header), "white")
        draw = ImageDraw.Draw(panel)
        for column, (label, rendered) in enumerate(zip(labels, images, strict=True)):
            panel.paste(rendered, (column * width, header))
            draw.text((column * width + 8, 8), label, fill="black")
        draw.text((width * len(images) - 72, 8), f"#{frame:04d}", fill="black")
        panel.save(frames_dir / f"frame_{frame:04d}.png")
        panels.append(panel)
        rows.append(
            {
                "frame": frame,
                "source_file": reference_frames[frame]["source_file"],
                **{
                    directory: {key: run_frames[frame][key] for key in METRIC_KEYS}
                    for (_, directory, _, _), (_, _, _, run_frames) in zip(
                        CONFIGS, runs, strict=True
                    )
                },
            }
        )

    if not panels:
        raise ValueError("no frames to compare")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = "gt_128x1_128x8_512x8"
    panels[0].save(
        args.output_dir / f"{stem}.gif",
        save_all=True,
        append_images=panels[1:],
        duration=300,
        loop=0,
    )
    columns = max(1, args.contact_columns)
    sheet_rows = math.ceil(len(panels) / columns)
    width, height = panels[0].size
    sheet = Image.new("RGB", (columns * width, sheet_rows * height), "white")
    for index, panel in enumerate(panels):
        sheet.paste(panel, ((index % columns) * width, (index // columns) * height))
    sheet.save(args.output_dir / f"{stem}_contact_sheet.jpg", quality=92)

    summary = {
        "format": "aegir-render-quality-triplet-v1",
        "views": len(panels),
        "means": {
            directory: {
                key: metrics[f"mean_{key}"] for key in ("psnr", "ssim", "lpips")
            }
            for (_, directory, _, _), (_, _, metrics, _) in zip(CONFIGS, runs, strict=True)
        },
        "frames": rows,
    }
    (args.output_dir / "comparison_metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"COMPARISON_OK views={len(panels)} output={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
