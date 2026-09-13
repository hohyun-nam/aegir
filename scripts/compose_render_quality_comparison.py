#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def load_metrics(path: Path) -> dict:
    return json.loads((path / "metrics.json").read_text(encoding="utf-8"))


def by_frame(payload: dict) -> dict[int, dict]:
    return {int(item["frame"]): item for item in payload["metrics"]}


def to_image(array: np.ndarray) -> Image.Image:
    pixels = (np.clip(array.astype(np.float32), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(pixels)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare matched 128/8 and 512/8 inspect renders.")
    parser.add_argument("--samples128-dir", type=Path, required=True)
    parser.add_argument("--samples512-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contact-columns", type=int, default=3)
    args = parser.parse_args()

    low = load_metrics(args.samples128_dir)
    high = load_metrics(args.samples512_dir)
    for payload, samples in ((low, 128), (high, 512)):
        evaluation = payload["evaluation"]
        if evaluation["training_sh_proxy_used"]:
            raise ValueError("comparison requires explicit secondary-bounce rendering")
        if int(evaluation["num_brdf_samples"]) != samples:
            raise ValueError(f"expected {samples} BRDF/light samples")
        if int(evaluation["num_bounce_samples"]) != 8:
            raise ValueError("expected 8 secondary samples")
        if evaluation["visualization_denoiser"] != "none":
            raise ValueError("comparison requires unfiltered renders")
    if low["views"] != high["views"] or low["split"] != high["split"]:
        raise ValueError("inspect outputs use different view selections")

    low_frames = by_frame(low)
    high_frames = by_frame(high)
    if low_frames.keys() != high_frames.keys():
        raise ValueError("inspect outputs contain different frame IDs")

    frames_dir = args.output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    panels: list[Image.Image] = []
    rows = []
    for frame in sorted(low_frames):
        low_item = low_frames[frame]
        high_item = high_frames[frame]
        if low_item["source_file"] != high_item["source_file"]:
            raise ValueError(f"frame {frame} source poses differ")
        with np.load(args.samples128_dir / "raw_metric_frames" / f"frame_{frame:04d}.npz") as a, np.load(
            args.samples512_dir / "raw_metric_frames" / f"frame_{frame:04d}.npz"
        ) as b:
            if not np.array_equal(a["target_rgb"], b["target_rgb"]):
                raise ValueError(f"frame {frame} targets differ")
            images = [to_image(a["target_rgb"]), to_image(a["render_rgb"]), to_image(b["render_rgb"])]
        width, height = images[0].size
        header = 28
        panel = Image.new("RGB", (width * 3, height + header), "white")
        draw = ImageDraw.Draw(panel)
        for column, (label, image) in enumerate(zip(("GT", "128 / 8", "512 / 8"), images, strict=True)):
            panel.paste(image, (column * width, header))
            draw.text((column * width + 8, 8), label, fill="black")
        draw.text((width * 3 - 72, 8), f"#{frame:04d}", fill="black")
        panel.save(frames_dir / f"frame_{frame:04d}.png")
        panels.append(panel)
        rows.append({
            "frame": frame,
            "source_file": low_item["source_file"],
            "samples128_bounce8": {key: low_item[key] for key in ("psnr", "ssim", "lpips", "log_hdr_l1")},
            "samples512_bounce8": {key: high_item[key] for key in ("psnr", "ssim", "lpips", "log_hdr_l1")},
        })

    if not panels:
        raise ValueError("no frames to compare")
    panels[0].save(
        args.output_dir / "gt_128x8_512x8.gif",
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
    sheet.save(args.output_dir / "gt_128x8_512x8_contact_sheet.jpg", quality=92)

    summary = {
        "format": "aegir-render-quality-comparison-v1",
        "views": len(panels),
        "samples128_bounce8_mean": {key: low[f"mean_{key}"] for key in ("psnr", "ssim", "lpips")},
        "samples512_bounce8_mean": {key: high[f"mean_{key}"] for key in ("psnr", "ssim", "lpips")},
        "frames": rows,
    }
    (args.output_dir / "comparison_metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"COMPARISON_OK views={len(panels)} output={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
