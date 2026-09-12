from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance


def _load_metrics(path: Path) -> dict:
    return json.loads((path / "metrics.json").read_text(encoding="utf-8"))


def _to_u8(value: np.ndarray) -> np.ndarray:
    return (np.clip(value.astype(np.float32), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare matching raw baseline and denoised inspect frames.")
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--denoised-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contact-columns", type=int, default=2)
    args = parser.parse_args()

    baseline = _load_metrics(args.baseline_dir)
    denoised = _load_metrics(args.denoised_dir)
    if baseline["checkpoint"] != denoised["checkpoint"]:
        raise ValueError("inspect directories use different checkpoints")
    for key in ("views", "split"):
        if baseline[key] != denoised[key]:
            raise ValueError(f"inspect directories disagree on {key}")
    if baseline["evaluation"]["visualization_denoiser"] != "none":
        raise ValueError("baseline inspect unexpectedly uses a denoiser")
    if denoised["evaluation"]["visualization_denoiser"] == "none":
        raise ValueError("denoised inspect does not record a denoiser")

    baseline_by_frame = {int(item["frame"]): item for item in baseline["metrics"]}
    denoised_by_frame = {int(item["frame"]): item for item in denoised["metrics"]}
    if baseline_by_frame.keys() != denoised_by_frame.keys():
        raise ValueError("inspect directories do not contain identical frame IDs")

    frames_dir = args.output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    panels: list[Image.Image] = []
    for frame in sorted(baseline_by_frame):
        before_path = args.baseline_dir / "raw_metric_frames" / f"frame_{frame:04d}.npz"
        after_path = args.denoised_dir / "raw_metric_frames" / f"frame_{frame:04d}.npz"
        with np.load(before_path) as before, np.load(after_path) as after:
            if not np.array_equal(before["target_rgb"], after["target_rgb"]):
                raise ValueError(f"frame {frame} targets differ")
            target = Image.fromarray(_to_u8(before["target_rgb"]))
            raw = Image.fromarray(_to_u8(before["render_rgb"]))
            filtered = Image.fromarray(_to_u8(after["render_rgb"]))
        difference = ImageEnhance.Brightness(ImageChops.difference(raw, filtered)).enhance(4.0)
        width, height = target.size
        header = 26
        panel = Image.new("RGB", (4 * width, height + header), "white")
        draw = ImageDraw.Draw(panel)
        for column, (label, image) in enumerate(
            (("GT", target), ("BASELINE", raw), ("DENOISED", filtered), ("|DIFF| x4", difference))
        ):
            panel.paste(image, (column * width, header))
            draw.text((column * width + 8, 7), label, fill="black")
        panel.save(frames_dir / f"frame_{frame:04d}.png")
        panels.append(panel)

    columns = max(1, args.contact_columns)
    rows = math.ceil(len(panels) / columns)
    width, height = panels[0].size
    contact = Image.new("RGB", (columns * width, rows * height), "white")
    for index, panel in enumerate(panels):
        contact.paste(panel, ((index % columns) * width, (index // columns) * height))
    contact.save(args.output_dir / "gt_baseline_denoised_contact_sheet.jpg", quality=92)

    summary = {
        "checkpoint": baseline["checkpoint"],
        "views": baseline["views"],
        "baseline": {key: baseline[f"mean_{key}"] for key in ("psnr", "ssim", "lpips")},
        "denoised": {key: denoised[f"mean_{key}"] for key in ("psnr", "ssim", "lpips")},
        "delta_denoised_minus_baseline": {
            key: denoised[f"mean_{key}"] - baseline[f"mean_{key}"]
            for key in ("psnr", "ssim", "lpips")
        },
        "denoiser": denoised["evaluation"]["visualization_denoiser"],
    }
    (args.output_dir / "comparison_metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
