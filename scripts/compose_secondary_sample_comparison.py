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


def metric_by_frame(payload: dict) -> dict[int, dict]:
    return {int(item["frame"]): item for item in payload["metrics"]}


def to_u8(value: np.ndarray) -> np.ndarray:
    return (np.clip(value.astype(np.float32), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def make_panel(
    target: np.ndarray,
    sample1: np.ndarray,
    sample8: np.ndarray,
    frame: int,
) -> Image.Image:
    arrays = [to_u8(target), to_u8(sample1), to_u8(sample8)]
    if any(array.shape != arrays[0].shape for array in arrays[1:]):
        raise ValueError(f"frame {frame} image shapes differ")
    height, width = arrays[0].shape[:2]
    header = 26
    output = Image.new("RGB", (3 * width, height + header), "white")
    draw = ImageDraw.Draw(output)
    labels = ("GT", "SECONDARY SAMPLE 1", "SECONDARY SAMPLE 8")
    for column, (label, array) in enumerate(zip(labels, arrays, strict=True)):
        left = column * width
        output.paste(Image.fromarray(array), (left, header))
        draw.text((left + 8, 7), label, fill="black")
    draw.text((3 * width - 70, 7), f"#{frame:04d}", fill="black")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compose identical-pose GT, secondary sample-1, and sample-8 renders."
    )
    parser.add_argument("--sample1-dir", type=Path, required=True)
    parser.add_argument("--sample8-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contact-columns", type=int, default=3)
    parser.add_argument("--gif-duration-ms", type=int, default=300)
    args = parser.parse_args()

    one_metrics = load_metrics(args.sample1_dir)
    eight_metrics = load_metrics(args.sample8_dir)
    for payload, count in ((one_metrics, 1), (eight_metrics, 8)):
        evaluation = payload["evaluation"]
        if evaluation["training_sh_proxy_used"]:
            raise ValueError("sample comparison requires explicit secondary-bounce inspections")
        if int(evaluation["num_bounce_samples"]) != count:
            raise ValueError(f"expected {count} bounce samples")
    for key in ("views", "split"):
        if one_metrics[key] != eight_metrics[key]:
            raise ValueError(f"inspect outputs disagree on {key}")

    one_by_frame = metric_by_frame(one_metrics)
    eight_by_frame = metric_by_frame(eight_metrics)
    if one_by_frame.keys() != eight_by_frame.keys():
        raise ValueError("inspect outputs do not contain identical frame IDs")

    frames_dir = args.output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    panels: list[Image.Image] = []
    manifest_frames: list[dict] = []
    for frame in sorted(one_by_frame):
        first = one_by_frame[frame]
        eighth = eight_by_frame[frame]
        if first["source_file"] != eighth["source_file"]:
            raise ValueError(f"frame {frame} source poses differ")
        one_path = args.sample1_dir / "raw_metric_frames" / f"frame_{frame:04d}.npz"
        eight_path = args.sample8_dir / "raw_metric_frames" / f"frame_{frame:04d}.npz"
        with np.load(one_path) as one, np.load(eight_path) as eight:
            if not np.array_equal(one["valid_mask"], eight["valid_mask"]):
                raise ValueError(f"frame {frame} valid masks differ")
            if not np.array_equal(one["target_rgb"], eight["target_rgb"]):
                raise ValueError(f"frame {frame} targets differ")
            panel = make_panel(one["target_rgb"], one["render_rgb"], eight["render_rgb"], frame)
        panel.save(frames_dir / f"frame_{frame:04d}.png")
        panels.append(panel)
        manifest_frames.append(
            {
                "frame": frame,
                "source_file": first["source_file"],
                "sample1": {key: first[key] for key in ("psnr", "ssim", "lpips", "log_hdr_l1")},
                "sample8": {key: eighth[key] for key in ("psnr", "ssim", "lpips", "log_hdr_l1")},
            }
        )

    if not panels:
        raise ValueError("no comparison frames were composed")
    panels[0].save(
        args.output_dir / "gt_secondary1_secondary8.gif",
        save_all=True,
        append_images=panels[1:],
        duration=args.gif_duration_ms,
        loop=0,
    )
    columns = max(1, args.contact_columns)
    rows = math.ceil(len(panels) / columns)
    width, height = panels[0].size
    contact = Image.new("RGB", (columns * width, rows * height), "white")
    for index, panel in enumerate(panels):
        contact.paste(panel, ((index % columns) * width, (index // columns) * height))
    contact.save(args.output_dir / "gt_secondary1_secondary8_contact_sheet.jpg", quality=92)
    summary = {
        "format": "aegir-secondary-sample-comparison-v1",
        "views": len(panels),
        "sample1_mean": {
            key: one_metrics[f"mean_{key}"] for key in ("psnr", "ssim", "lpips")
        },
        "sample8_mean": {
            key: eight_metrics[f"mean_{key}"] for key in ("psnr", "ssim", "lpips")
        },
        "frames": manifest_frames,
    }
    (args.output_dir / "comparison_metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"COMPARISON_OK views={len(panels)} output={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
