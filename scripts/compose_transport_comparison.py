from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _load_metrics(path: Path) -> dict:
    return json.loads((path / "metrics.json").read_text(encoding="utf-8"))


def _metric_by_frame(payload: dict) -> dict[int, dict]:
    return {int(item["frame"]): item for item in payload["metrics"]}


def _to_u8(value: np.ndarray) -> np.ndarray:
    return (np.clip(value.astype(np.float32), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def _labeled_triptych(
    target: np.ndarray,
    sh_proxy: np.ndarray,
    secondary: np.ndarray,
    frame_id: int,
) -> Image.Image:
    arrays = [_to_u8(target), _to_u8(sh_proxy), _to_u8(secondary)]
    height, width = arrays[0].shape[:2]
    if any(array.shape != arrays[0].shape for array in arrays[1:]):
        raise ValueError(f"frame {frame_id} comparison arrays have different shapes")
    header = 24
    canvas = Image.new("RGB", (width * 3, height + header), "white")
    draw = ImageDraw.Draw(canvas)
    labels = ("TARGET", "SH PROXY", "SECONDARY BOUNCE")
    for column, (label, array) in enumerate(zip(labels, arrays, strict=True)):
        left = column * width
        canvas.paste(Image.fromarray(array), (left, header))
        box = draw.textbbox((0, 0), label)
        text_width = box[2] - box[0]
        draw.text((left + (width - text_width) // 2, 6), label, fill="black")
    draw.text((4, 6), f"#{frame_id:04d}", fill="black")
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compose identical-pose target, SH-proxy, and secondary-bounce inspect outputs."
    )
    parser.add_argument("--sh-dir", type=Path, required=True)
    parser.add_argument("--secondary-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gif-duration-ms", type=int, default=300)
    parser.add_argument("--contact-columns", type=int, default=3)
    args = parser.parse_args()

    sh_metrics = _load_metrics(args.sh_dir)
    secondary_metrics = _load_metrics(args.secondary_dir)
    if not sh_metrics["evaluation"]["training_sh_proxy_used"]:
        raise ValueError("--sh-dir is not a training-SH-proxy inspection")
    if secondary_metrics["evaluation"]["training_sh_proxy_used"]:
        raise ValueError("--secondary-dir is not a secondary-bounce inspection")
    for key in ("views", "split"):
        if sh_metrics[key] != secondary_metrics[key]:
            raise ValueError(f"inspect outputs disagree on {key}")
    for key in ("seed", "frame_seed_rule", "num_brdf_samples", "num_light_samples"):
        if sh_metrics["evaluation"][key] != secondary_metrics["evaluation"][key]:
            raise ValueError(f"inspect outputs disagree on evaluation {key}")

    sh_by_frame = _metric_by_frame(sh_metrics)
    secondary_by_frame = _metric_by_frame(secondary_metrics)
    if sh_by_frame.keys() != secondary_by_frame.keys():
        raise ValueError("inspect outputs do not contain identical frame IDs")

    frames_dir = args.output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    composed: list[Image.Image] = []
    manifest_frames = []
    for frame_id in sorted(sh_by_frame):
        sh_item = sh_by_frame[frame_id]
        secondary_item = secondary_by_frame[frame_id]
        if sh_item["source_file"] != secondary_item["source_file"]:
            raise ValueError(f"frame {frame_id} source pose differs between inspections")
        if sh_item["seed"] != secondary_item["seed"]:
            raise ValueError(f"frame {frame_id} evaluation seed differs between inspections")

        sh_path = args.sh_dir / "raw_metric_frames" / f"frame_{frame_id:04d}.npz"
        secondary_path = args.secondary_dir / "raw_metric_frames" / f"frame_{frame_id:04d}.npz"
        with np.load(sh_path) as sh_raw, np.load(secondary_path) as secondary_raw:
            if not np.array_equal(sh_raw["valid_mask"], secondary_raw["valid_mask"]):
                raise ValueError(f"frame {frame_id} valid masks differ")
            if not np.allclose(sh_raw["target_rgb"], secondary_raw["target_rgb"], atol=0.0):
                raise ValueError(f"frame {frame_id} targets differ")
            image = _labeled_triptych(
                sh_raw["target_rgb"],
                sh_raw["render_rgb"],
                secondary_raw["render_rgb"],
                frame_id,
            )
        image.save(frames_dir / f"frame_{frame_id:04d}.png")
        composed.append(image)
        manifest_frames.append(
            {
                "frame": frame_id,
                "source_file": sh_item["source_file"],
                "seed": sh_item["seed"],
                "sh_proxy": {key: sh_item[key] for key in ("psnr", "ssim", "lpips", "log_hdr_l1")},
                "secondary_bounce": {
                    key: secondary_item[key] for key in ("psnr", "ssim", "lpips", "log_hdr_l1")
                },
            }
        )

    if not composed:
        raise ValueError("no comparison frames were composed")
    composed[0].save(
        args.output_dir / "target_sh_secondary.gif",
        save_all=True,
        append_images=composed[1:],
        duration=args.gif_duration_ms,
        loop=0,
    )

    columns = max(1, args.contact_columns)
    rows = math.ceil(len(composed) / columns)
    cell_width, cell_height = composed[0].size
    contact = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
    for index, image in enumerate(composed):
        contact.paste(image, ((index % columns) * cell_width, (index // columns) * cell_height))
    contact.save(args.output_dir / "target_sh_secondary_contact_sheet.png")

    manifest = {
        "format": "aegir-target-sh-secondary-comparison-v1",
        "views": len(composed),
        "split": sh_metrics["split"],
        "render_samples": sh_metrics["evaluation"]["num_brdf_samples"],
        "sh_proxy_mean": {
            key: sh_metrics[key] for key in ("mean_psnr", "mean_ssim", "mean_lpips")
        },
        "secondary_bounce_mean": {
            key: secondary_metrics[key] for key in ("mean_psnr", "mean_ssim", "mean_lpips")
        },
        "frames": manifest_frames,
    }
    (args.output_dir / "comparison_metrics.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"COMPARISON_OK views={len(composed)} output={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
