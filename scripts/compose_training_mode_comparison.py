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


def _load_raw(directory: Path, frame_id: int) -> dict[str, np.ndarray]:
    path = directory / "raw_metric_frames" / f"frame_{frame_id:04d}.npz"
    with np.load(path) as payload:
        return {key: payload[key].copy() for key in payload.files}


def _labeled_comparison(arrays: list[np.ndarray], frame_id: int) -> Image.Image:
    arrays_u8 = [_to_u8(array) for array in arrays]
    if any(array.shape != arrays_u8[0].shape for array in arrays_u8[1:]):
        raise ValueError(f"frame {frame_id} comparison arrays have different shapes")
    height, width = arrays_u8[0].shape[:2]
    labels = (
        "GT",
        "SECONDARY-TRAIN\nSH PROXY",
        "SH-TRAIN\nSH PROXY",
        "SECONDARY-TRAIN\nSECONDARY",
        "SH-TRAIN\nSECONDARY",
    )
    header = 34
    canvas = Image.new("RGB", (width * len(arrays_u8), height + header), "white")
    draw = ImageDraw.Draw(canvas)
    for column, (label, array) in enumerate(zip(labels, arrays_u8, strict=True)):
        left = column * width
        canvas.paste(Image.fromarray(array), (left, header))
        box = draw.multiline_textbbox((0, 0), label, spacing=1, align="center")
        text_width = box[2] - box[0]
        draw.multiline_text(
            (left + (width - text_width) // 2, 3),
            label,
            fill="black",
            spacing=1,
            align="center",
        )
    draw.text((4, 3), f"#{frame_id:04d}", fill="black")
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare SH and secondary renders from secondary-trained and SH-trained checkpoints."
    )
    parser.add_argument("--secondary-train-root", type=Path, required=True)
    parser.add_argument("--sh-train-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gif-duration-ms", type=int, default=400)
    parser.add_argument("--contact-columns", type=int, default=2)
    args = parser.parse_args()

    directories = {
        "secondary_train_sh": args.secondary_train_root / "inspect_sh_proxy",
        "sh_train_sh": args.sh_train_root / "inspect_sh_proxy",
        "secondary_train_secondary": args.secondary_train_root / "inspect_secondary_bounce",
        "sh_train_secondary": args.sh_train_root / "inspect_secondary_bounce",
    }
    metrics = {key: _load_metrics(path) for key, path in directories.items()}
    reference = metrics["secondary_train_sh"]
    for name, payload in metrics.items():
        for key in ("views", "split"):
            if payload[key] != reference[key]:
                raise ValueError(f"{name} disagrees on {key}")
        for key in (
            "seed",
            "frame_seed_rule",
            "num_brdf_samples",
            "num_light_samples",
            "num_bounce_samples",
        ):
            if payload["evaluation"][key] != reference["evaluation"][key]:
                raise ValueError(f"{name} disagrees on evaluation {key}")

    if not metrics["secondary_train_sh"]["evaluation"]["training_sh_proxy_used"]:
        raise ValueError("secondary-training SH input is not an SH-proxy inspection")
    if not metrics["sh_train_sh"]["evaluation"]["training_sh_proxy_used"]:
        raise ValueError("SH-training SH input is not an SH-proxy inspection")
    if metrics["secondary_train_secondary"]["evaluation"]["training_sh_proxy_used"]:
        raise ValueError("secondary-training secondary input used the SH proxy")
    if metrics["sh_train_secondary"]["evaluation"]["training_sh_proxy_used"]:
        raise ValueError("SH-training secondary input used the SH proxy")

    by_frame = {key: _metric_by_frame(payload) for key, payload in metrics.items()}
    reference_ids = by_frame["secondary_train_sh"].keys()
    if any(items.keys() != reference_ids for items in by_frame.values()):
        raise ValueError("inspect outputs do not contain identical frame IDs")

    frames_dir = args.output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    composed: list[Image.Image] = []
    manifest_frames: list[dict] = []
    for frame_id in sorted(reference_ids):
        reference_item = by_frame["secondary_train_sh"][frame_id]
        for name, items in by_frame.items():
            item = items[frame_id]
            if item["source_file"] != reference_item["source_file"]:
                raise ValueError(f"frame {frame_id} source pose differs for {name}")
            if item["seed"] != reference_item["seed"]:
                raise ValueError(f"frame {frame_id} seed differs for {name}")

        raw = {key: _load_raw(path, frame_id) for key, path in directories.items()}
        reference_raw = raw["secondary_train_sh"]
        for name, payload in raw.items():
            if not np.array_equal(payload["valid_mask"], reference_raw["valid_mask"]):
                raise ValueError(f"frame {frame_id} valid mask differs for {name}")
            if not np.array_equal(payload["target_rgb"], reference_raw["target_rgb"]):
                raise ValueError(f"frame {frame_id} target differs for {name}")

        image = _labeled_comparison(
            [
                reference_raw["target_rgb"],
                raw["secondary_train_sh"]["render_rgb"],
                raw["sh_train_sh"]["render_rgb"],
                raw["secondary_train_secondary"]["render_rgb"],
                raw["sh_train_secondary"]["render_rgb"],
            ],
            frame_id,
        )
        image.save(frames_dir / f"frame_{frame_id:04d}.png")
        composed.append(image)
        manifest_frames.append(
            {
                "frame": frame_id,
                "source_file": reference_item["source_file"],
                "seed": reference_item["seed"],
            }
        )

    if not composed:
        raise ValueError("no comparison frames were composed")
    gif_path = args.output_dir / "gt_training_transport_comparison.gif"
    composed[0].save(
        gif_path,
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
    contact_path = args.output_dir / "gt_training_transport_contact_sheet.png"
    contact.save(contact_path)

    mean_keys = ("mean_psnr", "mean_ssim", "mean_lpips")
    manifest = {
        "format": "aegir-training-transport-comparison-v1",
        "column_order": [
            "gt",
            "secondary_train_sh_proxy",
            "sh_train_sh_proxy",
            "secondary_train_secondary_bounce",
            "sh_train_secondary_bounce",
        ],
        "views": len(composed),
        "split": reference["split"],
        "evaluation": {
            key: reference["evaluation"][key]
            for key in (
                "seed",
                "frame_seed_rule",
                "num_brdf_samples",
                "num_light_samples",
                "num_bounce_samples",
            )
        },
        "means": {
            key: {metric: payload[metric] for metric in mean_keys}
            for key, payload in metrics.items()
        },
        "frames": manifest_frames,
    }
    (args.output_dir / "comparison_metrics.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"COMPARISON_OK views={len(composed)}")
    print(f"frames: {frames_dir.resolve()}")
    print(f"contact_sheet: {contact_path.resolve()}")
    print(f"gif: {gif_path.resolve()}")


if __name__ == "__main__":
    main()
