from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageChops, ImageDraw, ImageEnhance


def _load_metrics(path: Path) -> dict:
    return json.loads((path / "metrics.json").read_text(encoding="utf-8"))


def _to_u8(value: np.ndarray) -> np.ndarray:
    return (np.clip(value.astype(np.float32), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def _checkpoint(metrics: dict) -> tuple[Path, dict]:
    path = Path(metrics["checkpoint"]).resolve()
    return path, torch.load(path, map_location="cpu", weights_only=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare matched frozen-geometry and scale-only AEGIR inspections."
    )
    parser.add_argument("--frozen-dir", type=Path, required=True)
    parser.add_argument("--scale-only-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--difference-scale", type=float, default=4.0)
    parser.add_argument("--contact-columns", type=int, default=2)
    args = parser.parse_args()

    frozen_metrics = _load_metrics(args.frozen_dir)
    scale_metrics = _load_metrics(args.scale_only_dir)
    frozen_checkpoint, frozen_payload = _checkpoint(frozen_metrics)
    scale_checkpoint, scale_payload = _checkpoint(scale_metrics)
    frozen_training = frozen_payload["config"]["training"]
    scale_training = scale_payload["config"]["training"]
    if float(frozen_training["geometry_lr"]) != 0.0:
        raise ValueError("reference checkpoint does not use frozen geometry")
    expected = {
        "optimize_scene_means": False,
        "optimize_scene_scales": True,
        "optimize_scene_rotations": False,
        "optimize_scene_opacities": False,
        "constrain_scene_scales": True,
    }
    for key, value in expected.items():
        if scale_training.get(key) is not value:
            raise ValueError(f"scale-only checkpoint has {key}={scale_training.get(key)!r}")
    if float(scale_training["geometry_lr"]) <= 0.0:
        raise ValueError("scale-only checkpoint has no geometry learning rate")
    if scale_payload["config"]["scene_culling"]["enabled"]:
        raise ValueError("scale-only checkpoint unexpectedly enabled scene refinement")
    if int(frozen_payload["gaussian_count"]) != int(scale_payload["gaussian_count"]):
        raise ValueError("Gaussian count changed despite disabled densification/pruning")
    if (
        Path(frozen_payload["source_dnsplatter_checkpoint"]).resolve()
        != Path(scale_payload["source_dnsplatter_checkpoint"]).resolve()
    ):
        raise ValueError("checkpoints use different 2DGS geometry sources")

    for key in ("views", "split"):
        if frozen_metrics[key] != scale_metrics[key]:
            raise ValueError(f"inspections disagree on {key}")
    for key in (
        "seed",
        "frame_seed_rule",
        "num_brdf_samples",
        "num_light_samples",
        "num_bounce_samples",
        "metric_domain",
    ):
        if frozen_metrics["evaluation"].get(key) != scale_metrics["evaluation"].get(key):
            raise ValueError(f"inspections disagree on evaluation.{key}")

    frozen_by_frame = {int(item["frame"]): item for item in frozen_metrics["metrics"]}
    scale_by_frame = {int(item["frame"]): item for item in scale_metrics["metrics"]}
    if frozen_by_frame.keys() != scale_by_frame.keys() or not frozen_by_frame:
        raise ValueError("inspections do not contain the same nonempty frame set")

    frames_dir = args.output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    panels: list[Image.Image] = []
    for frame in sorted(frozen_by_frame):
        frozen_item = frozen_by_frame[frame]
        scale_item = scale_by_frame[frame]
        for key in ("source_file", "seed"):
            if frozen_item[key] != scale_item[key]:
                raise ValueError(f"frame {frame} disagrees on {key}")
        frozen_raw_path = args.frozen_dir / "raw_metric_frames" / f"frame_{frame:04d}.npz"
        scale_raw_path = args.scale_only_dir / "raw_metric_frames" / f"frame_{frame:04d}.npz"
        with np.load(frozen_raw_path) as frozen_raw, np.load(scale_raw_path) as scale_raw:
            if not np.array_equal(frozen_raw["target_rgb"], scale_raw["target_rgb"]):
                raise ValueError(f"frame {frame} targets differ")
            if not np.array_equal(frozen_raw["valid_mask"], scale_raw["valid_mask"]):
                raise ValueError(f"frame {frame} valid masks differ")
            target = Image.fromarray(_to_u8(frozen_raw["target_rgb"]))
            frozen = Image.fromarray(_to_u8(frozen_raw["render_rgb"]))
            scale = Image.fromarray(_to_u8(scale_raw["render_rgb"]))

        difference = ImageEnhance.Brightness(ImageChops.difference(frozen, scale)).enhance(
            args.difference_scale
        )
        width, height = target.size
        header = 26
        panel = Image.new("RGB", (4 * width, height + header), "white")
        draw = ImageDraw.Draw(panel)
        columns = (
            ("GT", target),
            ("GEOMETRY FIXED", frozen),
            ("SCALE ONLY", scale),
            (f"|SCALE - FIXED| x{args.difference_scale:g}", difference),
        )
        for column, (label, image) in enumerate(columns):
            panel.paste(image, (column * width, header))
            draw.text((column * width + 8, 7), label, fill="black")
        panel.save(frames_dir / f"frame_{frame:04d}.png")
        panels.append(panel)

    columns = max(1, args.contact_columns)
    rows = math.ceil(len(panels) / columns)
    panel_width, panel_height = panels[0].size
    contact = Image.new("RGB", (columns * panel_width, rows * panel_height), "white")
    for index, panel in enumerate(panels):
        contact.paste(panel, ((index % columns) * panel_width, (index // columns) * panel_height))
    contact_path = args.output_dir / "gt_fixed_scale_only_contact_sheet.jpg"
    contact.save(contact_path, quality=92)
    gif_path = args.output_dir / "gt_fixed_scale_only.gif"
    panels[0].save(
        gif_path,
        save_all=True,
        append_images=panels[1:],
        duration=700,
        loop=0,
    )

    def aggregate(metrics: dict) -> dict[str, float]:
        return {
            "mean_psnr": float(metrics["mean_psnr"]),
            "mean_ssim": float(metrics["mean_ssim"]),
            "mean_lpips": float(metrics["mean_lpips"]),
            "mean_log_hdr_l1": float(np.mean([float(item["log_hdr_l1"]) for item in metrics["metrics"]])),
        }

    summary = {
        "format": "aegir-geometry-scale-only-comparison-v1",
        "frozen_checkpoint": str(frozen_checkpoint),
        "scale_only_checkpoint": str(scale_checkpoint),
        "views": len(panels),
        "scale_only_training": {
            "geometry_lr": scale_training["geometry_lr"],
            "scale_min_ratio": scale_training["scene_scale_min_ratio"],
            "scale_max_ratio": scale_training["scene_scale_max_ratio"],
        },
        "frozen": aggregate(frozen_metrics),
        "scale_only": aggregate(scale_metrics),
    }
    (args.output_dir / "comparison_metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"COMPARISON_OK views={len(panels)}")
    print(f"contact_sheet: {contact_path.resolve()}")
    print(f"gif: {gif_path.resolve()}")


if __name__ == "__main__":
    main()
