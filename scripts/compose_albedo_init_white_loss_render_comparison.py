from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

EXPERIMENTS = (
    ("shdc_white0", "SH-DC / WHITE 0"),
    ("shdc_white01", "SH-DC / WHITE 0.1"),
    ("neutral05_white0", "ALBEDO 0.5 / WHITE 0"),
)


def _load(directory: Path) -> tuple[dict, dict, Path]:
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    checkpoint_path = Path(metrics["checkpoint"]).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return metrics, checkpoint, checkpoint_path


def _to_u8(value: np.ndarray) -> np.ndarray:
    return (np.clip(value.astype(np.float32), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def _mean_metrics(metrics: dict) -> dict[str, float]:
    return {key: float(metrics[f"mean_{key}"]) for key in ("psnr", "ssim", "lpips")}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare matching no-denoiser renders from the three albedo/white-loss ablations."
    )
    parser.add_argument("--shdc-white0-dir", type=Path, required=True)
    parser.add_argument("--shdc-white01-dir", type=Path, required=True)
    parser.add_argument("--neutral05-white0-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contact-columns", type=int, default=2)
    args = parser.parse_args()

    directories = {
        "shdc_white0": args.shdc_white0_dir,
        "shdc_white01": args.shdc_white01_dir,
        "neutral05_white0": args.neutral05_white0_dir,
    }
    loaded = {name: _load(directory) for name, directory in directories.items()}
    metrics_by_name = {name: value[0] for name, value in loaded.items()}
    checkpoints = {name: value[1] for name, value in loaded.items()}
    checkpoint_paths = {name: value[2] for name, value in loaded.items()}

    expected = {
        "shdc_white0": ("sh_dc", 0.0),
        "shdc_white01": ("sh_dc", 0.1),
        "neutral05_white0": ("neutral", 0.0),
    }
    for name, (albedo_initialization, light_white) in expected.items():
        config = checkpoints[name]["config"]
        if config["training"]["albedo_initialization"] != albedo_initialization:
            raise ValueError(f"{name} has the wrong albedo initialization")
        if float(config["losses"]["light_white"]) != light_white:
            raise ValueError(f"{name} has the wrong light-white weight")
        if int(config["training"].get("light_white_ramp_end", 0)) != 0:
            raise ValueError(f"{name} unexpectedly uses a light-white ramp")
    if float(checkpoints["neutral05_white0"]["config"]["training"]["neutral_albedo"]) != 0.5:
        raise ValueError("neutral05_white0 was not initialized at albedo 0.5")

    normalized_configs = {}
    for name, checkpoint in checkpoints.items():
        config = copy.deepcopy(checkpoint["config"])
        config["losses"].pop("light_white")
        config["training"].pop("albedo_initialization")
        config["training"].setdefault("light_white_ramp_end", 0)
        normalized_configs[name] = config
    reference_config = normalized_configs["shdc_white0"]
    for name, config in normalized_configs.items():
        if config != reference_config:
            raise ValueError(f"{name} differs in settings beyond albedo initialization/light-white weight")

    geometry_paths = {
        Path(checkpoint["source_dnsplatter_checkpoint"]).resolve()
        for checkpoint in checkpoints.values()
    }
    if len(geometry_paths) != 1:
        raise ValueError("the experiments use different geometry initialization checkpoints")

    reference_metrics = metrics_by_name["shdc_white0"]
    evaluation_keys = (
        "seed",
        "frame_seed_rule",
        "num_brdf_samples",
        "num_light_samples",
        "num_bounce_samples",
        "six_axis_emission",
        "emitter_axis_lobe_sharpness",
        "metric_domain",
    )
    for name, metrics in metrics_by_name.items():
        if metrics["views"] != reference_metrics["views"] or metrics["split"] != reference_metrics["split"]:
            raise ValueError(f"{name} uses a different view count or split")
        evaluation = metrics["evaluation"]
        if evaluation["visualization_denoiser"] != "none":
            raise ValueError(f"{name} unexpectedly uses a denoiser")
        if evaluation.get("training_sh_proxy_used") is not False:
            raise ValueError(f"{name} did not use physical secondary rendering")
        for key in evaluation_keys:
            if evaluation.get(key) != reference_metrics["evaluation"].get(key):
                raise ValueError(f"{name} disagrees on evaluation.{key}")

    frames_by_name = {
        name: {int(item["frame"]): item for item in metrics["metrics"]}
        for name, metrics in metrics_by_name.items()
    }
    reference_frames = frames_by_name["shdc_white0"]
    if not reference_frames:
        raise ValueError("inspect directories contain no frames")
    for name, frames in frames_by_name.items():
        if frames.keys() != reference_frames.keys():
            raise ValueError(f"{name} contains different frame IDs")

    frames_dir = args.output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    panels: list[Image.Image] = []
    for frame in sorted(reference_frames):
        raw_arrays = {}
        for name, _ in EXPERIMENTS:
            item = frames_by_name[name][frame]
            reference_item = reference_frames[frame]
            if item["source_file"] != reference_item["source_file"] or item["seed"] != reference_item["seed"]:
                raise ValueError(f"{name} frame {frame} does not match source/seed")
            raw_path = directories[name] / "raw_metric_frames" / f"frame_{frame:04d}.npz"
            with np.load(raw_path) as raw:
                raw_arrays[name] = {
                    "target_rgb": raw["target_rgb"].copy(),
                    "target_hdr_linear": raw["target_hdr_linear"].copy(),
                    "valid_mask": raw["valid_mask"].copy(),
                    "render_rgb": raw["render_rgb"].copy(),
                }
        reference_raw = raw_arrays["shdc_white0"]
        for name, raw in raw_arrays.items():
            for key in ("target_rgb", "target_hdr_linear", "valid_mask"):
                if not np.array_equal(raw[key], reference_raw[key]):
                    raise ValueError(f"{name} frame {frame} disagrees on {key}")

        images = [("GT", Image.fromarray(_to_u8(reference_raw["target_rgb"])))]
        images.extend(
            (label, Image.fromarray(_to_u8(raw_arrays[name]["render_rgb"])))
            for name, label in EXPERIMENTS
        )
        width, height = images[0][1].size
        header = 26
        panel = Image.new("RGB", (4 * width, height + header), "white")
        draw = ImageDraw.Draw(panel)
        for column, (label, image) in enumerate(images):
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
    contact.save(args.output_dir / "gt_three_training_comparison_contact_sheet.jpg", quality=92)
    panels[0].save(
        args.output_dir / "gt_three_training_comparison.gif",
        save_all=True,
        append_images=panels[1:],
        duration=700,
        loop=0,
    )

    scores = {name: _mean_metrics(metrics) for name, metrics in metrics_by_name.items()}
    reference_scores = scores["shdc_white0"]
    summary = {
        "checkpoints": {name: str(path) for name, path in checkpoint_paths.items()},
        "views": reference_metrics["views"],
        "evaluation": {key: reference_metrics["evaluation"].get(key) for key in evaluation_keys},
        "scores": scores,
        "delta_vs_shdc_white0": {
            name: {key: values[key] - reference_scores[key] for key in values}
            for name, values in scores.items()
            if name != "shdc_white0"
        },
    }
    (args.output_dir / "comparison_metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
