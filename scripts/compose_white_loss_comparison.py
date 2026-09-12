from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageChops, ImageDraw, ImageEnhance


def _load_metrics(path: Path) -> dict:
    return json.loads((path / "metrics.json").read_text(encoding="utf-8"))


def _load_checkpoint_config(metrics: dict) -> tuple[Path, dict, dict]:
    checkpoint_path = Path(metrics["checkpoint"]).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return checkpoint_path, checkpoint["config"], checkpoint


def _to_u8(value: np.ndarray) -> np.ndarray:
    return (np.clip(value.astype(np.float32), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def _mean_metrics(metrics: dict) -> dict[str, float]:
    return {key: float(metrics[f"mean_{key}"]) for key in ("psnr", "ssim", "lpips")}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare matching no-denoiser inspect renders trained with light-white loss 0 and 0.1."
    )
    parser.add_argument("--white0-dir", type=Path, required=True)
    parser.add_argument("--white01-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contact-columns", type=int, default=2)
    parser.add_argument("--difference-scale", type=float, default=4.0)
    args = parser.parse_args()

    white0 = _load_metrics(args.white0_dir)
    white01 = _load_metrics(args.white01_dir)
    checkpoint0, config0, payload0 = _load_checkpoint_config(white0)
    checkpoint01, config01, payload01 = _load_checkpoint_config(white01)

    loss0 = float(config0["losses"]["light_white"])
    loss01 = float(config01["losses"]["light_white"])
    if loss0 != 0.0 or loss01 != 0.1:
        raise ValueError(f"expected light-white weights 0 and 0.1, found {loss0} and {loss01}")
    if int(config0["training"].get("light_white_ramp_end", 0)) != 0:
        raise ValueError("white=0 checkpoint unexpectedly uses a white-loss ramp")
    if int(config01["training"].get("light_white_ramp_end", 0)) != 0:
        raise ValueError("white=0.1 checkpoint unexpectedly uses a white-loss ramp")

    comparable0 = copy.deepcopy(config0)
    comparable01 = copy.deepcopy(config01)
    comparable0["losses"].pop("light_white")
    comparable01["losses"].pop("light_white")
    comparable0["training"].setdefault("light_white_ramp_end", 0)
    comparable01["training"].setdefault("light_white_ramp_end", 0)
    if comparable0 != comparable01:
        raise ValueError("checkpoint configs differ in settings other than losses.light_white")
    if Path(payload0["source_dnsplatter_checkpoint"]).resolve() != Path(
        payload01["source_dnsplatter_checkpoint"]
    ).resolve():
        raise ValueError("checkpoints use different geometry initialization checkpoints")

    for key in ("views", "split"):
        if white0[key] != white01[key]:
            raise ValueError(f"inspect directories disagree on {key}")
    comparison_keys = (
        "seed",
        "frame_seed_rule",
        "num_brdf_samples",
        "num_light_samples",
        "num_bounce_samples",
        "six_axis_emission",
        "emitter_axis_lobe_sharpness",
        "metric_domain",
    )
    for key in comparison_keys:
        if white0["evaluation"].get(key) != white01["evaluation"].get(key):
            raise ValueError(f"inspect directories disagree on evaluation.{key}")
    for name, metrics in (("white=0", white0), ("white=0.1", white01)):
        evaluation = metrics["evaluation"]
        if evaluation["visualization_denoiser"] != "none":
            raise ValueError(f"{name} inspect unexpectedly uses a denoiser")
        if evaluation.get("training_sh_proxy_used") is not False:
            raise ValueError(f"{name} inspect did not use physical secondary rendering")

    white0_by_frame = {int(item["frame"]): item for item in white0["metrics"]}
    white01_by_frame = {int(item["frame"]): item for item in white01["metrics"]}
    if white0_by_frame.keys() != white01_by_frame.keys():
        raise ValueError("inspect directories do not contain identical frame IDs")
    if not white0_by_frame:
        raise ValueError("inspect directories contain no frames")

    frames_dir = args.output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    panels: list[Image.Image] = []
    for frame in sorted(white0_by_frame):
        item0 = white0_by_frame[frame]
        item01 = white01_by_frame[frame]
        for key in ("source_file", "seed"):
            if item0[key] != item01[key]:
                raise ValueError(f"frame {frame} disagrees on {key}")
        raw0_path = args.white0_dir / "raw_metric_frames" / f"frame_{frame:04d}.npz"
        raw01_path = args.white01_dir / "raw_metric_frames" / f"frame_{frame:04d}.npz"
        with np.load(raw0_path) as raw0, np.load(raw01_path) as raw01:
            if not np.array_equal(raw0["target_rgb"], raw01["target_rgb"]):
                raise ValueError(f"frame {frame} display-space targets differ")
            if not np.array_equal(raw0["target_hdr_linear"], raw01["target_hdr_linear"]):
                raise ValueError(f"frame {frame} linear-HDR targets differ")
            if not np.array_equal(raw0["valid_mask"], raw01["valid_mask"]):
                raise ValueError(f"frame {frame} valid masks differ")
            target = Image.fromarray(_to_u8(raw0["target_rgb"]))
            render0 = Image.fromarray(_to_u8(raw0["render_rgb"]))
            render01 = Image.fromarray(_to_u8(raw01["render_rgb"]))

        difference = ImageEnhance.Brightness(ImageChops.difference(render0, render01)).enhance(
            args.difference_scale
        )
        width, height = target.size
        header = 26
        panel = Image.new("RGB", (4 * width, height + header), "white")
        draw = ImageDraw.Draw(panel)
        columns = (
            ("GT", target),
            ("WHITE LOSS 0", render0),
            ("WHITE LOSS 0.1", render01),
            (f"|0.1 - 0| x{args.difference_scale:g}", difference),
        )
        for column, (label, image) in enumerate(columns):
            panel.paste(image, (column * width, header))
            draw.text((column * width + 8, 7), label, fill="black")
        panel.save(frames_dir / f"frame_{frame:04d}.png")
        panels.append(panel)

    contact_columns = max(1, args.contact_columns)
    rows = math.ceil(len(panels) / contact_columns)
    panel_width, panel_height = panels[0].size
    contact = Image.new(
        "RGB", (contact_columns * panel_width, rows * panel_height), "white"
    )
    for index, panel in enumerate(panels):
        contact.paste(
            panel,
            ((index % contact_columns) * panel_width, (index // contact_columns) * panel_height),
        )
    contact.save(args.output_dir / "gt_white0_white01_contact_sheet.jpg", quality=92)
    panels[0].save(
        args.output_dir / "gt_white0_white01.gif",
        save_all=True,
        append_images=panels[1:],
        duration=700,
        loop=0,
    )

    metrics0 = _mean_metrics(white0)
    metrics01 = _mean_metrics(white01)
    summary = {
        "white0_checkpoint": str(checkpoint0),
        "white01_checkpoint": str(checkpoint01),
        "views": white0["views"],
        "evaluation": {key: white0["evaluation"].get(key) for key in comparison_keys},
        "white0": metrics0,
        "white01": metrics01,
        "delta_white01_minus_white0": {
            key: metrics01[key] - metrics0[key] for key in metrics0
        },
    }
    (args.output_dir / "comparison_metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
