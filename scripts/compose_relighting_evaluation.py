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


def _scores(metrics: dict) -> dict[str, float]:
    result = {key: float(metrics[f"mean_{key}"]) for key in ("psnr", "ssim", "lpips")}
    result["log_hdr_l1"] = float(np.mean([item["log_hdr_l1"] for item in metrics["metrics"]]))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare identity and two-emitter relighting evaluations.")
    parser.add_argument("--all-on-dir", type=Path, required=True)
    parser.add_argument("--relight-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contact-columns", type=int, default=2)
    args = parser.parse_args()

    all_on = _load_metrics(args.all_on_dir)
    relight = _load_metrics(args.relight_dir)
    if all_on["checkpoint"] != relight["checkpoint"]:
        raise ValueError("evaluations use different checkpoints")
    if all_on["views"] != relight["views"] or all_on["split"] != relight["split"]:
        raise ValueError("evaluations use different view counts or splits")
    for key in (
        "seed",
        "frame_seed_rule",
        "num_brdf_samples",
        "num_light_samples",
        "num_bounce_samples",
        "six_axis_emission",
        "visualization_denoiser",
    ):
        if all_on["evaluation"].get(key) != relight["evaluation"].get(key):
            raise ValueError(f"evaluations disagree on {key}")
    if all_on["evaluation"]["visualization_denoiser"] != "none":
        raise ValueError("evaluation must use unfiltered baseline rendering")
    if all_on["evaluation"]["active_emitter_indices"] != [0, 1, 2, 3, 4, 5]:
        raise ValueError("all-on evaluation did not retain all six emitters")
    if relight["evaluation"]["active_emitter_indices"] != [0, 5]:
        raise ValueError("relighting evaluation did not retain emitters 0 and 5")

    matched_metrics = list(zip(all_on["metrics"], relight["metrics"], strict=True))
    if not matched_metrics:
        raise ValueError("evaluations do not contain any frames")
    for view_order, (all_item, relight_item) in enumerate(matched_metrics):
        if all_item.get("source_camera") != relight_item.get("source_camera"):
            raise ValueError(f"view {view_order} uses different source cameras")
        if all_item["seed"] != relight_item["seed"]:
            raise ValueError(f"view {view_order} uses different random seeds")

    frames_dir = args.output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    panels: list[Image.Image] = []
    delta_errors = []
    for view_order, (all_item, relight_item) in enumerate(matched_metrics):
        all_frame = int(all_item["frame"])
        relight_frame = int(relight_item["frame"])
        all_path = args.all_on_dir / "raw_metric_frames" / f"frame_{all_frame:04d}.npz"
        relight_path = args.relight_dir / "raw_metric_frames" / f"frame_{relight_frame:04d}.npz"
        with np.load(all_path) as all_raw, np.load(relight_path) as relight_raw:
            if not np.array_equal(all_raw["valid_mask"], relight_raw["valid_mask"]):
                raise ValueError(f"view {view_order} uses different valid masks")
            target_all = Image.fromarray(_to_u8(all_raw["target_rgb"]))
            render_all = Image.fromarray(_to_u8(all_raw["render_rgb"]))
            target_relight = Image.fromarray(_to_u8(relight_raw["target_rgb"]))
            render_relight = Image.fromarray(_to_u8(relight_raw["render_rgb"]))
            valid = relight_raw["valid_mask"].astype(bool)
            target_delta = relight_raw["target_hdr_linear"] - all_raw["target_hdr_linear"]
            render_delta = relight_raw["render_hdr_linear"] - all_raw["render_hdr_linear"]
            valid_rgb = np.broadcast_to(valid, target_delta.shape)
            delta_errors.append(float(np.abs(target_delta - render_delta)[valid_rgb].mean()))

        error_all = ImageEnhance.Brightness(ImageChops.difference(target_all, render_all)).enhance(4.0)
        error_relight = ImageEnhance.Brightness(
            ImageChops.difference(target_relight, render_relight)
        ).enhance(4.0)
        images = (
            ("GT ALL 6", target_all),
            ("AEGIR ALL 6", render_all),
            ("GT PANELS 0,5", target_relight),
            ("AEGIR EMITTERS 0,5", render_relight),
            ("ALL-6 ERROR x4", error_all),
            ("RELIGHT ERROR x4", error_relight),
        )
        width, height = target_all.size
        header = 26
        panel = Image.new("RGB", (6 * width, height + header), "white")
        draw = ImageDraw.Draw(panel)
        for column, (label, image) in enumerate(images):
            panel.paste(image, (column * width, header))
            draw.text((column * width + 8, 7), label, fill="black")
        panel.save(frames_dir / f"view_{view_order:04d}.png")
        panels.append(panel)

    columns = max(1, args.contact_columns)
    rows = math.ceil(len(panels) / columns)
    panel_width, panel_height = panels[0].size
    contact = Image.new("RGB", (columns * panel_width, rows * panel_height), "white")
    for index, panel in enumerate(panels):
        contact.paste(panel, ((index % columns) * panel_width, (index // columns) * panel_height))
    contact.save(args.output_dir / "all6_vs_panels05_contact_sheet.jpg", quality=92)
    panels[0].save(
        args.output_dir / "all6_vs_panels05.gif",
        save_all=True,
        append_images=panels[1:],
        duration=700,
        loop=0,
    )

    all_scores = _scores(all_on)
    relight_scores = _scores(relight)
    summary = {
        "checkpoint": all_on["checkpoint"],
        "views": all_on["views"],
        "all6_identity": all_scores,
        "emitters05_relighting": relight_scores,
        "relighting_minus_identity": {
            key: relight_scores[key] - all_scores[key] for key in all_scores
        },
        "mean_absolute_relighting_delta_error_linear": float(np.mean(delta_errors)),
        "all6_evaluation": all_on["evaluation"],
        "relighting_evaluation": relight["evaluation"],
    }
    (args.output_dir / "comparison_metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
