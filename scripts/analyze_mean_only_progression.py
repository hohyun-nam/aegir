#!/usr/bin/env python3
"""Summarize checkpoint metrics and background-Gaussian motion for a progression run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def displacement_stats(displacement: torch.Tensor) -> dict[str, float]:
    values = displacement.detach().float().cpu().numpy()
    return {
        "mean_m": float(values.mean()),
        "median_m": float(np.median(values)),
        "p90_m": float(np.quantile(values, 0.90)),
        "p95_m": float(np.quantile(values, 0.95)),
        "p99_m": float(np.quantile(values, 0.99)),
        "max_m": float(values.max()),
    }


def max_abs_delta(current: dict, initial: dict, key: str) -> float:
    return float((current[key].float() - initial[key].float()).abs().max())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    output = args.output or (root / "mean_only_progression_analysis.json")
    comparison = root / "inspect_checkpoint_progression/comparison/summary.json"
    metric_payload = json.loads(comparison.read_text(encoding="utf-8"))
    initial_path = root / "aegir_initialization.pt"
    initial_payload = torch.load(initial_path, map_location="cpu", weights_only=False)
    initial_scene = initial_payload["scene_state"]
    emitter_ids = initial_scene.get(
        "emitter_ids",
        torch.full((initial_scene["means"].shape[0],), -1, dtype=torch.long),
    )
    background = emitter_ids < 0

    checkpoints = []
    for run in metric_payload["runs"]:
        path = Path(run["checkpoint"])
        payload = torch.load(path, map_location="cpu", weights_only=False)
        scene = payload["scene_state"]
        if scene["means"].shape != initial_scene["means"].shape:
            raise ValueError(f"Gaussian count changed at {run['label']}: {path}")
        displacement = (scene["means"].float() - initial_scene["means"].float()).norm(dim=-1)
        item = {
            "label": run["label"],
            "checkpoint": str(path.resolve()),
            "step": int(payload["step"]),
            "psnr": float(run["mean_psnr"]),
            "ssim": float(run["mean_ssim"]),
            "lpips": float(run["mean_lpips"]),
            "background_mean_displacement": displacement_stats(displacement[background]),
            "tagged_mean_displacement": (
                displacement_stats(displacement[~background]) if bool((~background).any()) else None
            ),
            "frozen_parameter_max_abs_delta": {
                "log_scales": max_abs_delta(scene, initial_scene, "log_scales"),
                "raw_quats": max_abs_delta(scene, initial_scene, "raw_quats"),
                "opacity_logits": max_abs_delta(scene, initial_scene, "opacity_logits"),
            },
        }
        checkpoints.append(item)

    baseline = checkpoints[0]
    for item in checkpoints:
        item["metric_delta_from_initialization"] = {
            "psnr": item["psnr"] - baseline["psnr"],
            "ssim": item["ssim"] - baseline["ssim"],
            "lpips": item["lpips"] - baseline["lpips"],
        }
    report = {
        "root": str(root),
        "gaussian_count": int(initial_scene["means"].shape[0]),
        "background_count": int(background.sum()),
        "tagged_emitter_gaussian_count": int((~background).sum()),
        "trainability": initial_payload["config"]["training"],
        "checkpoints": checkpoints,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
