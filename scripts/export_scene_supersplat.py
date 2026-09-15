#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aegir.checkpoint import SUPPORTED_CHECKPOINT_FORMATS, emitter_state_dict, emitters_from_state
from aegir.gaussian_scene import (
    SH_C0,
    GaussianMaterialBackend,
    gaussian_surface_normals,
    planarize_gaussians,
    surfel_scales_for_3d_export,
)
from aegir.math_utils import safe_normalize
from scripts.export_supersplat_ply import filter_gaussians, write_binary_ply


def scene_from_checkpoint(payload: dict) -> dict[str, torch.Tensor]:
    state = payload["scene_state"]
    means = state["means"].float()
    log_scales = state["log_scales"].float()
    quats = safe_normalize(state["raw_quats"].float(), dim=-1)
    scales = log_scales.exp()
    if scales.shape[-1] == 3:
        quats, scales = planarize_gaussians(quats, scales)
    export_scales = surfel_scales_for_3d_export(scales).log()
    albedo = state["albedo_logits"].float().sigmoid()
    f_dc = (albedo - 0.5) / SH_C0
    return {
        "means": means,
        "normals": gaussian_surface_normals(quats),
        "f_dc": f_dc,
        "f_rest": torch.zeros((means.shape[0], 15, 3), dtype=torch.float32),
        "opacity": state["opacity_logits"].float(),
        "scale": export_scales,
        "rot": quats,
    }


def emitter_gaussians(state: dict, rest_channels: int, green_dc: float) -> dict[str, torch.Tensor]:
    centers = state["centers"].float()
    scales = state["scales"].float().clamp_min(1.0e-8)
    quats = safe_normalize(state["quaternions"].float(), dim=-1)
    count = centers.shape[0]
    if scales.shape[-1] == 2:
        thickness = scales.amin(dim=-1, keepdim=True) * 1.0e-3
        scales = torch.cat((scales, thickness.clamp_min(1.0e-6)), dim=-1)
    f_dc = torch.zeros((count, 3), dtype=torch.float32)
    f_dc[:, 1] = green_dc
    return {
        "means": centers,
        "normals": torch.zeros_like(centers),
        "f_dc": f_dc,
        "f_rest": torch.zeros((count, rest_channels, 3), dtype=torch.float32),
        "opacity": torch.full((count, 1), 10.0),
        "scale": scales.log(),
        "rot": quats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export learned shared Gaussian albedo plus fluorescent green emitters for SuperSplat."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-opacity", type=float, default=0.01)
    parser.add_argument("--max-gaussians", type=int, default=500_000)
    parser.add_argument("--emitter-green-dc", type=float, default=8.0)
    parser.add_argument(
        "--initial-state",
        action="store_true",
        help=(
            "Export the source DN geometry/material initialization together with "
            "the checkpoint's stored initial emitters instead of the learned state."
        ),
    )
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("format") not in SUPPORTED_CHECKPOINT_FORMATS:
        raise ValueError("expected a shared-Gaussian AEGIR checkpoint")
    if args.initial_state:
        emitter_state = payload.get("initial_emitter_state")
        if emitter_state is None:
            raise ValueError("checkpoint has no stored initial_emitter_state")
        source_scene = GaussianMaterialBackend(payload["source_dnsplatter_checkpoint"])
        scene = scene_from_checkpoint({"scene_state": source_scene.state_dict()})
    else:
        emitter_state = payload["emitter_state"]
        scene = scene_from_checkpoint(payload)
    emitter_state = emitter_state_dict(
        emitters_from_state(
            emitter_state,
            "cpu",
            preserve_3d=payload.get("format") == "aegir_shared_gaussian_v15",
        )
    )
    before = scene["means"].shape[0]
    scene = filter_gaussians(
        scene,
        min_opacity=args.min_opacity,
        stride=1,
        max_gaussians=args.max_gaussians,
        drop_radius_outliers=True,
        radius_outlier_quantile=0.999,
        drop_large_scales=True,
        large_scale_quantile=0.999,
    )
    emitters = emitter_gaussians(emitter_state, scene["f_rest"].shape[1], args.emitter_green_dc)
    merged = {key: torch.cat((scene[key], emitters[key]), dim=0) for key in scene}
    write_binary_ply(args.output, merged)
    print(f"scene_gaussians: {scene['means'].shape[0]} / {before}")
    print(f"emitter_gaussians: {emitters['means'].shape[0]}")
    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()
