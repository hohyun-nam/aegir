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


def scene_from_checkpoint(
    payload: dict,
    *,
    include_tagged_emitters: bool = False,
    tagged_emitter_green_dc: float | None = None,
) -> dict[str, torch.Tensor]:
    state = payload["scene_state"]
    emitter_ids = state.get("emitter_ids")
    if emitter_ids is None:
        emitter_ids = torch.full((state["means"].shape[0],), -1, dtype=torch.long)
    else:
        emitter_ids = emitter_ids.long()
    keep = (
        torch.ones_like(emitter_ids, dtype=torch.bool)
        if include_tagged_emitters
        else emitter_ids < 0
    )
    kept_emitter_ids = emitter_ids[keep]
    means = state["means"].float()[keep]
    log_scales = state["log_scales"].float()[keep]
    quats = safe_normalize(state["raw_quats"].float()[keep], dim=-1)
    scales = log_scales.exp()
    if scales.shape[-1] == 3:
        quats, scales = planarize_gaussians(quats, scales)
    export_scales = surfel_scales_for_3d_export(scales).log()
    albedo = state["albedo_logits"].float()[keep].sigmoid()
    f_dc = (albedo - 0.5) / SH_C0
    if tagged_emitter_green_dc is not None:
        tagged = kept_emitter_ids >= 0
        f_dc[tagged] = 0.0
        f_dc[tagged, 1] = float(tagged_emitter_green_dc)
    return {
        "means": means,
        "normals": gaussian_surface_normals(quats),
        "f_dc": f_dc,
        "f_rest": torch.zeros((means.shape[0], 15, 3), dtype=torch.float32),
        "opacity": state["opacity_logits"].float()[keep],
        "scale": export_scales,
        "rot": quats,
    }


def emitter_gaussians(
    state: dict,
    rest_channels: int,
    green_dc: float,
    visual_sigma_extent: float = 3.0,
) -> dict[str, torch.Tensor]:
    """Encode hard emitter radii as Gaussian sigmas for visualization.

    AEGIR emitter scales are physical semi-axis lengths, whereas Gaussian PLY
    viewers draw roughly ``visual_sigma_extent`` standard deviations.  Writing
    a hard radius directly as sigma therefore makes the debug emitter appear
    about three times too large.
    """
    if visual_sigma_extent <= 0:
        raise ValueError("visual_sigma_extent must be positive")
    centers = state["centers"].float()
    scales = state["scales"].float().div(visual_sigma_extent).clamp_min(1.0e-8)
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
        "--visualize-emitter-tags",
        action="store_true",
        help=(
            "retain source Gaussians associated with emitter IDs and color those "
            "rows fluorescent green"
        ),
    )
    parser.add_argument(
        "--omit-analytic-emitters",
        action="store_true",
        help="do not append analytic emitter visualization Gaussians",
    )
    parser.add_argument(
        "--emitter-visual-sigma-extent",
        type=float,
        default=3.0,
        help="Number of displayed Gaussian sigmas corresponding to one physical emitter radius.",
    )
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
        saved_ids = payload["scene_state"].get("emitter_ids")
        if saved_ids is not None:
            source_scene.emitter_ids.copy_(saved_ids.to(source_scene.emitter_ids))
        scene = scene_from_checkpoint(
            {"scene_state": source_scene.state_dict()},
            include_tagged_emitters=args.visualize_emitter_tags,
            tagged_emitter_green_dc=(
                args.emitter_green_dc if args.visualize_emitter_tags else None
            ),
        )
    else:
        emitter_state = payload["emitter_state"]
        scene = scene_from_checkpoint(
            payload,
            include_tagged_emitters=args.visualize_emitter_tags,
            tagged_emitter_green_dc=(
                args.emitter_green_dc if args.visualize_emitter_tags else None
            ),
        )
    emitter_state = emitter_state_dict(
        emitters_from_state(
            emitter_state,
            "cpu",
            preserve_3d=payload.get("format")
            in {"aegir_shared_gaussian_v15", "aegir_shared_gaussian_v16", "aegir_shared_gaussian_v17"},
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
    if args.omit_analytic_emitters:
        emitters = None
        merged = scene
    else:
        emitters = emitter_gaussians(
            emitter_state,
            scene["f_rest"].shape[1],
            args.emitter_green_dc,
            args.emitter_visual_sigma_extent,
        )
        merged = {key: torch.cat((scene[key], emitters[key]), dim=0) for key in scene}
    write_binary_ply(args.output, merged)
    print(f"scene_gaussians: {scene['means'].shape[0]} / {before}")
    print(f"analytic_emitter_gaussians: {0 if emitters is None else emitters['means'].shape[0]}")
    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()
