#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aegir.gaussian_scene import (
    gaussian_surface_normals,
    surfel_scales_for_3d_export,
)
from aegir.math_utils import safe_normalize
from scripts.export_supersplat_ply import filter_gaussians, write_binary_ply

PARAMETER_PREFIX = "_model.gauss_params."


def load_checkpoint_gaussians(path: Path) -> tuple[int, dict[str, torch.Tensor]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") == "aegir_2dgs_geometry_v1":
        quats = safe_normalize(payload["quats"].detach().cpu().float(), dim=-1)
        scales = payload["scales"].detach().cpu().float().clamp_min(1.0e-8)
        if payload.get("representation") != "2dgs" or scales.shape[-1] != 2:
            raise ValueError("AEGIR native geometry payload is not a 2DGS checkpoint")
        data = {
            "means": payload["means"].detach().cpu().float(),
            "normals": gaussian_surface_normals(quats),
            "f_dc": payload["features_dc_raw"].detach().cpu().float(),
            "f_rest": payload["features_rest_raw"].detach().cpu().float(),
            "opacity": payload["opacities_logits"].detach().cpu().float(),
            "scale": surfel_scales_for_3d_export(scales).log(),
            "rot": quats,
        }
    else:
        state = payload.get("pipeline")
        if state is None:
            raise ValueError(
                "unsupported Gaussian checkpoint: expected native aegir_2dgs_geometry_v1 "
                "or a Nerfstudio pipeline state"
            )

        def tensor(name: str) -> torch.Tensor:
            return state[PARAMETER_PREFIX + name].detach().cpu().float()

        data = {
            "means": tensor("means"),
            "normals": tensor("normals"),
            "f_dc": tensor("features_dc"),
            "f_rest": tensor("features_rest"),
            "opacity": tensor("opacities"),
            "scale": tensor("scales"),
            "rot": tensor("quats"),
        }
    count = data["means"].shape[0]
    if count == 0 or any(value.shape[0] != count for value in data.values()):
        raise ValueError("inconsistent or empty Gaussian tensors")
    if not all(bool(torch.isfinite(value).all()) for value in data.values()):
        raise ValueError("checkpoint contains non-finite Gaussian values")
    return int(payload["step"]), data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a raw DN-Splatter Nerfstudio checkpoint to SuperSplat/3DGS PLY."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-opacity", type=float, default=0.0)
    args = parser.parse_args()
    if not 0.0 <= args.min_opacity <= 1.0:
        raise ValueError("--min-opacity must be in [0, 1]")

    step, data = load_checkpoint_gaussians(args.checkpoint)
    before = data["means"].shape[0]
    data = filter_gaussians(
        data,
        min_opacity=args.min_opacity,
        stride=1,
        max_gaussians=None,
        drop_radius_outliers=False,
        radius_outlier_quantile=1.0,
        drop_large_scales=False,
        large_scale_quantile=1.0,
    )
    count = data["means"].shape[0]
    write_binary_ply(args.output, data)

    rest_properties = data["f_rest"].shape[1] * 3
    property_count = 6 + 3 + rest_properties + 1 + 3 + 4
    header_bytes = (
        len("ply\nformat binary_little_endian 1.0\n")
        + len(f"element vertex {count}\n")
        + sum(len(f"property float {name}\n") for name in (
            ["x", "y", "z", "nx", "ny", "nz"]
            + [f"f_dc_{i}" for i in range(3)]
            + [f"f_rest_{i}" for i in range(rest_properties)]
            + ["opacity"]
            + [f"scale_{i}" for i in range(3)]
            + [f"rot_{i}" for i in range(4)]
        ))
        + len("end_header\n")
    )
    expected_bytes = header_bytes + count * property_count * 4
    if args.output.stat().st_size != expected_bytes:
        raise RuntimeError("PLY size does not match its header and vertex payload")
    print(f"checkpoint_step: {step}")
    print(f"gaussians: {count} / {before}")
    print(f"properties: {property_count}")
    print(f"wrote: {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
