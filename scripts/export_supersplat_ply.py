from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def sigmoid_to_logit(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp(1.0e-6, 1.0 - 1.0e-6)
    return torch.log(value / (1.0 - value))


def load_gaussian_tensors(path: Path) -> dict[str, torch.Tensor]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    means = data["means"].detach().cpu().float()
    normals = data.get("normals", torch.zeros_like(means)).detach().cpu().float()
    f_dc = data.get("features_dc_raw", data.get("features_dc")).detach().cpu().float()
    f_rest = data.get("features_rest_raw")
    if f_rest is None:
        f_rest = torch.zeros((means.shape[0], 15, 3), dtype=torch.float32)
    else:
        f_rest = f_rest.detach().cpu().float()
    opacity = data.get("opacities_logits")
    if opacity is None:
        opacity = sigmoid_to_logit(data["opacities"].detach().cpu().float())
    else:
        opacity = opacity.detach().cpu().float()
    scale = data.get("log_scales")
    if scale is None:
        scale = data["scales"].detach().cpu().float().clamp_min(1.0e-8).log()
    else:
        scale = scale.detach().cpu().float()
    rot = data["quats"].detach().cpu().float()
    return {
        "means": means,
        "normals": normals,
        "f_dc": f_dc,
        "f_rest": f_rest,
        "opacity": opacity,
        "scale": scale,
        "rot": rot,
    }


def filter_gaussians(
    data: dict[str, torch.Tensor],
    min_opacity: float,
    stride: int,
    max_gaussians: int | None,
    drop_radius_outliers: bool,
    radius_outlier_quantile: float,
    drop_large_scales: bool,
    large_scale_quantile: float,
) -> dict[str, torch.Tensor]:
    n = data["means"].shape[0]
    finite = torch.ones(n, dtype=torch.bool)
    for value in data.values():
        finite &= torch.isfinite(value.reshape(n, -1)).all(dim=1)
    opacity = data["opacity"].sigmoid().reshape(-1)
    keep = finite & (opacity >= min_opacity)
    if drop_radius_outliers and bool(finite.any()):
        means = data["means"][finite]
        center = means.median(dim=0).values
        radius = torch.full((n,), torch.inf)
        radius[finite] = (data["means"][finite] - center).norm(dim=-1)
        cutoff = torch.quantile(radius[finite], radius_outlier_quantile)
        keep &= radius <= cutoff
    if drop_large_scales and bool(finite.any()):
        scale_max = data["scale"].exp().max(dim=-1).values
        cutoff = torch.quantile(scale_max[finite], large_scale_quantile)
        keep &= scale_max <= cutoff
    indices = torch.nonzero(keep, as_tuple=False).reshape(-1)
    if stride > 1:
        indices = indices[::stride]
    if max_gaussians is not None and indices.shape[0] > max_gaussians:
        sample_ids = torch.linspace(0, indices.shape[0] - 1, steps=max_gaussians).round().long()
        indices = indices[sample_ids]
    return {key: value[indices].contiguous() for key, value in data.items()}


def write_binary_ply(path: Path, data: dict[str, torch.Tensor]) -> None:
    means = data["means"].numpy().astype("<f4")
    normals = data["normals"].numpy().astype("<f4")
    f_dc = data["f_dc"].reshape(-1, 3).numpy().astype("<f4")
    f_rest = data["f_rest"].permute(0, 2, 1).reshape(data["means"].shape[0], -1).numpy().astype("<f4")
    opacity = data["opacity"].reshape(-1, 1).numpy().astype("<f4")
    scale = data["scale"].reshape(-1, 3).numpy().astype("<f4")
    rot = data["rot"].reshape(-1, 4).numpy().astype("<f4")

    names = ["x", "y", "z", "nx", "ny", "nz"]
    names += [f"f_dc_{i}" for i in range(3)]
    names += [f"f_rest_{i}" for i in range(f_rest.shape[1])]
    names += ["opacity"]
    names += [f"scale_{i}" for i in range(3)]
    names += [f"rot_{i}" for i in range(4)]

    values = np.concatenate([means, normals, f_dc, f_rest, opacity, scale, rot], axis=1)
    dtype = np.dtype([(name, "<f4") for name in names])
    vertices = np.empty(values.shape[0], dtype=dtype)
    for idx, name in enumerate(names):
        vertices[name] = values[:, idx]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(f"element vertex {vertices.shape[0]}\n".encode("ascii"))
        for name in names:
            f.write(f"property float {name}\n".encode("ascii"))
        f.write(b"end_header\n")
        vertices.tofile(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export DN-Splatter Gaussians to a SuperSplat/3DGS PLY file.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-opacity", type=float, default=0.01)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-gaussians", type=int, default=None)
    parser.add_argument("--drop-radius-outliers", action="store_true")
    parser.add_argument("--radius-outlier-quantile", type=float, default=0.995)
    parser.add_argument("--drop-large-scales", action="store_true")
    parser.add_argument("--large-scale-quantile", type=float, default=0.995)
    args = parser.parse_args()

    if args.stride < 1:
        raise ValueError("--stride must be >= 1")
    for name, value in (
        ("--radius-outlier-quantile", args.radius_outlier_quantile),
        ("--large-scale-quantile", args.large_scale_quantile),
    ):
        if not 0.0 < value <= 1.0:
            raise ValueError(f"{name} must be in (0, 1]")

    data = load_gaussian_tensors(args.input)
    before = data["means"].shape[0]
    data = filter_gaussians(
        data,
        args.min_opacity,
        args.stride,
        args.max_gaussians,
        args.drop_radius_outliers,
        args.radius_outlier_quantile,
        args.drop_large_scales,
        args.large_scale_quantile,
    )
    after = data["means"].shape[0]
    write_binary_ply(args.output, data)
    print(f"wrote {after} / {before} gaussians to {args.output}")


if __name__ == "__main__":
    main()
