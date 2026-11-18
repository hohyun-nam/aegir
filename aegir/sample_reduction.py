"""Opt-in reduction of per-ray linear RGB direct-light contributions."""
from __future__ import annotations

import math

from torch import Tensor
import torch


def central80_bounds(num_samples: int) -> tuple[int, int]:
    """Nearest symmetric integer 10% tails:128->13/102/13,512->51/410/51."""
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    tail = min(int(math.floor(num_samples * 0.1 + 0.5)), (num_samples - 1) // 2)
    return tail, num_samples - tail


def central_fraction_bounds(num_samples: int, keep_fraction: float) -> tuple[int, int]:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if not math.isfinite(keep_fraction) or not 0 < keep_fraction <= 1:
        raise ValueError("direct_sample_keep_fraction must be finite and in (0, 1]")
    # Round representation error, preserving symmetric nearest-integer tails.
    tail_fraction = round((1.0 - keep_fraction) * .5, 12)
    tail = min(int(math.floor(num_samples * tail_fraction + .5)), (num_samples - 1) // 2)
    return tail, num_samples - tail


def reduce_direct_samples_many(contributions: Tensor, keep_fractions) -> dict[float, tuple[Tensor, int]]:
    """Reduce identical rays at multiple fractions, sorting luminance only once."""
    if contributions.ndim < 2 or contributions.shape[-1] != 3 or contributions.shape[-2] <= 0:
        raise ValueError("contributions must have shape [..., positive samples, 3]")
    num_samples = contributions.shape[-2]
    bounds = {float(f): central_fraction_bounds(num_samples, float(f)) for f in keep_fractions}
    results = {}
    if 1.0 in bounds:
        results[1.0] = (contributions.sum(dim=-2), num_samples)
    if any(f < 1 for f in bounds):
        luminance = (contributions * contributions.new_tensor((.2126, .7152, .0722))).sum(-1)
        order = torch.argsort(luminance, dim=-1, stable=True)
        for fraction, (start, stop) in bounds.items():
            if fraction == 1:
                continue
            indices = order[..., start:stop]
            selected = contributions.gather(-2, indices[..., None].expand(indices.shape + (3,)))
            results[fraction] = (selected.sum(dim=-2), stop - start)
    return results


def reduce_direct_samples(contributions: Tensor, mode: str = "mean", keep_fraction: float = .8) -> tuple[Tensor, int]:
    """Return an RGB sum and its actual denominator for each shading point.

    Sort *all emitters together*, using final scene-linear HDR contribution
    luminance after BRDF, visibility, cosine and PDF weighting. Select the
    same sample indices for all RGB channels. Keep zero/occluded samples in
    the population. No exposure, tonemapping, spatial filtering, or RGB clamp.
    """
    if contributions.ndim < 2 or contributions.shape[-1] != 3:
        raise ValueError("contributions must have shape [..., samples, 3]")
    num_samples = contributions.shape[-2]
    if num_samples <= 0:
        raise ValueError("cannot reduce an empty sample population")
    if mode == "mean":
        # Preserve the historical summation and default floating-point result.
        return contributions.sum(dim=-2), num_samples
    if mode not in ("trimmed80", "trimmed"):
        raise ValueError("direct_sample_reduction must be mean, trimmed80 or trimmed")
    if mode == "trimmed" and keep_fraction == 1.0:
        return contributions.sum(dim=-2), num_samples
    start, stop = (central80_bounds(num_samples) if mode == "trimmed80" else
                   central_fraction_bounds(num_samples, keep_fraction))
    luminance = (contributions * contributions.new_tensor((0.2126, 0.7152, 0.0722))).sum(-1)
    order = torch.argsort(luminance, dim=-1, stable=True)[..., start:stop]
    selected = contributions.gather(-2, order[..., None].expand(order.shape + (3,)))
    return selected.sum(dim=-2), stop - start
