from __future__ import annotations

import math

import torch
from torch import Tensor

from .config import RenderConfig


def scene_scale_from_bounds(bounds: tuple[Tensor, Tensor]) -> float:
    """Return the scene-bounds diagonal used as one normalized distance unit."""

    bmin, bmax = bounds
    scene_scale = float((bmax - bmin).norm().detach().cpu())
    if not math.isfinite(scene_scale) or scene_scale <= 0.0:
        raise ValueError(f"scene bounds must have a finite positive diagonal, got {scene_scale}")
    return scene_scale


def resolve_visibility_parameters(config: RenderConfig, scene_scale: float) -> tuple[float, float]:
    """Convert normalized visibility parameters to the scene's coordinate units.

    ``visibility_bias`` is a fraction of the bounds diagonal. The sigmoid
    steepness has inverse-distance units, so it scales inversely with the same
    diagonal. This is equivalent to evaluating visibility after dividing all
    scene distances by ``scene_scale``.
    """

    scene_scale = float(scene_scale)
    if not math.isfinite(scene_scale) or scene_scale <= 0.0:
        raise ValueError(f"scene_scale must be finite and positive, got {scene_scale}")
    if config.visibility_bias < 0.0:
        raise ValueError("visibility_bias must be non-negative")
    if config.visibility_steepness <= 0.0:
        raise ValueError("visibility_steepness must be positive")
    return (
        float(config.visibility_bias) * scene_scale,
        float(config.visibility_steepness) / scene_scale,
    )


def soft_visibility(
    alpha_trace: Tensor,
    d_light: Tensor,
    d_trace: Tensor,
    bias: float,
    steepness: float,
) -> Tensor:
    """AEGIR Equation 3 with one bias subtraction and explicit steepness."""

    return 1.0 - alpha_trace * torch.sigmoid(steepness * (d_light - bias - d_trace))
