from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def _single_channel(value: Tensor, name: str, height: int, width: int) -> Tensor:
    if value.shape == (height, width):
        return value
    if value.shape == (height, width, 1):
        return value[..., 0]
    raise ValueError(f"{name} must have shape [H, W] or [H, W, 1]")


def _shifted(value: Tensor, dy: int, dx: int, radius: int, height: int, width: int) -> Tensor:
    return value[..., radius + dy : radius + dy + height, radius + dx : radius + dx + width]


def cross_bilateral_denoise(
    color: Tensor,
    depth: Tensor,
    normals: Tensor,
    valid_mask: Tensor | None = None,
    *,
    radius: int = 2,
    sigma_spatial: float = 2.0,
    sigma_depth: float = 1.0,
    normal_power: float = 32.0,
    depth_epsilon: float = 1.0e-3,
) -> Tensor:
    """Denoise an HWC radiance image while respecting depth and normal discontinuities.

    The depth term follows the gradient-aware cross-bilateral weighting used by
    SVGF-style filters. Invalid/background pixels keep their original value and
    cannot contribute to neighboring surface pixels.
    """
    if color.ndim != 3:
        raise ValueError("color must have shape [H, W, C]")
    height, width, _ = color.shape
    if normals.shape != (height, width, 3):
        raise ValueError("normals must have shape [H, W, 3]")
    depth_2d = _single_channel(depth, "depth", height, width)
    if valid_mask is None:
        requested_valid = torch.ones((height, width), dtype=torch.bool, device=color.device)
    else:
        requested_valid = _single_channel(valid_mask, "valid_mask", height, width).bool()
    if depth.device != color.device or normals.device != color.device or requested_valid.device != color.device:
        raise ValueError("color, depth, normals, and valid_mask must be on the same device")
    if radius < 0:
        raise ValueError("radius must be non-negative")
    if sigma_spatial <= 0.0 or sigma_depth <= 0.0 or normal_power < 0.0 or depth_epsilon <= 0.0:
        raise ValueError("denoiser scales and epsilon must be positive, and normal_power non-negative")
    if radius == 0 or height == 0 or width == 0:
        return color.clone()

    work_dtype = torch.float32 if color.dtype in (torch.float16, torch.bfloat16) else color.dtype
    work_color = color.to(work_dtype)
    work_depth = depth_2d.to(work_dtype)
    work_normals = normals.to(work_dtype)
    valid = (
        requested_valid
        & torch.isfinite(work_depth)
        & (work_depth > 0.0)
        & torch.isfinite(work_color).all(dim=-1)
        & torch.isfinite(work_normals).all(dim=-1)
    )
    safe_depth = torch.where(valid, work_depth, torch.zeros_like(work_depth))
    safe_normals = F.normalize(torch.nan_to_num(work_normals), dim=-1, eps=1.0e-8)
    safe_color = torch.nan_to_num(work_color)

    depth_chw = safe_depth[None, None]
    valid_chw = valid[None, None]
    one_pad = (1, 1, 1, 1)
    padded_depth_1 = F.pad(depth_chw, one_pad, mode="replicate")
    padded_valid_1 = F.pad(valid_chw.float(), one_pad, mode="replicate").bool()
    center = safe_depth
    left = padded_depth_1[0, 0, 1 : height + 1, 0:width]
    right = padded_depth_1[0, 0, 1 : height + 1, 2 : width + 2]
    up = padded_depth_1[0, 0, 0:height, 1 : width + 1]
    down = padded_depth_1[0, 0, 2 : height + 2, 1 : width + 1]
    left_valid = padded_valid_1[0, 0, 1 : height + 1, 0:width]
    right_valid = padded_valid_1[0, 0, 1 : height + 1, 2 : width + 2]
    up_valid = padded_valid_1[0, 0, 0:height, 1 : width + 1]
    down_valid = padded_valid_1[0, 0, 2 : height + 2, 1 : width + 1]
    left = torch.where(left_valid, left, center)
    right = torch.where(right_valid, right, center)
    up = torch.where(up_valid, up, center)
    down = torch.where(down_valid, down, center)
    grad_x = 0.5 * (right - left)
    grad_y = 0.5 * (down - up)

    pad = (radius, radius, radius, radius)
    padded_color = F.pad(safe_color.permute(2, 0, 1)[None], pad, mode="replicate")
    padded_depth = F.pad(depth_chw, pad, mode="replicate")
    padded_normals = F.pad(safe_normals.permute(2, 0, 1)[None], pad, mode="replicate")
    padded_valid = F.pad(valid_chw.float(), pad, mode="constant", value=0.0)
    numerator = torch.zeros_like(safe_color)
    denominator = torch.zeros((height, width), dtype=work_dtype, device=color.device)

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            neighbor_color = _shifted(padded_color, dy, dx, radius, height, width)[0].permute(1, 2, 0)
            neighbor_depth = _shifted(padded_depth, dy, dx, radius, height, width)[0, 0]
            neighbor_normal = _shifted(padded_normals, dy, dx, radius, height, width)[0].permute(1, 2, 0)
            neighbor_valid = _shifted(padded_valid, dy, dx, radius, height, width)[0, 0].bool()
            spatial_weight = math.exp(-(dx * dx + dy * dy) / (2.0 * sigma_spatial * sigma_spatial))
            depth_scale = sigma_depth * (grad_x.abs() * abs(dx) + grad_y.abs() * abs(dy)) + depth_epsilon
            depth_weight = torch.exp(-(safe_depth - neighbor_depth).abs() / depth_scale)
            normal_weight = (safe_normals * neighbor_normal).sum(dim=-1).clamp(0.0, 1.0).pow(normal_power)
            weight = valid & neighbor_valid
            weight = weight.to(work_dtype) * spatial_weight * depth_weight * normal_weight
            numerator.add_(neighbor_color * weight[..., None])
            denominator.add_(weight)

    filtered = numerator / denominator.clamp_min(torch.finfo(work_dtype).eps)[..., None]
    result = torch.where(valid[..., None], filtered, work_color)
    return result.to(color.dtype)


def edge_aware_vector_median_denoise(
    color: Tensor,
    depth: Tensor,
    normals: Tensor,
    valid_mask: Tensor | None = None,
    *,
    radius: int = 1,
    depth_multiplier: float = 3.0,
    normal_cosine: float = 0.8,
    depth_epsilon: float = 1.0e-3,
) -> Tensor:
    """Reject HDR fireflies with an edge-aware luminance vector median.

    The selected output is an existing RGB vector whose luminance is the median
    of compatible neighbors. This avoids the false colors produced by taking an
    independent median in each channel. Depth gradients and normals prevent
    samples from crossing surface discontinuities.
    """
    if color.ndim != 3:
        raise ValueError("color must have shape [H, W, C]")
    height, width, channels = color.shape
    if channels != 3 or normals.shape != (height, width, 3):
        raise ValueError("color and normals must have shape [H, W, 3]")
    depth_2d = _single_channel(depth, "depth", height, width)
    if valid_mask is None:
        requested_valid = torch.ones((height, width), dtype=torch.bool, device=color.device)
    else:
        requested_valid = _single_channel(valid_mask, "valid_mask", height, width).bool()
    if depth.device != color.device or normals.device != color.device or requested_valid.device != color.device:
        raise ValueError("color, depth, normals, and valid_mask must be on the same device")
    if radius < 0 or depth_multiplier <= 0.0 or not 0.0 <= normal_cosine <= 1.0 or depth_epsilon <= 0.0:
        raise ValueError("invalid median denoiser parameters")
    if radius == 0 or height == 0 or width == 0:
        return color.clone()

    work_dtype = torch.float32 if color.dtype in (torch.float16, torch.bfloat16) else color.dtype
    work_color = torch.nan_to_num(color.to(work_dtype))
    work_depth = depth_2d.to(work_dtype)
    work_normals = F.normalize(torch.nan_to_num(normals.to(work_dtype)), dim=-1, eps=1.0e-8)
    valid = (
        requested_valid
        & torch.isfinite(work_depth)
        & (work_depth > 0.0)
        & torch.isfinite(color).all(dim=-1)
        & torch.isfinite(normals).all(dim=-1)
    )
    safe_depth = torch.where(valid, work_depth, torch.zeros_like(work_depth))

    depth_chw = safe_depth[None, None]
    valid_chw = valid[None, None]
    padded_depth_1 = F.pad(depth_chw, (1, 1, 1, 1), mode="replicate")
    padded_valid_1 = F.pad(valid_chw.float(), (1, 1, 1, 1), mode="replicate").bool()
    center = safe_depth
    neighbors = (
        (padded_depth_1[0, 0, 1 : height + 1, :width], padded_valid_1[0, 0, 1 : height + 1, :width]),
        (padded_depth_1[0, 0, 1 : height + 1, 2 : width + 2], padded_valid_1[0, 0, 1 : height + 1, 2 : width + 2]),
        (padded_depth_1[0, 0, :height, 1 : width + 1], padded_valid_1[0, 0, :height, 1 : width + 1]),
        (padded_depth_1[0, 0, 2 : height + 2, 1 : width + 1], padded_valid_1[0, 0, 2 : height + 2, 1 : width + 1]),
    )
    left, right, up, down = [torch.where(ok, value, center) for value, ok in neighbors]
    grad_x = 0.5 * (right - left)
    grad_y = 0.5 * (down - up)

    pad = (radius, radius, radius, radius)
    padded_color = F.pad(work_color.permute(2, 0, 1)[None], pad, mode="replicate")
    padded_depth = F.pad(depth_chw, pad, mode="replicate")
    padded_normals = F.pad(work_normals.permute(2, 0, 1)[None], pad, mode="replicate")
    padded_valid = F.pad(valid_chw.float(), pad, mode="constant", value=0.0)
    colors = []
    eligible = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            neighbor_color = _shifted(padded_color, dy, dx, radius, height, width)[0].permute(1, 2, 0)
            neighbor_depth = _shifted(padded_depth, dy, dx, radius, height, width)[0, 0]
            neighbor_normal = _shifted(padded_normals, dy, dx, radius, height, width)[0].permute(1, 2, 0)
            neighbor_valid = _shifted(padded_valid, dy, dx, radius, height, width)[0, 0].bool()
            depth_scale = grad_x.abs() * abs(dx) + grad_y.abs() * abs(dy) + depth_epsilon
            depth_ok = (safe_depth - neighbor_depth).abs() <= depth_multiplier * depth_scale
            normal_ok = (work_normals * neighbor_normal).sum(dim=-1) >= normal_cosine
            colors.append(neighbor_color)
            eligible.append(valid & neighbor_valid & depth_ok & normal_ok)

    color_stack = torch.stack(colors)
    eligible_stack = torch.stack(eligible)
    luminance_weights = work_color.new_tensor((0.2126, 0.7152, 0.0722))
    luminance = (color_stack * luminance_weights).sum(dim=-1)
    order = luminance.masked_fill(~eligible_stack, torch.inf).argsort(dim=0)
    count = eligible_stack.sum(dim=0).clamp_min(1)
    rank = ((count - 1) // 2).unsqueeze(0)
    selected = order.gather(0, rank).squeeze(0)
    gather_index = selected[None, ..., None].expand(1, height, width, channels)
    filtered = color_stack.gather(0, gather_index).squeeze(0)
    return torch.where(valid[..., None], filtered, work_color).to(color.dtype)
