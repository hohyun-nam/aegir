from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from .color import display_srgb, hdr_log_image
from .config import LossWeights
from .emitters import AreaEmitterModel


def grayscale(rgb: Tensor) -> Tensor:
    weights = torch.tensor([0.299, 0.587, 0.114], device=rgb.device, dtype=rgb.dtype)
    return (rgb * weights).sum(dim=-1, keepdim=True)


def _mask_like(mask: Tensor | None, value: Tensor) -> Tensor | None:
    if mask is None:
        return None
    if mask.ndim == value.ndim - 1:
        mask = mask.unsqueeze(-1)
    return mask.to(device=value.device, dtype=value.dtype)


def _masked_mean(value: Tensor, mask: Tensor | None) -> Tensor:
    mask = _mask_like(mask, value)
    if mask is None:
        return value.mean()
    expanded = mask.expand_as(value)
    return (value * expanded).sum() / expanded.sum().clamp_min(1.0)


def focus_map(
    rgb: Tensor,
    focus_lambda: float = 5.0,
    kernel_size: int = 7,
    mask: Tensor | None = None,
) -> Tensor:
    gray = grayscale(rgb)
    x = gray.permute(0, 3, 1, 2) if gray.ndim == 4 else gray.unsqueeze(0).permute(0, 3, 1, 2)
    if mask is None:
        avg = F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        valid = None
    else:
        valid = mask if mask.ndim == 4 else mask.unsqueeze(0)
        valid = valid.permute(0, 3, 1, 2).to(device=x.device, dtype=x.dtype)
        pooled_value = F.avg_pool2d(
            x * valid,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
        pooled_valid = F.avg_pool2d(
            valid,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
        avg = pooled_value / pooled_valid.clamp_min(1.0e-6)
    delta = (x - avg).abs()
    if valid is not None:
        delta = delta * valid
    w = 1.0 + focus_lambda * delta / (delta.amax(dim=(-2, -1), keepdim=True).clamp_min(1.0e-6))
    w = w.permute(0, 2, 3, 1)
    return w if rgb.ndim == 4 else w[0]


def ssim(x: Tensor, y: Tensor, window: int = 11, mask: Tensor | None = None) -> Tensor:
    batched = x.ndim == 4
    if not batched:
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
    x_chw = x.permute(0, 3, 1, 2)
    y_chw = y.permute(0, 3, 1, 2)
    channels = x_chw.shape[1]
    weight = torch.ones(
        (channels, 1, window, window),
        device=x.device,
        dtype=x.dtype,
    )
    if mask is None:
        weight = weight / float(window * window)
        mu_x = F.conv2d(x_chw, weight, padding=window // 2, groups=channels)
        mu_y = F.conv2d(y_chw, weight, padding=window // 2, groups=channels)
        sigma_x = F.conv2d(x_chw * x_chw, weight, padding=window // 2, groups=channels) - mu_x.pow(2)
        sigma_y = F.conv2d(y_chw * y_chw, weight, padding=window // 2, groups=channels) - mu_y.pow(2)
        sigma_xy = F.conv2d(x_chw * y_chw, weight, padding=window // 2, groups=channels) - mu_x * mu_y
        output_mask = None
    else:
        valid = mask if mask.ndim == 4 else mask.unsqueeze(0)
        valid = valid.permute(0, 3, 1, 2).to(device=x.device, dtype=x.dtype)
        valid = valid.expand(-1, channels, -1, -1)
        count = F.conv2d(valid, weight, padding=window // 2, groups=channels)
        count = count.clamp_min(1.0)
        mu_x = F.conv2d(x_chw * valid, weight, padding=window // 2, groups=channels) / count
        mu_y = F.conv2d(y_chw * valid, weight, padding=window // 2, groups=channels) / count
        sigma_x = (
            F.conv2d(x_chw.square() * valid, weight, padding=window // 2, groups=channels) / count
            - mu_x.square()
        )
        sigma_y = (
            F.conv2d(y_chw.square() * valid, weight, padding=window // 2, groups=channels) / count
            - mu_y.square()
        )
        sigma_xy = (
            F.conv2d(x_chw * y_chw * valid, weight, padding=window // 2, groups=channels) / count
            - mu_x * mu_y
        )
        output_mask = valid
    c1 = 0.01**2
    c2 = 0.03**2
    val = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.pow(2) + mu_y.pow(2) + c1) * (sigma_x + sigma_y + c2)
    )
    if output_mask is None:
        out = val.mean()
    else:
        out = (val * output_mask).sum() / output_mask.sum().clamp_min(1.0)
        if not bool(output_mask.any()):
            out = out + 1.0
    return out if batched else out.squeeze()


def photometric_loss(
    pred: Tensor,
    target: Tensor,
    weights: LossWeights,
    mask: Tensor | None = None,
    focus_weights: Tensor | None = None,
    saturation_threshold: float | None = None,
) -> Tensor:
    if focus_weights is None:
        focus_weights = focus_map(target, weights.focus_lambda, mask=mask)
    elif focus_weights.shape != target.shape[:-1] + (1,):
        raise ValueError(
            "focus_weights must have one weight per target pixel; "
            f"got {tuple(focus_weights.shape)} for target {tuple(target.shape)}"
        )
    w = focus_weights.to(device=pred.device, dtype=pred.dtype)
    error = (pred - target).abs()
    if saturation_threshold is not None:
        saturated = target.amax(dim=-1, keepdim=True) >= saturation_threshold
        error = torch.where(saturated, (target - pred).clamp_min(0.0), error)
    l1 = _masked_mean(w * error, mask)
    return l1 + weights.dssim_lambda * (1.0 - ssim(pred.clamp(0.0, 1.0), target.clamp(0.0, 1.0), mask=mask))


def hdr_photometric_loss(
    pred_hdr: Tensor,
    target_hdr: Tensor,
    weights: LossWeights,
    mask: Tensor | None = None,
    focus_weights: Tensor | None = None,
    display_exposure: Tensor | float = 1.0,
) -> Tensor:
    """Compare scene-linear HDR through a symmetric log-domain objective."""

    pred_log = hdr_log_image(pred_hdr)
    target_log = hdr_log_image(target_hdr)
    if focus_weights is None:
        focus_weights = focus_map(target_log, weights.focus_lambda, mask=mask)
    elif focus_weights.shape != target_hdr.shape[:-1] + (1,):
        raise ValueError("focus_weights must have one weight per HDR target pixel")
    l1 = _masked_mean(focus_weights.to(pred_log) * (pred_log - target_log).abs(), mask)
    structural = ssim(
        display_srgb(pred_hdr, display_exposure),
        display_srgb(target_hdr, display_exposure),
        mask=mask,
    )
    return l1 + weights.dssim_lambda * (1.0 - structural)


def photometric_loss_sparse(
    pred: Tensor,
    target: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    return _masked_mean((pred - target).abs(), mask)


def material_prior_loss(
    rendered: dict[str, Tensor],
    priors: dict[str, Tensor],
    weights: LossWeights,
    mask: Tensor | None = None,
) -> Tensor:
    total = torch.zeros((), device=next(iter(rendered.values())).device)
    if "albedo" in priors:
        total = total + weights.prior_albedo * _masked_mean(
            (rendered["albedo"] - priors["albedo"]).abs(), mask
        )
    if "roughness" in priors:
        total = total + weights.prior_roughness * _masked_mean(
            (rendered["roughness"] - priors["roughness"]).abs(), mask
        )
    if "metallic" in priors:
        total = total + weights.prior_metallic * _masked_mean(
            (rendered["metallic"] - priors["metallic"]).abs(), mask
        )
    return total


def edge_aware_tv(
    rendered: dict[str, Tensor],
    target_rgb: Tensor,
    weights: LossWeights,
    mask: Tensor | None = None,
) -> Tensor:
    squeeze = target_rgb.ndim == 3
    if squeeze:
        target_rgb = target_rgb.unsqueeze(0)
        rendered = {key: value.unsqueeze(0) if value.ndim == 3 else value for key, value in rendered.items()}
        if mask is not None and mask.ndim == 3:
            mask = mask.unsqueeze(0)
    image_grad_x = (target_rgb[:, :, 1:, :] - target_rgb[:, :, :-1, :]).abs().mean(dim=-1, keepdim=True)
    image_grad_y = (target_rgb[:, 1:, :, :] - target_rgb[:, :-1, :, :]).abs().mean(dim=-1, keepdim=True)
    pair_mask_x = None if mask is None else mask[:, :, 1:, :] * mask[:, :, :-1, :]
    pair_mask_y = None if mask is None else mask[:, 1:, :, :] * mask[:, :-1, :, :]
    total = torch.zeros((), device=target_rgb.device)
    for key, gamma in (
        ("albedo", weights.tv_albedo),
        ("roughness", weights.tv_roughness),
        ("metallic", weights.tv_metallic),
    ):
        if key not in rendered:
            continue
        value = rendered[key]
        grad_x = (value[:, :, 1:, :] - value[:, :, :-1, :]).abs()
        grad_y = (value[:, 1:, :, :] - value[:, :-1, :, :]).abs()
        total = total + gamma * (
            _masked_mean(
                grad_x * torch.exp(-weights.tv_beta * image_grad_x),
                pair_mask_x,
            )
            + _masked_mean(
                grad_y * torch.exp(-weights.tv_beta * image_grad_y),
                pair_mask_y,
            )
        )
    return weights.tv_lambda * total


def emitter_regularization(
    emitters: AreaEmitterModel,
    bounds: tuple[Tensor, Tensor],
    weights: LossWeights,
    surface_points: Tensor | None = None,
    cameras: dict[str, Tensor] | None = None,
) -> Tensor:
    scales = emitters.scales
    colors = emitters.colors
    centers = emitters.centers
    bmin, bmax = bounds
    scene_scale = (bmax.to(centers) - bmin.to(centers)).norm().clamp_min(1.0e-6)
    # AEGIR Eq. 19 is the squared L2 norm of each emitter's scale in the
    # normalized scene coordinates used by the paper.  This code keeps scene
    # geometry in its source units, so normalize every scale axis first and
    # then square.  Using ||s|| / scene_scale here would be a first-power
    # penalty and would over-regularize small emitters relative to Eq. 19.
    normalized_scales = scales / scene_scale
    area = normalized_scales.square().sum(dim=-1).mean()
    white = (colors.max(dim=-1).values - colors.min(dim=-1).values).mean()
    lower = (bmin.to(centers) - centers).clamp_min(0.0)
    upper = (centers - bmax.to(centers)).clamp_min(0.0)
    bound = ((lower + upper).sum(dim=-1) / scene_scale).mean()
    surface = _surface_proximity_loss(centers, surface_points, scene_scale, weights)
    view = _camera_frustum_loss(centers, cameras, scene_scale, weights)
    return (
        weights.light_area * area
        + weights.light_white * white
        + weights.light_bounds * bound
        + weights.light_surface * surface
        + weights.light_view * view
    )


def _surface_proximity_loss(
    centers: Tensor,
    surface_points: Tensor | None,
    scene_scale: Tensor,
    weights: LossWeights,
) -> Tensor:
    if surface_points is None or weights.light_surface <= 0.0:
        return torch.zeros((), device=centers.device, dtype=centers.dtype)
    anchors = surface_points.to(device=centers.device, dtype=centers.dtype)
    finite = torch.isfinite(anchors).all(dim=-1)
    if not bool(finite.any()):
        return torch.zeros((), device=centers.device, dtype=centers.dtype)
    anchors = anchors[finite]
    distances = torch.cdist(centers.unsqueeze(0), anchors.unsqueeze(0)).squeeze(0).amin(dim=-1)
    allowed = scene_scale * weights.surface_distance_fraction
    return (distances - allowed).clamp_min(0.0).div(scene_scale).mean()


def _camera_frustum_loss(
    centers: Tensor,
    cameras: dict[str, Tensor] | None,
    scene_scale: Tensor,
    weights: LossWeights,
) -> Tensor:
    if cameras is None or weights.light_view <= 0.0:
        return torch.zeros((), device=centers.device, dtype=centers.dtype)
    if not cameras:
        return torch.zeros((), device=centers.device, dtype=centers.dtype)

    c2w = cameras["c2w"].to(device=centers.device, dtype=centers.dtype)
    fx = cameras["fx"].to(device=centers.device, dtype=centers.dtype).view(-1, 1)
    fy = cameras["fy"].to(device=centers.device, dtype=centers.dtype).view(-1, 1)
    cx = cameras["cx"].to(device=centers.device, dtype=centers.dtype).view(-1, 1)
    cy = cameras["cy"].to(device=centers.device, dtype=centers.dtype).view(-1, 1)
    width = cameras["width"].to(device=centers.device, dtype=centers.dtype).view(-1, 1).clamp_min(1.0)
    height = cameras["height"].to(device=centers.device, dtype=centers.dtype).view(-1, 1).clamp_min(1.0)
    convention = cameras["convention"].to(device=centers.device).view(-1, 1)

    w2c = torch.linalg.inv(c2w)
    homo = torch.cat(
        [centers, torch.ones((centers.shape[0], 1), device=centers.device, dtype=centers.dtype)], dim=-1
    )
    cam = homo.unsqueeze(0) @ w2c.transpose(-1, -2)
    cam_x = torch.where(convention == 2, -cam[..., 0], cam[..., 0])
    cam_y = torch.where(convention == 1, -cam[..., 1], cam[..., 1])
    cam_y = torch.where(convention == 2, -cam[..., 1], cam_y)
    depth = torch.where(convention == 1, -cam[..., 2], cam[..., 2])
    safe_depth = torch.where(depth.abs() < 1.0e-8, torch.full_like(depth, 1.0e-8), depth)
    u = fx * cam_x / safe_depth + cx
    v = fy * cam_y / safe_depth + cy

    depth_margin = scene_scale * weights.view_depth_margin_fraction
    behind = (depth_margin - depth).clamp_min(0.0).div(scene_scale)
    left = (-u).clamp_min(0.0).div(width)
    right = (u - (width - 1.0)).clamp_min(0.0).div(width)
    top = (-v).clamp_min(0.0).div(height)
    bottom = (v - (height - 1.0)).clamp_min(0.0).div(height)
    per_view = behind + left + right + top + bottom
    return per_view.amin(dim=0).mean()


def total_loss(
    rendered: dict[str, Tensor],
    target_rgb: Tensor,
    emitters: AreaEmitterModel,
    bounds: tuple[Tensor, Tensor],
    weights: LossWeights,
    priors: dict[str, Tensor] | None = None,
    surface_points: Tensor | None = None,
    cameras: dict[str, Tensor] | None = None,
    mask: Tensor | None = None,
) -> dict[str, Tensor]:
    photo = photometric_loss(rendered["rgb"], target_rgb, weights, mask)
    prior = material_prior_loss(rendered, priors or {}, weights, mask)
    tv = edge_aware_tv(rendered, target_rgb, weights, mask)
    light = emitter_regularization(emitters, bounds, weights, surface_points, cameras)
    return {
        "total": photo + prior + tv + light,
        "photo": photo,
        "prior": prior,
        "tv": tv,
        "light": light,
    }


def total_loss_sparse(
    rendered: dict[str, Tensor],
    target_rgb: Tensor,
    emitters: AreaEmitterModel,
    bounds: tuple[Tensor, Tensor],
    weights: LossWeights,
    priors: dict[str, Tensor] | None = None,
    surface_points: Tensor | None = None,
    cameras: dict[str, Tensor] | None = None,
    mask: Tensor | None = None,
) -> dict[str, Tensor]:
    photo = photometric_loss_sparse(rendered["rgb"], target_rgb, mask)
    prior = material_prior_loss(rendered, priors or {}, weights, mask)
    tv = torch.zeros((), device=target_rgb.device)
    light = emitter_regularization(emitters, bounds, weights, surface_points, cameras)
    return {
        "total": photo + prior + tv + light,
        "photo": photo,
        "prior": prior,
        "tv": tv,
        "light": light,
    }
