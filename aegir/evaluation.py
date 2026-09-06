from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .losses import ssim


class SpatialPerceptualMetric(Protocol):
    def __call__(self, prediction: Tensor, target: Tensor) -> Tensor: ...


def _valid_mask(mask: Tensor, value: Tensor) -> Tensor:
    if mask.ndim == value.ndim - 1:
        mask = mask.unsqueeze(-1)
    if mask.shape[:-1] != value.shape[:-1] or mask.shape[-1] not in (1, value.shape[-1]):
        raise ValueError(f"mask shape {tuple(mask.shape)} is incompatible with image {tuple(value.shape)}")
    valid = mask.to(device=value.device, dtype=torch.bool)
    if not bool(valid.any()):
        raise ValueError("metric mask has no valid pixels")
    return valid


def masked_psnr(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    valid = _valid_mask(mask, prediction).expand_as(prediction)
    mse = prediction.sub(target).square()[valid].mean().clamp_min(1.0e-12)
    return -10.0 * torch.log10(mse)


def masked_ssim(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    valid = _valid_mask(mask, prediction)
    return ssim(prediction.clamp(0.0, 1.0), target.clamp(0.0, 1.0), mask=valid)


def build_lpips_model(net: str, device: str | torch.device) -> nn.Module:
    try:
        import lpips
    except ImportError as exc:
        raise RuntimeError(
            "LPIPS evaluation requires the optional eval dependencies; "
            "install this project with `pip install -e '.[eval]'`"
        ) from exc
    if net not in {"alex", "vgg", "squeeze"}:
        raise ValueError("LPIPS network must be alex, vgg, or squeeze")
    model = lpips.LPIPS(net=net, spatial=True, verbose=False)
    return model.eval().requires_grad_(False).to(device)


def masked_lpips(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    model: SpatialPerceptualMetric,
    device: str | torch.device,
) -> Tensor:
    """LPIPS averaged only over valid positions in the spatial score map."""

    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    valid = _valid_mask(mask, prediction)
    expanded = valid.expand_as(prediction)
    # Equal inputs outside the mask ensure invalid RGB values cannot affect the
    # perceptual difference, including through convolutional receptive fields.
    prediction = torch.where(expanded, prediction, target)
    pred_nchw = prediction.permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
    target_nchw = target.permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
    score_map = model(pred_nchw.mul(2.0).sub(1.0), target_nchw.mul(2.0).sub(1.0))
    if score_map.ndim != 4:
        raise ValueError("LPIPS model must return a spatial [N,C,H,W] score map")
    valid_nchw = valid.permute(2, 0, 1).unsqueeze(0).to(device=score_map.device, dtype=score_map.dtype)
    valid_map = F.interpolate(valid_nchw, size=score_map.shape[-2:], mode="nearest")
    valid_map = valid_map.expand_as(score_map)
    return (score_map * valid_map).sum() / valid_map.sum().clamp_min(1.0)


def save_raw_metric_frame(
    path: str | Path,
    rendered_rgb: Tensor,
    rendered_hdr: Tensor,
    target_rgb: Tensor,
    valid_mask: Tensor,
    target_hdr: Tensor | None = None,
) -> None:
    """Persist float metric inputs separately from quantized visualization PNGs."""

    arrays = {
        "render_rgb": rendered_rgb.detach().float().cpu().numpy(),
        "render_hdr_linear": rendered_hdr.detach().float().cpu().numpy(),
        "target_rgb": target_rgb.detach().float().cpu().numpy(),
        "valid_mask": valid_mask.detach().bool().cpu().numpy(),
    }
    if target_hdr is not None:
        arrays["target_hdr_linear"] = target_hdr.detach().float().cpu().numpy()
    np.savez_compressed(Path(path), **arrays)
