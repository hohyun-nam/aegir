from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from aegir.config import RenderConfig
from aegir.emitters import AreaEmitterModel
from aegir.evaluation import masked_lpips, masked_psnr, masked_ssim, save_raw_metric_frame
from aegir.render import DeferredRenderer


class _PixelDistance(nn.Module):
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (prediction - target).square().mean(dim=1, keepdim=True)


def _masked_pair() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target = torch.zeros((16, 16, 3))
    prediction = target.clone()
    prediction[:, :8] = 0.25
    mask = torch.zeros((16, 16, 1), dtype=torch.bool)
    mask[:, :8] = True
    return prediction, target, mask


def test_all_metrics_ignore_invalid_prediction_pixels() -> None:
    prediction, target, mask = _masked_pair()
    altered = prediction.clone()
    altered[:, 8:] = 1_000.0
    model = _PixelDistance()

    assert masked_psnr(prediction, target, mask) == pytest.approx(masked_psnr(altered, target, mask))
    assert masked_ssim(prediction, target, mask) == pytest.approx(masked_ssim(altered, target, mask))
    assert masked_lpips(prediction, target, mask, model, "cpu") == pytest.approx(
        masked_lpips(altered, target, mask, model, "cpu")
    )


@pytest.mark.parametrize("metric", [masked_psnr, masked_ssim])
def test_pixel_metrics_reject_empty_mask(metric) -> None:
    image = torch.zeros((16, 16, 3))
    with pytest.raises(ValueError, match="no valid pixels"):
        metric(image, image, torch.zeros((16, 16, 1), dtype=torch.bool))


def test_raw_metric_frame_is_float_and_separate_from_png(tmp_path) -> None:
    prediction, target, mask = _masked_pair()
    hdr = prediction * 4.0
    path = tmp_path / "frame_0007.npz"
    save_raw_metric_frame(path, prediction, hdr, target, mask)

    payload = np.load(path)
    assert set(payload.files) == {"render_rgb", "render_hdr_linear", "target_rgb", "valid_mask"}
    assert payload["render_rgb"].dtype == np.float32
    assert payload["render_hdr_linear"].max() == pytest.approx(1.0)
    assert payload["valid_mask"].dtype == np.bool_


def test_fixed_generator_reproduces_monte_carlo_render() -> None:
    config = RenderConfig(
        num_brdf_samples=8,
        num_light_samples=8,
        point_visibility=False,
        indirect_bounce=False,
    )
    renderer = DeferredRenderer(config)
    emitters = AreaEmitterModel(
        centers=torch.tensor([[0.0, 0.0, 2.0]]),
        scales=torch.tensor([[0.5, 0.5, 0.5]]),
        colors=torch.tensor([[4.0, 4.0, 4.0]]),
    )
    gbuffer = {
        "points": torch.zeros((4, 3)),
        "normals": torch.tensor([[0.0, 0.0, 1.0]]).expand(4, -1),
        "view_dirs": torch.tensor([[0.0, 0.0, 1.0]]).expand(4, -1),
        "albedo": torch.full((4, 3), 0.5),
        "roughness": torch.full((4, 1), 0.6),
        "metallic": torch.full((4, 1), 0.2),
    }
    first = renderer.render(gbuffer, emitters, generator=torch.Generator().manual_seed(2026))
    second = renderer.render(gbuffer, emitters, generator=torch.Generator().manual_seed(2026))
    assert torch.equal(first["hdr"], second["hdr"])
