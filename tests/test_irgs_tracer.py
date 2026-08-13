from __future__ import annotations

import pytest
import torch
from torch import nn

from aegir.config import RenderConfig
from aegir.irgs_tracer import IRGSVisibilityTracer
from aegir.math_utils import safe_normalize


class _SingleSurfelScene(nn.Module):
    def __init__(self, coordinate_scale: float = 1.0) -> None:
        super().__init__()
        self.means = nn.Parameter(
            torch.tensor([[0.0, 0.0, coordinate_scale]], device="cuda")
        )
        self.log_scales = nn.Parameter(
            torch.tensor([[0.5, 0.5]], device="cuda").mul(coordinate_scale).log()
        )
        self.raw_quats = nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda"))
        self.opacity_logits = nn.Parameter(torch.logit(torch.tensor([[0.9]], device="cuda")))

    @property
    def gaussian_count(self) -> int:
        return 1

    @property
    def scales(self) -> torch.Tensor:
        return self.log_scales.exp()

    @property
    def quats(self) -> torch.Tensor:
        return safe_normalize(self.raw_quats, dim=-1)

    @property
    def opacities(self) -> torch.Tensor:
        return self.opacity_logits.sigmoid()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="IRGS OptiX test requires CUDA")
def test_irgs_visibility_hits_misses_and_backpropagates() -> None:
    pytest.importorskip("surfel_tracer")
    scene = _SingleSurfelScene()
    config = RenderConfig(visibility_steepness=80.0)
    tracer = IRGSVisibilityTracer(scene, config, scene_scale=1.0)
    origins = torch.zeros((1, 2, 3), device="cuda")
    directions = torch.tensor([[[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]], device="cuda")
    distances = torch.full((1, 2, 1), 2.0, device="cuda")

    visibility = tracer.visibility(origins, directions, distances)

    assert visibility.shape == (1, 2, 1)
    assert float(visibility[0, 0, 0].detach()) < 0.3
    assert float(visibility[0, 1, 0].detach()) > 0.99
    visibility[0, 0, 0].backward()
    assert scene.opacity_logits.grad is not None
    assert torch.isfinite(scene.opacity_logits.grad).all()
    assert float(scene.opacity_logits.grad.abs().max()) > 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="IRGS OptiX test requires CUDA")
@torch.inference_mode()
def test_irgs_visibility_runs_in_inspection_mode() -> None:
    pytest.importorskip("surfel_tracer")
    scene = _SingleSurfelScene()
    tracer = IRGSVisibilityTracer(scene, RenderConfig(), scene_scale=1.0)
    origins = torch.zeros((1, 1, 3), device="cuda")
    directions = torch.tensor([[[0.0, 0.0, 1.0]]], device="cuda")
    distances = torch.full((1, 1, 1), 2.0, device="cuda")

    visibility = tracer.visibility(origins, directions, distances)

    assert float(visibility[0, 0, 0]) < 0.3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="IRGS OptiX test requires CUDA")
def test_irgs_visibility_and_opacity_gradient_are_scale_invariant() -> None:
    pytest.importorskip("surfel_tracer")
    values = []
    gradients = []
    for coordinate_scale in (1.0, 10.0):
        scene = _SingleSurfelScene(coordinate_scale)
        tracer = IRGSVisibilityTracer(
            scene,
            RenderConfig(),
            scene_scale=coordinate_scale,
        )
        origins = torch.zeros((1, 1, 3), device="cuda")
        directions = torch.tensor([[[0.0, 0.0, 1.0]]], device="cuda")
        distances = torch.full((1, 1, 1), 2.0 * coordinate_scale, device="cuda")
        visibility = tracer.visibility(origins, directions, distances)
        visibility.sum().backward()
        values.append(visibility.detach())
        gradients.append(scene.opacity_logits.grad.detach())

    assert torch.allclose(values[0], values[1], atol=2.0e-5, rtol=2.0e-5)
    assert torch.allclose(gradients[0], gradients[1], atol=2.0e-5, rtol=2.0e-5)
