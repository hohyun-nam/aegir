from __future__ import annotations

import pytest
import torch

from aegir.config import RenderConfig
from aegir.visibility import (
    resolve_visibility_parameters,
    scene_scale_from_bounds,
    soft_visibility,
)


def test_paper_transmittance_termination_default() -> None:
    assert RenderConfig().trace_transmittance_min == pytest.approx(0.05)


def test_visibility_parameters_follow_scene_scale() -> None:
    config = RenderConfig(visibility_bias=1.0e-3, visibility_steepness=80.0)

    bias_6m, steepness_6m = resolve_visibility_parameters(config, 6.0)
    bias_33m, steepness_33m = resolve_visibility_parameters(config, 33.0)

    assert bias_6m == pytest.approx(0.006)
    assert steepness_6m == pytest.approx(80.0 / 6.0)
    assert bias_33m == pytest.approx(0.033)
    assert steepness_33m == pytest.approx(80.0 / 33.0)


def test_soft_visibility_subtracts_bias_once() -> None:
    visibility = soft_visibility(
        torch.tensor(0.8),
        torch.tensor(1.0),
        torch.tensor(0.9),
        bias=0.1,
        steepness=10.0,
    )

    # The sigmoid argument is exactly zero only when the bias is subtracted once.
    assert visibility.item() == pytest.approx(0.6)


def test_visibility_and_normalized_distance_gradient_are_scale_invariant() -> None:
    config = RenderConfig()

    def evaluate(scene_scale: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        alpha = torch.tensor(0.8, requires_grad=True)
        d_trace = torch.tensor(0.45 * scene_scale, requires_grad=True)
        bias, steepness = resolve_visibility_parameters(config, scene_scale)
        visibility = soft_visibility(
            alpha,
            torch.tensor(0.5 * scene_scale),
            d_trace,
            bias,
            steepness,
        )
        alpha_grad, distance_grad = torch.autograd.grad(visibility, (alpha, d_trace))
        return visibility.detach(), alpha_grad.detach(), distance_grad.detach() * scene_scale

    unit_result = evaluate(1.0)
    scaled_result = evaluate(10.0)
    for unit_value, scaled_value in zip(unit_result, scaled_result, strict=True):
        assert torch.allclose(unit_value, scaled_value, atol=1.0e-7, rtol=1.0e-7)


def test_scene_scale_is_bounds_diagonal() -> None:
    scale = scene_scale_from_bounds(
        (torch.tensor([-1.0, -2.0, -2.0]), torch.tensor([1.0, 2.0, 2.0]))
    )
    assert scale == pytest.approx(6.0)
