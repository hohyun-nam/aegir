from __future__ import annotations

import pytest
import torch

from aegir.cli import build_parser
from aegir.config import LossWeights
from aegir.emitters import AreaEmitterModel
from aegir.losses import emitter_regularization


def _emitter() -> AreaEmitterModel:
    return AreaEmitterModel(
        centers=torch.tensor([[2.0, 0.0, -1.0]]),
        scales=torch.tensor([[0.2, 0.2, 0.2]]),
        colors=torch.tensor([[2.0, 1.0, 1.0]]),
    )


def _camera_data() -> dict[str, torch.Tensor]:
    return {
        "c2w": torch.eye(4).unsqueeze(0),
        "fx": torch.tensor([100.0]),
        "fy": torch.tensor([100.0]),
        "cx": torch.tensor([50.0]),
        "cy": torch.tensor([50.0]),
        "width": torch.tensor([100.0]),
        "height": torch.tensor([100.0]),
        "convention": torch.tensor([0]),
    }


def test_non_paper_emitter_regularizers_are_disabled_by_default() -> None:
    weights = LossWeights(light_area=0.0, light_white=0.0, light_bounds=0.0)
    assert weights.light_surface == 0.0
    assert weights.light_view == 0.0

    loss = emitter_regularization(
        _emitter(),
        (torch.full((3,), -1.0), torch.full((3,), 1.0)),
        weights,
        surface_points=torch.tensor([[100.0, 100.0, 100.0]]),
        cameras=_camera_data(),
    )
    assert loss.item() == 0.0


@pytest.mark.parametrize(
    ("weight_name", "expected"),
    [
        ("light_area", 0.01),
        ("light_white", 1.0),
        ("light_bounds", 1.0 / (12.0**0.5)),
    ],
)
def test_paper_emitter_regularizers_remain_active(weight_name: str, expected: float) -> None:
    kwargs = {
        "light_area": 0.0,
        "light_white": 0.0,
        "light_bounds": 0.0,
        weight_name: 1.0,
    }
    loss = emitter_regularization(
        _emitter(),
        (torch.full((3,), -1.0), torch.full((3,), 1.0)),
        LossWeights(**kwargs),
    )
    assert loss.item() == pytest.approx(expected)


def test_paper_area_loss_is_squared_norm_in_normalized_scene_coordinates() -> None:
    weights = LossWeights(light_area=1.0, light_white=0.0, light_bounds=0.0)
    base = emitter_regularization(
        AreaEmitterModel(
            centers=torch.zeros((1, 3)),
            scales=torch.tensor([[0.2, 0.4, 0.6]]),
        ),
        (torch.full((3,), -1.0), torch.full((3,), 1.0)),
        weights,
    )
    scaled = emitter_regularization(
        AreaEmitterModel(
            centers=torch.zeros((1, 3)),
            scales=torch.tensor([[2.0, 4.0, 6.0]]),
        ),
        (torch.full((3,), -10.0), torch.full((3,), 10.0)),
        weights,
    )
    expected = (0.2**2 + 0.4**2 + 0.6**2) / (12.0)
    assert base.item() == pytest.approx(expected)
    assert scaled.item() == pytest.approx(expected)


def test_non_paper_emitter_regularizers_can_be_enabled_explicitly() -> None:
    bounds = (torch.full((3,), -1.0), torch.full((3,), 1.0))
    base = {"light_area": 0.0, "light_white": 0.0, "light_bounds": 0.0}
    surface = emitter_regularization(
        _emitter(),
        bounds,
        LossWeights(**base, light_surface=10.0),
        surface_points=torch.tensor([[100.0, 100.0, 100.0]]),
    )
    view = emitter_regularization(
        _emitter(),
        bounds,
        LossWeights(**base, light_view=1.0),
        cameras=_camera_data(),
    )
    assert surface.item() > 0.0
    assert view.item() > 0.0


def test_cli_requires_explicit_override_for_non_paper_regularizers() -> None:
    parser = build_parser()
    required = [
        "train-gaussian-scene",
        "--dnsplatter-checkpoint",
        "scene.ckpt",
        "--nerfstudio-dir",
        "scene",
        "--output",
        "output",
    ]
    defaults = parser.parse_args(required)
    assert defaults.light_surface == 0.0
    assert defaults.light_view == 0.0
    assert defaults.albedo_initialization == "sh_dc"
    assert defaults.albedo_lr == pytest.approx(3.0e-3)
    assert not defaults.optimize_emitter_centers
    assert not defaults.optimize_emitter_rotations
    assert not defaults.freeze_emitter_scales
    assert not defaults.unconstrained_emitter_scales
    assert not defaults.enable_emitter_adaptation

    overridden = parser.parse_args(required + ["--light-surface", "10", "--light-view", "1"])
    assert overridden.light_surface == 10.0
    assert overridden.light_view == 1.0
