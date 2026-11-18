"""Opt-in primary-pixel setting must not alter existing training defaults."""
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

from aegir.config import AEGIRConfig, config_from_dict
from aegir.trainer import SharedGaussianTrainer
from scripts.train_tagged_surface_scene import configure_training_patch


@pytest.mark.parametrize('saved_pixels', [256, 512])
def test_omitted_override_preserves_saved_patch(saved_pixels):
    config = AEGIRConfig()
    config.training.rays_per_step = saved_pixels
    args = SimpleNamespace(image_size=128, rays_per_step=None)
    shape = configure_training_patch(config, args)
    assert shape[0] * shape[1] == saved_pixels
    assert config.training.rays_per_step == saved_pixels


def test_1024_pixel_patch_is_saved_and_selected_exactly():
    config = AEGIRConfig()
    original = asdict(config)
    assert configure_training_patch(config, SimpleNamespace(image_size=128, rays_per_step=1024)) == (32, 32)
    changed = asdict(config)
    original['training']['rays_per_step'] = 1024
    assert changed == original
    restored = config_from_dict(changed)
    trainer = object.__new__(SharedGaussianTrainer)
    trainer.config = restored
    target = torch.arange(128 * 128 * 3).reshape(128, 128, 3).float()
    mask = torch.ones(128, 128, 1, dtype=torch.bool)
    focus = torch.ones(128, 128, 1)
    patch, selected_target, selected_mask, selected_focus = trainer._select_patch(
        {'points': target.clone()}, target, mask, focus, 1000,
    )
    assert selected_target.shape == (32, 32, 3)
    assert torch.equal(patch['points'], selected_target)
    assert selected_mask.shape == selected_focus.shape == (32, 32, 1)
    assert selected_mask.sum() == 1024


@pytest.mark.parametrize('pixels', [0, -1, 16385, 257])
def test_invalid_patch_fails_before_config_changes(pixels):
    config = AEGIRConfig()
    original = asdict(config)
    with pytest.raises(ValueError):
        configure_training_patch(config, SimpleNamespace(image_size=128, rays_per_step=pixels))
    assert asdict(config) == original
