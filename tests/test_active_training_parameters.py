from dataclasses import asdict

import pytest
import torch

from aegir.config import AEGIRConfig, config_from_dict
from aegir.trainer import SharedGaussianTrainer, training_sh_proxy_weight


def make_trainer():
    trainer = SharedGaussianTrainer.__new__(SharedGaussianTrainer)
    trainer.config = AEGIRConfig()
    trainer.config.training.geometry_lr = 0.0
    trainer.scene = torch.nn.ParameterDict({
        "geometry": torch.nn.Parameter(torch.ones(1)),
        "albedo": torch.nn.Parameter(torch.ones(1)),
        "material": torch.nn.Parameter(torch.ones(1)),
    })
    trainer.emitters = torch.nn.ParameterDict({"light": torch.nn.Parameter(torch.ones(1))})
    trainer.geometry_optimizer = torch.optim.Adam([trainer.scene["geometry"]])
    trainer.material_optimizer = torch.optim.Adam([
        {"params": [trainer.scene["albedo"]]},
        {"params": [trainer.scene["material"]]},
    ])
    trainer.emitter_optimizer = torch.optim.Adam(trainer.emitters.parameters())
    trainer.exposure_optimizer = None
    return trainer


def test_inactive_gradient_cannot_suppress_emitter_gradient():
    trainer = make_trainer()
    trainer._update_lrs(15_000)
    # Even stale gradients from frozen optimizer groups must not enter clipping.
    for parameter in trainer.scene.parameters():
        parameter.grad = torch.full_like(parameter, 100.0)
    light = trainer.emitters["light"]
    light.grad = torch.ones_like(light)
    trainer._sanitize_gradients()
    assert light.grad.item() == pytest.approx(1.0, abs=2e-6)


def test_positive_geometry_lr_respects_parameter_selection():
    trainer = make_trainer()
    geometry = trainer.scene["geometry"]
    trainer.config.training.geometry_lr = 1.0e-6
    trainer.scene_geometry_trainability = {id(geometry): False}

    trainer._update_lrs(0)

    assert not geometry.requires_grad
    assert trainer.geometry_optimizer.param_groups[0]["lr"] == pytest.approx(1.0e-6)


def test_freeze_removes_stale_gradients_and_keeps_light_transport_derivative():
    trainer = make_trainer()
    trainer._update_lrs(10_000)
    assert not trainer.scene["geometry"].requires_grad
    assert trainer.scene["albedo"].requires_grad
    assert trainer.scene["material"].requires_grad
    assert training_sh_proxy_weight(trainer.config, 10_000) == 0.0
    for parameter in trainer.scene.parameters():
        parameter.grad = torch.ones_like(parameter)
    trainer._update_lrs(15_000)
    assert all(not p.requires_grad and p.grad is None for p in trainer.scene.parameters())
    light = trainer.emitters["light"]
    before = trainer.scene["albedo"].detach().clone()
    loss = (trainer.scene["albedo"] * light).sum()
    loss.backward()
    trainer.material_optimizer.step()
    assert torch.equal(before, trainer.scene["albedo"])
    assert light.grad.item() == pytest.approx(1.0)


def test_frozen_material_regularizers_are_not_evaluated(monkeypatch):
    trainer = make_trainer()
    trainer._update_lrs(15_000)

    def unexpected(*args, **kwargs):
        raise AssertionError("regularizer evaluated after freeze")

    monkeypatch.setattr("aegir.trainer.full_view_material_prior_loss", unexpected)
    monkeypatch.setattr("aegir.trainer.edge_aware_tv", unexpected)
    prior, tv = trainer._material_regularization({}, {}, torch.ones(1), None)
    assert prior.item() == tv.item() == 0.0
    # Active geometry still needs prior/TV even if material LR is zero.
    trainer.scene["geometry"].requires_grad_(True)
    with pytest.raises(AssertionError, match="regularizer evaluated"):
        trainer._material_regularization({}, {}, torch.ones(1), None)


def test_independent_fade_roundtrip_and_legacy_config():
    config = AEGIRConfig()
    restored = config_from_dict(asdict(config))
    assert training_sh_proxy_weight(restored, 10_000) == 0.0
    assert restored.training.freeze_end == 15_000
    legacy = asdict(config)
    del legacy["training"]["sh_fade_start"]
    del legacy["training"]["sh_fade_end"]
    assert training_sh_proxy_weight(config_from_dict(legacy), 10_000) == 0.5
    config.training.sh_fade_end = config.training.sh_fade_start
    with pytest.raises(ValueError, match="SH fade"):
        training_sh_proxy_weight(config, 0)
