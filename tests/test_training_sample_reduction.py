"""Training must use the selected estimator, retain gradients, and save it."""
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

from aegir.config import AEGIRConfig, config_from_dict
from aegir.indirect import DIRECT_SH_MODE
from aegir.sample_reduction import reduce_direct_samples
from scripts.train_tagged_surface_scene import configure_training_sample_reduction


def options(mode=None, percent=None):
    return SimpleNamespace(direct_sample_reduction=mode, direct_sample_keep_percent=percent)


def test_omitted_training_option_preserves_defaults():
    cfg = AEGIRConfig()
    before = asdict(cfg)
    configure_training_sample_reduction(cfg, options())
    assert asdict(cfg) == before


def test_central90_is_saved_and_gradients_reach_only_retained_samples():
    cfg = AEGIRConfig()
    cfg.render.lighting_mode = DIRECT_SH_MODE
    cfg.render.num_light_samples = 128
    configure_training_sample_reduction(cfg, options(percent=90))
    restored = config_from_dict(asdict(cfg))
    assert restored.render.direct_sample_reduction == 'trimmed'
    assert restored.render.direct_sample_keep_fraction == .9
    samples = torch.arange(128.).reshape(1, 128, 1).expand(2, 128, 3).clone().requires_grad_()
    total, count = reduce_direct_samples(samples, restored.render.direct_sample_reduction,
                                          restored.render.direct_sample_keep_fraction)
    assert count == 116
    torch.testing.assert_close(total / count, torch.full((2, 3), 63.5))
    (total / count).sum().backward()
    assert torch.isfinite(samples.grad).all()
    assert (samples.grad[:, :6] == 0).all() and (samples.grad[:, 122:] == 0).all()
    torch.testing.assert_close(samples.grad[:, 6:122], torch.full((2, 116, 3), 1 / 116))


@pytest.mark.parametrize('mode,percent', [('mean', 90), ('trimmed80', 90), (None, 0),
                                        (None, 101), (None, float('nan'))])
def test_invalid_override_does_not_mutate_config(mode, percent):
    cfg = AEGIRConfig()
    cfg.render.lighting_mode = DIRECT_SH_MODE
    before = asdict(cfg)
    with pytest.raises(ValueError):
        configure_training_sample_reduction(cfg, options(mode, percent))
    assert asdict(cfg) == before


def test_explicit_transport_remains_untouched_and_rejects_trimming():
    cfg = AEGIRConfig()
    with pytest.raises(ValueError):
        configure_training_sample_reduction(cfg, options(percent=90))
    assert cfg.render.direct_sample_reduction == 'mean'


def test_explicit_mean_can_restore_all_sample_training():
    cfg = AEGIRConfig()
    cfg.render.lighting_mode = DIRECT_SH_MODE
    configure_training_sample_reduction(cfg, options(percent=90))
    configure_training_sample_reduction(cfg, options(mode='mean'))
    assert cfg.render.direct_sample_reduction == 'mean'
