from dataclasses import asdict

import pytest
import torch

from aegir.config import AEGIRConfig, config_from_dict
from aegir.cli import build_parser
from aegir.emitters import AreaEmitterModel
from aegir.math_utils import balanced_rqmc_uv, uniform_sample_sphere
from aegir.render import DeferredRenderer


def test_legacy_config_defaults_to_iid():
    payload = asdict(AEGIRConfig())
    del payload['render']['light_sample_pattern']
    assert config_from_dict(payload).render.light_sample_pattern == 'iid'
    payload['render']['light_sample_pattern'] = 'rqmc'
    assert config_from_dict(payload).render.light_sample_pattern == 'rqmc'


def test_inspect_cli_override_is_opt_in():
    args = ['inspect','--checkpoint','scene.pt','--nerfstudio-dir','scene',
            '--output-dir','inspection']
    assert build_parser().parse_args(args).light_sample_pattern is None
    assert build_parser().parse_args(args+['--light-sample-pattern','rqmc']).light_sample_pattern == 'rqmc'


def test_renderer_validates_pattern_and_routes_opt_in(monkeypatch):
    cfg = AEGIRConfig().render
    cfg.light_sample_pattern = 'invalid'
    with pytest.raises(ValueError, match='light_sample_pattern'):
        DeferredRenderer(cfg)
    cfg.light_sample_pattern = 'rqmc'
    cfg.num_brdf_samples = 0
    cfg.num_light_samples = 32
    model = AreaEmitterModel(torch.tensor([[0., 0., 2.]]), torch.ones(1, 3))
    original = model.sample_surface_total
    seen = []
    def capture(*args, **kwargs):
        seen.append(kwargs['pattern'])
        return original(*args, **kwargs)
    monkeypatch.setattr(model, 'sample_surface_total', capture)
    DeferredRenderer(cfg)._light_samples(
        torch.zeros(1, 3), torch.tensor([[0., 0., 1.]]), torch.tensor([[0., 0., 1.]]),
        torch.ones(1, 3), torch.ones(1, 1) * .5, torch.zeros(1, 1), model, None,
        generator=torch.Generator().manual_seed(7))
    assert seen == ['rqmc']


@pytest.mark.parametrize('axes', [2, 3])
@pytest.mark.parametrize('samples', [1, 7, 11, 128, 512])
def test_rqmc_preserves_budget_support_seed_and_pdf(axes, samples):
    model = AreaEmitterModel(torch.zeros(11, 3), torch.ones(11, axes))
    points = torch.zeros(3, 2, 3)
    iid = model.sample_surface_total(points, samples, torch.Generator().manual_seed(3))
    rqmc = model.sample_surface_total(points, samples, torch.Generator().manual_seed(3), pattern='rqmc')
    again = model.sample_surface_total(points, samples, torch.Generator().manual_seed(3), pattern='rqmc')
    assert torch.equal(iid[3], rqmc[3])
    for a, b in zip(rqmc, again):
        assert torch.equal(a, b)
    q, directions, distance, ids = rqmc
    assert q.shape == directions.shape == (3, 2, samples, 3)
    counts = torch.stack([torch.bincount(row, minlength=11) for row in ids.reshape(-1, samples)])
    assert (counts.max(-1).values - counts.min(-1).values <= 1).all()
    if axes == 3:
        torch.testing.assert_close(q.norm(dim=-1), torch.ones_like(q[..., 0]))
    else:
        assert (q[..., :2].norm(dim=-1) <= 1).all()
        assert (q[..., 2] == 0).all()
    torch.testing.assert_close(directions * distance, q - points[..., None, :])
    assert torch.isfinite(model.light_pdf_surrogate(distance, ids, directions)).all()


def test_iid_rng_matches_legacy_exactly():
    model = AreaEmitterModel(torch.zeros(11, 3), torch.ones(11, 3))
    points = torch.zeros(8, 3)
    generator = torch.Generator().manual_seed(17)
    ids = (torch.arange(512)[None] + torch.randint(11, (8, 1), generator=generator)) % 11
    expected = uniform_sample_sphere((8, 512), 'cpu', torch.float32, generator)
    actual, _, _, actual_ids = model.sample_surface_total(points, 512, torch.Generator().manual_seed(17))
    assert torch.equal(actual_ids, ids)
    assert torch.equal(actual, expected)


def test_rqmc_uniform_marginals_independent_shifts_and_lower_smooth_variance():
    ids = torch.zeros(2048, 47, dtype=torch.long)
    uv = balanced_rqmc_uv(ids, 1, torch.float64, torch.Generator().manual_seed(9))
    torch.testing.assert_close(uv.mean((0, 1)), torch.full((2,), .5, dtype=uv.dtype), atol=.003, rtol=0)
    assert not torch.equal(uv[0], uv[1])
    # Random shift makes the first sample (and not just the set average) uniform.
    torch.testing.assert_close(uv[:, 0].mean(0), torch.full((2,), .5, dtype=uv.dtype), atol=.025, rtol=0)
    f = lambda x: torch.cos(2 * torch.pi * x[..., 0]) * torch.sin(2 * torch.pi * x[..., 1])
    iid = torch.rand(uv.shape, dtype=uv.dtype, generator=torch.Generator().manual_seed(9))
    assert f(uv).mean(1).var() < .25 * f(iid).mean(1).var()


def test_rqmc_parameter_gradients_and_reject_invalid_mode():
    model = AreaEmitterModel(torch.zeros(2, 3), torch.ones(2, 3))
    q, _, _, _ = model.sample_surface_total(torch.zeros(4, 3), 32, pattern='rqmc')
    q.square().mean().backward()
    assert torch.isfinite(model.log_scales.grad).all()
    with pytest.raises(ValueError, match='unknown light sample pattern'):
        model.sample_surface_total(torch.zeros(4, 3), 32, pattern='typo')
