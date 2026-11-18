from dataclasses import asdict

import pytest
import torch

from aegir.cli import build_parser
from aegir.config import AEGIRConfig, RenderConfig, config_from_dict
from aegir.emitters import AreaEmitterModel
from aegir.indirect import DIRECT_SH_MODE, GlobalDiffuseSH
from aegir.render import DeferredRenderer
from aegir.sample_reduction import central80_bounds, reduce_direct_samples


def test_central80_uses_requested_population_and_actual_denominator():
    for count, tail, keep in ((128, 13, 102), (512, 51, 410)):
        values = torch.arange(count, dtype=torch.float64)[:, None].expand(count, 3)
        values = values[torch.randperm(count, generator=torch.Generator().manual_seed(1))]
        total, denominator = reduce_direct_samples(values[None], 'trimmed80')
        assert central80_bounds(count) == (tail, count - tail)
        assert denominator == keep
        torch.testing.assert_close(total / denominator, torch.full_like(total, (count - 1) / 2))
    assert central80_bounds(1) == (0, 1)


def test_mean_default_is_bitwise_legacy_and_rng_neutral():
    values = torch.rand((4, 512, 3), generator=torch.Generator().manual_seed(7))
    before = torch.random.get_rng_state().clone()
    total, count = reduce_direct_samples(values)
    assert count == 512 and torch.equal(total, values.sum(dim=1))
    assert torch.equal(before, torch.random.get_rng_state())


def test_hdr_luminance_ranking_preserves_whole_rgb_samples():
    # RGB-sum would rank red2 above green1, luminance correctly does the reverse.
    values = torch.tensor([[0., 0., 0.], [2., 0., 0.], [0., 1., 0.], [0., 0., 100.],
                           [0., 2., 0.], [0., 3., 0.], [0., 4., 0.], [0., 5., 0.],
                           [0., 6., 0.], [0., 0., 1000.]])
    total, keep = reduce_direct_samples(values, 'trimmed80')
    assert keep == 8
    torch.testing.assert_close(total, values[1:9].sum(0))
    # Every selected channel retains the same sample membership, including HDR>1.
    assert total[0] == 2 and total[2] == 100


def test_zero_visibility_samples_are_not_removed_before_ranking():
    values = torch.cat((torch.zeros(9, 3), torch.full((1, 3), 100.)), dim=0)
    total, keep = reduce_direct_samples(values, 'trimmed80')
    assert keep == 8 and torch.equal(total, torch.zeros(3))
    # This captures the legitimate-light loss risk, not just outlier suppression.
    assert reduce_direct_samples(values, 'mean')[0].sum() > 0


def test_ties_are_stable_and_selected_gradients_are_finite():
    values = torch.ones(2, 10, 3, requires_grad=True)
    total, keep = reduce_direct_samples(values, 'trimmed80')
    torch.testing.assert_close(total / keep, torch.ones(2, 3))
    total.sum().backward()
    assert torch.isfinite(values.grad).all()
    assert (values.grad[:, (0, 9)] == 0).all() and (values.grad[:, 1:9] == 1).all()


def test_invalid_or_empty_inputs_and_explicit_transport_are_guarded():
    with pytest.raises(ValueError):
        reduce_direct_samples(torch.zeros(2, 0, 3))
    with pytest.raises(ValueError):
        reduce_direct_samples(torch.zeros(2, 8, 2))
    with pytest.raises(ValueError):
        reduce_direct_samples(torch.zeros(2, 8, 3), 'invalid')
    with pytest.raises(ValueError, match='trimmed80 currently'):
        DeferredRenderer(RenderConfig(direct_sample_reduction='trimmed80'))


def test_config_and_inspect_cli_preserve_old_default():
    config = asdict(AEGIRConfig())
    del config['render']['direct_sample_reduction']
    assert config_from_dict(config).render.direct_sample_reduction == 'mean'
    args = ['inspect', '--checkpoint', 'a.pt', '--nerfstudio-dir', 'data', '--output-dir', 'out']
    assert build_parser().parse_args(args).direct_sample_reduction is None
    assert build_parser().parse_args(args + ['--direct-sample-reduction', 'trimmed80']).direct_sample_reduction == 'trimmed80'


@pytest.mark.parametrize('count', [128, 512])
def test_renderer_paired_modes_share_samples_visibility_and_indirect(monkeypatch, count):
    emitter = AreaEmitterModel(torch.tensor([[0., 0., 2.]]), torch.ones(1, 3) * .5)
    light = GlobalDiffuseSH(.3)
    observed = []
    def visibility(points, dirs, dist):
        observed.append((dirs.detach().clone(), dist.detach().clone()))
        return torch.ones_like(dist)
    gb = dict(points=torch.zeros(2, 3), normals=torch.tensor([[0., 0., 1.]]).expand(2, 3),
              view_dirs=torch.tensor([[0., 0., 1.]]).expand(2, 3), albedo=torch.ones(2, 3) * .5,
              roughness=torch.ones(2, 1) * .6, metallic=torch.zeros(2, 1), alpha=torch.ones(2, 1),
              valid_mask=torch.ones(2, 1, dtype=torch.bool))
    configs = [RenderConfig(lighting_mode=DIRECT_SH_MODE, num_light_samples=count,
                           camera_visible_emitters=False, direct_sample_reduction=mode) for mode in ('mean', 'trimmed80')]
    outputs, states = [], []
    for cfg in configs:
        generator = torch.Generator().manual_seed(81)
        outputs.append(DeferredRenderer(cfg, visibility_fn=visibility, indirect_light=light).render(
            gb, emitter, generator=generator, tonemap=False))
        states.append(generator.get_state())
    for first, second in zip(observed[0], observed[1]):
        assert torch.equal(first, second)
    assert torch.equal(states[0], states[1])
    assert torch.equal(outputs[0]['indirect_hdr'], outputs[1]['indirect_hdr'])
    assert torch.equal(outputs[0]['emission_hdr'], outputs[1]['emission_hdr'])
    assert torch.isfinite(outputs[1]['hdr']).all()


@pytest.mark.parametrize('count,tail,keep', [(128, 13, 102), (512, 51, 410)])
def test_full_renderer_divides_by_retained_count(monkeypatch, count, tail, keep):
    emitter = AreaEmitterModel(torch.tensor([[0., 0., 2.]]), torch.ones(1, 3))
    def sample(points, n, generator=None):
        direction = points.new_tensor((0., 0., 1.)).expand(len(points), n, 3)
        return points[:, None] + direction, direction, points.new_ones(len(points), n, 1), torch.zeros(len(points), n, dtype=torch.long)
    monkeypatch.setattr(emitter, 'sample_surface_total', sample)
    monkeypatch.setattr(emitter, 'light_pdf_surrogate', lambda distance, *args: torch.ones_like(distance))
    monkeypatch.setattr(emitter, 'radiance_for_ids', lambda dirs, ids, **kw:
                        torch.arange(count, dtype=dirs.dtype).square()[None, :, None].expand_as(dirs))
    cfg = RenderConfig(lighting_mode=DIRECT_SH_MODE, num_light_samples=count,
                       direct_sample_reduction='trimmed80', camera_visible_emitters=False)
    renderer = DeferredRenderer(cfg, indirect_light=GlobalDiffuseSH(.03))
    renderer.brdf = lambda view, dirs, *args: torch.ones_like(dirs)
    gb = dict(points=torch.zeros(1, 3), normals=torch.tensor([[0., 0., 1.]]), view_dirs=torch.tensor([[0., 0., 1.]]),
              albedo=torch.ones(1, 3), roughness=torch.ones(1, 1) * .6, metallic=torch.zeros(1, 1), alpha=torch.ones(1, 1),
              valid_mask=torch.ones(1, 1, dtype=torch.bool))
    result = renderer.render(gb, emitter, tonemap=False)
    expected = torch.arange(tail, count - tail, dtype=torch.float32).square().sum() / keep
    torch.testing.assert_close(result['direct_hdr'], expected.expand(1, 3))
