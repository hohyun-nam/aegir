from dataclasses import asdict

import pytest
import torch

from aegir.cli import build_parser
from aegir.config import AEGIRConfig, config_from_dict
from aegir.emitters import AreaEmitterModel
from aegir.indirect import DIRECT_SH_MODE
from aegir.render import DeferredRenderer


def sphere(dtype=torch.float64):
    return AreaEmitterModel(torch.zeros(1, 3, dtype=dtype), torch.ones(1, 3, dtype=dtype))


def test_config_and_cli_are_opt_in_and_explicit_transport_is_guarded():
    old = asdict(AEGIRConfig())
    del old['render']['light_sampling_strategy']
    assert config_from_dict(old).render.light_sampling_strategy == 'surface'
    args = ['inspect', '--checkpoint', 'a.pt', '--nerfstudio-dir', 'data', '--output-dir', 'out']
    assert build_parser().parse_args(args).light_sampling_strategy is None
    assert build_parser().parse_args(args + ['--light-sampling-strategy', 'solid_angle']).light_sampling_strategy == 'solid_angle'
    cfg = AEGIRConfig().render
    cfg.light_sampling_strategy = 'solid_angle'
    with pytest.raises(ValueError, match='explicit MIS'):
        DeferredRenderer(cfg)
    cfg.lighting_mode = DIRECT_SH_MODE
    DeferredRenderer(cfg)
    cfg.light_sampling_strategy = 'invalid'
    with pytest.raises(ValueError, match='light_sampling_strategy'):
        DeferredRenderer(cfg)


@pytest.mark.parametrize('pattern', ['iid', 'rqmc'])
@pytest.mark.parametrize('samples', [1, 7, 128, 512])
def test_budget_seed_support_and_front_surface(pattern, samples):
    model = AreaEmitterModel(torch.zeros(11, 3, dtype=torch.float64),
                             torch.tensor([2., .7, .3], dtype=torch.float64).expand(11, 3))
    points = torch.tensor([0., 0., -2.], dtype=torch.float64).expand(2, 3, 3)
    result = model.sample_solid_angle_total(points, samples, torch.Generator().manual_seed(14), pattern=pattern)
    again = model.sample_solid_angle_total(points, samples, torch.Generator().manual_seed(14), pattern=pattern)
    for a, b in zip(result, again):
        assert torch.equal(a, b)
        assert torch.isfinite(a).all()
    q, directions, distance, ids, pdf = result
    assert q.shape == directions.shape == (2, 3, samples, 3)
    assert distance.shape == pdf.shape == (2, 3, samples, 1)
    old_ids = model.sample_surface_total(points, samples, torch.Generator().manual_seed(14))[3]
    assert torch.equal(ids, old_ids)
    counts = torch.stack([torch.bincount(row, minlength=11) for row in ids.reshape(-1, samples)])
    assert (counts.max(-1).values - counts.min(-1).values <= 1).all()
    torch.testing.assert_close((q / model.scales[ids]).square().sum(-1), torch.ones_like(q[..., 0]))
    torch.testing.assert_close(directions * distance, q - points[..., None, :], atol=1e-12, rtol=1e-10)
    normal = q / model.scales[ids].square()
    assert (normal * -directions).sum(-1).min() >= -1e-10
    assert (pdf > 0).all()


@pytest.mark.parametrize('distance', [1.0001, 1.1, 2., 100000.])
def test_sphere_pdf_and_irradiance_are_analytic_and_bounded(distance):
    model = sphere()
    points = torch.tensor([[0., 0., -distance]], dtype=torch.float64)
    _, directions, _, _, pdf = model.sample_solid_angle_total(
        points, 16384, torch.Generator().manual_seed(13), pattern='rqmc')
    width = (1 / distance**2) / (1 + (1 - 1 / distance**2)**.5)
    torch.testing.assert_close(pdf, torch.full_like(pdf, 1 / (2 * torch.pi * width)))
    # Constant Le=1, normal points toward sphere. Integral cos(theta)domega=pi/r^2.
    estimator = directions[..., 2:3].clamp_min(0) / pdf
    torch.testing.assert_close(estimator.mean(), torch.tensor(torch.pi / distance**2, dtype=torch.float64),
                               atol=1e-12, rtol=8e-5)
    assert estimator.max() <= 2 * torch.pi * (1 + 1e-10)


@pytest.mark.parametrize('point', [[0., 0., 0.], [.7, 0., 0.], [0., 0., -1.]])
def test_inside_and_boundary_are_finite(point):
    model = sphere()
    points = torch.tensor([point], dtype=torch.float64)
    q, directions, distance, _, pdf = model.sample_solid_angle_total(points, 4096, pattern='rqmc')
    for value in (q, directions, distance, pdf):
        assert torch.isfinite(value).all()
    torch.testing.assert_close(q.norm(dim=-1), torch.ones_like(q[..., 0]), atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(directions * distance, q - points[:, None], atol=1e-10, rtol=1e-10)
    if points.norm() < 1:
        torch.testing.assert_close(pdf, torch.full_like(pdf, 1 / (4 * torch.pi)))


def test_anisotropic_world_pdf_normalizes_independently():
    model = AreaEmitterModel(torch.zeros(1, 3, dtype=torch.float64),
                             torch.tensor([[2., .7, .3]], dtype=torch.float64))
    _, directions, _, _, pdf = model.sample_solid_angle_total(
        torch.zeros(1, 3, dtype=torch.float64), 131072, torch.Generator().manual_seed(3), pattern='rqmc')
    # Inverse direction map gives the same density; E_p[1/p] must be 4pi.
    inv_length = (directions / model.scales[0]).norm(dim=-1, keepdim=True)
    reference_pdf = 1 / (4 * torch.pi * model.scales[0].prod() * inv_length.pow(3))
    torch.testing.assert_close(pdf, reference_pdf)
    torch.testing.assert_close(pdf.reciprocal().mean(), torch.tensor(4 * torch.pi, dtype=torch.float64), rtol=.002, atol=0)


def test_anisotropic_exterior_matches_independent_world_direction_quadrature():
    model = AreaEmitterModel(torch.zeros(1, 3, dtype=torch.float64), torch.tensor([[1.7, .8, .4]], dtype=torch.float64))
    point = torch.tensor([[0., 0., -1.3]], dtype=torch.float64)
    _, dirs, _, _, pdf = model.sample_solid_angle_total(point, 131072, torch.Generator().manual_seed(1), pattern='rqmc')
    estimate = (dirs[..., 2:3].clamp_min(0) / pdf).mean()
    # Independent uniform world-hemisphere grid + analytic ray intersection.
    z, phi = torch.meshgrid((torch.arange(1024, dtype=torch.float64) + .5) / 1024,
                            (torch.arange(512, dtype=torch.float64) + .5) * (2 * torch.pi / 512), indexing='ij')
    r = (1 - z.square()).sqrt()
    w = torch.stack((r * phi.cos(), r * phi.sin(), z), -1) / model.scales[0]
    o = point[0] / model.scales[0]
    a, b, c = w.square().sum(-1), (w * o).sum(-1), o.square().sum() - 1
    hits = b.square() >= a * c
    reference = (z * hits).mean() * 2 * torch.pi
    torch.testing.assert_close(estimate, reference, rtol=.003, atol=0)


def test_rotation_equivariance_and_finite_parameter_gradients():
    base = sphere()
    base.log_scales.data.copy_(torch.tensor([[1.2, .4, .2]], dtype=torch.float64).log())
    rotated = AreaEmitterModel(torch.tensor([[2., 3., 1.]], dtype=torch.float64), base.scales.detach(),
                               quaternions=torch.tensor([[.8, .0, .6, .0]], dtype=torch.float64))
    point = torch.tensor([[.3, -.2, -2.]], dtype=torch.float64)
    world_point = point @ rotated.rotations[0].T + rotated.centers
    a = base.sample_solid_angle_total(point, 1024, torch.Generator().manual_seed(1))
    b = rotated.sample_solid_angle_total(world_point, 1024, torch.Generator().manual_seed(1))
    torch.testing.assert_close(b[0], a[0] @ rotated.rotations[0].T + rotated.centers)
    torch.testing.assert_close(b[1], a[1] @ rotated.rotations[0].T)
    torch.testing.assert_close(b[4], a[4])
    (b[0].square().mean() + b[4].reciprocal().mean()).backward()
    for parameter in (rotated.centers, rotated.log_scales, rotated.raw_quaternions):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


def test_renderer_uses_returned_pdf_not_legacy_surrogate(monkeypatch):
    cfg = AEGIRConfig().render
    cfg.lighting_mode = DIRECT_SH_MODE
    cfg.light_sampling_strategy = 'solid_angle'
    cfg.num_light_samples = 32
    model = sphere(torch.float32)
    def fail(*args, **kwargs):
        raise AssertionError('Legacy surface/PDF must not be used for solid_angle')
    monkeypatch.setattr(model, 'sample_surface_total', fail)
    monkeypatch.setattr(model, 'light_pdf_surrogate', fail)
    renderer = DeferredRenderer(cfg)
    direct, indirect, count = renderer._light_samples(
        torch.tensor([[0., 0., -2.]]), torch.tensor([[0., 0., 1.]]), torch.tensor([[0., 0., 1.]]),
        torch.ones(1, 3), torch.ones(1, 1) * .5, torch.zeros(1, 1), model, None,
        generator=torch.Generator().manual_seed(7))
    assert count == 32 and torch.isfinite(direct).all() and (direct > 0).all()
    assert (indirect == 0).all()
    with pytest.raises(ValueError, match='ellipsoid_3d'):
        AreaEmitterModel(torch.zeros(1, 3), torch.ones(1, 2)).sample_solid_angle_total(torch.zeros(1, 3), 32)


def test_emitter_mixture_selection_pdf_and_near_surface_variance():
    model = sphere()
    points = torch.tensor([[0., 0., -1.0001]], dtype=torch.float64).expand(512, 3)
    _, wi, _, _, pdf = model.sample_solid_angle_total(points, 128, torch.Generator().manual_seed(2))
    new_estimates = (wi[..., 2:3].clamp_min(0) / pdf).mean(1)
    _, old_wi, old_distance, ids = model.sample_surface_total(points, 128, torch.Generator().manual_seed(2))
    old_estimates = (old_wi[..., 2:3].clamp_min(0) / model.light_pdf_surrogate(old_distance, ids, old_wi)).mean(1)
    # This is a controlled synthetic noise test, NOT evidence for scene quality.
    assert new_estimates.std() / new_estimates.mean() < .08
    assert old_estimates.std() / old_estimates.mean() > 3 * new_estimates.std() / new_estimates.mean()
    two = AreaEmitterModel(torch.zeros(2, 3, dtype=torch.float64), torch.ones(2, 3, dtype=torch.float64))
    _, _, _, _, two_pdf = two.sample_solid_angle_total(points[:1], 128, torch.Generator().manual_seed(2))
    torch.testing.assert_close(two_pdf, pdf[:1] / 2)


@pytest.mark.parametrize('distance', [1.00001, 2., 100000.])
def test_float32_tiny_cone_is_finite_and_hits_sphere(distance):
    model = sphere(torch.float32)
    q, directions, _, _, pdf = model.sample_solid_angle_total(
        torch.tensor([[0., 0., -distance]]), 1024, torch.Generator().manual_seed(4))
    assert torch.isfinite(pdf).all() and (pdf > 0).all()
    torch.testing.assert_close(q.norm(dim=-1), torch.ones_like(q[..., 0]), atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(directions.norm(dim=-1), torch.ones_like(directions[..., 0]))


def test_experiment_overrides_do_not_modify_base_config():
    from scripts.compare_solid_angle_render import experiment_config
    base = AEGIRConfig()
    base.render.lighting_mode = DIRECT_SH_MODE
    saved = asdict(base)
    new = experiment_config(base, 'solid_angle', 512)
    assert asdict(base) == saved
    assert new.render.num_light_samples == 512 and new.render.light_sampling_strategy == 'solid_angle'
    assert new.render.num_brdf_samples == 0 and not new.render.indirect_bounce
