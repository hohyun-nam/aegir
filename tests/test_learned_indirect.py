from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from aegir.checkpoint import load_checkpoint, save_checkpoint
from aegir.config import AEGIRConfig, RenderConfig, config_from_dict
from aegir.emitters import AreaEmitterModel
from aegir.gaussian_scene import GaussianMaterialBackend
from aegir.indirect import DIRECT_SH_MODE, GlobalDiffuseSH, load_indirect_light
from aegir.render import DeferredRenderer
from aegir.trainer import SharedGaussianTrainer, scheduled_indirect_lr, training_sh_proxy_weight
from scripts.train_tagged_surface_scene import configure_training_schedule


def gbuffer(count=2):
    return dict(points=torch.zeros(count, 3),
                normals=torch.tensor([[0., 0., 1.]]).repeat(count, 1),
                view_dirs=torch.tensor([[0., 0., 1.]]).repeat(count, 1),
                albedo=torch.full((count, 3), .5), roughness=torch.full((count, 1), .6),
                metallic=torch.zeros(count, 1), alpha=torch.ones(count, 1),
                valid_mask=torch.ones(count, 1, dtype=torch.bool))


def emitters(count=1):
    return AreaEmitterModel(centers=torch.tensor([[0., 0., 2.]]).repeat(count, 1),
                            scales=torch.full((count, 2), .1), colors=torch.ones(count, 3))


def test_global_sh_constant_directionality_and_material_response():
    light = GlobalDiffuseSH(.3)
    normals = torch.tensor([[0., 0., 1.], [0., 0., -1.], [1., 0., 0.]])
    assert torch.allclose(light(normals), torch.full((3, 3), .3))
    result = light.diffuse(normals, torch.full((3, 3), .5), torch.tensor([[0.], [.5], [1.]]))
    assert torch.allclose(result[0], torch.full((3,), .15 / torch.pi))
    assert torch.allclose(result[1], result[0] * .5)
    assert torch.equal(result[2], torch.zeros(3))
    with torch.no_grad():
        light.coefficients[2] = .1
    assert bool((light(normals)[0] > light(normals)[1]).all())
    assert torch.equal(light(normals[:1]), light(normals[:1].repeat(5, 1))[:1])
    light(normals).sum().backward()
    assert torch.isfinite(light.coefficients.grad).all()


def test_sh_nonnegative_energy_penalty_and_hdr_range():
    light = GlobalDiffuseSH(4.)
    normals = torch.tensor([[0., 0., 1.], [0., 0., -1.]])
    assert bool((light(normals) > 1).all())
    assert light.regularization(1., 0.).item() == pytest.approx(16.)
    with torch.no_grad():
        light.coefficients[2] = 100.
    assert bool((light(normals) >= 0).all())
    assert light.regularization(0., 1.).item() > 0


@pytest.mark.parametrize('sample_count', [1, 4, 32])
def test_light_only_estimator_uses_light_pdf_and_never_traces_secondary(monkeypatch, sample_count):
    config = RenderConfig(lighting_mode=DIRECT_SH_MODE, num_brdf_samples=128,
                          num_light_samples=sample_count, camera_visible_emitters=False)
    light = GlobalDiffuseSH(.03)
    lamps = emitters(2)
    observed_rays = []
    visibility = torch.nn.Parameter(torch.tensor(.5))

    def sample(points, samples, generator=None):
        shape = (len(points), samples)
        dirs = torch.tensor([0., 0., 1.]).expand(*shape, 3)
        return dirs * 2., dirs, torch.full((*shape, 1), 2.), torch.zeros(shape, dtype=torch.long)

    def shadow(origins, dirs, distances):
        observed_rays.append(dirs.numel() // 3)
        assert bool(torch.isfinite(distances).all())
        return visibility.expand_as(distances)

    def forbidden(*args, **kwargs):
        raise AssertionError('cosine/secondary transport was used')

    monkeypatch.setattr(lamps, 'sample_surface_total', sample)
    # Selected-light PDF includes emitter selection; the old BRDF mixture
    # would produce a different answer for these deterministic samples.
    monkeypatch.setattr(lamps, 'light_pdf_surrogate', lambda d, *_: torch.full_like(d, .25))
    monkeypatch.setattr(lamps, 'radiance_for_ids', lambda directions, *_args, **_kwargs: torch.full_like(directions, 2.))
    renderer = DeferredRenderer(config, visibility_fn=shadow, transport_fn=forbidden, indirect_light=light)
    monkeypatch.setattr(renderer, '_brdf_samples', forbidden)
    monkeypatch.setattr(renderer, '_explicit_secondary_bounce', forbidden)
    inputs = gbuffer()
    result = renderer.render(inputs, lamps, tonemap=False)
    fr = renderer.brdf(inputs['view_dirs'], inputs['normals'][:, None], inputs['normals'],
                       inputs['albedo'], inputs['roughness'], inputs['metallic'])[:, 0]
    assert torch.allclose(result['direct_hdr'], fr * 2. * .5 / .25)
    assert torch.allclose(result['indirect_hdr'], light.diffuse(inputs['normals'], inputs['albedo'], inputs['metallic']))
    assert observed_rays == [2 * sample_count]
    result['hdr'].sum().backward()
    assert visibility.grad.item() > 0
    assert bool(torch.isfinite(light.coefficients.grad).all())


def test_fast_mode_without_emitters_keeps_ambient_and_respects_masks():
    inputs = gbuffer(3)
    inputs['alpha'][0] = .5
    inputs['valid_mask'][1] = False
    inputs['metallic'][2] = 1.
    config = RenderConfig(lighting_mode=DIRECT_SH_MODE, point_visibility=False)
    result = DeferredRenderer(config, indirect_light=GlobalDiffuseSH(.3)).render(inputs, emitters(0))
    assert torch.count_nonzero(result['direct_hdr']) == 0
    assert torch.allclose(result['indirect_hdr'][0], torch.full((3,), .075 / torch.pi))
    assert torch.count_nonzero(result['indirect_hdr'][1:]) == 0


def test_camera_emission_stays_separate_in_fast_mode():
    inputs = gbuffer(1)
    inputs['normals'] *= -1  # No direct diffuse/specular reflection toward the light.
    inputs['emitter_weights'] = torch.ones(1, 1)
    config = RenderConfig(lighting_mode=DIRECT_SH_MODE, tagged_surface_emission=True,
                          point_visibility=False, num_light_samples=2)
    lamps = emitters()
    renderer = DeferredRenderer(config, indirect_light=GlobalDiffuseSH(.03))
    result = renderer.render(inputs, lamps, tonemap=False)
    expected, mask = renderer.camera_visible_emission(inputs, lamps)
    assert bool((expected > 0).all())
    assert torch.equal(result['emission_hdr'], expected)
    assert torch.equal(result['emitter_mask'], mask)
    assert torch.allclose(result['hdr'], result['direct_hdr'] + result['indirect_hdr'] + expected)


def test_missing_model_and_invalid_config_fail_explicitly():
    with pytest.raises(ValueError, match='indirect_light'):
        DeferredRenderer(RenderConfig(lighting_mode=DIRECT_SH_MODE)).render(gbuffer(), emitters())
    with pytest.raises(ValueError, match='positive num_light_samples'):
        DeferredRenderer(RenderConfig(lighting_mode=DIRECT_SH_MODE, num_light_samples=0))
    with pytest.raises(ValueError, match='lighting_mode'):
        DeferredRenderer(RenderConfig(lighting_mode='typo'))
    with pytest.raises(ValueError, match='indirect_light_state'):
        load_indirect_light({'config': {'render': {'lighting_mode': DIRECT_SH_MODE}}}, 'cpu')
    assert load_indirect_light({'config': {}}, 'cpu') is None


def make_source(path: Path):
    torch.save(dict(means=torch.tensor([[0., 0., -2.], [.1, .1, -2.]]),
                    quats=torch.tensor([[1., 0., 0., 0.]]).repeat(2, 1),
                    scales=torch.tensor([[.1, .1, .01]]).repeat(2, 1), opacities=torch.full((2, 1), .8),
                    features_dc_raw=torch.zeros(2, 1, 3), features_rest_raw=torch.zeros(2, 15, 3)), path)


def fake_dataset(root, source):
    frame = SimpleNamespace(target=torch.full((4, 4, 3), .4), radiance_hdr=None,
                            mask=torch.ones(4, 4, 1, dtype=torch.bool), priors={},
                            camera={}, emitter_mask=None, index=0, ldr_exposure_scale=1.,
                            source_frame={'file_path': 'rgb.png'})

    class Dataset:
        def __len__(self):
            return 1

        def __getitem__(self, index):
            return frame

        def camera_regularization_data(self, device):
            return {}

    data = Dataset()
    data.root, data.prior_dir, data.image_size = root, None, 4
    data.max_views, data.split, data.selected_indices, data.frames = None, 'train', [0], [frame]
    data.metadata = {}
    data.observation_policy = dict(domain='ldr', source='declared', exposure_policy='fixed',
                                   tone_mapper='reinhard', transfer_function='srgb',
                                   target_encoding='srgb', exposure_scale=1.)
    return data


def attach_cpu_raster(scene):
    def rasterize(*args, **kwargs):
        inputs = gbuffer(16)
        inputs['points'] = scene.means[:1].expand(16, 3)
        inputs['albedo'] = scene.albedo[:1].expand(16, 3)
        inputs['roughness'] = scene.roughness[:1].expand(16, 1)
        inputs['metallic'] = scene.metallic[:1].expand(16, 1)
        return {key: value.reshape(4, 4, -1) for key, value in inputs.items()}
    scene.rasterize = rasterize


@pytest.mark.parametrize('decay', [False, True])
def test_trainer_exact_resume_preserves_sh_optimizer_and_next_update(tmp_path, decay):
    source = tmp_path / 'source.pt'
    make_source(source)
    config = AEGIRConfig()
    config.render.lighting_mode = DIRECT_SH_MODE
    config.render.num_brdf_samples = 0
    config.render.num_light_samples = 2
    config.render.point_visibility = False
    config.render.indirect_bounce = False
    config.training.indirect_start_step = 0
    if decay:
        config.training.freeze_start = 0
        config.training.freeze_end = 2
        config.training.indirect_final_lr = .0001
    config.training.steps = 1
    config.training.checkpoint_every = 0
    config.training.rays_per_step = 4
    config.scene_culling.enabled = False
    config.adaptive.enabled = False
    dataset = fake_dataset(tmp_path, source)
    scene = GaussianMaterialBackend(source)
    attach_cpu_raster(scene)
    trainer = SharedGaussianTrainer(scene, dataset, emitters(), config, tmp_path/'first.pt', 'cpu')
    torch.manual_seed(17)
    trainer.fit()
    saved = torch.load(tmp_path/'first.pt', weights_only=False)
    assert saved['training_state']['optimizers']['indirect']['state']
    assert saved['losses'][0]['sh_proxy_weight'] == 0
    assert saved['losses'][0]['explicit_bounce_weight'] == 0
    config.training.steps = 2
    trainer.output_path = tmp_path/'continued.pt'
    trainer.fit(start_step=1)
    continued = torch.load(trainer.output_path, weights_only=False)

    reload_scene, reload_emitters, payload = load_checkpoint(tmp_path/'first.pt', 'cpu')
    attach_cpu_raster(reload_scene)
    restored_config = config_from_dict(payload['config'])
    restored_config.training.steps = 2
    resumed = SharedGaussianTrainer(reload_scene, dataset, reload_emitters, restored_config,
                                    tmp_path/'resumed.pt', 'cpu')
    assert resumed.restore_exact_state(payload, tmp_path/'first.pt') == 1
    resumed.fit(start_step=1)
    result = torch.load(resumed.output_path, weights_only=False)
    assert result['losses'] == continued['losses']
    for category in ('scene_state', 'emitter_raw_state', 'indirect_light_state'):
        assert all(torch.equal(result[category][key], value) for key, value in continued[category].items())
    for key in ('exp_avg', 'exp_avg_sq', 'step'):
        assert torch.equal(result['training_state']['optimizers']['indirect']['state'][0][key],
                           continued['training_state']['optimizers']['indirect']['state'][0][key])


def test_sh_training_delay_config_compatibility_and_legacy_ellipsoid(tmp_path):
    legacy = asdict(AEGIRConfig())
    del legacy['render']['lighting_mode']
    assert config_from_dict(legacy).render.lighting_mode == 'explicit_transport'
    source = tmp_path/'source.pt'
    make_source(source)
    config = AEGIRConfig()
    config.render.lighting_mode = DIRECT_SH_MODE
    config.render.point_visibility = False
    config.training.indirect_start_step = 10
    trainer = SharedGaussianTrainer(GaussianMaterialBackend(source), fake_dataset(tmp_path, source),
                                    emitters(), config, tmp_path/'unused.pt', 'cpu')
    trainer._update_lrs(9)
    assert trainer.indirect_optimizer.param_groups[0]['lr'] == 0
    assert not trainer.indirect_light.coefficients.requires_grad
    trainer._update_lrs(10)
    assert trainer.indirect_optimizer.param_groups[0]['lr'] == config.training.indirect_lr
    assert trainer.indirect_light.coefficients.requires_grad
    assert training_sh_proxy_weight(config, 0) == 0
    lamps = AreaEmitterModel(centers=torch.zeros(1, 3), scales=torch.tensor([[.1, .2, .3]]))
    save_checkpoint(tmp_path/'legacy.pt', trainer.scene, lamps, AEGIRConfig(), 0, [], {})
    payload = torch.load(tmp_path/'legacy.pt', weights_only=False)
    payload['format'] = 'aegir_shared_gaussian_v16'
    torch.save(payload, tmp_path/'legacy.pt')
    _, restored_lamps, _ = load_checkpoint(tmp_path/'legacy.pt', 'cpu')
    assert restored_lamps.representation == 'ellipsoid_3d'
    assert torch.equal(restored_lamps.scales, lamps.scales)


@pytest.mark.parametrize('step,expected_sh,material_factor', [
    (0, 0., 1.), (99, 0., 1.), (100, .001, 1.),
    (11999, .001, 1.), (12000, .001, 1.),
    (16000, .00055, .5), (20000, .0001, 0.), (21000, .0001, 0.),
])
def test_20k_optimizer_schedule_boundaries(tmp_path, step, expected_sh, material_factor):
    source = tmp_path/'source.pt'
    make_source(source)
    config = AEGIRConfig()
    config.render.lighting_mode = DIRECT_SH_MODE
    config.render.point_visibility = False
    args = SimpleNamespace(steps=20000, freeze_start=None, freeze_end=None,
                           indirect_start_step=100, indirect_lr=.001, indirect_final_lr=None,
                           lighting_mode=DIRECT_SH_MODE)
    configure_training_schedule(config, args)
    assert config.training.freeze_start == 12000
    assert config.training.freeze_end == 20000
    trainer = SharedGaussianTrainer(GaussianMaterialBackend(source), fake_dataset(tmp_path, source),
                                    emitters(), config, tmp_path/'unused.pt', 'cpu')
    lrs = trainer._update_lrs(step)
    assert lrs[0] == 0
    assert lrs[1] == pytest.approx(.003 * material_factor)
    assert lrs[2] == pytest.approx(.001 * material_factor)
    assert lrs[3] == pytest.approx(.0001 + .0009 * material_factor)
    assert trainer.indirect_optimizer.param_groups[0]['lr'] == pytest.approx(expected_sh)
    assert trainer.indirect_light.coefficients.requires_grad == (expected_sh > 0)
    assert trainer.scene.albedo_logits.requires_grad == (material_factor > 0)


def test_constant_sh_compatibility_and_schedule_overrides():
    config = AEGIRConfig()
    for step in (100, 5000, 20000):
        assert scheduled_indirect_lr(config, step) == .001
    legacy = asdict(config)
    del legacy['training']['indirect_final_lr']
    restored = config_from_dict(legacy)
    assert restored.training.indirect_final_lr is None
    assert scheduled_indirect_lr(restored, 20000) == .001
    args = SimpleNamespace(steps=5000, freeze_start=None, freeze_end=None,
                           indirect_start_step=100, indirect_lr=.001, indirect_final_lr=None,
                           lighting_mode=DIRECT_SH_MODE)
    configure_training_schedule(config, args)
    assert config.training.freeze_start == 3000
    assert config.training.freeze_end == 5000
    assert config.training.indirect_final_lr is None
    args.steps, args.freeze_start, args.freeze_end = 20000, 10000, 18000
    args.indirect_start_step, args.indirect_final_lr = 0, .0002
    configure_training_schedule(config, args)
    assert scheduled_indirect_lr(config, 0) == .001
    assert scheduled_indirect_lr(config, 14000) == pytest.approx(.0006)
    assert scheduled_indirect_lr(config, 20000) == .0002
