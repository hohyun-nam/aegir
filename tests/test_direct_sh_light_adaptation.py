import json

import numpy as np
from PIL import Image
import pytest
import torch

from aegir.color import resolve_observation_policy
from aegir.config import AEGIRConfig, config_from_dict
from aegir.provenance import validate_sam3_ldr_view
from aegir.trainer import curriculum_lrs, scheduled_indirect_lr
from scripts.adapt_direct_sh_lighting import adaptation_inspect_steps, assert_scene_frozen, configure_adaptation
from scripts.prepare_sam3_training_view import prepare
from scripts.filter_sam3_active_ldr import filter_components


def test_adaptation_freezes_source_and_decays_only_lights_and_new_sh():
    config = configure_adaptation(AEGIRConfig())
    assert config.training.steps == 5000
    assert config.raster.image_size == 128
    assert config.render.num_light_samples == 32
    assert config.render.num_brdf_samples == 0
    assert not config.render.indirect_bounce
    assert config.render.camera_visible_emitters and not config.render.tagged_surface_emission
    assert config.training.emitter_initialization == 'sam3_multiview'
    assert not config.training.associate_emitter_gaussians
    assert not config.adaptive.enabled and not config.scene_culling.enabled
    for name in ('means', 'rotations', 'scales', 'opacities'):
        assert not getattr(config.training, f'optimize_scene_{name}')
    for name in ('centers', 'rotations', 'scales'):
        assert not getattr(config.training, f'optimize_emitter_{name}')
    for step, emitter_lr, sh_lr in [(0, .001, 0.), (99, .001, 0.), (100, .001, .001),
                                    (3000, .001, .001), (4000, .00055, .00055),
                                    (5000, .0001, .0001)]:
        values = curriculum_lrs(config, step)
        assert values[:3] == (0., 0., 0.)
        assert values[3] == pytest.approx(emitter_lr)
        assert scheduled_indirect_lr(config, step) == pytest.approx(sh_lr)


def test_frozen_verification_allows_only_explicit_tag_reset():
    before = dict(means=torch.zeros(2, 3), albedo_logits=torch.zeros(2, 3),
                  emitter_ids=torch.tensor([0, -1]))
    after = {key: value.clone() for key, value in before.items()}
    after['emitter_ids'].fill_(-1)
    assert_scene_frozen(before, after, reset_ids=True)
    with pytest.raises(AssertionError, match='emitter_ids'):
        assert_scene_frozen(before, after)
    after['albedo_logits'][0, 0] = .01
    with pytest.raises(AssertionError, match='albedo_logits'):
        assert_scene_frozen(before, after, reset_ids=True)


def test_opt_in_patch1024_samples128_preserves_frozen_5k_policy():
    from dataclasses import asdict

    config = configure_adaptation(AEGIRConfig(), rays_per_step=1024, light_samples=128)
    assert config.training.rays_per_step == 1024
    assert config.render.num_light_samples == 128
    assert config.render.num_brdf_samples == 0
    assert config.training.freeze_start == 3000 and config.training.freeze_end == 5000
    assert config.training.geometry_lr == config.training.albedo_lr == config.training.material_lr == 0
    assert not config.training.optimize_emitter_scales
    assert not config.render.indirect_bounce
    restored = config_from_dict(asdict(config))
    assert restored.training.rays_per_step == 1024
    assert restored.render.num_light_samples == 128


def test_adaptation_without_patch_override_inherits_base_setting():
    config = AEGIRConfig()
    config.training.rays_per_step = 1024
    config = configure_adaptation(config)
    assert config.training.rays_per_step == 1024
    assert config.render.num_light_samples == 32  # Existing sample default is unchanged.


@pytest.mark.parametrize('overrides', [dict(rays_per_step=0), dict(rays_per_step=16385),
                                     dict(rays_per_step=257), dict(light_samples=0),
                                     dict(image_size=0)])
def test_invalid_adaptation_sampling_fails_before_initialization(overrides):
    with pytest.raises(ValueError):
        configure_adaptation(AEGIRConfig(), **overrides)


def test_opt_in_10k_schedule_extends_plateau_without_unfreezing_scene():
    config = configure_adaptation(AEGIRConfig(), steps=10000)
    assert config.training.steps == 10000
    assert config.training.checkpoint_every == 1000
    assert config.training.freeze_start == 7000
    assert config.training.freeze_end == 10000
    for step, emitter_lr, sh_lr in [(0, .001, 0.), (99, .001, 0.), (100, .001, .001),
                                    (3000, .001, .001), (5000, .001, .001), (7000, .001, .001),
                                    (8500, .00055, .00055), (10000, .0001, .0001)]:
        values = curriculum_lrs(config, step)
        assert values[:3] == (0., 0., 0.)
        assert values[3] == pytest.approx(emitter_lr)
        assert scheduled_indirect_lr(config, step) == pytest.approx(sh_lr)
    assert adaptation_inspect_steps(10000) == tuple(range(0, 10001, 1000))
    assert adaptation_inspect_steps(5000) == (0, 1000, 3000, 5000)


@pytest.mark.parametrize('steps', [0, 2000, 7000, 20000])
def test_unsupported_adaptation_schedules_fail_before_initialization(steps):
    with pytest.raises(ValueError, match='schedule'):
        configure_adaptation(AEGIRConfig(), steps=steps)
    with pytest.raises(ValueError, match='schedule'):
        adaptation_inspect_steps(steps)


def test_active_filter_rejects_dark_fixture_without_trimming_bright_boundaries():
    rgb = np.zeros((5, 6, 3))
    mask = np.zeros((5, 6), dtype=bool)
    mask[:2, :2] = True
    mask[3:, 3:] = True
    rgb[:2, :2] = .15
    rgb[3:, 3:] = .9
    rgb[3, 3] = .2  # Retain the dim edge of an otherwise bright emitter.
    filtered, records = filter_components(rgb, mask, .6)
    assert not filtered[:2, :2].any()
    assert filtered[3:, 3:].all()
    assert [record['kept'] for record in records] == [False, True]


def test_ldr_adaptation_view_keeps_validation_but_no_gt_or_hdr(tmp_path):
    source, run, view = [tmp_path / name for name in ('source', 'sam', 'view')]
    source.mkdir()
    (run / 'inputs').mkdir(parents=True)
    (run / 'sam_masks').mkdir()
    frames = []
    for index, split in enumerate(('train', 'val', 'test')):
        name = f'rgb{index}.png'
        image = Image.fromarray(np.full((8, 8, 3), 100, np.uint8))
        image.save(source / name)
        if split == 'train':
            image.save(run / 'inputs' / name)
            Image.fromarray(np.full((8, 8), 255, np.uint8)).save(run / 'sam_masks' / name)
        frames.append(dict(file_path=name, split=split, depth_file_path='DO_NOT_READ.npy',
                           radiance_hdr_file_path='DO_NOT_READ_hdr.npy',
                           emitter_mask_path='DO_NOT_READ_gt.png', object_id_file_path='GT.npy'))
    (source / 'transforms.json').write_text(json.dumps(dict(frames=frames, emitters=['GT'])))
    (run / 'manifest.json').write_text(json.dumps(dict(data=str(source), split='train',
                                                     selected_images=['rgb0.png'])))
    prepare(source, run, view, keep_validation=True, fixed_exposure=1.)
    meta = validate_sam3_ldr_view(view)
    assert len(meta['frames']) == 3
    assert len(meta['train_filenames']) == len(meta['val_filenames']) == len(meta['test_filenames']) == 1
    assert all('emitter_mask_path' not in frame for frame in meta['frames'][1:])
    policy = resolve_observation_policy(meta, meta['frames'])
    assert policy['exposure_policy'] == 'fixed_scene'
    assert policy['exposure_scale'] == 1.
