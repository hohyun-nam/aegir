"""Opt-in LDR/SAM lighting adaptation; legacy adapt-lighting is unchanged."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import subprocess

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aegir.checkpoint import load_checkpoint, module_state_dict_cpu
from aegir.color import apply_ldr_pipeline
from aegir.config import config_from_dict
from aegir.data import MultiViewSceneDataset
from aegir.indirect import DIRECT_SH_MODE, load_indirect_light
from aegir.inspect import inspect_checkpoint, render_full_frame
from aegir.provenance import sha256_file, training_inputs, validate_sam3_ldr_view, write_json
from aegir.render import DeferredRenderer
from aegir.trainer import SharedGaussianTrainer, _factor_patch_shape, initialize_emitters

POLICY = 'freeze_scene_geometry_and_brdf_fit_ldr_emitter_appearance_and_global_sh'


def configure_adaptation(config, steps=5000, image_size=128, light_samples=32,
                         rays_per_step=None, reuse_base_emitters=False):
    base_tagged_emission = config.render.tagged_surface_emission
    if steps not in (5000, 10000):
        raise ValueError('Lighting adaptation supports the default 5K or opt-in 10K schedule')
    if image_size <= 0 or light_samples <= 0:
        raise ValueError('Image size and light sample count must be positive')
    pixels = config.training.rays_per_step if rays_per_step is None else rays_per_step
    _factor_patch_shape(pixels, image_size, image_size)
    config.training.steps = steps
    config.training.rays_per_step = pixels
    config.training.checkpoint_every = 1000
    config.training.geometry_lr = config.training.albedo_lr = config.training.material_lr = 0.
    for component in ('means', 'scales', 'rotations', 'opacities'):
        setattr(config.training, f'optimize_scene_{component}', False)
    for component in ('centers', 'rotations', 'scales'):
        setattr(config.training, f'optimize_emitter_{component}', False)
    config.training.constrain_scene_scales = config.training.constrain_emitter_scales = False
    config.training.emitter_lr = config.training.indirect_lr = 1.e-3
    config.training.emitter_final_lr = config.training.indirect_final_lr = 1.e-4
    config.training.indirect_start_step = 100
    config.training.freeze_start = 3000 if steps == 5000 else 7000
    config.training.freeze_end = steps
    config.training.emitter_initialization = 'sam3_multiview'
    config.training.emitter_representation = 'ellipsoid_3d'
    config.training.associate_emitter_gaussians = False
    config.training.training_transport_mode = 'secondary_only'
    config.adaptive.enabled = config.scene_culling.enabled = False
    config.render.lighting_mode = DIRECT_SH_MODE
    config.render.num_light_samples = light_samples
    config.render.num_brdf_samples = 0
    config.render.indirect_bounce = False
    config.render.point_visibility = True
    config.render.bounce_chunk_size = 4096
    config.render.camera_visible_emitters = config.render.six_axis_emission = True
    # New fixtures have no source surfels: use the existing analytic camera
    # layer, not the source condition's tagged-surface emission weights.
    config.render.tagged_surface_emission = base_tagged_emission if reuse_base_emitters else False
    config.render.indirect_irradiance_init = .03
    config.raster.image_size = image_size
    config.losses.light_white = config.losses.light_surface = config.losses.light_view = 0.
    return config


def adaptation_inspect_steps(steps):
    if steps == 5000:
        return (0, 1000, 3000, 5000)  # Preserve the existing 5K runner outputs.
    if steps == 10000:
        return tuple(range(0, 10001, 1000))
    raise ValueError('Unsupported lighting adaptation schedule')


def assert_scene_frozen(before, after, *, reset_ids=False):
    for key, value in before.items():
        if key == 'emitter_ids' and reset_ids:
            if bool((after[key] != -1).any()):
                raise AssertionError('Old emitter associations were not reset')
        elif not torch.equal(value.cpu(), after[key].detach().cpu()):
            raise AssertionError(f'Frozen source scene/BRDF changed: {key}')


def dataset_for(data, source, size, max_views=None):
    dataset = MultiViewSceneDataset(data, source, prior_dir=None, image_size=size,
                                    max_views=max_views, split='train')
    if dataset.observation_policy.get('domain') != 'ldr':
        raise ValueError('This adaptation uses LDR observations only')
    if dataset.observation_policy.get('exposure_policy') != 'fixed_scene':
        raise ValueError('Declare fixed capture exposure in the LDR view')
    if dataset.emitter_definitions or any(
        frame.radiance_hdr is not None or frame.metric_depth is not None or frame.priors
        for frame in dataset.frames
    ):
        raise ValueError('Unexpected HDR/GT/depth/material supervision')
    dataset.training_input_provenance = training_inputs(dataset.root, Path(source), None)
    return dataset


@torch.no_grad()
def validate_camera_layer(scene, emitters, dataset, config, output):
    """Review the actual tagged-surface or analytic camera emission layer."""
    counts = torch.zeros(emitters.num_emitters, dtype=torch.long)
    boards = []
    rows = []
    renderer = DeferredRenderer(config.render)
    for frame in dataset.frames:
        camera = {key: value.cuda() if torch.is_tensor(value) else value
                  for key, value in frame.camera.items()}
        gb = scene.rasterize(camera, include_emitter_gaussians=True,
            camera_emitter_ids=tuple(range(emitters.num_emitters)) if config.render.tagged_surface_emission else None)
        emission, footprint = renderer.camera_visible_emission(gb, emitters)
        if config.render.tagged_surface_emission:
            weights = gb['emitter_weights']
            per_light = [int(((weights[...,i] > 1e-4) & frame.mask.to('cuda')[...,0]).sum())
                         for i in range(emitters.num_emitters)]
            visible = (footprint[...,0] > 1e-4) & frame.mask.to('cuda')[...,0]
        else:
            hit, distance, ids = emitters.intersect(gb['camera_origins'].reshape(-1, 3),
                                               gb['camera_ray_dirs'].reshape(-1, 3))
            visible = hit & (distance[:, 0] < gb['surface_distances'].reshape(-1))
            visible &= frame.mask.to('cuda').reshape(-1)
            per_light = [int((visible & (ids == index)).sum()) for index in range(emitters.num_emitters)]
        counts += torch.tensor(per_light)
        rows.append(dict(frame=frame.index, per_emitter_visible_pixels=per_light,
                         sam_pixels=int(frame.emitter_mask.sum())))
        if len(boards) < 8 and bool(visible.any()):
            rgb = frame.target.cpu().numpy()
            sam = frame.emitter_mask.cpu().numpy()[..., 0]
            predicted = footprint.cpu().numpy()[..., 0] > .5
            overlay = rgb.copy()
            overlay[sam] = .5 * overlay[sam] + .5 * np.array([1., 0., 1.])
            overlay[predicted] = .5 * overlay[predicted] + .5 * np.array([0., 1., 0.])
            emission_rgb = apply_ldr_pipeline(emission.cpu(), exposure=frame.ldr_exposure_scale,
                                              tone_mapper='reinhard', transfer_function='srgb').numpy()
            boards.append(np.concatenate((rgb, overlay, emission_rgb), axis=1))
    if boards:
        Image.fromarray((np.concatenate(boards, axis=0).clip(0, 1) * 255).astype('uint8')).save(
            output / 'initialization_camera_overlay.png')
    report = dict(camera_model=('tagged fixture surfel contribution' if config.render.tagged_surface_emission
                               else 'occlusion-tested analytic ellipsoid'),
                  overlay_columns=['LDR input', 'SAM magenta / analytic green', 'emission only'],
                  visible_pixels_per_emitter=counts.tolist(), views=rows,
                  all_emitters_camera_visible=bool((counts > 0).all()))
    write_json(output / 'camera_visibility.json', report)
    if not report['all_emitters_camera_visible']:
        raise ValueError(f'An initialized light is invisible in all cameras: {counts.tolist()}')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-checkpoint', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--initialization', type=Path,
                        help='Reuse a verified step-0 checkpoint from this script')
    parser.add_argument('--steps', type=int, choices=(5000, 10000), default=5000,
                        help='5K: decay 3K-5K; opt-in 10K: hold LR until 7K, decay 7K-10K')
    parser.add_argument('--expected-emitters', type=int, default=4,
                        help='Legacy count guard; 0 accepts the LDR/SAM-inferred count (no GT count)')
    parser.add_argument('--initialize-only', action='store_true')
    parser.add_argument('--reuse-base-emitters', action='store_true',
                        help='Color/energy-only changes: retain base SAM supports, IDs and camera emission representation')
    parser.add_argument('--smoke', action='store_true', help='Two bounded GPU updates, no main training')
    parser.add_argument('--inspect-samples', type=int, default=32)
    parser.add_argument('--render-samples', type=int, default=32,
                        help='Training emitter samples per primary surface pixel; default remains 32')
    parser.add_argument('--rays-per-step', type=int, default=None,
                        help='Primary patch pixels; default: inherit the base checkpoint')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Choose a fresh output: {args.output}')
    if args.smoke and args.initialize_only:
        parser.error('choose --smoke or --initialize-only')
    if args.expected_emitters < 0 or min(args.inspect_samples, args.render_samples) <= 0:
        parser.error('expected emitters must be nonnegative; sample counts must be positive')
    if args.rays_per_step is not None and args.rays_per_step <= 0:
        parser.error('rays-per-step must be positive')
    torch.set_num_threads(4)
    torch.manual_seed(2026)
    meta = validate_sam3_ldr_view(args.data)
    if meta['sam3_provenance'].get('partial_cache'):
        raise ValueError('Lighting adaptation requires masks for the full training split')
    base_hash = sha256_file(args.base_checkpoint)
    scene, base_emitters, base = load_checkpoint(args.base_checkpoint, 'cuda')
    source = Path(scene.source_checkpoint)
    config = configure_adaptation(config_from_dict(base['config']), steps=args.steps,
                                  light_samples=args.render_samples,
                                  rays_per_step=args.rays_per_step,
                                  reuse_base_emitters=args.reuse_base_emitters)
    source_state = module_state_dict_cpu(scene)
    bounds = tuple(value.cuda() for value in base['training_state']['bounds'])
    dataset = dataset_for(args.data, source, 128)
    provenance = dict(base_checkpoint=str(args.base_checkpoint.resolve()),
                      base_checkpoint_sha256=base_hash, policy=POLICY,
                      reset_previous_emitter_associations=not args.reuse_base_emitters,
                      reuse_base_emitters=args.reuse_base_emitters,
                      reset_indirect_sh=True, uses_hdr=False, uses_gt_emitters=False,
                      uses_gt_depth=False, camera_poses_fixed=True,
                      script_sha256=sha256_file(Path(__file__)),
                      requested_sampling=dict(primary_pixels=config.training.rays_per_step,
                                              light_samples=config.render.num_light_samples,
                                              inspect_samples=args.inspect_samples,
                                              train_image_size=config.raster.image_size,
                                              inspect_image_size=256),
                      requested_schedule=dict(steps=args.steps,
                                              indirect_start_step=config.training.indirect_start_step,
                                              decay_start=config.training.freeze_start,
                                              decay_end=config.training.freeze_end))
    if args.initialization:
        scene, emitters, initial = load_checkpoint(args.initialization, 'cuda')
        record = initial['dataset'].get('lighting_adaptation', {})
        if initial.get('step') != 0 or record.get('policy') != POLICY or record.get(
            'base_checkpoint_sha256') != base_hash:
            raise ValueError('Initialization has a different source scene/policy')
        if bool(record.get('reuse_base_emitters', False)) != args.reuse_base_emitters:
            raise ValueError('Initialization uses a different fixture-support policy')
        if initial['dataset'].get('input_provenance') != dataset.training_input_provenance:
            raise ValueError('Initialization LDR/SAM input content changed')
        if initial.get('losses'):
            raise ValueError('Initialization must not have optimizer updates')
        dataset.emitter_initialization_diagnostic = initial['dataset']['emitter_initialization_diagnostic']
        indirect_light = load_indirect_light(initial, 'cuda')
        provenance['initialization_checkpoint_sha256'] = sha256_file(args.initialization)
    elif args.reuse_base_emitters:
        emitters = base_emitters
        if base['config']['training']['emitter_initialization'] != 'sam3_multiview':
            raise ValueError('Shared fixtures require an LDR/SAM-initialized base checkpoint')
        dataset.emitter_initialization_diagnostic = dict(
            method='reuse_base_LDR_SAM_fixture_support', emitter_count=emitters.num_emitters,
            base_checkpoint_sha256=base_hash, uses_hdr_observations=False,
            uses_gt_emitters=False, uses_predicted_depth=False,
            source_sam_initialization=base['dataset'].get('emitter_initialization_diagnostic'))
        indirect_light = None
    else:
        scene.emitter_ids.fill_(-1)
        scene.requires_grad_(False)
        # Full native-size masks for initialization; only optimization is 128px.
        init_data = dataset_for(args.data, source, 512)
        emitters = initialize_emitters(scene, init_data, bounds, config, 'cuda')
        dataset.emitter_initialization_diagnostic = init_data.emitter_initialization_diagnostic
        indirect_light = None  # Fresh small DC, never the source condition's SH.
        del init_data
    assert_scene_frozen(source_state, scene.state_dict(), reset_ids=not args.reuse_base_emitters)
    args.output.mkdir(parents=True)
    write_json(args.output / 'emitter_initialization.json', dataset.emitter_initialization_diagnostic)
    if emitters.num_emitters <= 0:
        raise ValueError('SAM found no lights; no GT emitter fallback is permitted.')
    if args.expected_emitters > 0 and emitters.num_emitters != args.expected_emitters:
        raise ValueError(f'Expected {args.expected_emitters} SAM lights; found {emitters.num_emitters}. '
                         'No GT emitter fallback is permitted.')
    dataset.lighting_adaptation_provenance = provenance
    if args.smoke:
        config.training.steps = 2
        config.training.checkpoint_every = 2
        config.training.rays_per_step = 64
        config.training.indirect_start_step = 0  # Exercise both optimizers.
        config.render.num_light_samples = 2
    checkpoint = args.output / ('aegir_smoke.pt' if args.smoke else f'aegir_{args.steps}.pt')
    trainer = SharedGaussianTrainer(scene, dataset, emitters, config, checkpoint, 'cuda', bounds,
                                    indirect_light=indirect_light)
    assert_scene_frozen(source_state, scene.state_dict(), reset_ids=not args.reuse_base_emitters)
    if trainer.exposure_optimizer is not None:
        raise AssertionError('Capture exposure unexpectedly trainable')
    assert trainer.log_exposure.detach().exp().item() == 1.
    # Enable ONLY light appearance and new global SH; keep source materials frozen.
    trainer._update_lrs(0)
    assert not any(parameter.requires_grad for parameter in scene.parameters())
    support_before = {key: emitters.state_dict()[key].detach().clone()
                      for key in ('centers', 'raw_quaternions', 'log_scales')}
    sh_before = trainer.indirect_light.coefficients.detach().clone()
    appearance_before = emitters.log_colors.detach().clone()
    camera_report = validate_camera_layer(scene, emitters, dataset, config, args.output)
    _, _, bvh_ids = trainer.visibility_tracer._bounding_mesh()
    assert bool((scene.emitter_ids[bvh_ids] < 0).all())
    trainer._save(0, args.output / 'aegir_initialization.pt')
    write_json(args.output / 'emitter_initialization.json', dataset.emitter_initialization_diagnostic)
    if args.initialize_only:
        subprocess.run([sys.executable, str(Path(__file__).with_name('export_scene_supersplat.py')),
                        '--checkpoint', str(args.output / 'aegir_initialization.pt'),
                        '--output', str(args.output / 'map_with_new_emitters.ply')], check=True)
        print(f'INITIALIZE_ONLY_OK={args.output}', flush=True)
        return
    # A fast-mode regression guard: no cosine or secondary transport calls.
    def forbidden_transport(*_args, **_kwargs):
        raise AssertionError('Unexpected BRDF/cosine/secondary transport in direct+SH adaptation')
    trainer.renderer.transport_fn = forbidden_transport
    trainer.renderer._brdf_samples = forbidden_transport
    trainer.renderer._explicit_secondary_bounce = forbidden_transport
    trainer.fit()
    assert_scene_frozen(source_state, scene.state_dict(), reset_ids=not args.reuse_base_emitters)
    assert all(torch.equal(value, emitters.state_dict()[key]) for key, value in support_before.items())
    assert not torch.equal(sh_before, trainer.indirect_light.coefficients)
    assert not torch.equal(appearance_before, emitters.log_colors)
    result = dict(status='passed', steps=config.training.steps,
                  requested_schedule=provenance['requested_schedule'],
                  frozen_scene_geometry_and_brdf_bitwise_equal=True,
                  frozen_emitter_support_bitwise_equal=True, learned_indirect_updated=True,
                  emitter_appearance_updated=True, exposure_fixed=1.,
                  camera_visibility=camera_report['visible_pixels_per_emitter'],
                  architecture=list(torch.cuda.get_device_capability()),
                  peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3)
    if args.smoke:
        preview_data = MultiViewSceneDataset(args.data, source, image_size=256, split='val', max_views=1)
        frame = preview_data.frames[0]
        camera = {key: value.cuda() if torch.is_tensor(value) else value
                  for key, value in frame.camera.items()}
        with torch.inference_mode():
            rendered = render_full_frame(scene, emitters, camera, config, chunk_size=256,
                                         visibility_tracer=trainer.visibility_tracer,
                                         indirect_light=trainer.indirect_light)
        if not all(bool(torch.isfinite(value).all()) for value in rendered.values()):
            raise AssertionError('Non-finite GPU preview')
        assert rendered['hdr'].shape == (256, 256, 3)
        result['inspect_preview_image_size'] = 256
        result['smoke_indirect_warmup_override'] = 0
        rgb = apply_ldr_pipeline(rendered['hdr'], exposure=1., tone_mapper='reinhard',
                                transfer_function='srgb').numpy().clip(0, 1)
        Image.fromarray((rgb * 255).astype('uint8')).save(args.output / 'smoke_rgb.png')
        write_json(args.output / 'smoke_result.json', result)
        print(f'GPU_SMOKE_OK={args.output}', flush=True)
        return
    write_json(args.output / 'frozen_scene_verification.json', result)
    del trainer, scene, emitters
    torch.cuda.empty_cache()
    for step in adaptation_inspect_steps(args.steps):
        path = (args.output / 'aegir_initialization.pt' if step == 0 else checkpoint
                if step == args.steps else checkpoint.with_suffix(f'.step{step:05d}.pt'))
        inspect_checkpoint(path, args.data, None, args.output / f'inspect_{step:06d}',
                           'cuda', split='val', max_views=8, image_size=256,
                           render_samples=args.inspect_samples, evaluation_seed=2026,
                           evaluation_seed_by_view_order=True, median_indirect=True)
    print(f'CORNER4_DIRECT_SH_ADAPTATION_COMPLETE={args.output}', flush=True)


if __name__ == '__main__':
    main()
