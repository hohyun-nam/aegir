"""Fresh AEGIR optimization from SAM-retagged geometry and emitters, then inspect."""
import argparse
from pathlib import Path
import sys
import signal
import time
from dataclasses import asdict, replace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aegir.checkpoint import load_checkpoint
from aegir.config import config_from_dict
from aegir.data import MultiViewSceneDataset, image_dimensions
from aegir.inspect import inspect_checkpoint
from aegir.indirect import DIRECT_SH_MODE, LIGHTING_MODES, validate_lighting_config
from aegir.sample_reduction import central_fraction_bounds
from aegir.provenance import sha256_file, training_inputs, validate_emitter_inclusive_geometry, validate_sam3_ldr_view
from aegir.trainer import SharedGaussianTrainer, _factor_patch_shape, scheduled_indirect_lr


def configure_training_patch(config, args):
    """Override primary patch pixels only when explicitly requested."""
    pixels = config.training.rays_per_step if args.rays_per_step is None else args.rays_per_step
    size = ((args.image_height, args.image_width) if getattr(args, 'image_width', None) is not None
            else args.image_size)
    height, width = image_dimensions(size)
    shape = _factor_patch_shape(pixels, height, width)
    config.raster.image_size = size
    config.training.rays_per_step = pixels
    return shape


def configure_training_schedule(config, args):
    """Maintain the first 60% of a fresh run, then decay over the final 40%."""
    config.training.steps = args.steps
    config.training.freeze_start = (args.freeze_start if args.freeze_start is not None
                                    else int(args.steps * 0.6))
    config.training.freeze_end = args.freeze_end if args.freeze_end is not None else args.steps
    config.training.indirect_start_step = args.indirect_start_step
    config.training.indirect_lr = args.indirect_lr
    config.training.indirect_final_lr = args.indirect_final_lr
    if config.training.indirect_final_lr is None and args.steps > 5000 and args.lighting_mode == DIRECT_SH_MODE:
        config.training.indirect_final_lr = min(1.0e-4, args.indirect_lr)
    if not 0 <= config.training.freeze_start < config.training.freeze_end <= args.steps:
        raise ValueError('schedule requires 0 <= freeze_start < freeze_end <= steps')
    scheduled_indirect_lr(config, 0)


def configure_training_sample_reduction(config, args):
    """Opt-in trimmed direct-light training; leave unrelated defaults intact."""
    render = replace(config.render)
    mode = getattr(args, 'direct_sample_reduction', None)
    percent = getattr(args, 'direct_sample_keep_percent', None)
    if mode is not None:
        render.direct_sample_reduction = mode
    if percent is not None:
        if mode not in (None, 'trimmed'):
            raise ValueError('keep percent requires omitted reduction or trimmed')
        render.direct_sample_reduction = 'trimmed'
        render.direct_sample_keep_fraction = percent / 100.0
    validate_lighting_config(render)
    config.render.direct_sample_reduction = render.direct_sample_reduction
    config.render.direct_sample_keep_fraction = render.direct_sample_keep_fraction


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--initialization', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--priors', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=5000)
    p.add_argument('--freeze-start', type=int, default=None,
                   help='Default: 60%% of steps (3K at 5K, 12K at 20K)')
    p.add_argument('--freeze-end', type=int, default=None, help='Default: final step')
    p.add_argument('--image-size', type=int, default=128)
    p.add_argument('--image-width', type=int)
    p.add_argument('--image-height', type=int)
    p.add_argument('--prior-scale', type=float, default=1.0)
    p.add_argument('--gt-depth-ablation', action='store_true', help='Explicit oracle geometry branch; never GT-free')
    p.add_argument('--resume', type=Path, help='Exact optimizer/RNG resume; only patch pixels may change')
    p.add_argument('--checkpoint-every', type=int, default=1000)
    p.add_argument('--cooperative-stop', action='store_true', help='Save at a completed step on Ctrl-C/SIGTERM')
    p.add_argument('--no-inspect', action='store_true')
    p.add_argument('--rays-per-step', type=int, default=None,
                   help='Primary patch pixels, not emitter samples; default: saved initialization (usually 256)')
    p.add_argument('--inspect-image-size', type=int, default=256)
    p.add_argument('--render-samples', type=int, default=None,
                   help='Default: 128 explicit transport, 32 learned indirect')
    p.add_argument('--lighting-mode', choices=LIGHTING_MODES, default='explicit_transport')
    p.add_argument('--direct-sample-reduction', choices=('mean', 'trimmed80', 'trimmed'),
                   help='Training estimator override; omitted preserves initialization configuration')
    p.add_argument('--direct-sample-keep-percent', type=float,
                   help='Central HDR-luminance percentage used in training AND final inspect (e.g. 90)')
    p.add_argument('--indirect-irradiance-init', type=float, default=0.03)
    p.add_argument('--indirect-lr', type=float, default=1.0e-3)
    p.add_argument('--indirect-start-step', type=int, default=100)
    p.add_argument('--indirect-final-lr', type=float, default=None,
                   help='Default: 0.0001 for learned-indirect runs longer than 5K; constant otherwise')
    p.add_argument('--indirect-energy', type=float, default=0.1)
    p.add_argument('--indirect-directional', type=float, default=0.1)
    p.add_argument('--inspect-views', type=int, default=30)
    p.add_argument('--inspect-steps', type=int, nargs='+', default=None,
                   help='Inspect saved milestones after fitting; default is final step only')
    p.add_argument('--focus-lambda', type=float, default=5.0)
    p.add_argument('--allow-partial-sam', action='store_true',
                   help='Explicit subset experiment using a sampled SAM cache')
    args = p.parse_args()
    if (args.image_width is None) != (args.image_height is None):
        p.error('image-width and image-height must be specified together')
    if args.image_width is not None:
        image_dimensions((args.image_height, args.image_width))
    if not 0 < args.prior_scale <= 1 or args.checkpoint_every <= 0:
        p.error('prior scale must be in (0,1], checkpoint interval positive')
    if args.render_samples is None:
        args.render_samples = 32 if args.lighting_mode == DIRECT_SH_MODE else 128
    if min(args.steps, args.image_size, args.inspect_image_size, args.render_samples, args.inspect_views) <= 0:
        p.error('steps, training/inspect image sizes, samples and inspect views must be positive')
    if args.rays_per_step is not None and args.rays_per_step <= 0:
        p.error('rays-per-step must be positive')
    if args.focus_lambda < 0:
        p.error('focus lambda must be non-negative')
    inspect_steps = sorted(set(args.inspect_steps or []) | {args.steps})
    if any(step <= 0 or step > args.steps or (step != args.steps and step % 1000 != 0)
           for step in inspect_steps):
        p.error('inspect steps must be positive, <= steps, and multiples of 1000 except the final step')
    if args.output.exists() and args.resume is None:
        raise FileExistsError(f'Choose a fresh output directory: {args.output}')
    torch.manual_seed(2026)
    metadata = validate_sam3_ldr_view(args.data)
    if metadata['sam3_provenance'].get('partial_cache') and not args.allow_partial_sam:
        raise ValueError('Sampled SAM cache requires --allow-partial-sam or regenerated full masks')
    scene, emitters, payload = load_checkpoint(args.initialization, 'cuda')
    if not (scene.emitter_ids >= 0).any():
        raise ValueError('Initialization contains no tagged Gaussian rows')
    if payload.get('sam3_retagging', {}).get('method') != 'sam3_visible_contribution_multiview':
        raise ValueError('Requires SAM multiview contribution tags')
    if payload['sam3_retagging'].get('data_metadata_sha256') != sha256_file(args.data / 'transforms.json'):
        raise ValueError('SAM tagging input metadata differs; rebuild tags for this training view')
    if args.gt_depth_ablation:
        from aegir.gt_depth_ablation import validate_geometry
        geometry_provenance = validate_geometry(Path(scene.source_checkpoint))
    else:
        validate_emitter_inclusive_geometry(Path(scene.source_checkpoint))
    config = config_from_dict(payload['config'])
    config.render.tagged_surface_emission = True
    config.render.lighting_mode = args.lighting_mode
    config.render.indirect_irradiance_init = args.indirect_irradiance_init
    configure_training_schedule(config, args)
    config.losses.indirect_energy = args.indirect_energy
    config.losses.indirect_directional = args.indirect_directional
    for name in ('prior_albedo', 'prior_roughness', 'prior_metallic'):
        setattr(config.losses, name, getattr(config.losses, name) * args.prior_scale)
    config.render.point_visibility = True
    config.render.indirect_bounce = True
    config.render.num_brdf_samples = args.render_samples
    config.render.num_light_samples = args.render_samples
    if args.lighting_mode == DIRECT_SH_MODE:
        config.render.num_brdf_samples = 0
        config.render.indirect_bounce = False
    configure_training_sample_reduction(config, args)
    config.render.num_bounce_samples = 1
    config.render.bounce_chunk_size = 4096
    patch_shape = configure_training_patch(config, args)
    config.training.steps = args.steps
    config.training.checkpoint_every = args.checkpoint_every
    config.training.training_transport_mode = 'secondary_only'
    config.training.geometry_lr = 0
    config.training.optimize_scene_means = False
    config.training.optimize_scene_scales = False
    config.training.optimize_scene_rotations = False
    config.training.optimize_scene_opacities = False
    config.training.optimize_emitter_centers = False
    config.training.optimize_emitter_rotations = False
    # The camera footprint is fixed by source surfels, so keep the transport
    # emitter support fixed as well for this experiment.
    config.training.optimize_emitter_scales = False
    config.training.associate_emitter_gaussians = True
    config.scene_culling.enabled = False
    config.adaptive.enabled = False
    config.losses.light_white = 0
    config.losses.focus_lambda = args.focus_lambda
    print(f'Training raster (height,width)={image_dimensions(config.raster.image_size)}; patch {patch_shape[0]}x{patch_shape[1]} '
          f'({config.training.rays_per_step} pixels); {args.render_samples} samples/pixel; '
          f'lighting mode {args.lighting_mode}', flush=True)
    reduction = config.render.direct_sample_reduction
    fraction = (0.8 if reduction == 'trimmed80' else config.render.direct_sample_keep_fraction)
    low, high = ((0, args.render_samples) if reduction == 'mean'
                 else central_fraction_bounds(args.render_samples, fraction))
    print(f'Direct estimator: {reduction}; retain {high-low}/{args.render_samples} samples; '
          f'central fraction={fraction if reduction != "mean" else 1.0}', flush=True)
    dataset = MultiViewSceneDataset(args.data, scene.source_checkpoint, args.priors,
                                   image_size=config.raster.image_size, split='train')
    dataset.training_input_provenance = training_inputs(dataset.root, Path(scene.source_checkpoint), dataset.prior_dir)
    dataset.training_input_provenance['tagged_initialization'] = {
        'path': str(args.initialization.resolve()),
        'sha256': sha256_file(args.initialization),
        'sam3_retagging': payload['sam3_retagging'],
        'trainer_script_sha256': sha256_file(Path(__file__)),
    }
    if args.direct_sample_reduction is not None or args.direct_sample_keep_percent is not None:
        dataset.training_input_provenance['direct_sample_estimator'] = {
            'mode': reduction, 'requested_keep_fraction': fraction if reduction != 'mean' else 1.0,
            'drawn_samples': args.render_samples, 'retained_samples': high-low,
            'reducer_sha256': sha256_file(Path(__file__).resolve().parents[1] / 'aegir/sample_reduction.py'),
        }
    if args.gt_depth_ablation:
        dataset.training_input_provenance['gt_depth_ablation'] = geometry_provenance
    dataset.emitter_gaussian_association_diagnostic = payload['sam3_retagging']
    dataset.emitter_initialization_diagnostic = payload.get('dataset', {}).get('emitter_initialization_diagnostic')
    args.output.mkdir(parents=True, exist_ok=args.resume is not None)
    checkpoint = args.output / f'aegir_{args.steps}.pt'
    start_step = 0
    resume_payload = None
    if args.resume is not None:
        resume_scene, resume_emitters, resume_payload = load_checkpoint(args.resume, 'cuda')
        old_config = config_from_dict(resume_payload['config'])
        old_pixels = old_config.training.rays_per_step
        old_config.training.rays_per_step = config.training.rays_per_step
        if asdict(old_config) != asdict(config):
            raise ValueError('Resume may change only rays-per-step; preserve resolution/schedule/priors/samples')
        if resume_payload['dataset']['input_provenance'] != dataset.training_input_provenance:
            raise ValueError('Resume input content differs')
        if resume_payload['dataset']['frame_indices'] != list(dataset.selected_indices):
            raise ValueError('Resume view selection differs')
        if resume_payload['source_dnsplatter_checkpoint'] != scene.source_checkpoint:
            raise ValueError('Resume source geometry differs')
        scene, emitters = resume_scene, resume_emitters
        if old_pixels != config.training.rays_per_step:
            from aegir.provenance import write_json
            write_json(args.output / 'patch_change.json', dict(resume=str(args.resume.resolve()),
                resume_sha256=sha256_file(args.resume), old_pixels=old_pixels,
                new_pixels=config.training.rays_per_step, scientific_protocol_changed=True))
    trainer = SharedGaussianTrainer(scene, dataset, emitters, config, checkpoint, 'cuda')
    if resume_payload is not None:
        start_step = trainer.restore_exact_state(resume_payload, args.resume)
    else:
        trainer._save(0, args.output/'aegir_initialization.pt')
    stop = {'requested': False}
    def request_stop(signum, _frame):
        stop['requested'] = True
        print(f'STOP_REQUESTED signal={signum}; finish current optimizer step, then save. Please wait.', flush=True)
    if args.cooperative_stop:
        signal.signal(signal.SIGINT, request_stop)
        signal.signal(signal.SIGTERM, request_stop)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    finished = trainer.fit(start_step=start_step,
                           stop_requested=(lambda: stop['requested']) if args.cooperative_stop else None)
    print(f'FIT_WALL_SECONDS={time.monotonic() - started:.3f}', flush=True)
    print(f'PEAK_GPU_RESERVED_GIB={torch.cuda.max_memory_reserved()/1024**3:.3f}', flush=True)
    if not finished:
        raise SystemExit(130)
    if args.cooperative_stop:
        # Training is safely saved. Post-fit inspection can now stop directly.
        signal.signal(signal.SIGINT, signal.default_int_handler)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
    del trainer, scene, emitters
    torch.cuda.empty_cache()
    for step in ([] if args.no_inspect else inspect_steps):
        milestone = checkpoint if step == args.steps else checkpoint.with_suffix(f'.step{step:05d}.pt')
        inspect_dir = args.output / ('inspect' if step == args.steps else f'inspect_{step:06d}')
        inspect_checkpoint(milestone, args.data, args.priors, inspect_dir, 'cuda',
                           max_views=args.inspect_views, split='train', render_samples=args.render_samples,
                           bounce_samples=8, median_indirect=True,
                           image_size=args.inspect_image_size,
                           direct_sample_reduction=args.direct_sample_reduction,
                           direct_sample_keep_percent=args.direct_sample_keep_percent)
    if not args.no_inspect and args.gt_depth_ablation:
        inspect_checkpoint(checkpoint, args.data, args.priors, args.output / 'inspect_native', 'cuda',
                           max_views=args.inspect_views, split='train', render_samples=args.render_samples,
                           median_indirect=True, image_size=config.raster.image_size,
                           direct_sample_reduction=args.direct_sample_reduction,
                           direct_sample_keep_percent=args.direct_sample_keep_percent)
    print(f'COMPLETE: {checkpoint}', flush=True)
    if not args.no_inspect:
        print(f'Inspect: {args.output / "inspect"}', flush=True)


if __name__ == '__main__':
    main()
