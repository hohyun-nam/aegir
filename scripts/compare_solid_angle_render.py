"""Inference-only emitter sampling/reduction comparisons; fixed checkpoint/exposure."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aegir.checkpoint import load_checkpoint
from aegir.color import apply_ldr_pipeline, configure_ldr_fallback
from aegir.config import config_from_dict
from aegir.data import MultiViewSceneDataset
from aegir.evaluation import build_lpips_model, masked_lpips, masked_psnr, masked_ssim
from aegir.indirect import DIRECT_SH_MODE, load_indirect_light
from aegir.inspect import _camera_to, _select_flat, render_full_frame
from aegir.irgs_tracer import IRGSVisibilityTracer
from aegir.provenance import sha256_file, validate_sam3_ldr_view
from aegir.render import DeferredRenderer
from aegir.sample_reduction import central80_bounds
from aegir.visibility import scene_scale_from_bounds
from scripts.compare_legacy_transport_render import median_components, rgb
from scripts.inspect_completed_sample_override import compose


def experiment_config(base, strategy, samples, reduction='mean'):
    cfg = copy.deepcopy(base)
    cfg.raster.image_size = 256
    cfg.render.light_sampling_strategy = strategy
    cfg.render.light_sample_pattern = 'iid'
    cfg.render.direct_sample_reduction = reduction
    cfg.render.num_light_samples = samples
    cfg.render.num_brdf_samples = 0
    cfg.render.indirect_bounce = False
    if cfg.render.lighting_mode != DIRECT_SH_MODE:
        raise ValueError('This experiment requires the direct+SH checkpoint')
    return cfg


@torch.inference_mode()
def noise_probe(scene, emitters, payload, base, ds, tracer, scale, indirect, modes, output):
    frame = next(f for f in ds.frames if f.index == 131)
    active = tuple(range(emitters.num_emitters))
    gb = scene.rasterize(_camera_to(frame.camera, 'cuda'), alpha_threshold=base.raster.alpha_threshold,
        include_emitter_gaussians=True, excluded_emitter_ids=active, camera_emitter_ids=active)
    y, x = torch.meshgrid(torch.arange(8, 112, 8, device='cuda'),
                          torch.arange(8, 248, 12, device='cuda'), indexing='ij')
    y, x = y.flatten(), x.flatten()
    keep = gb['valid_mask'][y, x, 0] & (gb['emitter_weights'][y, x].sum(-1) < .01) & frame.mask.to('cuda')[y, x, 0]
    y, x = y[keep][:192], x[keep][:192]
    if len(x) < 16:
        raise RuntimeError('Too few ceiling surface probes')
    points = _select_flat(gb, y * 256 + x)
    arrays = {}
    seeds = list(range(9100, 9132))
    states = {}
    for name, cfg in modes.items():
        renderer = DeferredRenderer(cfg.render, visibility_fn=tracer.visibility, scene_scale=scale, indirect_light=indirect)
        values = {key: [] for key in ('direct_hdr', 'indirect_hdr', 'hdr')}
        for seed in seeds:
            generator = torch.Generator(device='cuda').manual_seed(seed)
            image = renderer.render(points, emitters, generator=generator,
                                    transport_mode='inference', tonemap=False)
            if cfg.render.light_sampling_strategy == 'surface':
                state_key = (cfg.render.num_light_samples, seed)
                state = generator.get_state()
                if state_key in states and not torch.equal(states[state_key], state):
                    raise RuntimeError('Different RNG consumption in paired sample reductions')
                states[state_key] = state
            for key in values:
                values[key].append(image[key].cpu().numpy())
        for key in values:
            arrays[name + '_' + key] = np.stack(values[key])
        print('NOISE_PROBE_COMPLETE', name, len(x), len(seeds), flush=True)
    results = {}
    weights = np.array([.2126, .7152, .0722], dtype=np.float32)
    for key in ('direct_hdr', 'indirect_hdr', 'hdr'):
        luma = {name: arrays[name + '_' + key] @ weights for name in modes}
        bright = np.logical_and.reduce([v.mean(0) > 1e-4 for v in luma.values()])
        results[key] = {'common_bright_probes': int(bright.sum()), 'modes': {}}
        for name, value in luma.items():
            mean = value.mean(0)
            constant = bool(np.array_equal(value, np.broadcast_to(value[0], value.shape)))
            std = np.zeros_like(mean) if constant else value.std(0, ddof=1)
            results[key]['modes'][name] = dict(mean_luminance=float(mean[bright].mean()) if bright.any() else None,
                median_cv=float(np.median(std[bright] / mean[bright])) if bright.any() else None,
                mean_seed_std=float(std[bright].mean()) if bright.any() else None,
                median_seed_std=float(np.median(std[bright])) if bright.any() else None,
                all_probe_mean_luminance=float(mean.mean()), all_probe_mean_seed_std=float(std.mean()),
                zero_mean_probes=int((mean <= 1e-4).sum()), seed_bitwise_constant=constant)
    # Also record how many shading points lie within the learned emitter proxies.
    relative = torch.einsum('pei,eij->pej', points['points'][:, None] - emitters.centers, emitters.rotations) / emitters.scales
    reduction_only = len({c.render.light_sampling_strategy for c in modes.values()}) == 1
    result = dict(frame=131, seeds=seeds, probes=len(x), pixel_xy=torch.stack((x, y), -1).cpu().tolist(),
                  probes_inside_any_ellipsoid=int((relative.square().sum(-1) < 1).any(-1).sum()), results=results,
                  paired_rng_state_identical=reduction_only,
                  notes=['Fixed non-emitting ceiling surface points; 32 seeds, raw linear HDR before any filter.',
                         'CV normalizes the noise by each estimator mean; brightness changes are reported separately.',
                         ('Same surface samples/PDF; central80 reduction is biased and may remove legitimate bright contributions.'
                          if reduction_only else 'Historical surface/surrogate-PDF and new exact first-hit estimator have different expected means.'),
                         'Common-bright CV excludes darkened probes; all-probe means/STD and zero counts are also reported.',
                         'Not a whole-image estimate; finite seeds can miss rare fireflies.'])
    (output / 'noise_probe.json').write_text(json.dumps(result, indent=2) + '\n')
    np.savez_compressed(output / 'noise_probe_seed_values.npz', **arrays)
    return result


@torch.inference_mode()
def main(default_experiment='solid_angle'):
    run = ROOT / 'outputs/conference_gtseed20k_native_bigpatch_20261008_gtseed_native01/aegir_5k'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=run / 'aegir_5000.pt')
    parser.add_argument('--data', type=Path, default=ROOT / 'outputs/server_sam3_ldr_inputs_20261006/conference/sam3_ldr_view')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, nargs='+', default=[131, 4, 0, 19, 34, 58, 77, 97, 116, 141])
    parser.add_argument('--experiment', choices=('solid_angle', 'trimmed80'), default=default_experiment)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if len(args.frames) != len(set(args.frames)):
        parser.error('Expected distinct frame indices')
    torch.set_num_threads(4)
    before = sha256_file(args.checkpoint)
    validate_sam3_ldr_view(args.data)
    scene, emitters, payload = load_checkpoint(args.checkpoint, 'cuda')
    scene.requires_grad_(False)
    emitters.requires_grad_(False)
    from aegir.gt_depth_ablation import validate_geometry
    validate_geometry(Path(payload['source_dnsplatter_checkpoint']))
    base = config_from_dict(payload['config'])
    indirect = load_indirect_light(payload, 'cuda')
    indirect.requires_grad_(False)
    if args.experiment == 'trimmed80':
        modes = {f'{reduction}{samples}': experiment_config(base, 'surface', samples, reduction)
                 for samples in (128, 512) for reduction in ('mean', 'trimmed80')}
    else:
        modes = {f'{strategy}{samples}': experiment_config(base, strategy, samples)
                 for samples in (128, 512) for strategy in ('surface', 'solid_angle')}
    ds = MultiViewSceneDataset(args.data, payload['source_dnsplatter_checkpoint'], image_size=256, split='train', max_views=30)
    frames = [next(f for f in ds.frames if f.index == idx) for idx in args.frames]
    policy = configure_ldr_fallback(ds.observation_policy, tone_mapper=base.color.ldr_fallback_tone_mapper,
                                  transfer_function=base.color.ldr_fallback_transfer_function)
    if policy != payload['dataset']['observation_policy'] or policy['domain'] != 'ldr':
        raise RuntimeError('Mismatched LDR observation policy')
    exposure = payload['training_state']['observation']['log_exposure'].exp()
    def display(hdr):
        return apply_ldr_pipeline(hdr, exposure=exposure, tone_mapper=policy['tone_mapper'], transfer_function=policy['transfer_function'])
    active = tuple(range(emitters.num_emitters))
    scale = scene_scale_from_bounds(scene.bounds(base.raster.bounds_quantile))
    tracer = IRGSVisibilityTracer(scene, base.render, scene_scale=scale, excluded_emitter_ids=active)
    lpips = build_lpips_model('alex', 'cuda')
    args.output.mkdir(parents=True)
    for name in (*modes, 'comparison', 'components', 'detail_crops'):
        (args.output / name).mkdir()
    summary = dict(status='running', checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=before,
        experiment=args.experiment,
        training_performed=False, checkpoint_unchanged=None, resolution=[256, 256], frames=args.frames,
        gaussian_count=scene.gaussian_count, emitters=emitters.num_emitters, emitter_scales=emitters.scales.cpu().tolist(),
        seed_rule='2026 + original frame index', learned_exposure=float(exposure), direct_filter='off',
        indirect_filter='same edge-aware3x3 median', geometry_fixed=True, indirect_model='same frozen global diffuse SH',
        gpu=torch.cuda.get_device_name(), architecture=list(torch.cuda.get_device_capability()),
        excluded_emitter_ids=list(active), render_chunk_size=256, cosine_samples=0, secondary_samples=0,
        timing_scope='CUDA-synchronized render_full_frame; includes geometry preparation; excludes setup/postprocess/I/O/metrics',
        warmup='16x16 inference smoke per mode', split='train',
        caveat=('GT-assisted TRAIN reconstruction. Trimmed mean removes both10% tails across all emitters; biased estimator, no retraining/brightness matching.'
                if args.experiment == 'trimmed80' else
                'GT-assisted geometry reconstruction. New exact first-hit PDF changes legacy estimator mean; no retraining/brightness matching.'),
        configs={name: dict(samples=c.render.num_light_samples, strategy=c.render.light_sampling_strategy,
                            pattern=c.render.light_sample_pattern, reduction=c.render.direct_sample_reduction,
                            retained_samples=(central80_bounds(c.render.num_light_samples)[1] - central80_bounds(c.render.num_light_samples)[0]
                                              if c.render.direct_sample_reduction == 'trimmed80' else c.render.num_light_samples))
                 for name, c in modes.items()},
        smoke={}, results={name: [] for name in modes})
    def save():
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    smoke = MultiViewSceneDataset(args.data, payload['source_dnsplatter_checkpoint'], image_size=16, split='train', max_views=30)
    sf = next(f for f in smoke.frames if f.index == 131)
    for name, cfg in modes.items():
        image = render_full_frame(scene, emitters, _camera_to(sf.camera, 'cuda'), cfg,
            visibility_tracer=tracer, generator=torch.Generator(device='cuda').manual_seed(2157),
            transport_mode='inference', excluded_emitter_ids=active, indirect_light=indirect)
        if not all(bool(torch.isfinite(v).all()) for v in image.values()):
            raise RuntimeError('Nonfinite GPU smoke')
        summary['smoke'][name] = {'finite': True}
        print('GPU_SMOKE_COMPLETE', name, flush=True)
    save()
    labels = ['Input RGB', 'Surface128 (historical PDF)', 'Solid angle128 (exact PDF)',
              'Surface512 (historical PDF)', 'Solid angle512 (exact PDF)']
    if args.experiment == 'trimmed80':
        labels = ['Input RGB', 'Mean128 (all128)', 'Central80%128 (keep102)',
                  'Mean512 (all512)', 'Central80%512 (keep410)']
    reference = run / 'render_solid_angle_128_512_20261009'
    if args.experiment == 'trimmed80':
        prior = json.loads((reference / 'summary.json').read_text())
        if prior['checkpoint_sha256'] != before or prior['learned_exposure'] != float(exposure):
            raise RuntimeError('Historical baseline checkpoint/exposure mismatch')
    for frame in frames:
        panels, invariant = {}, None
        rng_states = {}
        for name, cfg in modes.items():
            torch.cuda.synchronize()
            start = time.perf_counter()
            generator = torch.Generator(device='cuda').manual_seed(2026 + frame.index)
            image = render_full_frame(scene, emitters, _camera_to(frame.camera, 'cuda'), cfg,
                visibility_tracer=tracer, generator=generator,
                transport_mode='inference', excluded_emitter_ids=active, indirect_light=indirect)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if args.experiment == 'trimmed80':
                state = generator.get_state()
                sample_count = cfg.render.num_light_samples
                if sample_count in rng_states and not torch.equal(rng_states[sample_count], state):
                    raise RuntimeError('Different full-frame RNG consumption in paired reductions')
                rng_states[sample_count] = state
            if not all(bool(torch.isfinite(v).all()) for v in image.values()):
                raise RuntimeError('Nonfinite full frame')
            # Geometry, camera-visible emission, and unfiltered learned SH must be identical.
            keys = ('alpha', 'depth', 'normals', 'emitter_mask', 'emission_hdr', 'indirect_hdr')
            if invariant is None:
                invariant = {k: image[k].clone() for k in keys}
            elif any(not torch.equal(image[k], invariant[k]) for k in keys):
                raise RuntimeError('Sampling modified fixed geometry/emission/indirect SH')
            raw_indirect = median_components(image, cfg)
            pred = display(image['hdr'])
            row = dict(frame=frame.index, render_seconds=elapsed,
                psnr=float(masked_psnr(pred, frame.target, frame.mask)), ssim=float(masked_ssim(pred, frame.target, frame.mask)),
                lpips=float(masked_lpips(pred, frame.target, frame.mask, lpips, 'cuda')),
                mean_direct_hdr=float(image['direct_hdr'][frame.mask[..., 0].bool()].mean()),
                mean_total_hdr=float(image['hdr'][frame.mask[..., 0].bool()].mean()))
            summary['results'][name].append(row)
            prefix = f'frame_{frame.index:04d}'
            images = {k: rgb(display(image[k + '_hdr'])) for k in ('direct', 'indirect', 'emission')}
            images['total'] = rgb(pred)
            for key, im in images.items():
                im.save(args.output / name / f'{prefix}_{key}.png')
            arrays = {k: image[k].numpy() for k in ('hdr', 'direct_hdr', 'indirect_hdr', 'emission_hdr', 'alpha', 'depth', 'normals', 'emitter_mask')}
            arrays.update(indirect_hdr_before_median=raw_indirect.numpy(), render_rgb=pred.numpy(),
                          target_rgb=frame.target.numpy(), valid_mask=frame.mask.numpy())
            if args.experiment == 'trimmed80' and cfg.render.direct_sample_reduction == 'mean':
                with np.load(reference / f'surface{cfg.render.num_light_samples}' / f'{prefix}.npz') as prior:
                    if any(not np.array_equal(prior[k], value) for k, value in arrays.items()):
                        raise RuntimeError('Historical baseline changed')
            np.savez_compressed(args.output / name / f'{prefix}.npz', **arrays)
            panels[name] = images
            print('FRAME_COMPLETE', name, json.dumps(row), flush=True)
            save()
        prefix = f'frame_{frame.index:04d}'
        total_tiles = [rgb(frame.target)] + [panels[name]['total'] for name in modes]
        compose(total_tiles, labels, f'Conference {frame.index:04d}: fixed 5K /256x256 /same exposure /no direct filter').save(
            args.output / 'comparison' / f'{prefix}.png')
        rows = [compose([panels[name][k] for k in ('total', 'direct', 'indirect', 'emission')],
                        ['Total', 'Direct (unfiltered)', 'Indirect (median3x3)', 'Camera emission'], name) for name in modes]
        grid = Image.new('RGB', (rows[0].width, sum(r.height for r in rows)), 'white')
        offset = 0
        for row in rows:
            grid.paste(row, (0, offset))
            offset += row.height
        grid.save(args.output / 'components' / f'{prefix}.png')
        if frame.index in (131, 4):
            box = (0, 0, 200, 145) if frame.index == 131 else (0, 85, 230, 256)
            crops = [im.crop(box).resize(((box[2] - box[0]) * 3, (box[3] - box[1]) * 3), Image.Resampling.NEAREST) for im in total_tiles]
            compose(crops, labels, f'{frame.index:04d}:3x nearest crop /no additional filter').save(
                args.output / 'detail_crops' / f'{prefix}.png')
    summary['noise_probe'] = noise_probe(scene, emitters, payload, base, ds, tracer, scale, indirect, modes, args.output)
    if sha256_file(args.checkpoint) != before:
        raise RuntimeError('Checkpoint changed')
    summary['mean_results'] = {name: {key: statistics.mean(r[key] for r in rows) for key in
        ('psnr', 'ssim', 'lpips', 'render_seconds', 'mean_direct_hdr', 'mean_total_hdr')} for name, rows in summary['results'].items()}
    summary.update(status='complete', checkpoint_unchanged=True, invariants_bitwise_identical=True)
    if args.experiment == 'trimmed80':
        summary.update(historical_baseline_bitwise_identical=True, paired_full_frame_rng_state_identical=True)
    save()
    options = ''.join(f'<option value="{idx:04d}">{idx:04d}</option>' for idx in args.frames)
    page = '''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Solid angle sampling</title><style>body{background:#222;color:#eee;font-family:system-ui}header{position:sticky;top:0;background:#222;padding:10px}img{max-width:100%}a{color:#acf}</style><header><h2>Conference5K: surface vs affine solid angle</h2><select id="f">OPTIONS</select><p>256×256 /128·512 /같은 노출. 재학습 없음. 직접광 필터 없음 /간접광만 median3×3.</p><p>새 방식은 정확한 PDF·가까운 교차면을 사용하므로 기존 근사식의 평균 밝기와 다를 수 있음.</p><a href="summary.json">점수·시간·설정</a> /<a href="noise_probe.json">32 seed 노이즈</a></header><h3>Input /Surface128 /Solid angle128 /Surface512 /Solid angle512</h3><img id="total"><h3>Total /Direct /Indirect /Emission; each row follows the same order</h3><img id="components"><script>const f=document.getElementById('f');function show(){document.getElementById('total').src='comparison/frame_'+f.value+'.png';document.getElementById('components').src='components/frame_'+f.value+'.png'}f.onchange=show;show();</script></html>'''
    (args.output / 'index.html').write_text(page.replace('OPTIONS', options), encoding='utf-8')
    if args.experiment == 'trimmed80':
        page = page.replace('Solid angle sampling', 'Central80% sample mean')
        page = page.replace('Conference5K: surface vs affine solid angle', 'Conference5K: mean vs central80%')
        page = page.replace('새 방식은 정확한 PDF·가까운 교차면을 사용하므로 기존 근사식의 평균 밝기와 다를 수 있음.',
                            '동일한 샘플·PDF. 최종 HDR 기여도 밝기순으로 상하10%씩 제외.128→102개,512→410개 평균.')
        page = page.replace('Input /Surface128 /Solid angle128 /Surface512 /Solid angle512',
                            'Input /Mean128 /Central80%128 /Mean512 /Central80%512')
        (args.output / 'index.html').write_text(page.replace('OPTIONS', options), encoding='utf-8')
    print('EXPERIMENT_COMPLETE', json.dumps(summary['mean_results']), flush=True)


if __name__ == '__main__':
    main()
