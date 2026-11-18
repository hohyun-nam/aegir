"""Render a completed checkpoint at a new sample count; compare cached references."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aegir.inspect import inspect_checkpoint
from aegir.provenance import sha256_file, validate_sam3_ldr_view, write_json


def rgb(value):
    return Image.fromarray(np.rint(np.clip(value, 0, 1)*255).astype(np.uint8))


def compose(images, labels, title):
    w, h = images[0].size
    panel = Image.new('RGB', (len(images)*(w+10)+10, h+72), 'white')
    draw = ImageDraw.Draw(panel)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    draw.text((10, 5), title, font=font, fill='black')
    for i, (im, label) in enumerate(zip(images, labels)):
        x = 10+i*(w+10)
        draw.text((x, 34), label, font=font, fill='black')
        panel.paste(im, (x, 62))
    return panel


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--priors', type=Path, required=True)
    p.add_argument('--reference-256', type=Path, required=True)
    p.add_argument('--reference-native', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--samples', type=int, default=512)
    p.add_argument('--views', type=int, default=30)
    p.add_argument('--light-sample-pattern', choices=('iid','rqmc'), default='iid')
    p.add_argument('--resolutions', nargs='+', choices=('256','native'), default=['256','native'])
    p.add_argument('--frames', nargs='+', type=int, default=None)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if min(args.samples, args.views) <= 0:
        p.error('samples/views must be positive')
    torch.set_num_threads(4)
    protected = [args.checkpoint, args.reference_256/'metrics.json', args.reference_native/'metrics.json']
    hashes = {str(path.resolve()): sha256_file(path) for path in protected}
    validate_sam3_ldr_view(args.data)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if saved['config']['render']['lighting_mode'] != 'direct_only_learned_indirect':
        raise ValueError('Expected the completed direct-only + learned indirect checkpoint')
    from aegir.gt_depth_ablation import validate_geometry
    validate_geometry(Path(saved['source_dnsplatter_checkpoint']))
    native_size = saved['config']['raster']['image_size']
    if tuple(native_size) != (360, 540):
        raise ValueError('Expected native540x360')
    del saved
    args.output.mkdir(parents=True)
    summaries = {}
    for label, size, reference in [('256', 256, args.reference_256),
                                    ('native', (360, 540), args.reference_native)]:
        if label not in args.resolutions:
            continue
        output = args.output/f'inspect_{label}'
        print(f'RENDER_START size={label} samples={args.samples} views={args.views}', flush=True)
        started = time.monotonic()
        inspect_checkpoint(args.checkpoint, args.data, args.priors, output, 'cuda',
            max_views=args.views, render_samples=args.samples, bounce_samples=8,
            split='train', evaluation_seed=2026, median_indirect=True, image_size=size,
            light_sample_pattern=args.light_sample_pattern,
            frame_indices=tuple(args.frames) if args.frames is not None else None)
        elapsed = time.monotonic()-started
        old = json.loads((reference/'metrics.json').read_text())
        new = json.loads((output/'metrics.json').read_text())
        if args.frames is not None:
            old['metrics'] = [r for r in old['metrics'] if r['frame'] in args.frames]
        if [r['frame'] for r in old['metrics']] != [r['frame'] for r in new['metrics']]:
            raise ValueError('Old/new frame selection differs')
        if new['evaluation']['num_light_samples'] != args.samples or new['evaluation']['num_brdf_samples'] != 0:
            raise ValueError('Wrong direct-only sample override')
        compare = args.output/f'compare_{label}'
        compare.mkdir()
        for row in new['metrics']:
            index = row['frame']
            name = f'frame_{index:04d}'
            values = []
            for directory in (reference, output):
                with np.load(directory/'raw_metric_frames'/f'{name}.npz') as cache:
                    values.append({k: cache[k].copy() for k in ('render_rgb', 'target_rgb', 'valid_mask')})
            if not all(np.array_equal(values[0][k], values[1][k]) for k in ('target_rgb', 'valid_mask')):
                raise ValueError('Old/new RGB targets or masks differ')
            images = [rgb(values[0]['target_rgb']), rgb(values[0]['render_rgb']), rgb(values[1]['render_rgb'])]
            labels = ['Input RGB', f"IID {old['evaluation']['num_light_samples']}" if old['evaluation'].get('light_sample_pattern','iid') == 'iid' else f"RQMC {old['evaluation']['num_light_samples']}",
                      f'{args.light_sample_pattern.upper()} {args.samples}']
            compose(images, labels, f'Conference {index:04d}; {label}; same checkpoint / no direct denoiser').save(compare/f'{name}.png')
        summaries[label] = dict(reference={k: float(np.mean([r[k] for r in old['metrics']])) for k in ('psnr','ssim','lpips')},
                               rendered={k: new[f'mean_{k}'] for k in ('psnr','ssim','lpips')},
                               inspect_wall_seconds=elapsed, views=new['views'])
        print(f'RENDER_AND_COMPARE_COMPLETE size={label} seconds={elapsed:.3f}', flush=True)
    if any(sha256_file(path) != hashes[str(path.resolve())] for path in protected):
        raise RuntimeError('Protected checkpoint/reference changed')
    write_json(args.output/'summary.json', dict(training_performed=False, samples=args.samples,
        light_sample_pattern=args.light_sample_pattern, requested_frames=args.frames,
        checkpoint_sha256=hashes[str(args.checkpoint.resolve())], results=summaries,
        protocol='Matched TRAIN reconstruction views; GT-assisted, not held-out/paper NVS',
        direct_denoising=False, indirect_median_radius=1))
    frames = [r['frame'] for r in new['metrics']]
    options = ''.join(f'<option value="{i:04d}">{i:04d}</option>' for i in frames)
    page = '''<!doctype html><html lang="ko"><meta charset="utf-8"><title>128 vs SAMPLES samples</title><style>body{background:#222;color:#eee;font-family:system-ui}header{position:sticky;top:0;background:#222;padding:12px}img{max-width:100%;margin:12px 0}a{color:#acf}button,select{font-size:18px}</style><header><h2>Conference GT seed AEGIR5K:128 vs SAMPLES samples</h2><button onclick="move(-1)">←</button><select id="f">OPTIONS</select><button onclick="move(1)">→</button><p>동일30뷰·체크포인트·평가 seed. 입력 / samples128 / samplesSAMPLES. 직접광 추가 필터 없음.</p><a href="summary.json">점수</a></header><h3>원본540×360</h3><img id="native"><h3>256×256</h3><img id="small"><script>const f=document.getElementById('f');function show(){for(const [id,dir]of[['native','compare_native'],['small','compare_256']])document.getElementById(id).src=dir+'/frame_'+f.value+'.png'}function move(d){f.selectedIndex=(f.selectedIndex+d+f.options.length)%f.options.length;show()}f.onchange=show;show();</script></html>'''
    old_label = f"{old['evaluation'].get('light_sample_pattern','iid').upper()} {old['evaluation']['num_light_samples']}"
    new_label = f'{args.light_sample_pattern.upper()} {args.samples}'
    page = page.replace('128 vs SAMPLES', old_label+' vs '+new_label)
    page = page.replace('입력 / samples128 / samplesSAMPLES', '입력 / '+old_label+' / '+new_label)
    page = page.replace('동일30뷰', '동일'+str(new['views'])+'뷰')
    if '256' not in args.resolutions:
        page = page.replace('<h3>256×256</h3><img id="small">','').replace(
            "[['native','compare_native'],['small','compare_256']]", "[['native','compare_native']]")
    if 'native' not in args.resolutions:
        page = page.replace('<h3>원본540×360</h3><img id="native">','').replace(
            "[['native','compare_native'],['small','compare_256']]", "[['small','compare_256']]")
    (args.output/'index.html').write_text(page.replace('OPTIONS', options).replace('SAMPLES', str(args.samples)))
    print('SAMPLE_RENDER_COMPLETE', args.output, json.dumps(summaries), flush=True)


if __name__ == '__main__':
    main()
