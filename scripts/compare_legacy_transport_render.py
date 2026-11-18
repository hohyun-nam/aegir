"""Frozen-checkpoint comparison: direct+SH versus cosine128/light128/bounce8.

Inference-only overrides. No checkpoint/config/default is written back.
The SH-trained checkpoint is not a model trained with explicit transport.
"""
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

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from aegir.checkpoint import load_checkpoint
from aegir.color import apply_ldr_pipeline, configure_ldr_fallback
from aegir.config import config_from_dict
from aegir.data import MultiViewSceneDataset
from aegir.denoising import edge_aware_vector_median_denoise
from aegir.evaluation import build_lpips_model, masked_lpips, masked_psnr, masked_ssim
from aegir.indirect import DIRECT_SH_MODE, load_indirect_light
from aegir.inspect import _camera_to, render_full_frame
from aegir.irgs_tracer import IRGSVisibilityTracer
from aegir.provenance import sha256_file, validate_sam3_ldr_view
from aegir.render import DeferredRenderer
from aegir.visibility import scene_scale_from_bounds
from scripts.inspect_completed_sample_override import compose, rgb as numpy_rgb


def rgb(value):
    return numpy_rgb(value.detach().cpu().numpy() if torch.is_tensor(value) else value)


def transport_config(base, legacy=False):
    cfg=copy.deepcopy(base)
    cfg.raster.image_size=256
    cfg.render.num_light_samples=128
    cfg.render.light_sample_pattern='iid'
    cfg.render.num_bounce_samples=8
    cfg.render.num_brdf_samples=128 if legacy else 0
    cfg.render.lighting_mode='explicit_transport' if legacy else DIRECT_SH_MODE
    cfg.render.indirect_bounce=bool(legacy)
    return cfg


def median_components(rendered, cfg):
    raw_indirect=rendered['indirect_hdr'].clone()
    surface_mask=(rendered['alpha'][...,0]>=cfg.raster.alpha_threshold)&(rendered['emitter_mask'][...,0]<.5)
    rendered['indirect_hdr']=edge_aware_vector_median_denoise(rendered['indirect_hdr'],rendered['depth'],
        rendered['normals'],surface_mask,radius=1,depth_epsilon=1e-3)
    rendered['hdr']=rendered['direct_hdr']+rendered['indirect_hdr']+rendered['emission_hdr']
    return raw_indirect


@torch.inference_mode()
def main():
    run=ROOT/'outputs/conference_gtseed20k_native_bigpatch_20261008_gtseed_native01/aegir_5k'
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=run/'aegir_5000.pt')
    parser.add_argument('--data',type=Path,default=ROOT/'outputs/server_sam3_ldr_inputs_20261006/conference/sam3_ldr_view')
    parser.add_argument('--reference512',type=Path,default=run/'render_s512/inspect_256')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--frames',nargs='+',type=int,default=[131,4,0,19,34,58,77,97,116,141])
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.frames or len(set(args.frames))!=len(args.frames):
        parser.error('Expected distinct frame indices')
    torch.set_num_threads(4)
    before=sha256_file(args.checkpoint)
    validate_sam3_ldr_view(args.data)
    scene,emitters,payload=load_checkpoint(args.checkpoint,'cuda')
    scene.requires_grad_(False)
    emitters.requires_grad_(False)
    base=config_from_dict(payload['config'])
    if base.render.lighting_mode!=DIRECT_SH_MODE:
        raise ValueError('Expected the current direct+SH checkpoint')
    from aegir.gt_depth_ablation import validate_geometry
    validate_geometry(Path(payload['source_dnsplatter_checkpoint']))
    indirect=load_indirect_light(payload,'cuda')
    indirect.requires_grad_(False)
    ds=MultiViewSceneDataset(args.data,payload['source_dnsplatter_checkpoint'],image_size=256,split='train',max_views=30)
    frames=[next(f for f in ds.frames if f.index==idx) for idx in args.frames]
    policy=configure_ldr_fallback(ds.observation_policy,tone_mapper=base.color.ldr_fallback_tone_mapper,
                                  transfer_function=base.color.ldr_fallback_transfer_function)
    if policy!=payload['dataset']['observation_policy'] or policy['domain']!='ldr':
        raise ValueError('Expected the same LDR observation policy')
    exposure=payload['training_state']['observation']['log_exposure'].exp()
    def display(hdr):
        return apply_ldr_pipeline(hdr,exposure=exposure,tone_mapper=policy['tone_mapper'],transfer_function=policy['transfer_function'])
    reference=json.loads((args.reference512/'metrics.json').read_text())
    ev=reference['evaluation']
    if reference['checkpoint']!=str(args.checkpoint.resolve()) or ev['num_light_samples']!=512 or ev['lighting_mode']!=DIRECT_SH_MODE:
        raise ValueError('Cached reference512 must match this checkpoint/mode')
    if ev['learned_scene_exposure']!=float(exposure):
        raise ValueError('Exposure differs from cached512')
    refhash=sha256_file(args.reference512/'metrics.json')
    cfgs={'direct_sh128':transport_config(base),'legacy128_128_b8':transport_config(base,True)}
    active_ids=tuple(range(emitters.num_emitters))
    scale=scene_scale_from_bounds(scene.bounds(base.raster.bounds_quantile))
    tracer=IRGSVisibilityTracer(scene,base.render,scene_scale=scale,excluded_emitter_ids=active_ids)
    lpips=build_lpips_model('alex','cuda')
    args.output.mkdir(parents=True)
    for folder in ('direct_sh128','legacy128_128_b8','comparison','components','detail_crops'):
        (args.output/folder).mkdir()
    summary=dict(status='running',checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=before,
        training_performed=False,production_changes=False,resolution=[256,256],frames=args.frames,
        geometry_count=scene.gaussian_count,emitters=emitters.num_emitters,active_emitter_ids=list(active_ids),
        seed_rule='2026 + original frame index',render_chunk_size=256,bounce_chunk_size=base.render.bounce_chunk_size,
        gpu=torch.cuda.get_device_name(),architecture=list(torch.cuda.get_device_capability()),
        direct_denoiser='off',indirect_median='3x3, both modes',learned_exposure=float(exposure),
        timing_scope='CUDA-synchronized render_full_frame; excludes I/O, postprocess, metrics and setup',
        warmup='16x16 inference smoke per mode, not a full-frame benchmark warmup',
        caveat='SH-trained model switched to explicit inference, not a legacy-trained checkpoint; TRAIN GT-assisted reconstruction',
        configurations={name:dict(lighting_mode=c.render.lighting_mode,cosine_samples=c.render.num_brdf_samples,
            emitter_samples=c.render.num_light_samples,secondary_samples=c.render.num_bounce_samples if c.render.indirect_bounce else 0,
            indirect_bounce=c.render.indirect_bounce,global_sh=name=='direct_sh128',training_sh_proxy=False) for name,c in cfgs.items()},
        smoke={},results={name:[] for name in cfgs})
    def save():
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    current=dict(mode='',frame=None,calls=0,counts={})
    original_transport,original_visibility=tracer.trace_transport,tracer.visibility
    original_render=DeferredRenderer.render
    def transport(origins,directions,distances,training_proxy):
        if training_proxy or current['mode']!='legacy128_128_b8' or directions.shape[-2]!=128:
            raise RuntimeError('Unexpected primary transport routing')
        current['counts']['primary_transport_calls']=current['counts'].get('primary_transport_calls',0)+1
        current['counts']['primary_transport_rays']=current['counts'].get('primary_transport_rays',0)+directions.numel()//3
        return original_transport(origins,directions,distances,training_proxy)
    def visibility(origins,directions,distances):
        expected=8 if current['mode']=='legacy128_128_b8' else 128
        if directions.shape[-2]!=expected:
            raise RuntimeError(f'Unexpected visibility sample count {directions.shape}')
        key='secondary' if expected==8 else 'primary_direct'
        current['counts'][key+'_visibility_calls']=current['counts'].get(key+'_visibility_calls',0)+1
        current['counts'][key+'_visibility_rays']=current['counts'].get(key+'_visibility_rays',0)+directions.numel()//3
        return original_visibility(origins,directions,distances)
    def progress_render(renderer,*a,**kw):
        result=original_render(renderer,*a,**kw)
        current['calls']+=1
        if current['frame'] is not None and current['calls']%32==0:
            print(f"PROGRESS mode={current['mode']} frame={current['frame']} chunks={current['calls']}/256",flush=True)
        return result
    tracer.trace_transport=transport
    tracer.visibility=visibility
    DeferredRenderer.render=progress_render
    try:
        # Verify genuine primary cosine transport and secondary8 on GPU before full frames.
        smoke=MultiViewSceneDataset(args.data,payload['source_dnsplatter_checkpoint'],image_size=16,split='train',max_views=30)
        smoke_frame=next(f for f in smoke.frames if f.index==131)
        for name,cfg in cfgs.items():
            current.update(mode=name,frame=None,calls=0,counts={})
            image=render_full_frame(scene,emitters,_camera_to(smoke_frame.camera,'cuda'),cfg,
                visibility_tracer=tracer,generator=torch.Generator(device='cuda').manual_seed(2157),transport_mode='inference',
                excluded_emitter_ids=active_ids,indirect_light=indirect if name=='direct_sh128' else None)
            if not all(bool(torch.isfinite(v).all()) for v in image.values()):
                raise RuntimeError('Nonfinite GPU smoke output')
            if name=='legacy128_128_b8' and not current['counts'].get('secondary_visibility_rays',0):
                raise RuntimeError('GPU smoke did not execute secondary bounce')
            summary['smoke'][name]=dict(finite=True,**current['counts'])
            print('GPU_SMOKE_COMPLETE',name,current['counts'],flush=True)
            del image
        del smoke,smoke_frame
        save()
        for frame in frames:
            panels={}
            for name,cfg in cfgs.items():
                current.update(mode=name,frame=frame.index,calls=0,counts={})
                camera=_camera_to(frame.camera,'cuda')
                torch.cuda.synchronize()
                start=time.perf_counter()
                rendered=render_full_frame(scene,emitters,camera,cfg,visibility_tracer=tracer,
                    generator=torch.Generator(device='cuda').manual_seed(2026+frame.index),transport_mode='inference',
                    excluded_emitter_ids=active_ids,indirect_light=indirect if name=='direct_sh128' else None)
                torch.cuda.synchronize()
                elapsed=time.perf_counter()-start
                if not all(bool(torch.isfinite(v).all()) for v in rendered.values()):
                    raise RuntimeError('Nonfinite full frame')
                raw_indirect=median_components(rendered,cfg)
                pred=display(rendered['hdr'])
                row=dict(frame=frame.index,seed=2026+frame.index,render_seconds=elapsed,
                    psnr=float(masked_psnr(pred,frame.target,frame.mask)),ssim=float(masked_ssim(pred,frame.target,frame.mask)),
                    lpips=float(masked_lpips(pred,frame.target,frame.mask,lpips,'cuda')),counts=dict(current['counts']))
                if name=='legacy128_128_b8' and not row['counts'].get('secondary_visibility_rays',0):
                    raise RuntimeError('Explicit mode did not execute secondary transport')
                summary['results'][name].append(row)
                directory=args.output/name
                prefix=f'frame_{frame.index:04d}'
                images={key:rgb(display(rendered[key+'_hdr'])) for key in ('direct','indirect','emission')}
                images['total']=rgb(pred)
                images['indirect_raw']=rgb(display(raw_indirect))
                for key,im in images.items():
                    im.save(directory/f'{prefix}_{key}.png')
                arrays={key:rendered[key].numpy() for key in ('hdr','direct_hdr','indirect_hdr','emission_hdr','alpha','depth','normals','emitter_mask')}
                arrays.update(indirect_hdr_before_median=raw_indirect.numpy(),render_rgb=pred.numpy(),target_rgb=frame.target.numpy(),valid_mask=frame.mask.numpy())
                np.savez_compressed(directory/f'{prefix}.npz',**arrays)
                panels[name]=images
                print('FRAME_COMPLETE',name,json.dumps(row),flush=True)
                save()
                del rendered,raw_indirect,pred
            prefix=f'frame_{frame.index:04d}'
            with np.load(args.reference512/'raw_metric_frames'/f'{prefix}.npz') as ref:
                if not np.array_equal(ref['target_rgb'],frame.target.numpy()) or not np.array_equal(ref['valid_mask'],frame.mask.numpy()):
                    raise RuntimeError('Cached512 target/mask differs')
                refimage=rgb(ref['render_rgb'])
            labels=['Input RGB','Direct128 + learned SH','Direct512 + learned SH (cached)','Cos128 + light128 + bounce8']
            panel=compose([rgb(frame.target),panels['direct_sh128']['total'],refimage,panels['legacy128_128_b8']['total']],
                labels,f'Conference {frame.index:04d}: 256x256 / fixed AEGIR5K / no direct filter')
            panel.save(args.output/'comparison'/f'{prefix}.png')
            rows=[]
            for name,label in [('direct_sh128','Current direct+SH'),('legacy128_128_b8','Legacy explicit transport')]:
                rows.append(compose([panels[name][k] for k in ('total','direct','indirect','indirect_raw')],
                    ['Total RGB','Direct RGB (unfiltered)','Indirect RGB (median3x3)','Indirect RGB (raw)'],
                    f'{frame.index:04d} {label}: SAME scene exposure for all components'))
            grid=Image.new('RGB',(rows[0].width,sum(im.height for im in rows)),'white')
            for offset,im in zip((0,rows[0].height),rows):
                grid.paste(im,(0,offset))
            grid.save(args.output/'components'/f'{prefix}.png')
            if frame.index in (4,131):
                box=(0,0,200,145) if frame.index==131 else (0,85,230,256)
                tiles=[im.crop(box).resize(((box[2]-box[0])*3,(box[3]-box[1])*3),resample=Image.Resampling.NEAREST)
                       for im in (rgb(frame.target),panels['direct_sh128']['total'],refimage,panels['legacy128_128_b8']['total'])]
                compose(tiles,labels,f'{frame.index:04d}: 3x nearest crop / no extra filter').save(args.output/'detail_crops'/f'{prefix}.png')
            del panels
    finally:
        tracer.trace_transport=original_transport
        tracer.visibility=original_visibility
        DeferredRenderer.render=original_render
    if sha256_file(args.checkpoint)!=before or sha256_file(args.reference512/'metrics.json')!=refhash:
        raise RuntimeError('Protected checkpoint/reference changed')
    summary['mean_results']={name:{key:statistics.mean(row[key] for row in rows) for key in ('psnr','ssim','lpips','render_seconds')}
                             for name,rows in summary['results'].items()}
    summary['status']='complete'
    summary['checkpoint_unchanged']=True
    save()
    options=''.join(f'<option value="{idx:04d}">{idx:04d}</option>' for idx in args.frames)
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Conference legacy transport</title><style>body{background:#222;color:#eee;font-family:system-ui}header{position:sticky;top:0;background:#222;padding:10px}img{max-width:100%}a{color:#acf}</style><header><h2>Conference: current SH vs legacy128/128/bounce8</h2><select id="f">OPTIONS</select><p>256×256. 같은 SH-trained AEGIR5K. 재학습 없음. 직접광 필터 없음 / 간접광만 median3×3.</p><p>SH를 실제 secondary transport로 대체한 진단. 기존 방식으로 학습한 모델의 품질 비교는 아님.</p><a href="summary.json">설정·점수·시간</a></header><h3>Input / direct+SH128 / direct+SH512 cached / legacy128+128+b8</h3><img id="total"><h3>Total / direct / indirect filtered / indirect raw; top=current, bottom=legacy</h3><img id="components"><script>const f=document.getElementById('f');function show(){document.getElementById('total').src='comparison/frame_'+f.value+'.png';document.getElementById('components').src='components/frame_'+f.value+'.png'}f.onchange=show;show();</script></html>'''
    (args.output/'index.html').write_text(page.replace('OPTIONS',options),encoding='utf-8')
    print('LEGACY_TRANSPORT_COMPARISON_COMPLETE',json.dumps(summary['mean_results']),flush=True)


if __name__=='__main__':
    main()
