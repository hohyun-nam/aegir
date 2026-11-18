"""Make a train-only LDR RGB+SAM3 view without GT geometry or HDR observations."""
import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aegir.provenance import sha256_file, write_json
from aegir.data import select_frame_indices
from scripts.prepare_predicted_depth_view import GT_FRAME_KEYS, GT_TOP_KEYS


def prepare(source, sam_run, output, mask_dir=None, allow_partial_cache=False,
            keep_validation=False, fixed_exposure=None):
    if fixed_exposure is not None and (not np.isfinite(fixed_exposure) or fixed_exposure <= 0):
        raise ValueError('fixed exposure must be finite and positive')
    source, sam_run, output = [Path(p).resolve() for p in (source, sam_run, output)]
    mask_dir = Path(mask_dir).resolve() if mask_dir is not None else sam_run / 'sam_masks'
    meta = json.loads((source / 'transforms.json').read_text())
    manifest = json.loads((sam_run / 'manifest.json').read_text())
    recorded_split = manifest.get('split')
    if recorded_split is None and Path(manifest.get('metadata', '')).name == 'transforms_train.json':
        recorded_split = 'train'  # Legacy server inference manifest.
    if Path(manifest['data']).resolve() != source or recorded_split != 'train':
        raise ValueError('SAM3 run must use this source and the train split')
    frames = meta['frames']
    if any('split' in f for f in frames):
        train = [f for f in frames if f.get('split') == 'train']
    elif meta.get('train_filenames'):
        requested = {str(value) for value in meta['train_filenames']}
        requested_names = {Path(value).name for value in requested}
        train = [
            f for f in frames
            if str(f['file_path']) in requested or Path(str(f['file_path'])).name in requested_names
        ]
    else:
        train = list(frames)
    # Server smoke caches sample noncontiguous views. Keep the source frame
    # array in that case so frame_XXXX material priors retain their indices.
    if allow_partial_cache:
        available = {(source / str(value)).resolve() for value in manifest['selected_images']}
        training = {id(frames[i]) for i in select_frame_indices(meta, frames, 'train')}
        train = [f for f in frames if id(f) in training and (source / f['file_path']).resolve() in available]
        if not train or [(source / f['file_path']).resolve() for f in train] != [
            (source / str(value)).resolve() for value in manifest['selected_images']
        ]:
            raise ValueError('Partial SAM cache must select training frames in original metadata order')
    elif not train or [f['file_path'] for f in train] != manifest['selected_images']:
        raise ValueError('SAM3 run must cover every training image in metadata order')
    if not allow_partial_cache and meta['frames'][:len(train)] != train:
        raise ValueError('Existing prior indices require training frames first')
    clean = deepcopy(meta)
    for key in GT_TOP_KEYS:
        clean.pop(key, None)
    # SAM3 emitter initialization is intentionally an LDR-only contribution.
    # Remove the source HDR declaration as well as every per-frame HDR path so
    # downstream code cannot silently consume HDR radiance.
    clean['aegir_color'] = {
        'format': 'aegir-color-v1',
        'observation_domain': 'ldr',
        'target_png': {
            'tone_mapper': 'reinhard',
            'transfer_function': 'srgb',
            'exposure_policy': 'fixed_scene' if fixed_exposure is not None else 'learned_scene',
            'exposure_scale': float(fixed_exposure) if fixed_exposure is not None else 1.0,
            'source': 'LDR RGB input used by SAM3; HDR deliberately excluded',
        },
    }
    clean['frames'] = []
    records = []
    selected_paths = {f['file_path'] for f in train}
    for original_index, f in enumerate(frames if allow_partial_cache or keep_validation else train):
        f = deepcopy(f)
        stem = Path(f['file_path']).stem
        image = source / f['file_path']
        cached = sam_run / 'inputs' / (stem + '.png')
        mask = mask_dir / (stem + '.png')
        selected = f['file_path'] in selected_paths
        if selected:
            rgb = np.asarray(Image.open(image).convert('RGB'))
            if not np.array_equal(rgb, np.asarray(Image.open(cached).convert('RGB'))):
                raise ValueError(f'SAM input differs: {image}')
            mask_image = np.asarray(Image.open(mask).convert('L'))
            if mask_image.shape != rgb.shape[:2] or not set(np.unique(mask_image)).issubset({0, 255}):
                raise ValueError(f'Invalid SAM mask: {mask}')
        for key in GT_FRAME_KEYS:
            f.pop(key, None)
        f.pop('radiance_hdr_file_path', None)
        f.pop('radiance_exr_file_path', None)
        f.pop('emitter_mask_path', None)
        for key in ('file_path', 'mask_path'):
            if f.get(key):
                f[key] = str((source / f[key]).resolve())
        if selected:
            f['emitter_mask_path'] = str(mask)
        clean['frames'].append(f)
        if selected:
            records.append(dict(image=str(image), mask=str(mask), mask_sha256=sha256_file(mask),
                                rgb_sha256=sha256_file(image), original_frame_index=original_index))
    clean['train_filenames'] = [str((source / f['file_path']).resolve()) for f in train]
    for split in ('val', 'test'):
        clean[f'{split}_filenames'] = [
            str((source / f['file_path']).resolve())
            for f in ([f for f in frames if f.get('split') == split]
                      if any('split' in f for f in frames) else
                      [frames[i] for i in select_frame_indices(meta, frames, split)])
        ] if keep_validation else []
    clean['emitter_mask_source'] = 'sam3'
    clean['sam3_provenance'] = dict(
        run=str(sam_run),
        manifest_sha256=sha256_file(sam_run / 'manifest.json'),
        mask_dir=str(mask_dir),
        filter_manifest_sha256=(
            sha256_file(mask_dir.parent / 'manifest.json')
            if (mask_dir.parent / 'manifest.json').is_file() else None
        ),
        frames=records,
        partial_cache=allow_partial_cache,
        source_frame_count=len(frames),
        selected_frame_count=len(train),
    )
    target = output / 'transforms.json'
    if target.exists():
        if json.loads(target.read_text()) != clean:
            raise ValueError('Training view changed; choose a fresh output')
    else:
        output.mkdir(parents=True, exist_ok=True)
        write_json(target, clean)
    print(f'Verified SAM3 training view: {len(train)} images, {output}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--sam-run', type=Path, required=True)
    p.add_argument('--mask-dir', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--allow-partial-cache', action='store_true',
                   help='Explicit sampled-cache validation; preserve source frame/prior indices')
    p.add_argument('--keep-validation', action='store_true',
                   help='Retain RGB-only validation/test frames; SAM is still train-only')
    p.add_argument('--fixed-exposure', type=float,
                   help='Use a known capture exposure instead of learning exposure')
    a = p.parse_args()
    if a.fixed_exposure is not None and (not np.isfinite(a.fixed_exposure) or a.fixed_exposure <= 0):
        p.error('fixed exposure must be finite and positive')
    prepare(a.source, a.sam_run, a.output, a.mask_dir, a.allow_partial_cache,
            a.keep_validation, a.fixed_exposure)
