"""Explicit GT-depth diagnostic branch. Never accepted as practical GT-free geometry."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .provenance import dataset_signature, sha256_file, validate_practical_seed, write_json

FORMAT = 'aegir-gt-depth-ablation-v1'


def validate_gt_seed(seed: Path) -> dict:
    """Private oracle-only guard; never relax the practical seed validator."""
    stored = json.loads(seed.with_suffix(seed.suffix + '.provenance.json').read_text())
    request = stored.get('request', {})
    settings = request.get('settings', {})
    if (request.get('kind') != 'depth_seed' or settings.get('allow_gt_depth') is not True
            or settings.get('exclude_emitter_masks') is not False
            or stored.get('output_sha256') != sha256_file(seed)):
        raise ValueError('Unverified/non-emitter-inclusive GT depth seed')
    signature = request.get('inputs', {}).get('source', {})
    source = Path(signature['root'])
    view = validate_view(source)
    if view.get('seed_mode') != 'source_only' or signature != dataset_signature(source):
        raise ValueError('GT seed must come from a verified clean GT-depth source view')
    return stored


def prepare_view(source: Path, seed: Path | None, output: Path, *, allow_gt_seed=False) -> dict:
    source, output = source.resolve(), output.resolve()
    seed = seed.resolve() if seed is not None else None
    seed_mode = 'source_only' if seed is None else ('gt_depth' if allow_gt_seed else 'rgb_triangulated')
    seed_provenance = None
    if seed is not None:
        seed_provenance = validate_gt_seed(seed) if allow_gt_seed else validate_practical_seed(seed)
        request = seed_provenance['request']
        if not allow_gt_seed and (request['kind'] != 'fixed_pose_triangulated_seed'
                or request['settings'].get('exclude_emitter_masks') is not False):
            raise ValueError('GT ablation requires emitter-inclusive RGB triangulation unless GT seed explicitly enabled')
    meta = json.loads((source / 'transforms.json').read_text())
    if meta.get('aegir_depth_mode') != 'z' or meta.get('aegir_depth_convention') != 'opencv' or meta.get('depth_unit_scale_factor') != 1:
        raise ValueError('Reference depth must be metre-scale OpenCV Z-depth')
    if meta.get('applied_scale', 1) != 1 or meta.get('applied_transform') not in (None, [[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.]]):
        raise ValueError('This metric diagnostic expects identity saved world coordinates')
    source_hash = sha256_file(source / 'transforms.json')
    seed_hash = sha256_file(seed) if seed is not None else None
    if output.exists():
        stored = validate_view(output)
        if (stored['source_metadata_sha256'] != source_hash or stored['seed_sha256'] != seed_hash
                or stored.get('seed_mode', 'rgb_triangulated') != seed_mode):
            raise ValueError('GT ablation source changed; choose a fresh run')
        return stored
    frames, stats = [], []
    for index, original in enumerate(meta['frames']):
        image_path, depth_path = source / original['file_path'], source / original['depth_file_path']
        with Image.open(image_path) as image:
            width, height = image.size
        depth = np.load(depth_path, allow_pickle=False)
        if depth.shape != (height, width):
            raise ValueError(f'RGB/depth size mismatch: {depth_path}')
        valid = np.isfinite(depth) & (depth > .1)
        if valid.mean() < .1:
            raise ValueError(f'Too little valid reference depth: {depth_path}')
        stats.append(dict(index=index, valid_fraction=float(valid.mean()),
                          min_m=float(depth[valid].min()), median_m=float(np.median(depth[valid])),
                          max_m=float(depth[valid].max())))
        frame = {key: original[key] for key in ('file_path', 'depth_file_path', 'mask_path',
                 'transform_matrix', 'fl_x', 'fl_y', 'cx', 'cy', 'w', 'h') if key in original}
        for key in ('file_path', 'depth_file_path', 'mask_path'):
            if key in frame and (Path(frame[key]).is_absolute() or '..' in Path(frame[key]).parts):
                raise ValueError('Prepared native input paths must be relative')
        frames.append(frame)
    metadata = {key: meta[key] for key in ('camera_model', 'fl_x', 'fl_y', 'cx', 'cy', 'w', 'h',
                'k1', 'k2', 'p1', 'p2', 'train_filenames', 'val_filenames', 'test_filenames') if key in meta}
    metadata.update(frames=frames, depth_unit_scale_factor=1.,
                    aegir_depth_mode='z', aegir_depth_convention='opencv',
                    aegir_depth_provenance={'origin': 'dataset_gt_reference', 'gt_alignment': True,
                                            'unit': 'metre', 'mode': 'z', 'convention': 'opencv'},
                    aegir_experiment={'kind': 'gt_depth_ablation', 'gt_free': False})
    output.mkdir(parents=True)
    folders = {Path(frame[key]).parts[0] for frame in frames
               for key in ('file_path', 'depth_file_path', 'mask_path') if key in frame}
    for folder in folders:
        if folder == 'point_cloud.ply' or not (source / folder).is_dir():
            raise ValueError(f'Expected source directory: {folder}')
        (output / folder).symlink_to(source / folder, target_is_directory=True)
    if seed is not None:
        metadata['ply_file_path'] = 'point_cloud.ply'
        (output / 'point_cloud.ply').symlink_to(seed)
        (output / 'point_cloud.ply.provenance.json').symlink_to(seed.with_suffix('.ply.provenance.json'))
    write_json(output / 'transforms.json', metadata)
    manifest = dict(format=FORMAT, gt_free=False, source=str(source), source_metadata_sha256=source_hash,
                    seed_sha256=seed_hash, seed_provenance=seed_provenance, seed_mode=seed_mode,
                    inputs=dataset_signature(output), depth_statistics=stats,
                    rgb_mask_independent_of_depth=True, exclude_emitters=False,
                    gt_normals_used=False, hdr_used=False, gt_emitter_used=False)
    write_json(output / 'gt_depth_ablation.json', manifest)
    return manifest


def validate_view(root: Path) -> dict:
    stored = json.loads((root / 'gt_depth_ablation.json').read_text())
    if stored.get('format') != FORMAT or stored.get('gt_free') is not False or stored.get('inputs') != dataset_signature(root):
        raise ValueError('Changed/unverified GT-depth ablation view')
    mode = stored.get('seed_mode', 'rgb_triangulated')
    if mode == 'source_only':
        if stored.get('seed_sha256') is not None or stored.get('seed_provenance') is not None:
            raise ValueError('Unexpected seed in GT depth source view')
    elif mode in ('gt_depth', 'rgb_triangulated'):
        seed = root / 'point_cloud.ply'
        proof = validate_gt_seed(seed) if mode == 'gt_depth' else validate_practical_seed(seed)
        if stored['seed_sha256'] != sha256_file(seed) or stored['seed_provenance'] != proof:
            raise ValueError('GT diagnostic seed provenance changed')
    else:
        raise ValueError('Unknown GT diagnostic seed mode')
    return stored


def validate_geometry(path: Path) -> dict:
    stored = json.loads(path.with_suffix(path.suffix + '.gt_depth_ablation.json').read_text())
    if (stored.get('format') != FORMAT or stored.get('gt_free') is not False
            or stored.get('output_sha256') != sha256_file(path)
            or stored.get('rgb_mask_independent_of_depth') is not True
            or stored.get('depth_loss') != 'l1'):
        raise ValueError('Unverified/changed GT-depth ablation geometry')
    if stored.get('view_manifest') != validate_view(Path(stored['view'])):
        raise ValueError('GT ablation geometry input content changed')
    if stored['view_manifest'].get('seed_mode') == 'source_only':
        raise ValueError('A geometry checkpoint requires an actual certified seed')
    return stored


def independent_rgb_masks(source: str, cameras) -> None:
    """Attach RGB-only masks, independent of missing/invalid depth values."""
    root = Path(source)
    meta = json.loads((root / 'transforms.json').read_text())
    frames = {Path(frame['file_path']).stem: frame for frame in meta['frames']}
    for camera in cameras:
        frame = frames[camera.image_name]
        shape = (camera.image_height, camera.image_width)
        if frame.get('mask_path'):
            with Image.open(root / frame['mask_path']) as mask:
                mask = mask.convert('L').resize((shape[1], shape[0]), Image.Resampling.NEAREST)
                value = torch.from_numpy(np.asarray(mask).copy() > 127).float()[None]
        else:
            value = torch.ones((1, *shape))
        camera.rgb_supervision_mask = value.to(camera.original_image.device)
