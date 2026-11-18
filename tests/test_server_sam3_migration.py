import json

import numpy as np
import pytest
import torch
from PIL import Image

from aegir.data import MultiViewSceneDataset
from aegir.provenance import validate_sam3_ldr_view
from scripts.prepare_sam3_training_view import prepare


def sampled_view(tmp_path):
    source, cache, output = [tmp_path / name for name in ('source', 'cache', 'view')]
    source.mkdir()
    (cache / 'inputs').mkdir(parents=True)
    (cache / 'sam_masks').mkdir()
    frames = []
    for index in range(4):
        name = f'frame_{index:04d}.png'
        image = Image.fromarray(np.full((8, 8, 3), 80 + index, np.uint8))
        image.save(source / name)
        frames.append(dict(file_path=name, transform_matrix=np.eye(4).tolist(),
                           depth_file_path='FORBIDDEN_GT.npy',
                           radiance_hdr_file_path='FORBIDDEN_HDR.npy',
                           emitter_mask_path='FORBIDDEN_GT.png'))
        if index in (0, 2):
            image.save(cache / 'inputs' / name)
            Image.fromarray(np.full((8, 8), 255, np.uint8)).save(cache / 'sam_masks' / name)
    meta = dict(fl_x=8, fl_y=8, cx=4, cy=4, frames=frames,
                train_filenames=[frame['file_path'] for frame in frames[:3]],
                val_filenames=[frames[3]['file_path']], emitters=['FORBIDDEN'])
    (source / 'transforms.json').write_text(json.dumps(meta))
    (cache / 'manifest.json').write_text(json.dumps(dict(
        data=str(source), metadata=str(source / 'transforms_train.json'),
        selected_images=[frames[i]['file_path'] for i in (0, 2)])))
    prepare(source, cache, output, allow_partial_cache=True)
    return source, cache, output


def test_sampled_sam_cache_keeps_original_material_prior_indices(tmp_path):
    _, _, output = sampled_view(tmp_path)
    meta = validate_sam3_ldr_view(output)
    assert len(meta['frames']) == 4
    assert [record['original_frame_index'] for record in meta['sam3_provenance']['frames']] == [0, 2]
    checkpoint = tmp_path / 'geometry/nerfstudio_models/step.pt'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    (checkpoint.parent.parent / 'dataparser_transforms.json').write_text(
        json.dumps(dict(transform=np.eye(4)[:3].tolist(), scale=1)))
    priors = tmp_path / 'priors'
    priors.mkdir()
    for index in (0, 2):
        torch.save(dict(prior_albedo=torch.full((8, 8, 3), index / 4),
                        prior_roughness=torch.full((8, 8, 1), .6),
                        prior_metallic=torch.zeros(8, 8, 1),
                        prior_albedo_space='linear_reflectance',
                        radiance_hdr=torch.full((8, 8, 3), 999)),
                   priors / f'frame_{index:04d}.pt')
    dataset = MultiViewSceneDataset(output, checkpoint, priors, image_size=8)
    assert dataset.selected_indices == (0, 2)
    assert dataset.frames[1].priors['albedo'].mean() == .5
    assert all(frame.metric_depth is None and frame.radiance_hdr is None for frame in dataset.frames)
    assert dataset.emitter_definitions == []


@pytest.mark.parametrize('key', ['depth_file_path', 'radiance_hdr_file_path', 'radiance_exr_file_path'])
def test_sam_surface_guard_rejects_forbidden_paths_before_loading(tmp_path, key):
    _, _, output = sampled_view(tmp_path)
    path = output / 'transforms.json'
    meta = json.loads(path.read_text())
    meta['frames'][0][key] = 'must_not_be_read.npy'
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='forbidden HDR/GT'):
        validate_sam3_ldr_view(output)


def test_sam_surface_guard_checks_mask_hash(tmp_path):
    _, cache, output = sampled_view(tmp_path)
    Image.fromarray(np.zeros((8, 8), np.uint8)).save(cache / 'sam_masks/frame_0000.png')
    with pytest.raises(ValueError, match='input content changed'):
        validate_sam3_ldr_view(output)


def test_partial_cache_requires_explicit_choice(tmp_path):
    source, cache, _ = sampled_view(tmp_path)
    with pytest.raises(ValueError, match='every training image'):
        prepare(source, cache, tmp_path / 'strict_view')
