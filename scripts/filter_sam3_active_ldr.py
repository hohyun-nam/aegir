"""Reject dark SAM fixture components using LDR RGB only, never GT/HDR."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import label

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aegir.provenance import sha256_file, write_json


def filter_components(rgb, mask, min_median_luma):
    components, count = label(mask)
    luminance = rgb @ np.array([.2126, .7152, .0722])
    filtered = np.zeros_like(mask)
    records = []
    for index in range(1, count + 1):
        selected = components == index
        median = float(np.median(luminance[selected]))
        kept = median >= min_median_luma
        if kept:
            filtered |= selected
        records.append(dict(component=index, pixels=int(selected.sum()),
                            median_ldr_luma=median, kept=kept))
    return filtered, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sam-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--min-median-luma', type=float, default=.6)
    args = parser.parse_args()
    if not 0 <= args.min_median_luma <= 1:
        parser.error('LDR luminance threshold must be in [0,1]')
    if args.output.exists():
        raise FileExistsError(f'Choose a fresh filter output: {args.output}')
    manifest = json.loads((args.sam_run / 'manifest.json').read_text())
    if manifest.get('gt_mask_mode') != 'none' or manifest.get('split') != 'train':
        raise ValueError('This filter requires a GT-free, train-only SAM run')
    (args.output / 'sam_masks').mkdir(parents=True)
    frames = []
    for value in manifest['selected_images']:
        name = Path(value).stem + '.png'
        rgb_path, mask_path = args.sam_run / 'inputs' / name, args.sam_run / 'sam_masks' / name
        rgb = np.asarray(Image.open(rgb_path).convert('RGB'), dtype=np.float32) / 255.
        mask = np.asarray(Image.open(mask_path).convert('L')) > 127
        if mask.shape != rgb.shape[:2]:
            raise ValueError('SAM mask/input shape mismatch')
        filtered, components = filter_components(rgb, mask, args.min_median_luma)
        Image.fromarray(filtered.astype('uint8') * 255).save(args.output / 'sam_masks' / name)
        frames.append(dict(image=value, rgb_sha256=sha256_file(rgb_path),
                           mask_sha256=sha256_file(mask_path), components=components,
                           input_pixels=int(mask.sum()), output_pixels=int(filtered.sum())))
    report = dict(method='sam_connected_component_median_ldr_luma',
                  min_median_luma=args.min_median_luma, uses_hdr=False, uses_gt=False,
                  warning='Appearance heuristic, not proof of physical emission; dim lights may be missed',
                  sam_manifest_sha256=sha256_file(args.sam_run / 'manifest.json'), frames=frames)
    write_json(args.output / 'manifest.json', report)
    print(f'LDR_ACTIVE_FILTER_OK={args.output}', flush=True)


if __name__ == '__main__':
    main()
