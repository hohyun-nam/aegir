#!/usr/bin/env python3
"""Import DiffusionRenderer material PNGs as strict AEGIR frame priors."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load_map(path: Path, channels: int) -> torch.Tensor:
    image = Image.open(path)
    if channels == 3:
        array = np.asarray(image.convert("RGB")).copy()
    else:
        array = np.asarray(image.convert("L")).copy()[..., None]
    return torch.from_numpy(array).float().div(255.0).clamp(0.0, 1.0)


def srgb_to_linear(value: torch.Tensor) -> torch.Tensor:
    low = value / 12.92
    high = ((value + 0.055) / 1.055).clamp_min(0.0).pow(2.4)
    return torch.where(value <= 0.04045, low, high)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="prepared RGB input directory")
    parser.add_argument("--result-dir", type=Path, required=True, help="DiffusionRenderer output directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--basecolor-space",
        choices=("model", "srgb"),
        default="model",
        help="keep DiffusionRenderer numeric output, or decode it from sRGB to linear reflectance",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest_path = args.input_dir.resolve() / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "aegir-diffusionrenderer-input-v1":
        raise ValueError(f"unsupported manifest: {manifest_path}")
    result_dir = args.result_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output} already exists; pass --overwrite")
        shutil.rmtree(output)
    staging = output.with_name(output.name + ".tmp")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    try:
        for frame in manifest["frames"]:
            prefix = f"0000.{frame['local_index']:04d}"
            folder = result_dir / frame["chunk_name"]
            paths = {
                name: folder / f"{prefix}.{source}.png"
                for name, source in (
                    ("albedo", "basecolor"),
                    ("roughness", "roughness"),
                    ("metallic", "metallic"),
                )
            }
            missing = [path for path in paths.values() if not path.is_file()]
            if missing:
                raise FileNotFoundError(missing[0])
            albedo = load_map(paths["albedo"], 3)
            if args.basecolor_space == "srgb":
                albedo = srgb_to_linear(albedo)
            payload = {
                "prior_albedo": albedo,
                "prior_roughness": load_map(paths["roughness"], 1),
                "prior_metallic": load_map(paths["metallic"], 1),
                "frame_index": int(frame["frame_index"]),
                "source_file_path": frame["source_file_path"],
                "prior_source": "DiffusionRenderer inverse SVD",
                "basecolor_space": args.basecolor_space,
                "uses_gt_depth": False,
                "uses_gt_normal": False,
            }
            torch.save(payload, staging / f"frame_{frame['frame_index']:04d}.pt")

        output_manifest = {
            "format": "aegir-material-priors-v1",
            "input_manifest": str(manifest_path),
            "diffusionrenderer_result_dir": str(result_dir),
            "frame_count": manifest["frame_count"],
            "frame_indices": [frame["frame_index"] for frame in manifest["frames"]],
            "fields": ["prior_albedo", "prior_roughness", "prior_metallic"],
            "basecolor_space": args.basecolor_space,
            "uses_gt_depth": False,
            "uses_gt_normal": False,
        }
        (staging / "manifest.json").write_text(json.dumps(output_manifest, indent=2) + "\n", encoding="utf-8")
        staging.rename(output)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    print(f"wrote {manifest['frame_count']} strict material priors to {output}")
    print("fields: prior_albedo prior_roughness prior_metallic")
    print("GT depth/normal used: no")


if __name__ == "__main__":
    main()
