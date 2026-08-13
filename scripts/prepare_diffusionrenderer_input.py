#!/usr/bin/env python3
"""Prepare frame-identity-preserving RGB chunks for DiffusionRenderer."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def _normalized(path: str) -> str:
    value = Path(path.replace("\\", "/")).as_posix()
    return value.removeprefix("./")


def select_indices(metadata: dict, split: str) -> list[int]:
    frames = metadata.get("frames", [])
    if split == "all":
        return list(range(len(frames)))
    key = f"{split}_filenames"
    if split == "test" and key not in metadata and "val_filenames" in metadata:
        key = "val_filenames"
    if key not in metadata:
        if split == "train" and not any(
            name in metadata for name in ("train_filenames", "val_filenames", "test_filenames")
        ):
            return list(range(len(frames)))
        raise ValueError(f"transforms.json has no {key}")
    requested = {_normalized(value) for value in metadata[key]}
    selected = [index for index, frame in enumerate(frames) if _normalized(frame["file_path"]) in requested]
    if len(selected) != len(requested):
        raise ValueError(f"{key} does not align with transforms.json frames")
    return selected


def publish_image(source: Path, destination: Path, mask: Path | None) -> float:
    if mask is None:
        try:
            destination.hardlink_to(source)
        except OSError:
            shutil.copy2(source, destination)
        return 1.0
    image = Image.open(source).convert("RGB")
    valid = np.asarray(Image.open(mask).convert("L")) > 0
    rgb = np.asarray(image).copy()
    if valid.shape != rgb.shape[:2]:
        raise ValueError(f"mask {mask} shape {valid.shape} does not match image {source} {rgb.shape[:2]}")
    rgb[~valid] = 0
    Image.fromarray(rgb).save(destination)
    return float(valid.mean())


def prepare(
    nerfstudio_dir: Path,
    output_dir: Path,
    chunk_size: int,
    split: str,
    apply_masks: bool,
    overwrite: bool,
) -> dict:
    root = nerfstudio_dir.resolve()
    transforms = root / "transforms.json"
    metadata = json.loads(transforms.read_text(encoding="utf-8"))
    indices = select_indices(metadata, split)
    if not indices:
        raise ValueError(f"no frames selected for split {split!r}")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    output = output_dir.resolve()
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"{output} already exists; pass --overwrite")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    records = []
    frames = metadata["frames"]
    for sequence_index, frame_index in enumerate(indices):
        frame = frames[frame_index]
        source = (root / frame["file_path"]).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        chunk_index, local_index = divmod(sequence_index, chunk_size)
        chunk_name = f"chunk_{chunk_index:04d}"
        chunk_dir = output / chunk_name
        chunk_dir.mkdir(exist_ok=True)
        destination = chunk_dir / f"frame_{frame_index:04d}.png"
        mask = None
        if apply_masks and frame.get("mask_path"):
            mask = (root / frame["mask_path"]).resolve()
            if not mask.is_file():
                raise FileNotFoundError(mask)
        valid_fraction = publish_image(source, destination, mask)
        records.append(
            {
                "frame_index": frame_index,
                "source_file_path": _normalized(frame["file_path"]),
                "input_file_path": destination.relative_to(output).as_posix(),
                "chunk_name": chunk_name,
                "chunk_index": chunk_index,
                "local_index": local_index,
                "mask_applied": mask is not None,
                "valid_fraction": valid_fraction,
            }
        )

    manifest = {
        "format": "aegir-diffusionrenderer-input-v1",
        "nerfstudio_dir": str(root),
        "transforms_json": str(transforms),
        "split": split,
        "chunk_size": chunk_size,
        "frame_count": len(records),
        "material_passes": ["basecolor", "roughness", "metallic"],
        "uses_gt_depth": False,
        "uses_gt_normal": False,
        "frames": records,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nerfstudio-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=24)
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="all")
    parser.add_argument(
        "--apply-masks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="black out invalid dataset pixels before inverse rendering",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = prepare(
        args.nerfstudio_dir,
        args.output_dir,
        args.chunk_size,
        args.split,
        args.apply_masks,
        args.overwrite,
    )
    print(
        f"prepared {manifest['frame_count']} RGB frames in "
        f"{(manifest['frame_count'] + manifest['chunk_size'] - 1) // manifest['chunk_size']} chunks"
    )
    print(f"manifest: {args.output_dir.resolve() / 'manifest.json'}")


if __name__ == "__main__":
    main()
