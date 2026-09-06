from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from aegir.conversion_common import (
    atomic_output_dir,
    independent_val_test_split,
    resolve_suffix,
    save_mask,
    validate_c2w,
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _split_source_stems(scene_dir: Path, filename: str) -> set[str]:
    path = scene_dir / filename
    if not path.is_file():
        return set()
    metadata = _load_json(path)
    return {Path(frame["file_path"]).name for frame in metadata.get("frames", [])}


def convert_isaacsim_scene(
    scene_dir: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
    max_frames: int | None = None,
) -> dict[str, Any]:
    scene_dir = Path(scene_dir).resolve()
    if not scene_dir.is_dir():
        raise FileNotFoundError(scene_dir)
    source_meta = _load_json(scene_dir / "transforms.json")
    source_frames = source_meta.get("frames", [])
    if max_frames is not None:
        source_frames = source_frames[: max(0, int(max_frames))]
    if not source_frames:
        raise ValueError(f"no frames in {scene_dir / 'transforms.json'}")

    width = int(source_meta["w"])
    height = int(source_meta["h"])
    fl_x = float(source_meta["fl_x"])
    fl_y = float(source_meta["fl_y"])
    cx = float(source_meta["cx"])
    cy = float(source_meta["cy"])
    train_stems = _split_source_stems(scene_dir, "transforms_train.json")
    eval_stems = _split_source_stems(scene_dir, "transforms_test.json")

    with atomic_output_dir(output_dir, overwrite=overwrite) as (staging, output):
        for name in ("images", "radiance_hdr", "depths", "masks"):
            (staging / name).mkdir()

        frames: list[dict[str, Any]] = []
        train_filenames: list[str] = []
        eval_filenames: list[str] = []
        for output_index, source_frame in enumerate(source_frames):
            source_stem = Path(source_frame["file_path"]).name
            image_source = resolve_suffix(scene_dir / source_frame["file_path"], ".png")
            depth_source = resolve_suffix(scene_dir / source_frame["depth_file_path"], ".npy")
            hdr_key = "hdr_npy_file_path"
            if hdr_key not in source_frame:
                raise KeyError(f"missing {hdr_key} for Isaac frame {source_stem}")
            hdr_source = resolve_suffix(scene_dir / source_frame[hdr_key], ".npy")
            for path in (image_source, depth_source, hdr_source):
                if not path.is_file():
                    raise FileNotFoundError(path)

            name = f"frame_{output_index:04d}"
            image_output = staging / "images" / f"{name}.png"
            depth_output = staging / "depths" / f"{name}.npy"
            hdr_output = staging / "radiance_hdr" / f"{name}.npy"
            mask_output = staging / "masks" / f"{name}.png"

            with Image.open(image_source) as image:
                if image.size != (width, height):
                    raise ValueError(f"unexpected Isaac image size in {image_source}: {image.size}")
            depth = np.load(depth_source, allow_pickle=False).astype(np.float32)
            hdr = np.load(hdr_source, allow_pickle=False).astype(np.float32)
            if depth.shape != (height, width) or hdr.shape != (height, width, 3):
                raise ValueError(
                    f"Isaac modality shape mismatch for {source_stem}: depth={depth.shape}, hdr={hdr.shape}"
                )
            valid = np.isfinite(depth) & (depth > 0.0)
            if not np.any(valid):
                raise ValueError(f"Isaac frame has no valid depth: {depth_source}")
            depth = np.where(valid, depth, 0.0).astype(np.float32)
            hdr = np.where(np.isfinite(hdr), hdr, 0.0).astype(np.float32)
            hdr[~valid] = 0.0

            transform = np.asarray(source_frame["transform_matrix"], dtype=np.float64)
            validate_c2w(transform, label=f"Isaac frame {source_stem}")
            shutil.copy2(image_source, image_output)
            np.save(depth_output, depth, allow_pickle=False)
            np.save(hdr_output, hdr, allow_pickle=False)
            save_mask(mask_output, valid)

            image_rel = image_output.relative_to(staging).as_posix()
            frame = {
                "file_path": image_rel,
                "radiance_hdr_file_path": hdr_output.relative_to(staging).as_posix(),
                "depth_file_path": depth_output.relative_to(staging).as_posix(),
                "mask_path": mask_output.relative_to(staging).as_posix(),
                "transform_matrix": transform.tolist(),
                "fl_x": fl_x,
                "fl_y": fl_y,
                "cx": cx,
                "cy": cy,
                "w": width,
                "h": height,
                "source_frame": source_stem,
            }
            frames.append(frame)
            if source_stem in train_stems:
                train_filenames.append(image_rel)
            if source_stem in eval_stems:
                eval_filenames.append(image_rel)
            if (output_index + 1) % 25 == 0 or output_index + 1 == len(source_frames):
                print(
                    f"[{scene_dir.name}] converted {output_index + 1}/{len(source_frames)} frames", flush=True
                )

        metadata: dict[str, Any] = {
            "camera_model": "OPENCV",
            "fl_x": fl_x,
            "fl_y": fl_y,
            "cx": cx,
            "cy": cy,
            "w": width,
            "h": height,
            "k1": 0.0,
            "k2": 0.0,
            "p1": 0.0,
            "p2": 0.0,
            "aegir_note": "Isaac Sim capture normalized without changing its OpenGL c2w poses",
            "aegir_source_camera_convention": "opengl_c2w",
            "aegir_camera_convention": "opengl_c2w",
            "aegir_depth_convention": "opencv",
            "aegir_depth_mode": "z",
            "source_depth_mode": "z",
            "depth_unit_scale_factor": 1.0,
            "aegir_color": {
                "format": "aegir-color-v1",
                "observation_domain": "linear_hdr",
                "hdr_encoding": "scene_linear_rgb",
                "target_png": {"source": "Isaac Sim source PNG camera response unknown"},
            },
            "source_scene": str(scene_dir),
            "frames": frames,
        }
        if train_filenames and eval_filenames:
            val_filenames, test_filenames, split_metadata = independent_val_test_split(
                train_filenames, eval_filenames
            )
            metadata["train_filenames"] = train_filenames
            metadata["val_filenames"] = val_filenames
            metadata["test_filenames"] = test_filenames
            metadata["aegir_evaluation_split"] = split_metadata
        (staging / "transforms.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    return {
        "scene": scene_dir.name,
        "output_dir": str(output),
        "frames": len(frames),
        "train_frames": len(train_filenames),
        "eval_frames": len(eval_filenames),
        "width": width,
        "height": height,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize an Isaac Sim capture for AEGIR/Nerfstudio.")
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-frames", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = convert_isaacsim_scene(
        args.scene_dir,
        args.output_dir,
        overwrite=args.overwrite,
        max_frames=args.max_frames,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
