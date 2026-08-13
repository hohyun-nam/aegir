from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


TOP_LEVEL_SUPERVISION_KEYS = {
    "aegir_depth_convention",
    "aegir_depth_mode",
    "depth_unit_scale_factor",
    "normal_camera_convention",
    "normal_encoding",
    "normal_space",
    "source_depth_mode",
    "source_normal_space",
}

FRAME_SUPERVISION_KEYS = {
    "albedo_gt_file_path",
    "depth_file_path",
    "normal_camera_file_path",
    "normal_file_path",
    "radiance_hdr_file_path",
    "roughness_gt_file_path",
}


def ensure_relative_symlink(source_item: Path, output_dir: Path, name: str) -> None:
    if not source_item.exists():
        return
    destination = output_dir / name
    expected = source_item.resolve()
    if destination.is_symlink():
        if destination.resolve() != expected:
            raise RuntimeError(f"{destination} points to the wrong target")
        return
    if destination.exists():
        raise RuntimeError(f"refusing to replace existing {destination}")
    relative_target = os.path.relpath(source_item, start=output_dir)
    destination.symlink_to(relative_target, target_is_directory=source_item.is_dir())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a Nerfstudio view containing RGB, fixed poses, masks, and COLMAP seed only."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mono-depth-source",
        type=Path,
        default=None,
        help="Optional RGB-derived mono-depth directory to expose as output/mono_depth.",
    )
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    if source == output:
        raise ValueError("--source and --output must be different")
    transforms_path = source / "transforms.json"
    if not transforms_path.is_file():
        raise FileNotFoundError(transforms_path)
    if not (source / "colmap" / "sparse_pc.ply").is_file():
        raise FileNotFoundError(source / "colmap" / "sparse_pc.ply")

    with transforms_path.open("r", encoding="utf-8") as stream:
        transforms = json.load(stream)
    for key in TOP_LEVEL_SUPERVISION_KEYS:
        transforms.pop(key, None)
    for frame in transforms["frames"]:
        for key in FRAME_SUPERVISION_KEYS:
            frame.pop(key, None)

    forbidden = TOP_LEVEL_SUPERVISION_KEYS.intersection(transforms)
    forbidden.update(
        key
        for frame in transforms["frames"]
        for key in FRAME_SUPERVISION_KEYS.intersection(frame)
    )
    if forbidden:
        raise RuntimeError(f"supervision metadata survived sanitization: {sorted(forbidden)}")

    output.mkdir(parents=True, exist_ok=True)
    for name in ("images", "masks", "colmap"):
        ensure_relative_symlink(source / name, output, name)
    if args.mono_depth_source is not None:
        mono_depth_source = args.mono_depth_source.resolve()
        ensure_relative_symlink(mono_depth_source, output, "mono_depth")
    with (output / "transforms.json").open("w", encoding="utf-8") as stream:
        json.dump(transforms, stream, indent=2)
        stream.write("\n")
    print(f"frames: {len(transforms['frames'])}")
    print(f"wrote: {output / 'transforms.json'}")


if __name__ == "__main__":
    main()
