from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a raw ZoeDepth-only directory, excluding COLMAP-aligned and GT depth files."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = args.data.resolve()
    raw_dir = data / "mono_depth"
    output = args.output.resolve()
    if not raw_dir.is_dir():
        raise FileNotFoundError(raw_dir)
    with (data / "transforms.json").open("r", encoding="utf-8") as stream:
        transforms = json.load(stream)

    stems = [Path(frame["file_path"]).stem for frame in transforms["frames"]]
    if len(stems) != len(set(stems)):
        raise ValueError("image stems must be unique")
    expected_names = {f"{stem}.npy" for stem in stems}

    output.mkdir(parents=True, exist_ok=True)
    unexpected = {
        path.name for path in output.glob("*.npy") if path.name not in expected_names
    }
    if unexpected:
        raise RuntimeError(f"unexpected depth files in {output}: {sorted(unexpected)}")

    mins = []
    medians = []
    maxes = []
    for stem in stems:
        source = raw_dir / f"{stem}.npy"
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.name.endswith("_aligned.npy"):
            raise RuntimeError(f"aligned depth cannot be used as raw ZoeDepth: {source}")
        depth = np.load(source, mmap_mode="r")
        if depth.ndim not in (2, 3) or depth.size == 0:
            raise ValueError(f"invalid depth shape {depth.shape}: {source}")
        if not np.isfinite(depth).all() or not bool((depth > 0).all()):
            raise ValueError(f"raw ZoeDepth must be finite and positive: {source}")
        mins.append(float(depth.min()))
        medians.append(float(np.median(depth)))
        maxes.append(float(depth.max()))

        destination = output / source.name
        if destination.is_symlink():
            if destination.resolve() != source.resolve():
                raise RuntimeError(f"{destination} points to the wrong source")
            continue
        if destination.exists():
            raise RuntimeError(f"refusing to replace existing {destination}")
        destination.symlink_to(os.path.relpath(source, start=output))

    print(f"model: ZoeD_N (RGB-derived; no GT depth)")
    print(f"frames: {len(stems)}")
    print(
        "raw_stored_depth_range: "
        f"{min(mins):.6f} .. {max(maxes):.6f}; median_of_frame_medians={np.median(medians):.6f}"
    )
    print(f"wrote: {output}")


if __name__ == "__main__":
    main()
