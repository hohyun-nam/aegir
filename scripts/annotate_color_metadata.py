"""Add explicit AEGIR color-domain metadata to existing converted scenes."""

from __future__ import annotations

import argparse
import json
import os
import stat
import tempfile
from pathlib import Path


def infer_metadata(path: Path, payload: dict) -> dict:
    frames = payload.get("frames", [])
    if not frames:
        raise ValueError(f"no frames in {path}")
    hdr_flags = ["radiance_hdr_file_path" in frame for frame in frames]
    if any(hdr_flags) and not all(hdr_flags):
        raise ValueError(f"mixed HDR/LDR frames in {path}")
    if all(hdr_flags):
        scene_name = path.parent.name.lower()
        if "hypersim" in scene_name:
            png_source = (
                "generated HyperSim 90th-percentile gamma-2.2 display image; "
                "legacy conversion did not retain per-frame exposure"
            )
        elif "fipt_synthetic" in scene_name:
            png_source = "FIPT source PNG or generated display image; frame response not retained"
        elif "fipt_real" in scene_name:
            png_source = "FIPT-Real source PNG camera response unknown"
        elif "isaacsim" in scene_name:
            png_source = "Isaac Sim source PNG camera response unknown"
        else:
            png_source = "legacy PNG camera response unknown"
        return {
            "format": "aegir-color-v1",
            "observation_domain": "linear_hdr",
            "hdr_encoding": "scene_linear_rgb",
            "target_png": {"source": png_source},
        }
    return {
        "format": "aegir-color-v1",
        "observation_domain": "ldr",
        "target_png": {
            "tone_mapper": "reinhard",
            "transfer_function": "srgb",
            "exposure_policy": "learned_scene",
            "exposure_scale": 1.0,
            "source": "explicit fallback: legacy RGB-only data has no camera-response metadata",
        },
    }


def annotate(path: Path, *, overwrite: bool) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "aegir_color" in payload and not overwrite:
        return "skipped"
    payload["aegir_color"] = infer_metadata(path, payload)
    original_stat = path.stat()
    descriptor, temporary_name = tempfile.mkstemp(prefix=".transforms-color-", dir=path.parent)
    try:
        os.fchmod(descriptor, stat.S_IMODE(original_stat.st_mode))
        try:
            os.fchown(descriptor, original_stat.st_uid, original_stat.st_gid)
        except PermissionError:
            pass
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
        raise
    return payload["aegir_color"]["observation_domain"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="scene directory or parent containing scenes")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    paths = (
        [root / "transforms.json"]
        if (root / "transforms.json").is_file()
        else sorted(root.glob("*/transforms.json"))
    )
    if not paths:
        raise FileNotFoundError(f"no transforms.json below {root}")
    counts = {"linear_hdr": 0, "ldr": 0, "skipped": 0}
    for path in paths:
        result = annotate(path, overwrite=args.overwrite)
        counts[result] += 1
        print(f"{result}\t{path}")
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
