#!/usr/bin/env python3
"""Generate certified metre-scale ZoeDepth from RGB and a GT-free training view."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from aegir.provenance import DEPTH_FORMAT, fingerprint, sha256_file, validate_predicted_depth, write_json

# Masks and observed radiance are allowed. Geometric/material answers are not.
GT_FRAME_KEYS = {
    "depth_file_path",
    "normal_world_file_path",
    "normal_camera_file_path",
    "normal_file_path",
    "albedo_gt_file_path",
    "roughness_gt_file_path",
    "metallic_gt_file_path",
    "object_id_file_path",
}
GT_TOP_KEYS = {
    "emitters",
    "emitter_definitions",
    "emitters_file_path",
    "lighting_gt",
    "ply_file_path",
    "aegir_intrinsics_correction",
    "normal_encoding",
    "normal_space",
    "source_normal_space",
    "normal_camera_convention",
    "source_depth_mode",
    "aegir_depth_provenance",
    "aegir_pipeline_provenance",
}


def input_request(source: Path, metadata: dict) -> dict:
    clean = deepcopy(metadata)
    for key in GT_TOP_KEYS:
        clean.pop(key, None)
    for frame in clean["frames"]:
        for key in GT_FRAME_KEYS:
            frame.pop(key, None)
        # Hash inputs actually used; never read source GT depth or emitter bbox.
        for key in ("file_path", "radiance_hdr_file_path", "mask_path", "emitter_mask_path"):
            if frame.get(key):
                frame[key + "_sha256"] = sha256_file(source / frame[key])
    weights = Path(torch.hub.get_dir()).parent / "checkpoints/ZoeD_M12_N.pt"
    return {
        "generator_sha256": sha256_file(Path(__file__)),
        "weights_sha256": sha256_file(weights) if weights.exists() else None,
        "source": str(source.resolve()),
        "model": "ZoeD_N",
        "model_output_unit": "metre",
        "gt_alignment": False,
        "input_signature": fingerprint(clean),
    }


def metre_z_depth(prediction: torch.Tensor | np.ndarray) -> np.ndarray:
    if torch.is_tensor(prediction):
        prediction = prediction.detach().cpu().numpy()
    depth = np.squeeze(np.asarray(prediction, dtype=np.float32))
    if depth.ndim != 2 or not np.isfinite(depth).all() or not (depth > 0).all():
        raise ValueError("Depth model must return finite positive HxW axial depth")
    # ZoeD_N.infer outputs metres. Unlike DN's exporter, do not divide by .001.
    return depth


def prepare(source: Path, output: Path, device: str = "cuda", model=None) -> Path:
    source, output = source.resolve(), output.resolve()
    if source == output:
        raise ValueError("Source and output must differ")
    metadata = json.loads((source / "transforms.json").read_text())
    request = input_request(source, metadata)
    if (output / "transforms.json").exists():
        stored = json.loads((output / "predicted_depth/manifest.json").read_text())
        if stored.get("request") != request:
            raise ValueError(f"Predicted-depth cache input/model mismatch: {output}; choose a fresh path")
        validate_predicted_depth(output)
        print(f"Verified predicted-depth cache: {output}")
        return output
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Incomplete/unverified prediction cache: {output}; choose a fresh path")
    if model is None:
        model = (
            torch.hub.load("isl-org/ZoeDepth", "ZoeD_N", pretrained=True, trust_repo=True).to(device).eval()
        )
    request = input_request(source, metadata)  # Weights may have been downloaded above.
    output.mkdir(parents=True, exist_ok=True)
    depth_dir = output / "predicted_depth"
    depth_dir.mkdir()
    records = []
    clean = deepcopy(metadata)
    for key in GT_TOP_KEYS:
        clean.pop(key, None)
    for index, frame in enumerate(clean["frames"]):
        for key in GT_FRAME_KEYS:
            frame.pop(key, None)
        for key in (
            "file_path",
            "radiance_hdr_file_path",
            "radiance_exr_file_path",
            "mask_path",
            "emitter_mask_path",
        ):
            if frame.get(key):
                # Absolute paths support any source layout; no directories containing GT are linked.
                frame[key] = str((source / frame[key]).resolve())
        rgb_path = source / metadata["frames"][index]["file_path"]
        rgb = np.array(Image.open(rgb_path).convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(rgb).permute(2, 0, 1)[None].to(device)
        with torch.inference_mode():
            depth = metre_z_depth(model.infer(tensor))
        if depth.shape != rgb.shape[:2]:
            raise ValueError("Depth prediction resolution differs from input RGB")
        # Native 2DGS resolves mono_depth by image stem; prevent ambiguous naming.
        name = Path(metadata["frames"][index]["file_path"]).stem + ".npy"
        if (depth_dir / name).exists():
            raise ValueError(f"Duplicate RGB stem: {name}")
        np.save(depth_dir / name, depth)
        frame["depth_file_path"] = f"predicted_depth/{name}"
        records.append(
            {
                "file_path": frame["file_path"],
                "depth_file_path": frame["depth_file_path"],
                "rgb_sha256": sha256_file(rgb_path),
                "depth_sha256": sha256_file(depth_dir / name),
            }
        )
        print(f"ZoeD_N {index + 1}/{len(clean['frames'])}: median={np.median(depth):.3f} m", flush=True)
    # Preserve filename splits after making frame image paths absolute.
    for key in ("train_filenames", "val_filenames", "test_filenames"):
        if key in clean:
            clean[key] = [str((source / value).resolve()) for value in clean[key]]
    clean.update(aegir_depth_convention="opencv", aegir_depth_mode="z", depth_unit_scale_factor=1.0)
    manifest = {
        "format": DEPTH_FORMAT,
        "origin": "rgb_model",
        "model": "ZoeD_N",
        "unit": "metre",
        "mode": "z",
        "convention": "opencv",
        "gt_alignment": False,
        "request": request,
        "frames": records,
    }
    clean["aegir_depth_provenance"] = {
        key: manifest[key] for key in ("origin", "model", "unit", "mode", "convention", "gt_alignment")
    }
    clean["aegir_depth_provenance"]["manifest"] = "predicted_depth/manifest.json"
    manifest["metadata_signature"] = fingerprint(clean)
    write_json(depth_dir / "manifest.json", manifest)
    write_json(output / "transforms.json", clean)
    validate_predicted_depth(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(prepare(args.source, args.output, args.device))


if __name__ == "__main__":
    main()
