#!/usr/bin/env python3
"""Run the official DiffusionRenderer inverse model for AEGIR material priors."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

MATERIAL_PASSES = ("basecolor", "roughness", "metallic")


def expected_outputs(input_manifest: dict, output_dir: Path) -> list[Path]:
    paths = []
    for frame in input_manifest["frames"]:
        prefix = f"0000.{frame['local_index']:04d}"
        for material_pass in MATERIAL_PASSES:
            paths.append(output_dir / frame["chunk_name"] / f"{prefix}.{material_pass}.png")
    return paths


def build_command(args: argparse.Namespace) -> list[str]:
    repo = args.repo_dir.resolve()
    inference = repo / "inference_svd_rgbx.py"
    config = repo / "configs" / "rgbx_inference.yaml"
    weights = (
        args.model_weights.resolve()
        if args.model_weights
        else repo / "checkpoints" / "diffusion_renderer-inverse-svd"
    )
    for required in (inference, config, weights / "unet" / "diffusion_pytorch_model.safetensors"):
        if not required.exists():
            raise FileNotFoundError(required)
    return [
        sys.executable,
        str(inference),
        "--config",
        str(config),
        f"inference_model_weights={weights}",
        f"inference_input_dir={args.input_dir.resolve()}",
        f"inference_save_dir={args.output_dir.resolve()}",
        f"inference_n_frames={args.frames}",
        f"inference_n_steps={args.steps}",
        f"inference_res=[{args.height},{args.width}]",
        f"decode_chunk_size={args.decode_chunk_size}",
        "model_passes=['basecolor','roughness','metallic']",
        "chunk_mode=first",
        "overlap_n_frames=0",
        "image_group_mode=folder",
        "save_image=true",
        "save_video=false",
        f"seed={args.seed}",
        "cond_mode=skip",
        "weight_dtype=fp16",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-weights", type=Path, default=None)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--decode-chunk-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest_path = args.input_dir.resolve() / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "aegir-diffusionrenderer-input-v1":
        raise ValueError(f"unsupported input manifest: {manifest_path}")
    if args.frames != int(manifest["chunk_size"]):
        raise ValueError(f"--frames {args.frames} must equal prepared chunk_size {manifest['chunk_size']}")
    if args.height % 64 or args.width % 64:
        raise ValueError("DiffusionRenderer height and width must be divisible by 64")

    command = build_command(args)
    print("command:")
    print(" ".join(command))
    if args.dry_run:
        return
    args.output_dir.resolve().mkdir(parents=True, exist_ok=True)
    subprocess.run(command, cwd=args.repo_dir.resolve(), check=True)

    missing = [path for path in expected_outputs(manifest, args.output_dir.resolve()) if not path.is_file()]
    if missing:
        raise RuntimeError(
            f"DiffusionRenderer finished but {len(missing)} output(s) are missing, e.g. {missing[0]}"
        )
    run_manifest = {
        "format": "aegir-diffusionrenderer-output-v1",
        "input_manifest": str(manifest_path),
        "frame_count": manifest["frame_count"],
        "model_weights": next(
            value.split("=", 1)[1] for value in command if value.startswith("inference_model_weights=")
        ),
        "material_passes": list(MATERIAL_PASSES),
        "inference_frames": args.frames,
        "inference_steps": args.steps,
        "inference_resolution": [args.height, args.width],
        "decode_chunk_size": args.decode_chunk_size,
        "seed": args.seed,
        "uses_gt_depth": False,
        "uses_gt_normal": False,
    }
    path = args.output_dir.resolve() / "run_manifest.json"
    path.write_text(json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8")
    print(f"validated {manifest['frame_count']} frames x {len(MATERIAL_PASSES)} material passes")
    print(f"run manifest: {path}")


if __name__ == "__main__":
    main()
