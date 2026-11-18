#!/usr/bin/env python3
"""Guard training/restart against changed inputs, settings, code and stale GT artifacts."""

import argparse
import json
import os
from pathlib import Path

from aegir.provenance import (
    artifact_request,
    dataset_signature,
    publish_artifact,
    sha256_file,
    validate_emitter_inclusive_geometry,
    validate_native_view,
    validate_practical_geometry,
    validate_predicted_depth,
    validate_sam3_ldr_view,
    verify_cache,
    write_json,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("2dgs", "aegir"), required=True)
    p.add_argument("--action", choices=("begin", "complete"), required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--geometry", type=Path)
    p.add_argument("--priors", type=Path)
    args = p.parse_args()
    if args.stage == "2dgs":
        names = [
            "ITERATIONS",
            "DEPTH_LOSS_TYPE",
            "LAMBDA_DEPTH",
            "LAMBDA_INVALID_OPACITY",
            "INVALID_OPACITY_PROTECTION_RADIUS",
            "LAMBDA_DIST",
            "DENSIFY_UNTIL_ITER",
            "DENSIFY_GRAD_THRESHOLD",
            "ALLOW_UNVERIFIED_DEPTH",
        ]
        if os.getenv("ALLOW_UNVERIFIED_DEPTH", "0") != "1":
            view_provenance = validate_native_view(args.data)
            meta = json.loads((args.data / "transforms.json").read_text())
            if any(
                any(
                    key in frame
                    for key in (
                        "depth_file_path",
                        "normal_world_file_path",
                        "normal_camera_file_path",
                        "albedo_gt_file_path",
                        "roughness_gt_file_path",
                        "metallic_gt_file_path",
                        "radiance_hdr_file_path",
                        "radiance_exr_file_path",
                    )
                )
                for frame in meta["frames"]
            ):
                raise ValueError("Native 2DGS input contains forbidden GT/supervision paths")
    else:
        names = [
            "STEPS",
            "EMITTER_INITIALIZATION",
            "ASSOCIATE_EMITTER_GAUSSIANS",
            "EMITTER_GAUSSIAN_OFF_ALBEDO",
            "EMITTER_PERSISTENCE_RADIUS_FRACTION",
            "EMITTER_REPRESENTATION",
            "GEOMETRY_LR",
            "OPTIMIZE_SCENE_MEANS",
            "OPTIMIZE_SCENE_SCALES",
            "OPTIMIZE_SCENE_ROTATIONS",
            "OPTIMIZE_SCENE_OPACITIES",
            "CONSTRAIN_SCENE_SCALES",
            "SCENE_SCALE_MIN_RATIO",
            "SCENE_SCALE_MAX_RATIO",
            "SCENE_CULLING",
            "BOUNCE_SAMPLES",
            "RASTER_SIZE",
            "ALBEDO_INITIALIZATION",
            "NEUTRAL_ALBEDO",
            "ALBEDO_LR",
            "SH_FADE_START",
            "SH_FADE_END",
            "TRAINING_TRANSPORT_MODE",
            "LIGHT_WHITE",
            "LIGHT_WHITE_RAMP_END",
            "SIX_AXIS_EMISSION",
            "EMITTER_AXIS_LOBE_SHARPNESS",
            "OPTIMIZE_EMITTER_CENTERS",
            "OPTIMIZE_EMITTER_ROTATIONS",
            "OPTIMIZE_EMITTER_SCALES",
            "CONSTRAIN_EMITTER_SCALES",
            "EMITTER_SCALE_MIN_RATIO",
            "EMITTER_SCALE_MAX_RATIO",
            "EMITTER_ADAPTATION",
        ]
        initialization = os.getenv("EMITTER_INITIALIZATION", "hdr_multiview")
        if initialization == "sam3_multiview":
            validate_sam3_ldr_view(args.data)
        if initialization in {
            "hdr_multiview",
            "hdr_predicted_depth",
            "hdr_gbuffer",
            "sam3_mask_depth",
            "sam3_multiview",
        }:
            if initialization not in {"hdr_multiview", "sam3_multiview"}:
                validate_predicted_depth(args.data)
            associate_emitters = os.getenv("ASSOCIATE_EMITTER_GAUSSIANS", "1") == "1"
            if initialization == "hdr_gbuffer" or associate_emitters:
                validate_emitter_inclusive_geometry(args.geometry)
            else:
                validate_practical_geometry(args.geometry)
    root = Path(__file__).resolve().parents[1]
    code = {str(path.relative_to(root)): sha256_file(path) for path in sorted((root / "aegir").glob("*.py"))}
    for name in (
        "check_training_provenance.py",
        "run_2dgs_depth_pearson_10k.sh",
        "run_2dgs_checkpoint_ablation_10k20k30k.sh",
        "run_conference_frozen_origin015_bounce8_train_inspect.sh",
    ):
        path = root / "scripts" / name
        code[name] = sha256_file(path)
    if args.stage == "2dgs":
        external = Path(os.getenv("TWO_DGS_ROOT", str(root.parent / "external/dn-splatter-2dgs")))
        for path in [
            external / "train.py",
            external / "scene/dataset_readers.py",
            root / "scripts/export_2dgs_geometry_checkpoint.py",
            root / "scripts/patch_2dgs_emitter_exclusion.py",
        ]:
            code[str(path)] = sha256_file(path)
    inputs = {"data": dataset_signature(args.data), "code": code}
    if args.stage == "2dgs" and os.getenv("ALLOW_UNVERIFIED_DEPTH", "0") != "1":
        inputs["view_provenance"] = view_provenance
    if args.geometry:
        inputs["geometry_sha256"] = sha256_file(args.geometry)
    if args.priors:
        inputs["priors"] = {
            str(path.relative_to(args.priors)): sha256_file(path)
            for path in sorted(args.priors.rglob("*"))
            if path.is_file()
        }
    request = artifact_request(args.stage, {key: os.getenv(key) for key in names}, inputs)
    manifest = args.output.with_suffix(args.output.suffix + ".provenance.json")
    pending = args.output.with_suffix(args.output.suffix + ".pending.json")
    if args.action == "complete":
        if not pending.exists() or json.loads(pending.read_text()) != request:
            raise ValueError("Training inputs/settings changed while running")
        publish_artifact(manifest, request, args.output)
        pending.unlink()
        return
    if verify_cache(manifest, request, args.output):
        print("1")
        return
    if pending.exists():
        if json.loads(pending.read_text()) != request:
            raise ValueError("Interrupted training has different inputs/settings; choose a fresh output")
    elif list(args.output.parent.glob(args.output.stem + ".step*.pt")):
        raise ValueError("Unverified intermediate checkpoint; choose a fresh output")
    else:
        write_json(pending, request)
    print("0")


if __name__ == "__main__":
    main()
