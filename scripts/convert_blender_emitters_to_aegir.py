#!/usr/bin/env python3
"""Convert Blender emissive-mesh metadata to AEGIR planar emitter state."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from aegir.data import (
    infer_dataparser_transform_path,
    load_dataparser_transform,
    saved_pose_to_model_transform,
)
from aegir.initialization import _fit_planar_bbox
from aegir.math_utils import quat_to_matrix, safe_normalize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert enabled Blender emissive meshes in emitters.json into "
            "AEGIR bounded planar-ellipse emitter parameters."
        )
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dnsplatter-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--aegir-checkpoint",
        type=Path,
        help=(
            "Use the training scene and initial emitter state in this checkpoint "
            "to calibrate Blender emission strength to AEGIR radiance units."
        ),
    )
    parser.add_argument(
        "--emitter-name",
        action="append",
        default=[],
        help="Convert only this emitter name; repeat for multiple emitters.",
    )
    parser.add_argument(
        "--include-disabled",
        action="store_true",
        help="Also convert emitters disabled in the captured lighting condition.",
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-pt", type=Path)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def declared_radiance(emitter: dict[str, Any]) -> torch.Tensor:
    condition = emitter.get("condition", {})
    color = torch.as_tensor(condition.get("color", [1.0, 1.0, 1.0]), dtype=torch.float32)
    strength = float(condition.get("strength", 1.0))
    return color * strength


def radiance_calibration(
    aegir_checkpoint: Path | None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if aegir_checkpoint is None:
        return torch.ones(3), {
            "method": "identity_no_reference_checkpoint",
            "rgb_scale": [1.0, 1.0, 1.0],
        }

    checkpoint = torch.load(aegir_checkpoint, map_location="cpu", weights_only=False)
    initial = checkpoint.get("initial_emitter_state", {})
    colors = initial.get("colors")
    if not torch.is_tensor(colors) or colors.ndim != 2 or colors.shape[-1] != 3:
        raise ValueError(f"{aegir_checkpoint} has no valid initial_emitter_state.colors")
    dataset_root_value = checkpoint.get("dataset", {}).get("nerfstudio_dir")
    if not dataset_root_value:
        raise ValueError(f"{aegir_checkpoint} has no dataset.nerfstudio_dir")
    reference_path = Path(dataset_root_value) / "emitters.json"
    reference_payload = read_json(reference_path)
    reference_emitters = [
        emitter
        for emitter in reference_payload.get("emitters", [])
        if emitter.get("condition", {}).get("enabled", True)
    ]
    if not reference_emitters:
        raise ValueError(f"{reference_path} has no enabled emitters")

    reference_declared = torch.stack([declared_radiance(value) for value in reference_emitters])
    declared_center = reference_declared.median(dim=0).values.clamp_min(1.0e-8)
    initial_center = colors.detach().float().median(dim=0).values
    scale = initial_center / declared_center
    return scale, {
        "method": "training_checkpoint_initial_radiance",
        "rgb_scale": scale.tolist(),
        "reference_aegir_checkpoint": str(aegir_checkpoint.resolve()),
        "reference_emitters_json": str(reference_path.resolve()),
        "reference_declared_radiance": declared_center.tolist(),
        "reference_initial_radiance": initial_center.tolist(),
        "uses_target_relight_images": False,
    }


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    dnsplatter_checkpoint = args.dnsplatter_checkpoint.resolve()
    emitter_payload = read_json(dataset / "emitters.json")
    metadata = read_json(dataset / "transforms.json")

    requested_names = set(args.emitter_name)
    source_emitters = []
    for emitter in emitter_payload.get("emitters", []):
        if requested_names and emitter.get("name") not in requested_names:
            continue
        if not args.include_disabled and not emitter.get("condition", {}).get("enabled", True):
            continue
        if emitter.get("world_bbox_corners") is None:
            continue
        source_emitters.append(emitter)
    if requested_names:
        found = {value.get("name") for value in source_emitters}
        missing = requested_names - found
        if missing:
            raise ValueError(f"requested emitters not found or disabled: {sorted(missing)}")
    if not source_emitters:
        raise ValueError(f"no emitters selected from {dataset / 'emitters.json'}")

    dataparser = load_dataparser_transform(infer_dataparser_transform_path(dnsplatter_checkpoint))
    saved_to_model = saved_pose_to_model_transform(dataparser, metadata)
    matrix, model_scale = saved_to_model
    rotation, translation = matrix[:, :3], matrix[:, 3]
    color_scale, calibration = radiance_calibration(
        args.aegir_checkpoint.resolve() if args.aegir_checkpoint else None
    )

    records: list[dict[str, Any]] = []
    centers, scales, quaternions, colors = [], [], [], []
    for emitter in source_emitters:
        world = torch.as_tensor(emitter["world_bbox_corners"], dtype=torch.float32)
        if world.ndim != 2 or world.shape[-1] != 3:
            raise ValueError(f"invalid bbox for emitter {emitter.get('name')!r}")
        model = (world @ rotation.T + translation) * float(model_scale)
        center, ellipse_scale, quaternion = _fit_planar_bbox(model)
        quaternion = safe_normalize(quaternion, dim=0)
        axes = quat_to_matrix(quaternion[None])[0]
        color = declared_radiance(emitter) * color_scale
        area = math.pi * float(ellipse_scale.prod())
        records.append(
            {
                "name": emitter.get("name"),
                "emitter_id": emitter.get("emitter_id"),
                "representation": "planar_ellipse_2d",
                "center_model": center.tolist(),
                "semi_axes_model": ellipse_scale.tolist(),
                "quaternion_wxyz": quaternion.tolist(),
                "local_x_axis_model": axes[:, 0].tolist(),
                "local_y_axis_model": axes[:, 1].tolist(),
                "local_z_axis_model": axes[:, 2].tolist(),
                "area_model_squared": area,
                "radiance_rgb": color.tolist(),
                "axis_gain_order": ["+X", "-X", "+Y", "-Y", "+Z", "-Z"],
                "axis_gains": [1.0] * 6,
                "spreads": [32.0] * 3,
                "falloff_power": 1.0,
                "source_condition": emitter.get("condition", {}),
            }
        )
        centers.append(center)
        scales.append(ellipse_scale)
        quaternions.append(quaternion)
        colors.append(color)

    n = len(records)
    state = {
        "centers": torch.stack(centers),
        "quaternions": torch.stack(quaternions),
        "scales": torch.stack(scales),
        "colors": torch.stack(colors),
        "axis_gains": torch.ones((n, 6), dtype=torch.float32),
        "spreads": torch.full((n, 3), 32.0, dtype=torch.float32),
        "falloff_powers": torch.ones((n, 1), dtype=torch.float32),
    }
    output_json = (args.output_json or dataset / "aegir_emitter_override.json").resolve()
    output_pt = (args.output_pt or output_json.with_suffix(".pt")).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_pt.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "format": "aegir-runtime-emitter-v1",
        "source_dataset": str(dataset),
        "source_condition": emitter_payload.get("condition"),
        "dnsplatter_checkpoint": str(dnsplatter_checkpoint),
        "coordinate_system": "AEGIR model coordinates",
        "saved_to_model_transform": matrix.tolist(),
        "saved_to_model_scale": float(model_scale),
        "radiance_calibration": calibration,
        "emitter_count": n,
        "emitters": records,
        "tensor_state": str(output_pt),
    }
    output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    torch.save(
        {
            "format": result["format"],
            "source_dataset": str(dataset),
            "source_condition": emitter_payload.get("condition"),
            "emitter_names": [record["name"] for record in records],
            "emitter_ids": [record["emitter_id"] for record in records],
            "emitter_state": state,
            "radiance_calibration": calibration,
        },
        output_pt,
    )
    print(f"converted {n} emitter(s)")
    print(f"json: {output_json}")
    print(f"tensor state: {output_pt}")


if __name__ == "__main__":
    main()
