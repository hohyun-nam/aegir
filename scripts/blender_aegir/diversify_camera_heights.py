from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import bpy
from mathutils import Vector


CAMERA_COLLECTIONS = {
    "train": "AEGIR_TrainCameras",
    "val": "AEGIR_ValCameras",
    "test": "AEGIR_TestCameras",
}


def script_arguments() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def cameras_for_split(split: str) -> list[bpy.types.Object]:
    collection = bpy.data.collections.get(CAMERA_COLLECTIONS[split])
    if collection is None:
        raise RuntimeError(f"missing camera collection {CAMERA_COLLECTIONS[split]!r}")
    cameras = sorted((obj for obj in collection.objects if obj.type == "CAMERA"), key=lambda x: x.name)
    if not cameras:
        raise RuntimeError(f"no cameras in {collection.name}")
    if any(obj.parent is not None or obj.constraints for obj in cameras):
        raise RuntimeError(f"{collection.name} cameras must be unparented and unconstrained")
    return cameras


def location_key(camera: bpy.types.Object) -> tuple[float, float]:
    location = camera.matrix_basis.translation
    return round(float(location.x), 4), round(float(location.y), 4)


def old_target(camera: bpy.types.Object, depsgraph, fallback_distance: float) -> Vector:
    matrix = camera.matrix_basis
    origin = matrix.translation.copy()
    forward = -(matrix.to_3x3() @ Vector((0.0, 0.0, 1.0))).normalized()
    hit, location, *_ = bpy.context.scene.ray_cast(
        depsgraph,
        origin,
        forward,
        distance=float(camera.data.clip_end),
    )
    return location.copy() if hit else origin + forward * fallback_distance


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Distribute paired AEGIR cameras over metric height bands while preserving view targets."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--levels", type=float, nargs="+", default=[0.95, 1.25, 1.55, 1.80])
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--fallback-target-distance", type=float, default=4.0)
    args = parser.parse_args(script_arguments())

    levels = sorted(set(float(value) for value in args.levels))
    if len(levels) < 2 or levels[0] <= 0.0:
        raise ValueError("--levels needs at least two distinct positive heights")
    if args.fallback_target_distance <= 0.0:
        raise ValueError("--fallback-target-distance must be positive")

    depsgraph = bpy.context.evaluated_depsgraph_get()
    rng = random.Random(args.seed)
    report: dict[str, object] = {
        "levels_m": levels,
        "seed": args.seed,
        "splits": {},
    }

    for split in CAMERA_COLLECTIONS:
        cameras = cameras_for_split(split)
        groups: dict[tuple[float, float], list[bpy.types.Object]] = defaultdict(list)
        for camera in cameras:
            groups[location_key(camera)].append(camera)

        targets = {camera.name: old_target(camera, depsgraph, args.fallback_target_distance) for camera in cameras}
        ordered_groups = sorted(groups.items())
        shuffled_indices = list(range(len(ordered_groups)))
        rng.shuffle(shuffled_indices)
        assigned_level: dict[int, float] = {}
        for assignment_index, group_index in enumerate(shuffled_indices):
            assigned_level[group_index] = levels[assignment_index % len(levels)]

        counts = {str(level): 0 for level in levels}
        records = []
        for group_index, (xy, group) in enumerate(ordered_groups):
            height = assigned_level[group_index]
            counts[str(height)] += 1
            for camera in group:
                target = targets[camera.name]
                camera.location.z = height
                direction = target - camera.location
                if direction.length <= 1.0e-6:
                    raise RuntimeError(f"camera target collapsed after height update: {camera.name}")
                camera.rotation_mode = "QUATERNION"
                camera.rotation_quaternion = direction.to_track_quat("-Z", "Y")
                records.append(
                    {
                        "camera": camera.name,
                        "xy": [float(xy[0]), float(xy[1])],
                        "height_m": height,
                        "target": [float(value) for value in target],
                    }
                )

        report["splits"][split] = {
            "cameras": len(cameras),
            "unique_xy_positions": len(groups),
            "position_height_counts": counts,
            "records": records,
        }

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output))
    report_path = output.with_suffix(".camera_heights.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"AEGIR_CAMERA_HEIGHTS levels={levels}")
    for split, payload in report["splits"].items():
        print(
            f"AEGIR_CAMERA_HEIGHTS split={split} cameras={payload['cameras']} "
            f"positions={payload['unique_xy_positions']} counts={payload['position_height_counts']}"
        )
    print(f"AEGIR_CAMERA_HEIGHTS wrote={output}")
    print(f"AEGIR_CAMERA_HEIGHTS report={report_path}")


if __name__ == "__main__":
    main()
