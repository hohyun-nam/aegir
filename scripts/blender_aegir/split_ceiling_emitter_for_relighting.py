"""Split the six loose ceiling panels into independently controlled emitters."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector

REFERENCE_CENTERS = (
    (-0.5784845352, -0.6657509208, 2.2737267017),
    (1.9258534908, -0.6724909544, 2.2677693367),
    (-0.5425602794, 1.2579993010, 2.2859625816),
    (1.9116386175, 1.2956045866, 2.2793312073),
    (-0.5178974271, -2.5934870243, 2.2828898430),
    (1.9189769030, -2.6151566505, 2.2729516029),
)
DEFAULT_RGB = (1.0, 0.78, 0.52)
DEFAULT_STRENGTH = 50.0


def script_arguments() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def world_center(obj: bpy.types.Object) -> Vector:
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    return sum(corners, Vector()) / len(corners)


def match_panels(objects: list[bpy.types.Object]) -> dict[int, bpy.types.Object]:
    centers = [world_center(obj) for obj in objects]
    references = [Vector(value) for value in REFERENCE_CENTERS]
    best_cost = float("inf")
    best_permutation = None
    for permutation in itertools.permutations(range(len(objects))):
        cost = sum((centers[permutation[index]] - references[index]).length_squared for index in range(6))
        if cost < best_cost:
            best_cost = cost
            best_permutation = permutation
    if best_permutation is None:
        raise RuntimeError("failed to match ceiling panels")
    result = {index: objects[object_index] for index, object_index in enumerate(best_permutation)}
    errors = {
        index: (world_center(obj) - references[index]).length for index, obj in result.items()
    }
    if max(errors.values()) > 0.25:
        raise RuntimeError(f"panel-to-learned-emitter match exceeds 0.25 m: {errors}")
    return result


def condition(active_indices: set[int]) -> dict[str, dict]:
    return {
        f"CeilingPanel_{index}": {
            "enabled": index in active_indices,
            "rgb": list(DEFAULT_RGB),
            "strength": DEFAULT_STRENGTH,
        }
        for index in range(6)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-blend", type=Path, required=True)
    parser.add_argument("--output-lighting", type=Path, required=True)
    parser.add_argument("--source-object", default="CeilingEmitter")
    args = parser.parse_args(script_arguments())

    output_blend = args.output_blend.resolve()
    output_lighting = args.output_lighting.resolve()
    for path in (output_blend, output_lighting):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    source = bpy.data.objects.get(args.source_object)
    if source is None or source.type != "MESH":
        raise RuntimeError(f"missing mesh object {args.source_object!r}")
    if source.library is not None:
        raise RuntimeError(f"source emitter is linked and cannot be edited: {source.library.filepath}")
    if len(source.data.materials) != 1 or source.data.materials[0] is None:
        raise RuntimeError("source emitter must use exactly one material")

    bpy.ops.object.mode_set(mode="OBJECT") if bpy.context.object and bpy.context.object.mode != "OBJECT" else None
    bpy.ops.object.select_all(action="DESELECT")
    source.select_set(True)
    bpy.context.view_layer.objects.active = source
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.separate(type="LOOSE")
    bpy.ops.object.mode_set(mode="OBJECT")
    panels = [obj for obj in bpy.context.selected_objects if obj.type == "MESH"]
    if len(panels) != 6:
        raise RuntimeError(f"expected six loose ceiling panels, found {len(panels)}")
    bpy.context.view_layer.update()

    matched = match_panels(panels)
    emitter_records = []
    for index, obj in matched.items():
        obj.name = f"CeilingPanel_{index}"
        obj.data.name = f"CeilingPanel_{index}_Mesh"
        material = obj.data.materials[0].copy()
        material.name = f"AEGIR_CeilingPanel_{index}_Material"
        obj.data.materials[0] = material
        obj["aegir_emitter_id"] = index + 1
        obj["aegir_learned_emitter_index"] = index
        center = world_center(obj)
        error = (center - Vector(REFERENCE_CENTERS[index])).length
        emitter_records.append(
            {
                "name": obj.name,
                "aegir_emitter_id": index + 1,
                "aegir_learned_emitter_index": index,
                "material": material.name,
                "world_center": list(center),
                "learned_center": list(REFERENCE_CENTERS[index]),
                "center_distance_m": error,
                "default_rgb": list(DEFAULT_RGB),
                "default_strength": DEFAULT_STRENGTH,
            }
        )

    all_indices = set(range(6))
    conditions = {
        "train": condition(all_indices),
        "all_on": condition(all_indices),
        "left_only": condition({0, 2, 4}),
        "right_only": condition({1, 3, 5}),
        "checker_a": condition({1, 2, 4}),
        "checker_b": condition({0, 3, 5}),
        "all_off": condition(set()),
    }
    for index in range(6):
        conditions[f"panel_{index}_only"] = condition({index})
        conditions[f"without_panel_{index}"] = condition(all_indices - {index})

    lighting = {
        "schema_version": 1,
        "scene": output_blend.name,
        "emitters": emitter_records,
        "conditions": conditions,
        "notes": [
            "Created by splitting CeilingEmitter into its six existing loose mesh components.",
            "Panel indices minimize distance to the learned SH-DC/white-0 AEGIR emitter centers.",
            "All panels retain the original RGB and strength; conditions only toggle enabled state.",
            "Other Blender LIGHT objects remain render-disabled with zero energy and World strength is zero.",
        ],
    }
    output_lighting.write_text(json.dumps(lighting, indent=2) + "\n", encoding="utf-8")
    bpy.ops.wm.save_as_mainfile(filepath=str(output_blend), compress=True)

    print(f"RELIGHT_BLEND={output_blend}")
    print(f"RELIGHT_LIGHTING={output_lighting}")
    for record in emitter_records:
        print(
            "PANEL_MATCH "
            f"index={record['aegir_learned_emitter_index']} "
            f"center={record['world_center']} "
            f"distance_m={record['center_distance_m']:.6f}"
        )


if __name__ == "__main__":
    main()
