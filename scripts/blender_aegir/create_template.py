"""Create a small editable Blender scene that follows the AEGIR capture contract.

Run inside Blender, for example::

    blender -b --python create_template.py -- --output room_template.blend
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def script_arguments() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def material(
    name: str,
    base_color: tuple[float, float, float, float],
    *,
    roughness: float = 0.55,
    metallic: float = 0.0,
    emission_color: tuple[float, float, float, float] | None = None,
    emission_strength: float = 0.0,
) -> bpy.types.Material:
    value = bpy.data.materials.new(name)
    value.use_nodes = True
    principled = value.node_tree.nodes.get("Principled BSDF")
    principled.inputs["Base Color"].default_value = base_color
    principled.inputs["Roughness"].default_value = roughness
    principled.inputs["Metallic"].default_value = metallic
    if emission_color is not None:
        principled.inputs["Emission Color"].default_value = emission_color
        principled.inputs["Emission Strength"].default_value = emission_strength
    return value


def cube(
    name: str,
    location: tuple[float, float, float],
    dimensions: tuple[float, float, float],
    value: bpy.types.Material,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(location=location)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = dimensions
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(value)
    return obj


def collection(name: str) -> bpy.types.Collection:
    value = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(value)
    return value


def move_to_collection(obj: bpy.types.Object, destination: bpy.types.Collection) -> None:
    for current in tuple(obj.users_collection):
        current.objects.unlink(obj)
    destination.objects.link(obj)


def look_at(camera: bpy.types.Object, target: Vector) -> None:
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()


def add_cameras(
    destination: bpy.types.Collection,
    prefix: str,
    count: int,
    *,
    phase: float,
) -> None:
    golden = math.pi * (3.0 - math.sqrt(5.0))
    for index in range(count):
        angle = phase + golden * index
        radius = 2.15 + 0.55 * (0.5 + 0.5 * math.sin(index * 1.71 + phase))
        location = Vector(
            (
                radius * math.cos(angle),
                0.72 * radius * math.sin(angle),
                1.15 + 0.65 * (0.5 + 0.5 * math.sin(index * 0.83 + phase)),
            )
        )
        target = Vector(
            (
                0.55 * math.sin(index * 0.47 + phase),
                0.40 * math.cos(index * 0.61 + phase),
                0.85 + 0.35 * math.sin(index * 0.37),
            )
        )
        camera_data = bpy.data.cameras.new(f"{prefix}_{index:04d}_data")
        camera_data.lens = 28.0
        camera_data.sensor_width = 36.0
        camera_data.clip_start = 0.05
        camera_data.clip_end = 100.0
        camera = bpy.data.objects.new(f"{prefix}_{index:04d}", camera_data)
        destination.objects.link(camera)
        camera.location = location
        look_at(camera, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-cameras", type=int, default=120)
    parser.add_argument("--val-cameras", type=int, default=20)
    parser.add_argument("--test-cameras", type=int, default=40)
    args = parser.parse_args(script_arguments())

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.name = "AEGIR_Room"
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.render.image_settings.color_mode = "RGBA"
    scene.world = bpy.data.worlds.new("AEGIR_World")
    scene.world.color = (0.005, 0.005, 0.005)
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    scene["aegir_scene_format"] = "aegir-blender-v1"
    scene["aegir_metric_units"] = True

    geometry = collection("AEGIR_Geometry")
    emitters = collection("AEGIR_Emitters")
    train_cameras = collection("AEGIR_TrainCameras")
    val_cameras = collection("AEGIR_ValCameras")
    test_cameras = collection("AEGIR_TestCameras")

    white = material("Wall_Material", (0.72, 0.72, 0.70, 1.0), roughness=0.72)
    floor_material = material("Floor_Material", (0.16, 0.20, 0.24, 1.0), roughness=0.42)
    wood = material("Wood_Material", (0.28, 0.075, 0.025, 1.0), roughness=0.38)
    red = material("Red_Material", (0.55, 0.035, 0.025, 1.0), roughness=0.32)
    metal = material("Metal_Material", (0.38, 0.42, 0.48, 1.0), roughness=0.20, metallic=0.85)

    room_objects = [
        cube("Floor", (0.0, 0.0, -0.05), (7.0, 5.4, 0.1), floor_material),
        cube("BackWall", (0.0, 2.75, 1.5), (7.0, 0.1, 3.0), white),
        cube("FrontWall", (0.0, -2.75, 1.5), (7.0, 0.1, 3.0), white),
        cube("LeftWall", (-3.55, 0.0, 1.5), (0.1, 5.4, 3.0), white),
        cube("RightWall", (3.55, 0.0, 1.5), (0.1, 5.4, 3.0), white),
        cube("Ceiling", (0.0, 0.0, 3.05), (7.0, 5.4, 0.1), white),
        cube("TableTop", (0.0, 0.15, 0.78), (2.4, 1.25, 0.14), wood),
        cube("DisplayBox", (-0.55, 0.1, 1.12), (0.58, 0.58, 0.58), red),
        cube("MetalPedestal", (0.72, 0.12, 1.05), (0.45, 0.45, 0.45), metal),
    ]
    for obj in room_objects:
        move_to_collection(obj, geometry)

    ceiling_material = material(
        "CeilingEmitter_Material",
        (0.8, 0.8, 0.8, 1.0),
        roughness=0.35,
        emission_color=(1.0, 0.78, 0.58, 1.0),
        emission_strength=18.0,
    )
    stand_material = material(
        "StandEmitter_Material",
        (0.8, 0.8, 0.8, 1.0),
        roughness=0.35,
        emission_color=(0.55, 0.72, 1.0, 1.0),
        emission_strength=10.0,
    )
    ceiling = cube("CeilingEmitter", (-1.15, 0.55, 2.72), (1.35, 0.55, 0.05), ceiling_material)
    stand = cube("StandEmitter", (1.65, -0.25, 1.35), (0.34, 0.34, 0.06), stand_material)
    stand.rotation_euler = (0.0, math.radians(58.0), math.radians(18.0))
    ceiling["aegir_emitter_id"] = 1
    stand["aegir_emitter_id"] = 2
    move_to_collection(ceiling, emitters)
    move_to_collection(stand, emitters)

    add_cameras(train_cameras, "train", args.train_cameras, phase=0.0)
    add_cameras(val_cameras, "val", args.val_cameras, phase=0.37)
    add_cameras(test_cameras, "test", args.test_cameras, phase=0.91)
    scene.camera = train_cameras.objects[0]

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output))

    lighting = {
        "train": {
            "CeilingEmitter": {"strength": 18.0, "color": [1.0, 0.78, 0.58]},
            "StandEmitter": {"strength": 10.0, "color": [0.55, 0.72, 1.0]},
        },
        "ceiling_only": {
            "CeilingEmitter": {"strength": 18.0, "color": [1.0, 0.78, 0.58]},
            "StandEmitter": {"strength": 0.0, "color": [0.55, 0.72, 1.0]},
        },
        "stand_only": {
            "CeilingEmitter": {"strength": 0.0, "color": [1.0, 0.78, 0.58]},
            "StandEmitter": {"strength": 10.0, "color": [0.55, 0.72, 1.0]},
        },
        "cool_relight": {
            "CeilingEmitter": {"strength": 13.0, "color": [0.58, 0.72, 1.0]},
            "StandEmitter": {"strength": 16.0, "color": [1.0, 0.42, 0.18]},
        },
    }
    lighting_path = output.with_name(f"{output.stem}_lighting.json")
    lighting_path.write_text(json.dumps(lighting, indent=2) + "\n", encoding="utf-8")
    print(f"AEGIR_TEMPLATE={output}")
    print(f"AEGIR_LIGHTING={lighting_path}")


if __name__ == "__main__":
    main()
