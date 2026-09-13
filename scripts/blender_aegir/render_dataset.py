"""Render a Blender scene directly into the AEGIR Nerfstudio/HDR layout."""

from __future__ import annotations

import argparse
import binascii
import json
import struct
import sys
import time
import zlib
from pathlib import Path

import bpy
import numpy as np
import OpenImageIO as oiio
from mathutils import Vector


CAMERA_COLLECTIONS = {
    "train": "AEGIR_TrainCameras",
    "val": "AEGIR_ValCameras",
    "test": "AEGIR_TestCameras",
}
EMITTER_COLLECTION = "AEGIR_Emitters"


def script_arguments() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def png_chunk(name: bytes, payload: bytes) -> bytes:
    body = name + payload
    return struct.pack(">I", len(payload)) + body + struct.pack(">I", binascii.crc32(body) & 0xFFFFFFFF)


def write_png(path: Path, image: np.ndarray) -> None:
    image = np.ascontiguousarray(image)
    if image.dtype != np.uint8 or image.ndim not in (2, 3):
        raise ValueError("PNG input must be uint8 HxW or HxWxC")
    if image.ndim == 2:
        color_type = 0
    elif image.shape[2] == 3:
        color_type = 2
    elif image.shape[2] == 4:
        color_type = 6
    else:
        raise ValueError(f"unsupported PNG channels: {image.shape}")
    height, width = image.shape[:2]
    scanlines = b"".join(b"\x00" + image[row].tobytes() for row in range(height))
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0))
        + png_chunk(b"IDAT", zlib.compress(scanlines, level=6))
        + png_chunk(b"IEND", b"")
    )
    path.write_bytes(payload)


def linear_to_srgb(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return np.where(value <= 0.0031308, value * 12.92, 1.055 * np.power(value, 1.0 / 2.4) - 0.055)


def display_image(linear_hdr: np.ndarray, exposure: float) -> np.ndarray:
    exposed = np.clip(linear_hdr, 0.0, None) * exposure
    mapped = exposed / (1.0 + exposed)
    return np.rint(linear_to_srgb(mapped) * 255.0).astype(np.uint8)


def recursive_objects(collection: bpy.types.Collection) -> list[bpy.types.Object]:
    result = list(collection.objects)
    for child in collection.children:
        result.extend(recursive_objects(child))
    return result


def camera_objects(split: str) -> list[bpy.types.Object]:
    name = CAMERA_COLLECTIONS[split]
    collection = bpy.data.collections.get(name)
    if collection is None:
        raise RuntimeError(f"missing camera collection {name!r}")
    cameras = [obj for obj in recursive_objects(collection) if obj.type == "CAMERA"]
    if not cameras:
        raise RuntimeError(f"camera collection {name!r} has no camera objects")
    return sorted(cameras, key=lambda obj: obj.name)


def emitter_objects() -> list[bpy.types.Object]:
    collection = bpy.data.collections.get(EMITTER_COLLECTION)
    if collection is None:
        raise RuntimeError(f"missing emitter collection {EMITTER_COLLECTION!r}")
    values = [obj for obj in recursive_objects(collection) if obj.type in {"MESH", "LIGHT"}]
    if not values:
        raise RuntimeError(f"emitter collection {EMITTER_COLLECTION!r} is empty")
    return sorted(values, key=lambda obj: obj.name)


def validate_external_images() -> None:
    missing = []
    for image in bpy.data.images:
        if image.source != "FILE" or image.packed_file is not None or not image.filepath:
            continue
        path = Path(bpy.path.abspath(image.filepath))
        if not path.is_file():
            missing.append(f"{image.name}: {path}")
    if missing:
        raise FileNotFoundError(
            "missing external image textures; use File > External Data > Pack Resources before upload: "
            + "; ".join(missing)
        )


def principled_nodes(obj: bpy.types.Object) -> list[bpy.types.Node]:
    result = []
    if obj.type != "MESH":
        return result
    for slot in obj.material_slots:
        material = slot.material
        if material is None or not material.use_nodes:
            continue
        result.extend(node for node in material.node_tree.nodes if node.bl_idname == "ShaderNodeBsdfPrincipled")
    return result


def set_mesh_emission(obj: bpy.types.Object, strength: float, color: list[float]) -> None:
    nodes = principled_nodes(obj)
    if not nodes:
        raise RuntimeError(f"mesh emitter {obj.name!r} needs a Principled BSDF material")
    for node in nodes:
        node.inputs["Emission Color"].default_value = (*map(float, color), 1.0)
        node.inputs["Emission Strength"].default_value = float(strength)


def apply_lighting(path: Path, condition: str, emitters: list[bpy.types.Object]) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    # Accept both the original compact mapping and the versioned local-editor
    # schema, whose condition mappings live below a top-level ``conditions``
    # key.  Normalizing here keeps the capture metadata identical.
    conditions = payload.get("conditions", payload)
    if not isinstance(conditions, dict) or condition not in conditions:
        choices = sorted(conditions) if isinstance(conditions, dict) else []
        raise KeyError(f"lighting condition {condition!r} not found; choices={choices}")
    raw_values = conditions[condition]
    if not isinstance(raw_values, dict):
        raise TypeError(f"lighting condition {condition!r} must map emitter names to settings")
    missing = [obj.name for obj in emitters if obj.name not in raw_values]
    if missing:
        raise KeyError(f"condition {condition!r} is missing emitters: {missing}")
    values = {}
    for obj in emitters:
        config = raw_values[obj.name]
        enabled = bool(config.get("enabled", True))
        strength = float(config["strength"]) if enabled else 0.0
        raw_color = config.get("color", config.get("rgb"))
        if raw_color is None:
            raise KeyError(f"emitter config for {obj.name!r} needs 'color' or 'rgb'")
        color = [float(value) for value in raw_color]
        if len(color) != 3 or strength < 0.0:
            raise ValueError(f"invalid emitter config for {obj.name}")
        values[obj.name] = {
            "enabled": enabled,
            "strength": strength,
            "color": color,
        }
        if obj.type == "LIGHT":
            obj.data.energy = strength
            obj.data.color = color
        else:
            set_mesh_emission(obj, strength, color)
    return values


def enable_optix() -> list[str]:
    preferences = bpy.context.preferences.addons["cycles"].preferences
    preferences.compute_device_type = "OPTIX"
    preferences.get_devices()
    names = []
    for device in preferences.devices:
        use = device.type == "OPTIX"
        device.use = use
        if use:
            names.append(device.name)
    if not names:
        raise RuntimeError("Cycles did not expose an OptiX device")
    bpy.context.scene.cycles.device = "GPU"
    return names


def ensure_aov(view_layer: bpy.types.ViewLayer, name: str, kind: str) -> None:
    existing = view_layer.aovs.get(name)
    if existing is None:
        existing = view_layer.aovs.add()
        existing.name = name
    existing.type = kind


def connect_aov(material: bpy.types.Material, name: str, input_name: str, kind: str) -> None:
    if not material.use_nodes:
        return
    tree = material.node_tree
    principled = next((node for node in tree.nodes if node.bl_idname == "ShaderNodeBsdfPrincipled"), None)
    if principled is None:
        return
    source = principled.inputs.get(input_name)
    if source is None:
        return
    node = tree.nodes.new("ShaderNodeOutputAOV")
    node.name = f"AEGIR_Output_{name}"
    node.aov_name = name
    destination = node.inputs["Color" if kind == "COLOR" else "Value"]
    if source.is_linked:
        tree.links.new(source.links[0].from_socket, destination)
    else:
        destination.default_value = source.default_value


def configure_material_aovs(view_layer: bpy.types.ViewLayer) -> None:
    definitions = [
        ("AEGIR_BaseColor", "Base Color", "COLOR"),
        ("AEGIR_Roughness", "Roughness", "VALUE"),
        ("AEGIR_Metallic", "Metallic", "VALUE"),
    ]
    for name, _, kind in definitions:
        ensure_aov(view_layer, name, kind)
    for material in bpy.data.materials:
        for name, input_name, kind in definitions:
            connect_aov(material, name, input_name, kind)


def configure_multilayer_output(scene: bpy.types.Scene, base_path: Path) -> bpy.types.Node:
    scene.render.use_compositing = True
    tree = bpy.data.node_groups.new("AEGIR_Capture_Compositor", "CompositorNodeTree")
    scene.compositing_node_group = tree
    render_layers = tree.nodes.new("CompositorNodeRLayers")
    output = tree.nodes.new("CompositorNodeOutputFile")
    output.directory = str(base_path)
    output.file_name = "passes"
    output.use_file_extension = True
    output.format.file_format = "OPEN_EXR_MULTILAYER"
    output.format.color_mode = "RGBA"
    output.format.color_depth = "32"
    output.format.exr_codec = "ZIP"

    definitions = [
        ("Combined", "Image", "RGBA"),
        ("Depth", "Depth", "FLOAT"),
        ("Normal", "Normal", "VECTOR"),
        ("IndexOB", "Object Index", "FLOAT"),
        ("AEGIR_BaseColor", "AEGIR_BaseColor", "RGBA"),
        ("AEGIR_Roughness", "AEGIR_Roughness", "FLOAT"),
        ("AEGIR_Metallic", "AEGIR_Metallic", "FLOAT"),
    ]
    for layer_name, socket_name, socket_type in definitions:
        source = render_layers.outputs.get(socket_name)
        if source is None:
            raise KeyError(
                f"missing compositor pass {socket_name!r}; "
                f"available={[socket.name for socket in render_layers.outputs]}"
            )
        output.file_output_items.new(socket_type, layer_name)
        destination = output.inputs.get(layer_name)
        if destination is None:
            raise RuntimeError(f"file output did not create socket {layer_name!r}")
        tree.links.new(source, destination)
    return output


def read_multilayer_exr(path: Path) -> tuple[np.ndarray, list[str]]:
    source = oiio.ImageInput.open(str(path))
    if source is None:
        raise RuntimeError(f"OpenImageIO could not open {path}: {oiio.geterror()}")
    try:
        images = []
        channels = []
        subimage = 0
        expected_shape = None
        while source.seek_subimage(subimage, 0):
            spec = source.spec()
            value = np.asarray(source.read_image(format=oiio.FLOAT), dtype=np.float32)
            if expected_shape is None:
                expected_shape = value.shape[:2]
            elif value.shape[:2] != expected_shape:
                raise ValueError(f"inconsistent multilayer EXR shape in {path}")
            images.append(value)
            channels.extend(spec.channelnames)
            subimage += 1
        if not images:
            raise ValueError(f"multilayer EXR has no subimages: {path}")
    finally:
        source.close()
    return np.concatenate(images, axis=-1), list(channels)


def exr_pass(
    value: np.ndarray,
    channels: list[str],
    names: tuple[str, ...],
    components: tuple[tuple[str, ...], ...],
) -> np.ndarray:
    indices = []
    for aliases in components:
        selected = None
        for pass_name in names:
            for component in aliases:
                suffix = f".{pass_name}.{component}"
                exact = f"{pass_name}.{component}"
                selected = next(
                    (
                        index
                        for index, name in enumerate(channels)
                        if name == exact or name.endswith(suffix)
                    ),
                    None,
                )
                if selected is not None:
                    break
            if selected is not None:
                break
        if selected is None:
            raise KeyError(
                f"missing EXR pass names={names} components={aliases}; available={channels}"
            )
        indices.append(selected)
    result = value[..., indices]
    return result[..., 0] if len(indices) == 1 else result


def intrinsics(scene: bpy.types.Scene, camera: bpy.types.Object) -> tuple[float, float, float, float]:
    width = scene.render.resolution_x
    height = scene.render.resolution_y
    data = camera.data
    if data.type != "PERSP":
        raise RuntimeError("AEGIR capture requires a perspective camera")
    if abs(scene.render.pixel_aspect_x - scene.render.pixel_aspect_y) > 1e-8:
        raise RuntimeError("AEGIR capture requires square pixels")
    if abs(data.shift_x) > 1e-8 or abs(data.shift_y) > 1e-8:
        raise RuntimeError("AEGIR capture currently requires zero camera lens shift")

    # Blender's Camera.angle_y follows the physical sensor aspect, not
    # necessarily the aspect ratio of the rendered image.  In particular, an
    # AUTO-fit 36x24 mm camera rendered to a square image has square-pixel
    # fx == fy even though angle_x != angle_y.  view_frame(scene=...) is the
    # authoritative render frustum after sensor-fit and output-aspect handling.
    frame = data.view_frame(scene=scene)
    normalized_x = [float(corner.x / -corner.z) for corner in frame]
    normalized_y = [float(corner.y / -corner.z) for corner in frame]
    x_min, x_max = min(normalized_x), max(normalized_x)
    y_min, y_max = min(normalized_y), max(normalized_y)
    if x_max <= x_min or y_max <= y_min:
        raise RuntimeError(f"invalid Blender camera view frame: {frame}")

    fx = width / (x_max - x_min)
    fy = height / (y_max - y_min)
    cx = -x_min * fx
    cy = y_max * fy
    return fx, fy, cx, cy


def emitter_metadata(obj: bpy.types.Object, config: dict) -> dict:
    world_corners = (
        [list(obj.matrix_world @ Vector(corner)) for corner in obj.bound_box]
        if obj.type == "MESH"
        else []
    )
    return {
        "name": obj.name,
        "type": obj.type,
        "emitter_id": int(obj.get("aegir_emitter_id", 0)),
        "object_pass_index": int(obj.pass_index),
        "transform_matrix": [list(row) for row in obj.matrix_world],
        "world_bbox_corners": world_corners,
        "condition": config,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lighting-config", type=Path, required=True)
    parser.add_argument("--condition", default="train")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--exposure", type=float, default=1.0)
    parser.add_argument("--splits", nargs="+", choices=tuple(CAMERA_COLLECTIONS), default=list(CAMERA_COLLECTIONS))
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--seed", type=int, default=20260826)
    args = parser.parse_args(script_arguments())

    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {args.output}")
    if args.resolution <= 0 or args.samples <= 0 or args.exposure <= 0.0:
        raise ValueError("resolution, samples, and exposure must be positive")

    output = args.output.resolve()
    directories = {
        name: output / name
        for name in (
            "images", "radiance_hdr", "depths", "normals_world", "albedo",
            "roughness", "metallic", "masks", "emitter_masks", "object_ids",
        )
    }
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=False)

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = args.resolution
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = 1.0
    scene.render.film_transparent = False
    scene.render.use_file_extension = True
    scene.render.use_persistent_data = True
    scene.cycles.samples = args.samples
    scene.cycles.seed = args.seed
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.01
    scene.cycles.use_denoising = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"

    validate_external_images()
    devices = enable_optix()
    view_layer = bpy.context.view_layer
    view_layer.use_pass_z = True
    view_layer.use_pass_normal = True
    view_layer.use_pass_object_index = True
    view_layer.use_pass_diffuse_color = True
    view_layer.pass_alpha_threshold = 0.0
    configure_material_aovs(view_layer)
    bpy.context.view_layer.update()
    multilayer_output = configure_multilayer_output(scene, output / "_multilayer_tmp")

    emitters = emitter_objects()
    lighting = apply_lighting(args.lighting_config.resolve(), args.condition, emitters)
    emitter_ids = {}
    for index, obj in enumerate(emitters, start=1):
        emitter_id = int(obj.get("aegir_emitter_id", index))
        if emitter_id <= 0 or emitter_id in emitter_ids.values():
            raise ValueError("emitter IDs must be unique positive integers")
        emitter_ids[obj.name] = emitter_id

    for index, obj in enumerate(sorted(bpy.context.scene.objects, key=lambda value: value.name), start=1):
        obj.pass_index = index
    emitter_pass_indices = {obj.pass_index for obj in emitters}

    entries = []
    for split in args.splits:
        entries.extend((split, camera) for camera in camera_objects(split))
    if args.max_frames is not None:
        entries = entries[: args.max_frames]
    if not entries:
        raise RuntimeError("no cameras selected")

    frames = []
    split_files = {split: [] for split in CAMERA_COLLECTIONS}
    started = time.perf_counter()

    for frame_index, (split, camera) in enumerate(entries):
        frame_started = time.perf_counter()
        scene.camera = camera
        stem = f"frame_{frame_index:04d}"
        exr_path = directories["radiance_hdr"] / f"{stem}.exr"
        temporary_exr_dir = output / "_multilayer_tmp" / stem
        temporary_exr_dir.mkdir(parents=True, exist_ok=False)
        multilayer_output.directory = str(temporary_exr_dir)
        scene.render.filepath = str(output / "_unused.png")
        bpy.ops.render.render(write_still=False)
        temporary_exrs = list(temporary_exr_dir.glob("*.exr"))
        if len(temporary_exrs) != 1:
            raise RuntimeError(
                f"expected one multilayer EXR in {temporary_exr_dir}, found {temporary_exrs}"
            )
        temporary_exrs[0].replace(exr_path)
        temporary_exr_dir.rmdir()

        exr, channels = read_multilayer_exr(exr_path)
        combined = exr_pass(
            exr, channels, ("Combined",), (("R",), ("G",), ("B",))
        )
        combined = np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)
        combined = np.clip(combined, 0.0, None).astype(np.float32)
        # Cycles' Z pass is camera-space axial depth already.  It is not the
        # Euclidean length of the primary ray, so applying a pixel-dependent
        # ray-length correction contracts off-axis geometry toward the camera.
        camera_z_depth = exr_pass(
            exr, channels, ("Depth", "Z"), (("Z", "V", "R", "X"),)
        )
        normal = exr_pass(
            exr, channels, ("Normal",), (("X", "R"), ("Y", "G"), ("Z", "B"))
        )
        object_index = exr_pass(
            exr, channels, ("IndexOB", "Object Index"), (("X", "V", "R"),)
        )
        albedo = exr_pass(
            exr, channels, ("AEGIR_BaseColor", "DiffCol"), (("R",), ("G",), ("B",))
        )
        roughness = exr_pass(
            exr, channels, ("AEGIR_Roughness",), (("X", "V", "R"),)
        )
        metallic = exr_pass(
            exr, channels, ("AEGIR_Metallic",), (("X", "V", "R"),)
        )

        valid = (
            np.isfinite(camera_z_depth)
            & (camera_z_depth > camera.data.clip_start)
            & (camera_z_depth < camera.data.clip_end)
        )
        fx, fy, cx, cy = intrinsics(scene, camera)
        depth_z = np.where(valid, camera_z_depth, 0.0).astype(np.float32)
        normal = np.where(valid[..., None], normal, 0.0).astype(np.float32)
        albedo = np.where(valid[..., None], albedo, 0.0).astype(np.float32)
        roughness = np.where(valid, roughness, 0.0).astype(np.float32)
        metallic = np.where(valid, metallic, 0.0).astype(np.float32)
        ids = np.rint(object_index).astype(np.int32)
        ids[~valid] = 0
        emitter_mask = np.isin(ids, list(emitter_pass_indices))

        write_png(directories["images"] / f"{stem}.png", display_image(combined, args.exposure))
        write_png(directories["masks"] / f"{stem}.png", valid.astype(np.uint8) * 255)
        write_png(directories["emitter_masks"] / f"{stem}.png", emitter_mask.astype(np.uint8) * 255)
        np.save(directories["radiance_hdr"] / f"{stem}.npy", combined, allow_pickle=False)
        np.save(directories["depths"] / f"{stem}.npy", depth_z, allow_pickle=False)
        np.save(directories["normals_world"] / f"{stem}.npy", normal, allow_pickle=False)
        np.save(directories["albedo"] / f"{stem}.npy", albedo, allow_pickle=False)
        np.save(directories["roughness"] / f"{stem}.npy", roughness, allow_pickle=False)
        np.save(directories["metallic"] / f"{stem}.npy", metallic, allow_pickle=False)
        np.save(directories["object_ids"] / f"{stem}.npy", ids, allow_pickle=False)
        image_path = f"images/{stem}.png"
        frame = {
            "file_path": image_path,
            "radiance_hdr_file_path": f"radiance_hdr/{stem}.npy",
            "radiance_exr_file_path": f"radiance_hdr/{stem}.exr",
            "depth_file_path": f"depths/{stem}.npy",
            "normal_world_file_path": f"normals_world/{stem}.npy",
            "albedo_gt_file_path": f"albedo/{stem}.npy",
            "roughness_gt_file_path": f"roughness/{stem}.npy",
            "metallic_gt_file_path": f"metallic/{stem}.npy",
            "mask_path": f"masks/{stem}.png",
            "emitter_mask_path": f"emitter_masks/{stem}.png",
            "object_id_file_path": f"object_ids/{stem}.npy",
            "transform_matrix": [list(row) for row in camera.matrix_world],
            "fl_x": fx,
            "fl_y": fy,
            "cx": cx,
            "cy": cy,
            "w": args.resolution,
            "h": args.resolution,
            "split": split,
            "source_camera": camera.name,
        }
        frames.append(frame)
        split_files[split].append(image_path)
        elapsed = time.perf_counter() - frame_started
        total = time.perf_counter() - started
        average = total / (frame_index + 1)
        remaining = average * (len(entries) - frame_index - 1)
        print(
            f"AEGIR_RENDER {frame_index + 1}/{len(entries)} split={split} camera={camera.name} "
            f"frame_seconds={elapsed:.2f} eta_minutes={remaining / 60.0:.1f}",
            flush=True,
        )

    metadata = {
        "camera_model": "OPENCV",
        "fl_x": frames[0]["fl_x"],
        "fl_y": frames[0]["fl_y"],
        "cx": frames[0]["cx"],
        "cy": frames[0]["cy"],
        "w": args.resolution,
        "h": args.resolution,
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "aegir_source_camera_convention": "blender_opengl_c2w",
        "aegir_camera_convention": "opengl_c2w",
        "aegir_depth_convention": "opencv",
        "aegir_depth_mode": "z",
        "source_depth_mode": "ray_distance_converted_to_z",
        "depth_unit_scale_factor": float(scene.unit_settings.scale_length),
        "normal_space": "world",
        "aegir_color": {
            "format": "aegir-color-v1",
            "observation_domain": "linear_hdr",
            "hdr_encoding": "scene_linear_rgb",
            "target_png": {
                "tone_mapper": "reinhard",
                "transfer_function": "srgb",
                "exposure_policy": "fixed_scene",
                "exposure_scale": args.exposure,
                "source": "deterministic AEGIR Blender display pipeline",
            },
        },
        "aegir_blender": {
            "format": "aegir-blender-v1",
            "blender_version": bpy.app.version_string,
            "engine": "CYCLES",
            "cycles_device": "OPTIX",
            "devices": devices,
            "samples": args.samples,
            "adaptive_threshold": scene.cycles.adaptive_threshold,
            "denoising": False,
            "lighting_condition": args.condition,
            "seed": args.seed,
        },
        "train_filenames": split_files["train"],
        "val_filenames": split_files["val"],
        "test_filenames": split_files["test"],
        "frames": frames,
    }
    (output / "transforms.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    for split in CAMERA_COLLECTIONS:
        subset = [frame for frame in frames if frame["split"] == split]
        (output / f"transforms_{split}.json").write_text(
            json.dumps({**metadata, "frames": subset}, indent=2) + "\n",
            encoding="utf-8",
        )
    emitter_payload = {
        "condition": args.condition,
        "emitters": [emitter_metadata(obj, lighting[obj.name]) for obj in emitters],
    }
    (output / "emitters.json").write_text(json.dumps(emitter_payload, indent=2) + "\n", encoding="utf-8")
    elapsed = time.perf_counter() - started
    (output / "capture_summary.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "frames": len(frames),
                "elapsed_seconds": elapsed,
                "seconds_per_frame": elapsed / len(frames),
                "output": str(output),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"AEGIR_DATASET_COMPLETE frames={len(frames)} elapsed_minutes={elapsed / 60.0:.2f}")


if __name__ == "__main__":
    main()
