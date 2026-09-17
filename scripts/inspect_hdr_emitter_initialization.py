#!/usr/bin/env python3
"""Validate HDR-only multi-view emitter initialization before expensive training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aegir.checkpoint import emitter_state_dict, save_checkpoint
from aegir.config import AEGIRConfig
from aegir.data import MultiViewSceneDataset
from aegir.gaussian_scene import GaussianMaterialBackend
from aegir.hdr_triangulation import project
from aegir.initialization import associate_gaussians_with_emitters
from aegir.math_utils import quat_to_matrix
from aegir.provenance import (
    training_inputs,
    validate_emitter_inclusive_geometry,
    validate_practical_geometry,
    validate_predicted_depth,
    write_json,
)
from aegir.trainer import initialize_emitters
from scripts.export_scene_supersplat import emitter_gaussians, scene_from_checkpoint
from scripts.export_supersplat_ply import filter_gaussians, write_binary_ply


def write_overlays(dataset, emitters, diagnostic, output):
    by_frame = {}
    for emitter in diagnostic["emitters"]:
        for observation in emitter["observations"]:
            by_frame.setdefault(observation["frame"], []).append((emitter, observation))
    frames = {f.index: f for f in dataset.frames}
    selected = sorted(by_frame, key=lambda k: (-len(by_frame[k]), k))[:8]
    theta = np.linspace(0, 2 * np.pi, 96, endpoint=True)
    rotations = quat_to_matrix(emitters.state().quaternions).detach().numpy()
    centers = emitters.centers.detach().numpy()
    scales = emitters.scales.detach().numpy()
    tiles = []
    directory = output / "projection_overlays"
    directory.mkdir(parents=True, exist_ok=True)
    for index in selected:
        frame = frames[index]
        rgb = np.clip(frame.target.numpy() * 255, 0, 255).astype(np.uint8)
        hdr = frame.radiance_hdr.numpy()
        display = np.log1p(hdr) / np.log1p(diagnostic["high_threshold"])
        heat = (np.clip(display, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)
        h, w = rgb.shape[:2]
        canvas = Image.new("RGB", (w * 2, h + 40))
        canvas.paste(Image.fromarray(rgb), (0, 40))
        canvas.paste(Image.fromarray(heat), (w, 40))
        draw = ImageDraw.Draw(canvas)
        draw.text((6, 4), f"Frame {index} | RGB / observed HDR", fill="white")
        draw.text((6, 20), "Cyan: HDR region  Green: emitter  Red: center", fill="white")
        for emitter, obs in by_frame[index]:
            i = emitter["id"]
            cloud = (
                centers[i]
                + (np.cos(theta) * scales[i, 0])[:, None] * rotations[i, :, 0]
                + (np.sin(theta) * scales[i, 1])[:, None] * rotations[i, :, 1]
            )
            boundary, depth = project(cloud, frame.camera)
            uv, center_depth = project(centers[i : i + 1], frame.camera)
            if np.any(depth <= 0) or center_depth[0] <= 0:
                continue
            for offset in (0, w):
                polygon = [(p[0] + offset, p[1] + 40) for p in obs["polygon"]]
                draw.line(polygon + polygon[:1], fill="cyan", width=1)
                ellipse = [(p[0] + offset, p[1] + 40) for p in boundary]
                draw.line(ellipse, fill="#40ff40", width=1)
                x, y = uv[0] + [offset, 40]
                draw.line([(x - 3, y), (x + 3, y)], fill="red", width=1)
                draw.line([(x, y - 3), (x, y + 3)], fill="red", width=1)
                draw.text((x + 4, y - 10), str(i), fill="red")
        canvas.save(directory / f"frame_{index:04d}.png")
        tiles.append(canvas)
    if tiles:
        width, height = tiles[0].size
        contact = Image.new("RGB", (width * 2, height * ((len(tiles) + 1) // 2)))
        for k, tile in enumerate(tiles):
            contact.paste(tile, ((k % 2) * width, (k // 2) * height))
        contact.save(output / "projection_contact.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument(
        "--representation", choices=("planar_ellipse_2d", "ellipsoid_3d"), default="ellipsoid_3d"
    )
    parser.add_argument(
        "--initialization",
        choices=("hdr_multiview", "hdr_gbuffer"),
        default="hdr_multiview",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    validate_predicted_depth(args.data)
    if args.initialization == "hdr_gbuffer":
        validate_emitter_inclusive_geometry(args.geometry)
    else:
        validate_practical_geometry(args.geometry)
    config = AEGIRConfig()
    config.raster.image_size = args.image_size
    config.training.emitter_initialization = args.initialization
    config.training.emitter_representation = args.representation
    config.training.training_transport_mode = "secondary_only"
    config.losses.light_white = 0
    dataset = MultiViewSceneDataset(args.data, args.geometry, image_size=args.image_size, split="train")
    scene = GaussianMaterialBackend(args.geometry).to(args.device)
    bounds = scene.bounds(config.raster.bounds_quantile)
    emitters = initialize_emitters(scene, dataset, bounds, config, args.device)
    association = associate_gaussians_with_emitters(
        scene,
        emitters,
        bounds,
        tangent_margin=config.training.emitter_gaussian_tangent_margin,
        normal_fraction=config.training.emitter_gaussian_normal_fraction,
    )
    diagnostic = dataset.emitter_initialization_diagnostic
    diagnostic["gaussian_association"] = association
    diagnostic["input_provenance"] = training_inputs(args.data.resolve(), args.geometry.resolve(), None)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "emitter_initialization.json", diagnostic)
    checkpoint = args.output / "initialization.pt"
    save_checkpoint(
        checkpoint,
        scene,
        emitters,
        config,
        0,
        [],
        {
            "nerfstudio_dir": str(args.data.resolve()),
            "prior_dir": None,
            "image_size": args.image_size,
            "max_views": None,
            "observation_policy": dataset.observation_policy,
            "split_protocol": dataset.split_protocol,
            "input_provenance": diagnostic["input_provenance"],
            "emitter_initialization_diagnostic": diagnostic,
            "preflight_only": True,
        },
        initial_emitter_state=emitter_state_dict(emitters),
    )
    saved_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    exported = filter_gaussians(
        scene_from_checkpoint(saved_payload),
        min_opacity=0.01,
        stride=1,
        max_gaussians=500_000,
        drop_radius_outliers=True,
        radius_outlier_quantile=0.999,
        drop_large_scales=True,
        large_scale_quantile=0.999,
    )
    lights = emitter_gaussians(emitter_state_dict(emitters), exported["f_rest"].shape[1], 8.0)
    write_binary_ply(
        args.output / "map_with_initial_emitters.ply",
        {key: torch.cat((exported[key], lights[key]), dim=0) for key in exported},
    )
    if args.initialization == "hdr_multiview":
        write_overlays(dataset, emitters.cpu(), diagnostic, args.output)
        for light in diagnostic["emitters"]:
            print(
                f"emitter {light['id']}: center={light['center']} views={light['views']} "
                f"median_error={light['reprojection_median_px']:.3f}px",
                flush=True,
            )
    print(
        f"associated source Gaussians: {association['tagged']} counts={association['counts']}",
        flush=True,
    )
    print(f"PREFLIGHT_COMPLETE: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
