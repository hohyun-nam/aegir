#!/usr/bin/env python3
"""Render matched camera poses and measure synchronized full-frame wall times."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aegir.checkpoint import load_checkpoint
from aegir.color import apply_ldr_pipeline, configure_ldr_fallback, display_srgb
from aegir.config import config_from_dict
from aegir.data import MultiViewSceneDataset
from aegir.inspect import _camera_to, _to_u8, render_full_frame
from aegir.irgs_tracer import IRGSVisibilityTracer
from aegir.visibility import scene_scale_from_bounds


def parse_setting(value):
    try:
        samples, bounces = map(int, value.split("/"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use samples/bounces, for example 128/8") from error
    if samples < 1 or bounces < 1:
        raise argparse.ArgumentTypeError("Sample counts must be positive")
    return samples, bounces


def write_summary(output, summary):
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output / "median_seconds.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["setting", "images", "median_seconds"])
        for row in summary["results"]:
            writer.writerow([row["setting"], row["images"], f"{row['median_seconds']:.6f}"])


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="test")
    parser.add_argument("--views", type=int, default=10)
    parser.add_argument("--settings", type=parse_setting, nargs="+", default=[(128, 1), (128, 8), (512, 8)])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.views < 1 or args.warmup < 1:
        parser.error("views and warmup must be positive")
    if len(set(args.settings)) != len(args.settings):
        parser.error("Settings must be distinct")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("This benchmark requires CUDA")
    scene, emitters, payload = load_checkpoint(args.checkpoint, args.device)
    base = config_from_dict(payload["config"])
    dataset = MultiViewSceneDataset(
        args.data,
        payload["source_dnsplatter_checkpoint"],
        image_size=base.raster.image_size,
        max_views=args.views,
        split=args.split,
    )
    if len(dataset.frames) != args.views:
        raise ValueError(f"Requested {args.views} poses but split {args.split!r} has {len(dataset.frames)}")
    policy = configure_ldr_fallback(
        dataset.observation_policy,
        tone_mapper=base.color.ldr_fallback_tone_mapper,
        transfer_function=base.color.ldr_fallback_transfer_function,
    )
    if payload.get("dataset", {}).get("observation_policy", policy) != policy:
        raise ValueError("Benchmark observation policy differs from the checkpoint")
    learned_exposure = None
    if policy["domain"] == "ldr" and policy["exposure_policy"] == "learned_scene":
        learned_exposure = payload["training_state"]["observation"]["log_exposure"].exp()
    args.output.mkdir(parents=True, exist_ok=True)
    scale = scene_scale_from_bounds(scene.bounds(base.raster.bounds_quantile))
    summary = {
        "status": "running",
        "checkpoint": str(args.checkpoint.resolve()),
        "data": str(args.data.resolve()),
        "split": args.split,
        "resolution": [base.raster.image_size, base.raster.image_size],
        "frames": [f.index for f in dataset.frames],
        "seed": args.seed,
        "gpu": torch.cuda.get_device_name(device),
        "emitter_count": emitters.num_emitters,
        "render_chunk_size": 256,
        "bounce_chunk_size": base.render.bounce_chunk_size,
        "warmup_frames_per_setting": args.warmup,
        "denoiser": "off",
        "sh_proxy": "off",
        "timing_scope": "CUDA-synchronized render_full_frame wall time: G-buffer, shading/ray tracing, CPU output transfer; excludes loading, warmup, PNG/NPZ I/O and image metrics",
        "results": [],
    }
    write_summary(args.output, summary)
    targets = {}
    for frame in dataset.frames:
        if policy["domain"] == "linear_hdr":
            targets[frame.index] = _to_u8(display_srgb(frame.radiance_hdr, base.render.tonemap_exposure))
        else:
            targets[frame.index] = _to_u8(frame.target)
    cameras = [_camera_to(frame.camera, args.device) for frame in dataset.frames]
    with (args.output / "timings.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["setting", "view_order", "frame", "seed", "render_seconds"]
        )
        writer.writeheader()
        for samples, bounces in args.settings:
            label = f"{samples}/{bounces}"
            directory = args.output / f"render_{samples}x{bounces}"
            directory.mkdir(exist_ok=True)
            config = copy.deepcopy(base)
            config.render.num_brdf_samples = config.render.num_light_samples = samples
            config.render.num_bounce_samples = bounces
            tracer = (
                IRGSVisibilityTracer(scene, config.render, scene_scale=scale)
                if config.render.point_visibility
                else None
            )

            def render(camera, seed, config=config, tracer=tracer):
                generator = torch.Generator(device=device).manual_seed(seed)
                return render_full_frame(
                    scene,
                    emitters,
                    camera,
                    config,
                    visibility_tracer=tracer,
                    generator=generator,
                    transport_mode="inference",
                    chunk_size=256,
                )

            print(f"[{label}] warmup: {args.warmup} full frame(s), excluded from median", flush=True)
            for _ in range(args.warmup):
                render(cameras[0], args.seed + dataset.frames[0].index)
            torch.cuda.synchronize(device)
            times = []
            for order, (frame, camera) in enumerate(zip(dataset.frames, cameras, strict=True)):
                frame_seed = args.seed + frame.index
                torch.cuda.synchronize(device)
                started = time.perf_counter()
                rendered = render(camera, frame_seed)
                torch.cuda.synchronize(device)
                seconds = time.perf_counter() - started
                times.append(seconds)
                writer.writerow(
                    {
                        "setting": label,
                        "view_order": order,
                        "frame": frame.index,
                        "seed": frame_seed,
                        "render_seconds": f"{seconds:.9f}",
                    }
                )
                handle.flush()
                if policy["domain"] == "linear_hdr":
                    prediction = display_srgb(rendered["hdr"], config.render.tonemap_exposure)
                else:
                    exposure = learned_exposure if learned_exposure is not None else frame.ldr_exposure_scale
                    prediction = apply_ldr_pipeline(
                        rendered["hdr"],
                        exposure=exposure,
                        tone_mapper=policy["tone_mapper"],
                        transfer_function=policy["transfer_function"],
                    )
                rgb = _to_u8(prediction)
                Image.fromarray(rgb).save(directory / f"frame_{frame.index:04d}.png")
                Image.fromarray(np.concatenate((targets[frame.index], rgb), axis=1)).save(
                    directory / f"compare_{frame.index:04d}.png"
                )
                np.savez_compressed(directory / f"hdr_{frame.index:04d}.npz", hdr=rendered["hdr"].numpy())
                print(f"[{label}] {order + 1}/{args.views}: frame={frame.index}, {seconds:.4f} s", flush=True)
                del rendered
            median = statistics.median(times)
            summary["results"].append(
                {
                    "setting": label,
                    "images": len(times),
                    "median_seconds": median,
                    "min_seconds": min(times),
                    "max_seconds": max(times),
                    "frame_seconds": times,
                }
            )
            write_summary(args.output, summary)
            print(f"[{label}] MEDIAN = {median:.6f} seconds/image", flush=True)
            del tracer
            del render
    comparison = args.output / "comparison"
    comparison.mkdir(exist_ok=True)
    for frame in dataset.frames:
        tiles = [targets[frame.index]]
        labels = ["GT"]
        for samples, bounces in args.settings:
            path = args.output / f"render_{samples}x{bounces}" / f"frame_{frame.index:04d}.png"
            tiles.append(np.asarray(Image.open(path).convert("RGB")))
            labels.append(f"{samples}/{bounces}")
        h, w = tiles[0].shape[:2]
        canvas = Image.new("RGB", (w * len(tiles), h + 24))
        draw = ImageDraw.Draw(canvas)
        for k, (tile, label) in enumerate(zip(tiles, labels, strict=True)):
            canvas.paste(Image.fromarray(tile), (k * w, 24))
            draw.text((k * w + 6, 6), label, fill="white")
        canvas.save(comparison / f"frame_{frame.index:04d}.png")
    summary["status"] = "complete"
    write_summary(args.output, summary)
    print("BENCHMARK_COMPLETE", args.output.resolve(), flush=True)
    print((args.output / "median_seconds.tsv").read_text(), flush=True)


if __name__ == "__main__":
    main()
