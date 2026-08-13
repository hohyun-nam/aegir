from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .config import AEGIRConfig
from .inspect import inspect_checkpoint, inspect_emitter_evolution, render_camera_path
from .trainer import train_shared_gaussians


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aegir")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser(
        "train-gaussian-scene",
        help="jointly optimize one shared 2DGS geometry, per-Gaussian BRDFs, and emitters",
    )
    train.add_argument("--dnsplatter-checkpoint", type=Path, required=True)
    train.add_argument("--nerfstudio-dir", type=Path, required=True)
    train.add_argument("--prior-dir", type=Path, default=None)
    train.add_argument("--emitter-checkpoint", type=Path, default=None)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--device", default="cuda")
    train.add_argument("--steps", type=int, default=20_000)
    train.add_argument("--image-size", type=int, default=128)
    train.add_argument("--max-views", type=int, default=None)
    train.add_argument("--max-gaussians", type=int, default=None)
    train.add_argument("--max-emitters", type=int, default=16)
    train.add_argument("--geometry-lr", type=float, default=1.0e-5)
    train.add_argument("--material-lr", type=float, default=1.0e-3)
    train.add_argument("--emitter-lr", type=float, default=1.0e-3)
    train.add_argument("--emitter-final-lr", type=float, default=1.0e-4)
    train.add_argument("--freeze-start", type=int, default=5_000)
    train.add_argument("--freeze-end", type=int, default=15_000)
    train.add_argument("--checkpoint-every", type=int, default=5_000)
    train.add_argument("--num-brdf-samples", type=int, default=128)
    train.add_argument("--num-light-samples", type=int, default=128)
    train.add_argument("--rays-per-step", type=int, default=1_024)
    train.add_argument("--visibility-occluders", type=int, default=2048)
    train.add_argument("--no-visibility", action="store_true")
    train.add_argument("--no-indirect", action="store_true")

    inspect = subparsers.add_parser(
        "inspect",
        help="re-rasterize the shared Gaussian scene in every requested camera",
    )
    inspect.add_argument("--checkpoint", type=Path, required=True)
    inspect.add_argument("--nerfstudio-dir", type=Path, required=True)
    inspect.add_argument("--prior-dir", type=Path, default=None)
    inspect.add_argument("--output-dir", type=Path, required=True)
    inspect.add_argument("--device", default="cuda")
    inspect.add_argument("--max-views", type=int, default=None)
    inspect.add_argument(
        "--split",
        choices=("train", "val", "test", "all"),
        default="test",
    )
    inspect.add_argument("--render-samples", type=int, default=16)

    inspect_emitters = subparsers.add_parser(
        "inspect-emitters",
        help="compare pre-training and final anisotropic emitter positions and scales",
    )
    inspect_emitters.add_argument("--checkpoint", type=Path, required=True)
    inspect_emitters.add_argument("--nerfstudio-dir", type=Path, required=True)
    inspect_emitters.add_argument("--prior-dir", type=Path, default=None)
    inspect_emitters.add_argument("--output-dir", type=Path, required=True)
    inspect_emitters.add_argument("--device", default="cuda")
    inspect_emitters.add_argument("--max-views", type=int, default=None)
    inspect_emitters.add_argument(
        "--split",
        choices=("train", "val", "test", "all"),
        default="all",
    )

    camera_path = subparsers.add_parser(
        "render-camera-path",
        help="render arbitrary Nerfstudio transforms.json cameras from the shared scene",
    )
    camera_path.add_argument("--checkpoint", type=Path, required=True)
    camera_path.add_argument("--transforms", type=Path, required=True)
    camera_path.add_argument("--output-dir", type=Path, required=True)
    camera_path.add_argument("--device", default="cuda")
    camera_path.add_argument("--max-frames", type=int, default=None)
    camera_path.add_argument("--render-samples", type=int, default=16)
    camera_path.add_argument("--light-scale", type=float, default=1.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "train-gaussian-scene":
        if not str(args.device).startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("shared Gaussian training requires CUDA and diff-surfel-rasterization")
        config = AEGIRConfig()
        config.training.steps = args.steps
        config.raster.image_size = args.image_size
        config.training.geometry_lr = args.geometry_lr
        config.training.material_lr = args.material_lr
        config.training.emitter_lr = args.emitter_lr
        config.training.emitter_final_lr = args.emitter_final_lr
        config.training.freeze_start = args.freeze_start
        config.training.freeze_end = args.freeze_end
        config.training.checkpoint_every = args.checkpoint_every
        config.adaptive.max_emitters = args.max_emitters
        config.render.num_brdf_samples = args.num_brdf_samples
        config.render.num_light_samples = args.num_light_samples
        config.training.rays_per_step = args.rays_per_step
        config.render.visibility_occluders = args.visibility_occluders
        config.render.point_visibility = not args.no_visibility
        config.render.indirect_bounce = not args.no_indirect
        train_shared_gaussians(
            args.dnsplatter_checkpoint,
            args.nerfstudio_dir,
            args.prior_dir,
            args.output,
            config,
            args.device,
            emitter_checkpoint=args.emitter_checkpoint,
            max_views=args.max_views,
            max_gaussians=args.max_gaussians,
        )
    elif args.command == "inspect":
        inspect_checkpoint(
            args.checkpoint,
            args.nerfstudio_dir,
            args.prior_dir,
            args.output_dir,
            args.device,
            max_views=args.max_views,
            render_samples=args.render_samples,
            split=args.split,
        )
    elif args.command == "inspect-emitters":
        inspect_emitter_evolution(
            args.checkpoint,
            args.nerfstudio_dir,
            args.prior_dir,
            args.output_dir,
            args.device,
            max_views=args.max_views,
            split=args.split,
        )
    elif args.command == "render-camera-path":
        render_camera_path(
            args.checkpoint,
            args.transforms,
            args.output_dir,
            args.device,
            max_frames=args.max_frames,
            render_samples=args.render_samples,
            light_scale=args.light_scale,
        )


if __name__ == "__main__":
    main()
