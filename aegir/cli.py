from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .config import AEGIRConfig
from .inspect import inspect_checkpoint, inspect_emitter_evolution, render_camera_path
from .trainer import resume_shared_gaussians, train_shared_gaussians


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
    initialization = train.add_mutually_exclusive_group()
    initialization.add_argument(
        "--emitter-checkpoint",
        type=Path,
        default=None,
        help="warm-start emitter values only (not an exact training resume)",
    )
    initialization.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="resume an exact-resume v14-v16 checkpoint; checkpoint config remains authoritative",
    )
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--device", default="cuda")
    train.add_argument("--steps", type=int, default=5_000, help="final global-step target")
    train.add_argument("--image-size", type=int, default=128)
    train.add_argument("--max-views", type=int, default=None)
    train.add_argument("--max-gaussians", type=int, default=None)
    train.add_argument("--max-emitters", type=int, default=16)
    train.add_argument("--geometry-lr", type=float, default=1.0e-5)
    train.add_argument("--freeze-scene-means", action="store_true")
    train.add_argument("--freeze-scene-scales", action="store_true")
    train.add_argument("--freeze-scene-rotations", action="store_true")
    train.add_argument("--freeze-scene-opacities", action="store_true")
    train.add_argument(
        "--constrain-scene-scales",
        action="store_true",
        help="clamp learned background-Gaussian scales relative to their initialization",
    )
    train.add_argument("--scene-scale-min-ratio", type=float, default=0.9)
    train.add_argument("--scene-scale-max-ratio", type=float, default=1.1)
    train.add_argument(
        "--albedo-lr",
        type=float,
        default=3.0e-3,
        help="albedo-only learning rate; roughness/metallic use --material-lr",
    )
    train.add_argument("--material-lr", type=float, default=1.0e-3)
    train.add_argument(
        "--albedo-initialization",
        choices=("sh_dc", "neutral"),
        default="sh_dc",
        help="initialize physical albedo from DN SH DC or a uniform linear gray",
    )
    train.add_argument(
        "--neutral-albedo",
        type=float,
        default=0.5,
        help="scene-linear RGB value used by --albedo-initialization neutral",
    )
    train.add_argument("--emitter-lr", type=float, default=1.0e-3)
    train.add_argument("--emitter-final-lr", type=float, default=1.0e-4)
    train.add_argument(
        "--emitter-representation",
        choices=("planar_ellipse_2d", "ellipsoid_3d"),
        default="planar_ellipse_2d",
        help="analytic emitter support used for training; the Gaussian scene remains 2DGS",
    )
    train.add_argument(
        "--optimize-emitter-centers",
        action="store_true",
        help="optimize emitter centers instead of fixing their initialized positions",
    )
    train.add_argument(
        "--optimize-emitter-rotations",
        action="store_true",
        help="optimize emitter rotations instead of fixing their initialized orientations",
    )
    train.add_argument(
        "--freeze-emitter-scales",
        action="store_true",
        help="fix emitter scales as well as center/rotation",
    )
    train.add_argument(
        "--unconstrained-emitter-scales",
        action="store_true",
        help="allow learned scales outside the configured initialization-relative bounds",
    )
    train.add_argument("--emitter-scale-min-ratio", type=float, default=0.1)
    train.add_argument("--emitter-scale-max-ratio", type=float, default=1.0)
    train.add_argument(
        "--enable-emitter-adaptation",
        action="store_true",
        help="enable the historical emitter split/prune schedule",
    )
    train.add_argument("--freeze-start", type=int, default=5_000)
    train.add_argument("--freeze-end", type=int, default=15_000)
    train.add_argument("--sh-fade-start", type=int, default=5_000)
    train.add_argument("--sh-fade-end", type=int, default=10_000)
    train.add_argument(
        "--training-transport-mode",
        choices=("hybrid", "secondary_only", "sh_proxy_only"),
        default="hybrid",
        help="training-time indirect transport schedule stored in the checkpoint",
    )
    train.add_argument(
        "--light-white",
        type=float,
        default=0.0,
        help="final white-light regularization weight",
    )
    train.add_argument(
        "--light-white-ramp-end",
        type=int,
        default=0,
        help="smoothly ramp white-light weight from zero until this step; 0 keeps it fixed",
    )
    train.add_argument("--emitter-initialization", choices=("hdr_multiview", "hdr_predicted_depth", "hdr_gbuffer", "gt_mask_depth"), default="hdr_multiview", help="GT mask/depth initialization requires explicit diagnostic opt-in")
    train.add_argument(
        "--no-merge-collinear-emitter-fragments",
        action="store_true",
        help="keep disconnected HDR pieces as separate emitter candidates",
    )
    train.add_argument(
        "--no-associate-emitter-gaussians",
        action="store_true",
        help="leave source Gaussians inside analytic emitters in the background scene",
    )
    train.add_argument("--emitter-gaussian-tangent-margin", type=float, default=1.05)
    train.add_argument("--emitter-gaussian-normal-fraction", type=float, default=0.001)
    train.add_argument("--checkpoint-every", type=int, default=5_000)
    train.add_argument("--num-brdf-samples", type=int, default=128)
    train.add_argument("--num-light-samples", type=int, default=128)
    train.add_argument(
        "--num-bounce-samples",
        type=int,
        default=1,
        help="explicit emitter NEE samples at each first secondary hit",
    )
    train.add_argument(
        "--bounce-chunk-size",
        type=int,
        default=65_536,
        help=(
            "secondary NEE rays per OptiX launch; execution-only and safe to "
            "override during exact resume"
        ),
    )
    train.add_argument(
        "--rays-per-step",
        type=int,
        default=256,
        help=(
            "primary surface points per iteration (historical option name); "
            "total transport rays are this value times the sum of BRDF and light samples"
        ),
    )
    train.add_argument("--visibility-occluders", type=int, default=2048)
    train.add_argument(
        "--trace-origin-offset",
        type=float,
        default=0.05,
        help="absolute scene-unit ray-start offset used to suppress Gaussian self-hits",
    )
    train.add_argument(
        "--no-scene-culling",
        action="store_true",
        help="disable DN-style scene split/duplicate refinement and opacity culling",
    )
    train.add_argument("--scene-cull-opacity", type=float, default=0.005)
    train.add_argument("--scene-densify-grad", type=float, default=8.0e-4)
    train.add_argument("--scene-split-scale", type=float, default=0.01)
    train.add_argument("--scene-split-screen-size", type=float, default=0.05)
    train.add_argument("--scene-max-gaussians", type=int, default=500_000)
    train.add_argument("--scene-cull-scale", type=float, default=None, help=argparse.SUPPRESS)
    train.add_argument(
        "--scene-cull-screen-size", type=float, default=None, help=argparse.SUPPRESS
    )
    train.add_argument(
        "--light-surface",
        type=float,
        default=0.0,
        help="non-paper emitter-to-surface regularizer weight (default: disabled)",
    )
    train.add_argument(
        "--light-view",
        type=float,
        default=0.0,
        help="non-paper camera-frustum regularizer weight (default: disabled)",
    )
    train.add_argument("--no-visibility", action="store_true")
    train.add_argument(
        "--no-camera-emitters",
        action="store_true",
        help="disable camera-visible emitter self-emission compositing (ablation only)",
    )
    train.add_argument(
        "--symmetric-emission",
        action="store_true",
        help="disable learned signed-axis gains while retaining symmetric spread/falloff",
    )
    train.add_argument(
        "--emitter-axis-lobe-sharpness",
        type=float,
        default=12.0,
        help="soft assignment sharpness for the six learned local-axis gains",
    )
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
    inspect.add_argument("--render-samples", type=int, default=128)
    inspect.add_argument(
        "--bounce-samples",
        type=int,
        default=None,
        help="override explicit secondary-bounce samples for this inspection only",
    )
    inspect.add_argument("--evaluation-seed", type=int, default=2026)
    inspect.add_argument(
        "--evaluation-seed-by-view-order",
        action="store_true",
        help="use selected camera order for seeds when matching separately indexed datasets",
    )
    inspect.add_argument("--lpips-net", choices=("alex", "vgg", "squeeze"), default="alex")
    inspect.add_argument(
        "--training-sh-proxy",
        action="store_true",
        help=(
            "use the frozen DN full-SH training proxy instead of explicit secondary-bounce "
            "tracing; fast diagnostic only, not the paper inference estimator"
        ),
    )
    inspect.add_argument(
        "--symmetric-emission",
        action="store_true",
        help="render without the checkpoint's learned signed-axis gains",
    )
    inspect.add_argument(
        "--denoise-indirect",
        action="store_true",
        help="cross-bilateral filter indirect HDR using rendered depth and normals",
    )
    inspect.add_argument("--denoiser-radius", type=int, default=2)
    inspect.add_argument("--denoiser-sigma-spatial", type=float, default=2.0)
    inspect.add_argument("--denoiser-sigma-depth", type=float, default=1.0)
    inspect.add_argument("--denoiser-normal-power", type=float, default=32.0)
    inspect.add_argument("--denoiser-depth-epsilon", type=float, default=1.0e-3)
    inspect.add_argument(
        "--active-emitter-indices",
        type=int,
        nargs="+",
        default=None,
        help="zero-based checkpoint emitter indices to retain for relighting",
    )

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
    camera_path.add_argument(
        "--symmetric-emission",
        action="store_true",
        help="render without the checkpoint's learned signed-axis gains",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "train-gaussian-scene":
        if not str(args.device).startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("shared Gaussian training requires CUDA and diff-surfel-rasterization")
        if args.resume is not None:
            resume_shared_gaussians(
                args.resume,
                args.dnsplatter_checkpoint,
                args.nerfstudio_dir,
                args.prior_dir,
                args.output,
                args.steps,
                args.device,
                bounce_chunk_size=args.bounce_chunk_size,
            )
            return
        config = AEGIRConfig()
        config.training.steps = args.steps
        config.raster.image_size = args.image_size
        config.training.geometry_lr = args.geometry_lr
        config.training.optimize_scene_means = not args.freeze_scene_means
        config.training.optimize_scene_scales = not args.freeze_scene_scales
        config.training.optimize_scene_rotations = not args.freeze_scene_rotations
        config.training.optimize_scene_opacities = not args.freeze_scene_opacities
        config.training.constrain_scene_scales = args.constrain_scene_scales
        config.training.scene_scale_min_ratio = args.scene_scale_min_ratio
        config.training.scene_scale_max_ratio = args.scene_scale_max_ratio
        config.training.albedo_lr = args.albedo_lr
        config.training.material_lr = args.material_lr
        config.training.albedo_initialization = args.albedo_initialization
        config.training.neutral_albedo = args.neutral_albedo
        config.training.emitter_lr = args.emitter_lr
        config.training.emitter_final_lr = args.emitter_final_lr
        config.training.emitter_representation = args.emitter_representation
        config.training.emitter_initialization = args.emitter_initialization
        config.training.merge_collinear_emitter_fragments = (
            not args.no_merge_collinear_emitter_fragments
        )
        config.training.associate_emitter_gaussians = not args.no_associate_emitter_gaussians
        config.training.emitter_gaussian_tangent_margin = args.emitter_gaussian_tangent_margin
        config.training.emitter_gaussian_normal_fraction = args.emitter_gaussian_normal_fraction
        config.training.optimize_emitter_centers = args.optimize_emitter_centers
        config.training.optimize_emitter_rotations = args.optimize_emitter_rotations
        config.training.optimize_emitter_scales = not args.freeze_emitter_scales
        config.training.constrain_emitter_scales = not args.unconstrained_emitter_scales
        config.training.emitter_scale_min_ratio = args.emitter_scale_min_ratio
        config.training.emitter_scale_max_ratio = args.emitter_scale_max_ratio
        config.training.freeze_start = args.freeze_start
        config.training.freeze_end = args.freeze_end
        config.training.sh_fade_start = args.sh_fade_start
        config.training.sh_fade_end = args.sh_fade_end
        config.training.training_transport_mode = args.training_transport_mode
        config.losses.light_white = args.light_white
        config.training.light_white_ramp_end = args.light_white_ramp_end
        config.training.checkpoint_every = args.checkpoint_every
        config.adaptive.max_emitters = args.max_emitters
        config.adaptive.enabled = args.enable_emitter_adaptation
        config.render.num_brdf_samples = args.num_brdf_samples
        config.render.num_light_samples = args.num_light_samples
        config.render.num_bounce_samples = args.num_bounce_samples
        config.render.bounce_chunk_size = args.bounce_chunk_size
        config.training.rays_per_step = args.rays_per_step
        config.render.visibility_occluders = args.visibility_occluders
        config.render.trace_origin_offset = args.trace_origin_offset
        config.scene_culling.enabled = not args.no_scene_culling
        config.scene_culling.opacity_threshold = args.scene_cull_opacity
        config.scene_culling.densify_grad_threshold = args.scene_densify_grad
        config.scene_culling.densify_size_threshold = args.scene_split_scale
        config.scene_culling.split_screen_size_threshold = args.scene_split_screen_size
        config.scene_culling.max_gaussians = args.scene_max_gaussians
        config.losses.light_surface = args.light_surface
        config.losses.light_view = args.light_view
        config.render.point_visibility = not args.no_visibility
        config.render.camera_visible_emitters = not args.no_camera_emitters
        config.render.six_axis_emission = not args.symmetric_emission
        if args.emitter_axis_lobe_sharpness <= 0.0:
            raise ValueError("--emitter-axis-lobe-sharpness must be positive")
        config.render.emitter_axis_lobe_sharpness = args.emitter_axis_lobe_sharpness
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
            bounce_samples=args.bounce_samples,
            split=args.split,
            evaluation_seed=args.evaluation_seed,
            evaluation_seed_by_view_order=args.evaluation_seed_by_view_order,
            lpips_net=args.lpips_net,
            training_sh_proxy=args.training_sh_proxy,
            symmetric_emission=args.symmetric_emission,
            denoise_indirect=args.denoise_indirect,
            denoiser_radius=args.denoiser_radius,
            denoiser_sigma_spatial=args.denoiser_sigma_spatial,
            denoiser_sigma_depth=args.denoiser_sigma_depth,
            denoiser_normal_power=args.denoiser_normal_power,
            denoiser_depth_epsilon=args.denoiser_depth_epsilon,
            active_emitter_indices=(
                tuple(args.active_emitter_indices)
                if args.active_emitter_indices is not None
                else None
            ),
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
            symmetric_emission=args.symmetric_emission,
        )


if __name__ == "__main__":
    main()
