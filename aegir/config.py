from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RasterConfig:
    image_size: int = 128
    alpha_threshold: float = 1.0e-3
    opacity_threshold: float = 1.0e-4
    rasterize_mode: str = "classic"
    bounds_quantile: float = 0.005


@dataclass(slots=True)
class RenderConfig:
    num_brdf_samples: int = 128
    num_light_samples: int = 128
    eps: float = 1.0e-6
    # Normalized-coordinate values. At runtime bias is multiplied by the scene
    # bounds diagonal and steepness is divided by it.
    visibility_bias: float = 1.0e-3
    visibility_steepness: float = 80.0
    trace_alpha_min: float = 1.0 / 255.0
    trace_transmittance_min: float = 0.05
    # Official IRGS shifts surface-to-light ray origins by light_t_min=0.05.
    # Converted AEGIR datasets use metres, so keep this as an absolute metric
    # self-intersection offset; it is not Equation 3's endpoint bias.
    trace_origin_offset: float = 0.05
    tonemap_exposure: float = 1.0
    point_visibility: bool = True
    # Composite learned emitter self-emission when analytic support is hit
    # before the camera-visible Gaussian surface.
    camera_visible_emitters: bool = True
    # Blend learned local +X/-X/+Y/-Y/+Z/-Z gains into the emitter's
    # full-sphere directional radiance. Disable only for the symmetric ablation.
    six_axis_emission: bool = True
    emitter_axis_lobe_sharpness: float = 12.0
    visibility_occluders: int = 2048
    visibility_radius: float = 0.035
    visibility_chunk_size: int = 256
    indirect_bounce: bool = True
    # Emitter next-event samples at the first secondary Gaussian hit. Training
    # cross-fades from the frozen DN SH proxy to this explicit estimator.
    num_bounce_samples: int = 1
    # Execution-only batching control. The default fits the complete 2^16
    # first-hit NEE batch in one OptiX launch under the 24 GiB job limit.
    bounce_chunk_size: int = 65_536


@dataclass(slots=True)
class LossWeights:
    focus_lambda: float = 5.0
    dssim_lambda: float = 0.2
    prior_albedo: float = 1.0
    prior_roughness: float = 1.0
    prior_metallic: float = 1.0
    tv_lambda: float = 1.0
    tv_albedo: float = 0.5
    tv_roughness: float = 0.25
    tv_metallic: float = 0.25
    tv_beta: float = 1.0
    light_area: float = 0.5
    # Blender ablations showed that 0.1 forces genuinely warm emitters toward
    # gray and degrades reconstruction. Keep color neutrality opt-in.
    light_white: float = 0.0
    light_bounds: float = 0.5
    # Non-paper stabilization priors. Keep disabled for paper-faithful runs;
    # enable explicitly if emitters drift away from surfaces/observed views.
    light_surface: float = 0.0
    light_view: float = 0.0
    surface_distance_fraction: float = 0.003
    view_depth_margin_fraction: float = 0.001


@dataclass(slots=True)
class ColorConfig:
    hdr_loss: str = "log_l1"
    ldr_fallback_tone_mapper: str = "reinhard"
    ldr_fallback_transfer_function: str = "srgb"
    ldr_exposure_init: float = 1.0
    ldr_exposure_lr: float = 1.0e-3
    ldr_saturation_threshold: float = 0.98


@dataclass(slots=True)
class TrainingConfig:
    steps: int = 20_000
    geometry_lr: float = 1.0e-5
    # Keep the faster albedo optimizer group for both SH DC and neutral seeds.
    # Roughness and metallic continue to use ``material_lr``.
    albedo_lr: float = 3.0e-3
    material_lr: float = 1.0e-3
    emitter_lr: float = 1.0e-3
    emitter_final_lr: float = 1.0e-4
    # Keep planar emitters as the compatibility default. The ellipsoid option
    # restores the pre-v13 three-axis analytic emitter without changing the
    # underlying 2D Gaussian scene representation.
    emitter_representation: str = "planar_ellipse_2d"
    # Keep the initialized light proxy attached to the detected fixture. Scale
    # may shrink, but cannot grow beyond its initialized extent by default.
    optimize_emitter_centers: bool = False
    optimize_emitter_rotations: bool = False
    optimize_emitter_scales: bool = True
    constrain_emitter_scales: bool = True
    emitter_scale_min_ratio: float = 0.1
    emitter_scale_max_ratio: float = 1.0
    freeze_start: int = 5_000
    freeze_end: int = 15_000
    sh_fade_start: int = 5_000
    sh_fade_end: int = 10_000
    # Zero applies losses.light_white from the first step. A positive step uses
    # a smoothstep ramp from zero at step 0 to that target at this step.
    light_white_ramp_end: int = 0
    # ``hybrid`` fades from frozen DN SH to explicit secondary NEE.
    # Ablations persist their transport choice in the checkpoint instead of
    # monkey-patching the trainer at runtime.
    training_transport_mode: str = "hybrid"
    adaptive_every: int = 1_000
    # Historical name kept for checkpoint/CLI compatibility. This is the
    # number of primary surface points, not the total Monte Carlo ray count.
    # 256 primary points * (128 BRDF + 128 light samples) = 2^16 transport rays.
    rays_per_step: int = 256
    surface_anchor_points: int = 65_536
    # ``sh_dc`` preserves the historical appearance-based seed. ``neutral``
    # starts every Gaussian from a scene-linear gray reflectance instead, so
    # source-light shadows/highlights are not copied into physical albedo.
    albedo_initialization: str = "sh_dc"
    neutral_albedo: float = 0.5
    emitter_init_score: str = "hdr"
    emitter_init_quantile: float = 0.995
    emitter_persistence_min_views: int = 3
    emitter_persistence_min_fraction: float = 0.05
    emitter_persistence_voxel_fraction: float = 0.005
    # Keep the PCA seed floor small enough to retain thin light fixtures.
    # This is a radius fraction of the scene-bounds diagonal.
    emitter_min_scale_fraction: float = 0.002
    grad_clip: float = 1.0
    checkpoint_every: int = 1_000


@dataclass(slots=True)
class AdaptiveConfig:
    # Split/prune changes emitter identity and geometry. It is disabled in the
    # current light-separation setup; enable it explicitly for the historical
    # AEGIR adaptive-emitter baseline.
    enabled: bool = False
    prune_energy: float = 0.1
    split_energy: float = 5.0
    split_scale_fraction: float = 0.2
    min_emitters: int = 1
    max_emitters: int = 16


@dataclass(slots=True)
class SceneCullingConfig:
    """DN-style split/duplicate refinement plus conservative opacity culling."""

    enabled: bool = True
    refine_every: int = 100
    warmup_length: int = 500
    opacity_threshold: float = 0.005
    densify_grad_threshold: float = 8.0e-4
    densify_size_threshold: float = 0.01
    split_screen_size_threshold: float = 0.05
    n_split_samples: int = 2
    split_scale_divisor: float = 1.6
    stop_split_at: int = 15_000
    screen_size_stop: int = 4_000
    max_gaussians: int = 500_000
    # Legacy v6/v7 fields retained only for config deserialization. Large
    # Gaussians are no longer deleted by either threshold.
    scale_threshold: float = 0.5
    scale_start: int = 3_000
    screen_size_threshold: float = 0.15


@dataclass(slots=True)
class AEGIRConfig:
    raster: RasterConfig = field(default_factory=RasterConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    losses: LossWeights = field(default_factory=LossWeights)
    color: ColorConfig = field(default_factory=ColorConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    adaptive: AdaptiveConfig = field(default_factory=AdaptiveConfig)
    scene_culling: SceneCullingConfig = field(default_factory=SceneCullingConfig)


def config_from_dict(payload: dict[str, Any]) -> AEGIRConfig:
    def known_fields(config_type: type, values: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in values.items() if key in config_type.__dataclass_fields__}

    # Checkpoints written before scene culling existed must resume with the old
    # behavior. Fresh AEGIRConfig instances enable DN-style culling by default.
    legacy_scene_culling = "scene_culling" not in payload
    scene_culling_values = payload.get("scene_culling", {})
    if legacy_scene_culling:
        scene_culling_values = {"enabled": False}
    training_values = dict(payload.get("training", {}))
    # Old checkpoints coupled the SH fade to the material freeze schedule.
    training_values.setdefault("sh_fade_start", training_values.get("freeze_start", 5_000))
    training_values.setdefault("sh_fade_end", training_values.get("freeze_end", 15_000))
    return AEGIRConfig(
        raster=RasterConfig(**known_fields(RasterConfig, payload.get("raster", {}))),
        render=RenderConfig(**known_fields(RenderConfig, payload.get("render", {}))),
        losses=LossWeights(**known_fields(LossWeights, payload.get("losses", {}))),
        color=ColorConfig(**known_fields(ColorConfig, payload.get("color", {}))),
        training=TrainingConfig(**known_fields(TrainingConfig, training_values)),
        adaptive=AdaptiveConfig(**known_fields(AdaptiveConfig, payload.get("adaptive", {}))),
        scene_culling=SceneCullingConfig(
            **known_fields(SceneCullingConfig, scene_culling_values)
        ),
    )
