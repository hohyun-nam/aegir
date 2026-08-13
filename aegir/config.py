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
    tonemap_exposure: float = 1.0
    point_visibility: bool = True
    visibility_occluders: int = 2048
    visibility_radius: float = 0.035
    visibility_chunk_size: int = 256
    indirect_bounce: bool = True
    num_bounce_samples: int = 4
    bounce_proxy_points: int = 1024
    bounce_radius_fraction: float = 0.01
    bounce_max_distance_fraction: float = 0.5
    bounce_strength: float = 0.25
    bounce_chunk_size: int = 256


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
    light_white: float = 0.1
    light_bounds: float = 0.5
    light_surface: float = 10.0
    light_view: float = 1.0
    surface_distance_fraction: float = 0.003
    view_depth_margin_fraction: float = 0.001


@dataclass(slots=True)
class TrainingConfig:
    steps: int = 20_000
    geometry_lr: float = 1.0e-5
    material_lr: float = 1.0e-3
    emitter_lr: float = 1.0e-3
    emitter_final_lr: float = 1.0e-4
    freeze_start: int = 5_000
    freeze_end: int = 15_000
    adaptive_every: int = 1_000
    rays_per_step: int = 1_024
    surface_anchor_points: int = 65_536
    emitter_init_score: str = "hdr"
    emitter_init_quantile: float = 0.995
    emitter_persistence_min_views: int = 3
    emitter_persistence_min_fraction: float = 0.05
    emitter_persistence_voxel_fraction: float = 0.005
    emitter_min_scale_fraction: float = 0.02
    grad_clip: float = 1.0
    checkpoint_every: int = 1_000


@dataclass(slots=True)
class AdaptiveConfig:
    prune_energy: float = 0.1
    split_energy: float = 5.0
    split_scale_fraction: float = 0.2
    min_emitters: int = 1
    max_emitters: int = 16


@dataclass(slots=True)
class AEGIRConfig:
    raster: RasterConfig = field(default_factory=RasterConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    losses: LossWeights = field(default_factory=LossWeights)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    adaptive: AdaptiveConfig = field(default_factory=AdaptiveConfig)


def config_from_dict(payload: dict[str, Any]) -> AEGIRConfig:
    def known_fields(config_type: type, values: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in values.items() if key in config_type.__dataclass_fields__}

    return AEGIRConfig(
        raster=RasterConfig(**known_fields(RasterConfig, payload.get("raster", {}))),
        render=RenderConfig(**known_fields(RenderConfig, payload.get("render", {}))),
        losses=LossWeights(**known_fields(LossWeights, payload.get("losses", {}))),
        training=TrainingConfig(**known_fields(TrainingConfig, payload.get("training", {}))),
        adaptive=AdaptiveConfig(**known_fields(AdaptiveConfig, payload.get("adaptive", {}))),
    )
