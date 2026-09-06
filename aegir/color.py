from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from .math_utils import linear_to_srgb, srgb_to_linear, tonemap_reinhard

COLOR_METADATA_FORMAT = "aegir-color-v1"
SUPPORTED_TONE_MAPPERS = {"reinhard", "clip"}
SUPPORTED_TRANSFER_FUNCTIONS = {"srgb", "gamma22", "linear"}
SUPPORTED_EXPOSURE_POLICIES = {"fixed_scene", "fixed_per_frame", "learned_scene"}


def resolve_observation_policy(
    metadata: dict[str, Any],
    frames: list[dict[str, Any]],
) -> dict[str, Any]:
    """Resolve one strict observation domain for a complete scene."""

    if not frames:
        raise ValueError("cannot resolve a color policy for an empty scene")
    hdr_flags = ["radiance_hdr_file_path" in frame for frame in frames]
    if any(hdr_flags) and not all(hdr_flags):
        raise ValueError("scene mixes HDR and LDR-only frames; use one observation domain per scene")

    declared = metadata.get("aegir_color")
    if declared is not None and declared.get("format") != COLOR_METADATA_FORMAT:
        raise ValueError("unsupported or missing aegir_color metadata format")
    declared_domain = None if declared is None else declared.get("observation_domain")

    if all(hdr_flags):
        if declared_domain not in {None, "linear_hdr"}:
            raise ValueError("scene has complete HDR observations but metadata declares an LDR domain")
        return {
            "format": COLOR_METADATA_FORMAT,
            "domain": "linear_hdr",
            "hdr_encoding": "scene_linear_rgb",
            "loss_domain": "log1p_scene_linear_rgb",
            "source": (
                "dataset_metadata"
                if declared_domain == "linear_hdr"
                else "legacy_inferred_from_complete_radiance_hdr_paths"
            ),
        }

    if declared_domain == "linear_hdr":
        raise ValueError("metadata declares linear HDR but the scene has no HDR observations")
    target_png = {} if declared is None else dict(declared.get("target_png", {}))
    if not target_png:
        target_png = {
            "tone_mapper": "reinhard",
            "transfer_function": "srgb",
            "exposure_policy": "learned_scene",
            "exposure_scale": 1.0,
            "source": "explicit_aegir_fallback_for_uncharacterized_ldr",
        }
    tone_mapper = target_png.get("tone_mapper")
    transfer_function = target_png.get("transfer_function")
    exposure_policy = target_png.get("exposure_policy")
    if tone_mapper not in SUPPORTED_TONE_MAPPERS:
        raise ValueError(f"unsupported LDR tone mapper {tone_mapper!r}")
    if transfer_function not in SUPPORTED_TRANSFER_FUNCTIONS:
        raise ValueError(f"unsupported LDR transfer function {transfer_function!r}")
    if exposure_policy not in SUPPORTED_EXPOSURE_POLICIES:
        raise ValueError(f"unsupported LDR exposure policy {exposure_policy!r}")
    exposure_scale = float(target_png.get("exposure_scale", 1.0))
    if not math.isfinite(exposure_scale) or exposure_scale <= 0.0:
        raise ValueError("LDR exposure_scale must be finite and positive")
    if exposure_policy == "fixed_per_frame":
        missing = [index for index, frame in enumerate(frames) if "ldr_exposure_scale" not in frame]
        if missing:
            raise ValueError(f"fixed_per_frame LDR metadata is missing frame exposure at index {missing[0]}")
    return {
        "format": COLOR_METADATA_FORMAT,
        "domain": "ldr",
        "tone_mapper": tone_mapper,
        "transfer_function": transfer_function,
        "exposure_policy": exposure_policy,
        "exposure_scale": exposure_scale,
        "target_encoding": f"{transfer_function}_ldr",
        "source": target_png.get("source", "dataset_metadata"),
    }


def frame_exposure_scale(policy: dict[str, Any], source_frame: dict[str, Any]) -> float:
    if policy["domain"] != "ldr":
        return 1.0
    if policy["exposure_policy"] == "fixed_per_frame":
        value = float(source_frame["ldr_exposure_scale"])
    else:
        value = float(policy["exposure_scale"])
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("frame LDR exposure must be finite and positive")
    return value


def configure_ldr_fallback(
    policy: dict[str, Any],
    *,
    tone_mapper: str,
    transfer_function: str,
) -> dict[str, Any]:
    """Apply run config only to explicitly uncharacterized LDR observations."""

    configured = dict(policy)
    is_fallback = configured.get("domain") == "ldr" and "fallback" in configured.get("source", "")
    if not is_fallback:
        return configured
    if tone_mapper not in SUPPORTED_TONE_MAPPERS:
        raise ValueError(f"unsupported configured fallback tone mapper {tone_mapper!r}")
    if transfer_function not in SUPPORTED_TRANSFER_FUNCTIONS:
        raise ValueError(f"unsupported configured fallback transfer function {transfer_function!r}")
    configured["tone_mapper"] = tone_mapper
    configured["transfer_function"] = transfer_function
    configured["target_encoding"] = f"{transfer_function}_ldr"
    return configured


def apply_ldr_pipeline(
    linear_hdr: Tensor,
    *,
    exposure: Tensor | float,
    tone_mapper: str,
    transfer_function: str,
) -> Tensor:
    exposure_tensor = torch.as_tensor(exposure, device=linear_hdr.device, dtype=linear_hdr.dtype)
    exposed = linear_hdr.clamp_min(0.0) * exposure_tensor
    if tone_mapper == "reinhard":
        linear_ldr = tonemap_reinhard(exposed)
    elif tone_mapper == "clip":
        linear_ldr = exposed.clamp(0.0, 1.0)
    else:
        raise ValueError(f"unsupported tone mapper {tone_mapper!r}")

    if transfer_function == "srgb":
        return linear_to_srgb(linear_ldr)
    if transfer_function == "gamma22":
        return linear_ldr.clamp(0.0, 1.0).pow(1.0 / 2.2)
    if transfer_function == "linear":
        return linear_ldr.clamp(0.0, 1.0)
    raise ValueError(f"unsupported transfer function {transfer_function!r}")


def hdr_log_image(linear_hdr: Tensor) -> Tensor:
    return torch.log1p(linear_hdr.clamp_min(0.0))


def display_srgb(linear_hdr: Tensor, exposure: Tensor | float = 1.0) -> Tensor:
    return apply_ldr_pipeline(
        linear_hdr,
        exposure=exposure,
        tone_mapper="reinhard",
        transfer_function="srgb",
    )


def decode_prior_albedo(value: Tensor, payload: dict[str, Any]) -> Tensor:
    """Return DiffusionRenderer base color as linear BRDF reflectance."""

    output_space = payload.get("prior_albedo_space")
    if output_space == "linear_reflectance":
        return value.clamp(0.0, 1.0)
    legacy_source = payload.get("basecolor_space")
    if legacy_source in {"model", "srgb"}:
        return srgb_to_linear(value.clamp(0.0, 1.0))
    raise ValueError(
        "material prior must declare prior_albedo_space='linear_reflectance'; "
        "legacy DiffusionRenderer priors require basecolor_space='model' or 'srgb'"
    )
