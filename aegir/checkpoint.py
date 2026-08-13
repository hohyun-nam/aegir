from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .config import AEGIRConfig
from .emitters import AreaEmitterModel
from .gaussian_scene import GaussianMaterialBackend, planarize_gaussians

CHECKPOINT_FORMAT = "aegir_shared_gaussian_v3"
SUPPORTED_CHECKPOINT_FORMATS = {
    "aegir_shared_gaussian_v1",
    "aegir_shared_gaussian_v2",
    CHECKPOINT_FORMAT,
}


def emitter_state_dict(emitters: AreaEmitterModel) -> dict[str, torch.Tensor]:
    state = emitters.state()
    return {
        "centers": state.centers.cpu(),
        "quaternions": state.quaternions.cpu(),
        "scales": state.scales.cpu(),
        "colors": state.colors.cpu(),
        "spreads": state.spreads.cpu(),
        "falloff_powers": state.falloff_powers.cpu(),
    }


def emitters_from_state(payload: dict[str, torch.Tensor], device: str) -> AreaEmitterModel:
    return AreaEmitterModel(
        centers=payload["centers"].to(device),
        scales=payload["scales"].to(device),
        colors=payload["colors"].to(device),
        quaternions=payload["quaternions"].to(device),
        spreads=payload["spreads"].to(device),
        falloff_powers=payload["falloff_powers"].to(device),
    )


def load_legacy_emitters(path: str | Path, device: str) -> AreaEmitterModel:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "emitter_state" in payload:
        return emitters_from_state(payload["emitter_state"], device)
    required = (
        "emitter_centers",
        "emitter_scales",
        "emitter_colors",
        "emitter_quaternions",
        "emitter_spreads",
        "emitter_falloff_powers",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(f"{path} is missing emitter keys: {missing}")
    state = {
        "centers": payload["emitter_centers"],
        "scales": payload["emitter_scales"],
        "colors": payload["emitter_colors"],
        "quaternions": payload["emitter_quaternions"],
        "spreads": payload["emitter_spreads"],
        "falloff_powers": payload["emitter_falloff_powers"],
    }
    return emitters_from_state(state, device)


def save_checkpoint(
    path: str | Path,
    scene: GaussianMaterialBackend,
    emitters: AreaEmitterModel,
    config: AEGIRConfig,
    step: int,
    losses: list[dict[str, float]],
    dataset_meta: dict[str, Any],
    initial_emitter_state: dict[str, torch.Tensor] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "step": int(step),
        "source_dnsplatter_checkpoint": scene.source_checkpoint,
        "gaussian_count": scene.gaussian_count,
        "scene_state": {key: value.detach().cpu() for key, value in scene.state_dict().items()},
        "emitter_state": emitter_state_dict(emitters),
        "config": asdict(config),
        "losses": losses,
        "dataset": dataset_meta,
    }
    if initial_emitter_state is not None:
        payload["initial_emitter_state"] = {
            key: value.detach().cpu() for key, value in initial_emitter_state.items()
        }
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    device: str,
) -> tuple[GaussianMaterialBackend, AreaEmitterModel, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") not in SUPPORTED_CHECKPOINT_FORMATS:
        raise ValueError(f"{path} is not a shared-Gaussian AEGIR checkpoint")
    scene_state = dict(payload["scene_state"])
    legacy_log_scales = scene_state["log_scales"]
    if legacy_log_scales.shape[-1] == 3:
        quats, scales = planarize_gaussians(scene_state["raw_quats"].float(), legacy_log_scales.float().exp())
        scene_state["raw_quats"] = quats.to(scene_state["raw_quats"])
        scene_state["log_scales"] = scales.log().to(legacy_log_scales)
    scene = GaussianMaterialBackend.from_state(payload["source_dnsplatter_checkpoint"], scene_state).to(
        device
    )
    emitters = emitters_from_state(payload["emitter_state"], device)
    return scene, emitters, payload
