from __future__ import annotations

import random
from dataclasses import asdict
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
import torch

from .config import AEGIRConfig
from .emitters import AreaEmitterModel
from .gaussian_scene import GaussianMaterialBackend, planarize_gaussians
from .math_utils import matrix_to_quat, quat_to_matrix

CHECKPOINT_FORMAT = "aegir_shared_gaussian_v13"
SUPPORTED_CHECKPOINT_FORMATS = {
    "aegir_shared_gaussian_v1",
    "aegir_shared_gaussian_v2",
    "aegir_shared_gaussian_v3",
    "aegir_shared_gaussian_v4",
    "aegir_shared_gaussian_v5",
    "aegir_shared_gaussian_v6",
    "aegir_shared_gaussian_v7",
    "aegir_shared_gaussian_v8",
    "aegir_shared_gaussian_v9",
    "aegir_shared_gaussian_v10",
    "aegir_shared_gaussian_v11",
    "aegir_shared_gaussian_v12",
    CHECKPOINT_FORMAT,
}
EXACT_RESUME_FORMATS = {CHECKPOINT_FORMAT}


def emitter_state_dict(emitters: AreaEmitterModel) -> dict[str, torch.Tensor]:
    state = emitters.state()
    return {
        "centers": state.centers.cpu(),
        "quaternions": state.quaternions.cpu(),
        "scales": state.scales.cpu(),
        "colors": state.colors.cpu(),
    }


def module_state_dict_cpu(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _tree_to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _tree_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_tree_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_tree_to_cpu(item) for item in value)
    return value


def optimizer_state_dict_cpu(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    return _tree_to_cpu(optimizer.state_dict())


def capture_rng_state(device: str | torch.device) -> dict[str, Any]:
    device = torch.device(device)
    cuda_states = []
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("cannot capture CUDA RNG state because CUDA is unavailable")
        cuda_states = [state.cpu().clone() for state in torch.cuda.get_rng_state_all()]
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().cpu().clone(),
        "torch_cuda_all": cuda_states,
        "cuda_device_count": torch.cuda.device_count() if device.type == "cuda" else 0,
    }


def restore_rng_state(state: dict[str, Any], device: str | torch.device) -> None:
    device = torch.device(device)
    random.setstate(state["python"])
    np.random.set_state(tuple(state["numpy"]))
    torch.set_rng_state(state["torch_cpu"].cpu())
    cuda_states = state.get("torch_cuda_all", [])
    if device.type == "cuda":
        expected = int(state.get("cuda_device_count", len(cuda_states)))
        current = torch.cuda.device_count()
        if expected != current or len(cuda_states) != current:
            raise RuntimeError(
                "exact resume requires the same visible CUDA device count "
                f"(checkpoint={expected}, current={current})"
            )
        torch.cuda.set_rng_state_all([value.cpu() for value in cuda_states])


def require_exact_resume_state(payload: dict[str, Any], path: str | Path | None = None) -> dict[str, Any]:
    location = f" {path}" if path is not None else ""
    if payload.get("format") not in EXACT_RESUME_FORMATS or "training_state" not in payload:
        raise ValueError(
            f"checkpoint{location} has no exact-resume optimizer/RNG state; "
            "only v13 planar-emitter checkpoints are exactly resumable; "
            "v1-v12 are inference-only or explicit warm starts"
        )
    training_state = payload["training_state"]
    required = {
        "next_step",
        "optimizers",
        "rng",
        "bounds",
        "surface_points",
        "adaptation",
        "sampling",
    }
    missing = sorted(required.difference(training_state))
    if missing:
        raise ValueError(f"checkpoint{location} is missing exact-resume fields: {missing}")
    if payload.get("format") == CHECKPOINT_FORMAT:
        emitter_optimization = training_state.get("emitter_optimization")
        if emitter_optimization is None or "scale_anchors" not in emitter_optimization:
            raise ValueError(
                f"checkpoint{location} is missing constrained-emitter resume state"
            )
        refinement = training_state.get("scene_culling")
        if refinement is None:
            raise ValueError(f"checkpoint{location} is missing scene-refinement resume state")
        refinement_required = {
            "state",
            "max_2d_size",
            "xy_grad_norm",
            "xy_grad_vis_counts",
            "refinement_observations",
        }
        missing_refinement = sorted(refinement_required.difference(refinement))
        if missing_refinement:
            raise ValueError(
                f"checkpoint{location} is missing scene-refinement fields: "
                f"{missing_refinement}"
            )
    return training_state


def _legacy_planar_emitter_state(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Collapse a legacy ellipsoid and angular lobe into one planar ellipse."""

    scales = payload["scales"]
    if scales.shape[-1] == 2:
        return {
            "centers": payload["centers"],
            "quaternions": payload["quaternions"],
            "scales": scales,
            "colors": payload["colors"],
        }
    if scales.shape[-1] != 3:
        raise ValueError(f"unsupported emitter scale shape {tuple(scales.shape)}")

    rotations = quat_to_matrix(payload["quaternions"].float())
    order = scales.argsort(dim=-1, descending=True)
    rotations = rotations.gather(-1, order[:, None, :].expand(-1, 3, -1))
    planar_scales = scales.gather(-1, order)[..., :2]
    normal_axis = order[..., 2]

    axis_gains = payload.get("axis_gains")
    front_sign = torch.ones_like(normal_axis, dtype=rotations.dtype)
    if axis_gains is not None:
        positive = axis_gains.gather(-1, (2 * normal_axis).unsqueeze(-1)).squeeze(-1)
        negative = axis_gains.gather(-1, (2 * normal_axis + 1).unsqueeze(-1)).squeeze(-1)
        front_sign = torch.where(negative > positive, -front_sign, front_sign)
    # Flip one tangent with the normal so the basis remains right-handed.
    rotations[..., :, 0] *= front_sign[:, None]
    rotations[..., :, 2] *= front_sign[:, None]

    colors = payload["colors"].float()
    spreads = payload.get("spreads")
    powers = payload.get("falloff_powers")
    if spreads is not None and powers is not None:
        selected_spread = spreads.gather(-1, normal_axis.unsqueeze(-1)).clamp_min(1.0e-6)
        colors = colors * torch.exp(-(selected_spread.reciprocal().square()).pow(powers))
    if axis_gains is not None:
        gain_index = 2 * normal_axis + (front_sign < 0).long()
        colors = colors * axis_gains.gather(-1, gain_index.unsqueeze(-1))

    return {
        "centers": payload["centers"],
        "quaternions": matrix_to_quat(rotations).to(payload["quaternions"]),
        "scales": planar_scales,
        "colors": colors.to(payload["colors"]),
    }


def emitters_from_state(payload: dict[str, Any], device: str) -> AreaEmitterModel:
    planar = _legacy_planar_emitter_state(payload)
    return AreaEmitterModel(
        centers=planar["centers"].to(device),
        scales=planar["scales"].to(device),
        colors=planar["colors"].to(device),
        quaternions=planar["quaternions"].to(device),
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
    training_state: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "step": int(step),
        "source_dnsplatter_checkpoint": scene.source_checkpoint,
        "gaussian_count": scene.gaussian_count,
        "scene_state": module_state_dict_cpu(scene),
        "emitter_state": emitter_state_dict(emitters),
        "emitter_raw_state": module_state_dict_cpu(emitters),
        "config": asdict(config),
        "losses": losses,
        "dataset": dataset_meta,
    }
    if initial_emitter_state is not None:
        payload["initial_emitter_state"] = {
            key: value.detach().cpu() for key, value in initial_emitter_state.items()
        }
    if training_state is not None:
        payload["training_state"] = _tree_to_cpu(training_state)
    with NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
    try:
        torch.save(payload, temporary_path)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


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
    if payload.get("format") == CHECKPOINT_FORMAT and "emitter_raw_state" in payload:
        raw_state = dict(payload["emitter_raw_state"])
        emitters.load_state_dict(raw_state, strict=True)
    return scene, emitters, payload
