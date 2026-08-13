from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

GAUSSIAN_KEYS = {
    "features_dc": "features_dc_raw",
    "features_rest": "features_rest_raw",
    "means": "means",
    "normals": "normals",
    "opacities": "opacities_logits",
    "quats": "quats",
    "scales": "log_scales",
}


def load_dnsplatter_gaussians(path: str | Path, max_gaussians: int | None = None) -> dict[str, Any]:
    """Load either an exported Gaussian `.pt` or a Nerfstudio DN-Splatter checkpoint."""

    path = Path(path)
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if "pipeline" in obj:
        out = _load_nerfstudio_checkpoint(obj, path)
    elif "means" in obj:
        out = obj
    else:
        raise ValueError(f"{path} is neither an exported Gaussian file nor a Nerfstudio checkpoint")

    if max_gaussians is not None:
        for key, value in list(out.items()):
            if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] >= max_gaussians:
                out[key] = value[:max_gaussians].clone()
    return out


def _load_nerfstudio_checkpoint(obj: dict[str, Any], path: Path) -> dict[str, Any]:
    state = obj["pipeline"]
    out: dict[str, Any] = {
        "source_checkpoint": str(path),
        "step": int(obj.get("step", -1)),
    }
    prefix = "_model.gauss_params."
    for source_key, target_key in GAUSSIAN_KEYS.items():
        key = prefix + source_key
        if key not in state:
            raise KeyError(f"Missing {key} in {path}")
        out[target_key] = state[key].detach().cpu().clone()

    out["scales"] = out["log_scales"].exp()
    out["opacities"] = out["opacities_logits"].sigmoid()
    out["features_dc"] = out["features_dc_raw"].sigmoid()
    return out


def load_nerfstudio_frame(
    nerfstudio_dir: str | Path,
    frame_index: int = 0,
    image_size: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float | int | str], dict[str, Any]]:
    """Load one Nerfstudio camera frame and resized RGB target."""

    nerfstudio_dir = Path(nerfstudio_dir)
    meta_path = nerfstudio_dir / "transforms.json"
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    frames = meta.get("frames", [])
    if not frames:
        raise ValueError(f"{meta_path} has no frames")
    frame = frames[frame_index]

    image_path = nerfstudio_dir / frame["file_path"]
    image = _image_to_tensor(image_path)
    source_h, source_w = image.shape[:2]
    if image_size is not None:
        image = _resize_image(image, image_size)
    height, width = image.shape[:2]
    sx = width / float(source_w)
    sy = height / float(source_h)

    fx = float(frame.get("fl_x", meta.get("fl_x")))
    fy = float(frame.get("fl_y", meta.get("fl_y", fx)))
    cx = float(frame.get("cx", meta.get("cx", source_w * 0.5)))
    cy = float(frame.get("cy", meta.get("cy", source_h * 0.5)))
    camera = {
        "c2w": torch.as_tensor(frame["transform_matrix"], dtype=torch.float32),
        "fx": fx * sx,
        "fy": fy * sy,
        "cx": cx * sx,
        "cy": cy * sy,
        "width": width,
        "height": height,
        "camera_convention": "auto",
        "depth_convention": _resolve_depth_convention(meta, frame),
        "depth_mode": _resolve_depth_mode(meta, frame),
    }
    return image, camera, frame


def _resolve_depth_convention(meta: dict[str, Any], frame: dict[str, Any]) -> str:
    convention = frame.get("aegir_depth_convention", meta.get("aegir_depth_convention"))
    if convention is not None:
        return str(convention)
    note = str(meta.get("aegir_note", ""))
    if "FIPT" in note:
        return "fipt"
    return "opencv"


def _resolve_depth_mode(meta: dict[str, Any], frame: dict[str, Any]) -> str:
    mode = frame.get("aegir_depth_mode", meta.get("aegir_depth_mode"))
    if mode is not None:
        return str(mode)
    note = str(meta.get("aegir_note", ""))
    if "FIPT" in note:
        return "ray"
    return "z"


def load_priors_from_sample(sample_path: str | Path, image_size: tuple[int, int]) -> dict[str, torch.Tensor]:
    sample = torch.load(sample_path, map_location="cpu", weights_only=False)
    priors: dict[str, torch.Tensor] = {}
    for key in ("albedo", "roughness", "metallic"):
        prior_key = f"prior_{key}"
        if prior_key in sample:
            value = sample[prior_key].float()
            if value.shape[:2] != image_size:
                value = _resize_image_to(value, image_size)
            priors[key] = value
    return priors


def _image_to_tensor(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    return torch.from_numpy(np.asarray(image).copy()).float().div(255.0).contiguous()


def _resize_image(x: torch.Tensor, size: int) -> torch.Tensor:
    return _resize_image_to(x, (size, size))


def _resize_image_to(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if x.ndim == 2:
        x = x[..., None]
    y = x.permute(2, 0, 1).unsqueeze(0)
    y = F.interpolate(y, size=size, mode="bilinear", align_corners=False)
    return y.squeeze(0).permute(1, 2, 0).contiguous()
