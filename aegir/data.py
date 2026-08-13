from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor


@dataclass(slots=True)
class FrameData:
    index: int
    target: Tensor
    mask: Tensor
    camera: dict[str, Any]
    priors: dict[str, Tensor]
    radiance_hdr: Tensor | None
    source_frame: dict[str, Any]


def infer_dataparser_transform_path(checkpoint: Path) -> Path:
    path = checkpoint.parent.parent / "dataparser_transforms.json"
    if not path.exists():
        raise FileNotFoundError(f"cannot infer dataparser transform next to {checkpoint}")
    return path


def load_dataparser_transform(path: Path) -> tuple[Tensor, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    matrix = torch.as_tensor(payload["transform"], dtype=torch.float32)
    if matrix.shape != (3, 4):
        raise ValueError(f"expected 3x4 transform in {path}, got {tuple(matrix.shape)}")
    return matrix, float(payload.get("scale", 1.0))


def saved_pose_to_model_transform(
    dataparser_transform: tuple[Tensor, float],
    dataset_meta: dict[str, Any],
) -> tuple[Tensor, float]:
    matrix, scale = dataparser_transform
    applied = dataset_meta.get("applied_transform")
    if applied is None:
        return matrix, scale
    applied = torch.as_tensor(applied, dtype=matrix.dtype)
    dataparser_4x4 = torch.eye(4, dtype=matrix.dtype)
    dataparser_4x4[:3] = matrix
    applied_4x4 = torch.eye(4, dtype=matrix.dtype)
    applied_4x4[:3] = applied
    saved_to_model = dataparser_4x4 @ torch.linalg.inv(applied_4x4)
    return saved_to_model[:3], scale


def transform_camera(camera: dict[str, Any], world_to_model: tuple[Tensor, float]) -> dict[str, Any]:
    matrix, scale = world_to_model
    c2w = camera["c2w"].clone()
    matrix = matrix.to(c2w)
    rotation = matrix[:, :3]
    c2w[:3, :3] = rotation @ c2w[:3, :3]
    c2w[:3, 3] = (rotation @ c2w[:3, 3] + matrix[:, 3]) * scale
    return {**camera, "c2w": c2w}


def resize_map(value: Tensor, size: tuple[int, int]) -> Tensor:
    if value.ndim == 2:
        value = value[..., None]
    if value.shape[:2] == size:
        return value.contiguous()
    mode = "bilinear"
    result = F.interpolate(value.permute(2, 0, 1)[None], size=size, mode=mode, align_corners=False)
    return result[0].permute(1, 2, 0).contiguous()


def resize_mask(value: Tensor, size: tuple[int, int]) -> Tensor:
    if value.ndim == 2:
        value = value[..., None]
    if value.shape[:2] == size:
        return value.bool().contiguous()
    result = F.interpolate(
        value.float().permute(2, 0, 1)[None],
        size=size,
        mode="nearest",
    )
    return result[0].permute(1, 2, 0).bool().contiguous()


def _normalized_relative_path(value: str | Path) -> str:
    normalized = Path(str(value).replace("\\", "/")).as_posix()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def select_frame_indices(
    metadata: dict[str, Any],
    frames: list[dict[str, Any]],
    split: str,
) -> list[int]:
    """Select a dataset split while retaining each frame's original index."""
    if split not in {"train", "val", "test", "all"}:
        raise ValueError(f"unsupported split {split!r}; expected train, val, test, or all")
    if split == "all":
        return list(range(len(frames)))

    split_key = f"{split}_filenames"
    if split == "test" and split_key not in metadata and "val_filenames" in metadata:
        split_key = "val_filenames"
    filename_keys = {"train_filenames", "val_filenames", "test_filenames"}
    if split_key in metadata:
        requested = {_normalized_relative_path(path) for path in metadata[split_key]}
        frame_paths = {
            _normalized_relative_path(frame["file_path"]): index for index, frame in enumerate(frames)
        }
        unknown = sorted(requested.difference(frame_paths))
        if unknown:
            raise ValueError(f"{split_key} references {len(unknown)} unknown frame(s), e.g. {unknown[0]!r}")
        return [
            index
            for index, frame in enumerate(frames)
            if _normalized_relative_path(frame["file_path"]) in requested
        ]
    if any(key in metadata for key in filename_keys):
        raise ValueError(f"dataset has split metadata but no {split_key}")

    source_splits = {str(frame.get("source_split", "")).lower() for frame in frames}
    source_splits.discard("")
    if source_splits:
        requested_split = "val" if split == "test" and "test" not in source_splits else split
        return [
            index
            for index, frame in enumerate(frames)
            if str(frame.get("source_split", "")).lower() == requested_split
        ]
    if split == "train":
        return list(range(len(frames)))
    raise ValueError(
        f"dataset has no split metadata for {split!r}; "
        "define split filenames or request split='all' explicitly"
    )


class MultiViewSceneDataset:
    """RGB cameras and 2D material priors; no trainable per-view G-buffers."""

    def __init__(
        self,
        nerfstudio_dir: str | Path,
        dnsplatter_checkpoint: str | Path,
        prior_dir: str | Path | None = None,
        image_size: int = 128,
        max_views: int | None = None,
        split: str = "train",
    ) -> None:
        self.root = Path(nerfstudio_dir).resolve()
        self.checkpoint = Path(dnsplatter_checkpoint).resolve()
        self.prior_dir = Path(prior_dir).resolve() if prior_dir is not None else None
        self.image_size = int(image_size)
        self.split = split
        metadata = json.loads((self.root / "transforms.json").read_text(encoding="utf-8"))
        source_frames = metadata.get("frames", [])
        selected_indices = select_frame_indices(metadata, source_frames, split)
        if max_views is not None:
            selected_indices = selected_indices[:max_views]
        if not selected_indices:
            raise ValueError(f"{self.root}/transforms.json has no frames for split {split!r}")
        dataparser = load_dataparser_transform(infer_dataparser_transform_path(self.checkpoint))
        saved_to_model = saved_pose_to_model_transform(dataparser, metadata)
        self.frames = [
            self._load_frame(index, source_frames[index], metadata, saved_to_model)
            for index in selected_indices
        ]

    def _load_frame(
        self,
        index: int,
        frame: dict[str, Any],
        metadata: dict[str, Any],
        saved_to_model: tuple[Tensor, float],
    ) -> FrameData:
        image = Image.open(self.root / frame["file_path"]).convert("RGB")
        source_w, source_h = image.size
        target = torch.from_numpy(np.asarray(image).copy()).float().div(255.0)
        target = resize_map(target, (self.image_size, self.image_size))
        sx = self.image_size / float(source_w)
        sy = self.image_size / float(source_h)
        mask_path = frame.get("mask_path")
        if mask_path is None:
            mask = torch.ones(
                (self.image_size, self.image_size, 1),
                dtype=torch.bool,
            )
        else:
            mask_image = Image.open(self.root / mask_path).convert("L")
            if mask_image.size != (source_w, source_h):
                raise ValueError(
                    f"mask {self.root / mask_path} has size {mask_image.size}, "
                    f"expected {(source_w, source_h)}"
                )
            mask = torch.from_numpy(np.asarray(mask_image).copy()) > 0
            mask = resize_mask(mask, (self.image_size, self.image_size))
        fx = float(frame.get("fl_x", metadata.get("fl_x")))
        fy = float(frame.get("fl_y", metadata.get("fl_y", fx)))
        cx = float(frame.get("cx", metadata.get("cx", source_w * 0.5)))
        cy = float(frame.get("cy", metadata.get("cy", source_h * 0.5)))
        camera = {
            "c2w": torch.as_tensor(frame["transform_matrix"], dtype=torch.float32),
            "fx": fx * sx,
            "fy": fy * sy,
            "cx": cx * sx,
            "cy": cy * sy,
            "width": self.image_size,
            "height": self.image_size,
            "camera_convention": "opengl",
        }
        camera = transform_camera(camera, saved_to_model)
        priors: dict[str, Tensor] = {}
        radiance_hdr = None
        if self.prior_dir is not None:
            prior_path = self.prior_dir / f"frame_{index:04d}.pt"
            if not prior_path.exists():
                raise FileNotFoundError(prior_path)
            payload = torch.load(prior_path, map_location="cpu", weights_only=False)
            required = {"prior_albedo", "prior_roughness", "prior_metallic"}
            missing = sorted(required.difference(payload))
            if missing:
                raise ValueError(
                    f"material prior {prior_path} is missing required field(s): {', '.join(missing)}"
                )
            for name in ("albedo", "roughness", "metallic"):
                key = f"prior_{name}"
                if key in payload:
                    priors[name] = resize_map(payload[key].float(), target.shape[:2])
            if "radiance_hdr" in payload:
                radiance_hdr = resize_map(payload["radiance_hdr"].float(), target.shape[:2])
        if radiance_hdr is None and "radiance_hdr_file_path" in frame:
            hdr = np.load(self.root / frame["radiance_hdr_file_path"]).astype(np.float32)
            radiance_hdr = resize_map(torch.from_numpy(hdr.copy()), target.shape[:2])
        return FrameData(index, target, mask, camera, priors, radiance_hdr, frame)

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> FrameData:
        return self.frames[index]

    def camera_regularization_data(self, device: str) -> dict[str, Tensor]:
        return {
            "c2w": torch.stack([frame.camera["c2w"] for frame in self.frames]).to(device),
            "fx": torch.tensor([frame.camera["fx"] for frame in self.frames], device=device),
            "fy": torch.tensor([frame.camera["fy"] for frame in self.frames], device=device),
            "cx": torch.tensor([frame.camera["cx"] for frame in self.frames], device=device),
            "cy": torch.tensor([frame.camera["cy"] for frame in self.frames], device=device),
            "width": torch.tensor([frame.camera["width"] for frame in self.frames], device=device),
            "height": torch.tensor([frame.camera["height"] for frame in self.frames], device=device),
            "convention": torch.ones(len(self.frames), device=device, dtype=torch.long),
        }
