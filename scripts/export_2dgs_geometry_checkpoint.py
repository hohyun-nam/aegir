from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData


def _ordered_properties(vertex, prefix: str) -> list[str]:
    names = [item.name for item in vertex.properties if item.name.startswith(prefix)]
    return sorted(names, key=lambda value: int(value.rsplit("_", 1)[-1]))


def _stack(vertex, names: list[str]) -> torch.Tensor:
    if not names:
        raise ValueError("checkpoint PLY is missing required properties")
    array = np.stack([np.asarray(vertex[name], dtype=np.float32) for name in names], axis=-1)
    return torch.from_numpy(array.copy())


def load_2dgs_ply(path: Path) -> dict[str, torch.Tensor | str | int]:
    vertex = PlyData.read(path)["vertex"]
    means = _stack(vertex, ["x", "y", "z"])
    features_dc = _stack(vertex, ["f_dc_0", "f_dc_1", "f_dc_2"])[:, None, :]
    rest_names = _ordered_properties(vertex, "f_rest_")
    features_rest_flat = _stack(vertex, rest_names)
    if features_rest_flat.shape[-1] % 3:
        raise ValueError("2DGS PLY has an invalid number of SH-rest properties")
    features_rest = features_rest_flat.reshape(means.shape[0], 3, -1).transpose(1, 2).contiguous()
    log_scales = _stack(vertex, _ordered_properties(vertex, "scale_"))
    if log_scales.shape[-1] != 2:
        raise ValueError(f"expected native 2DGS PLY with two scales, got {tuple(log_scales.shape)}")
    quats = _stack(vertex, _ordered_properties(vertex, "rot_"))
    if quats.shape[-1] != 4:
        raise ValueError(f"expected four quaternion properties, got {tuple(quats.shape)}")
    opacity_logits = _stack(vertex, ["opacity"])
    tensors = (means, features_dc, features_rest, log_scales, quats, opacity_logits)
    if means.shape[0] == 0 or any(value.shape[0] != means.shape[0] for value in tensors):
        raise ValueError("2DGS PLY contains empty or inconsistent Gaussian arrays")
    if not all(bool(torch.isfinite(value).all()) for value in tensors):
        raise ValueError("2DGS PLY contains non-finite values")
    return {
        "format": "aegir_2dgs_geometry_v1",
        "representation": "2dgs",
        "means": means,
        "quats": quats,
        "scales": log_scales.exp(),
        "opacities_logits": opacity_logits,
        "features_dc_raw": features_dc,
        "features_rest_raw": features_rest,
        "gaussian_count": int(means.shape[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a native Inria/AGS-Mesh 2DGS PLY into an AEGIR geometry checkpoint."
    )
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    args = parser.parse_args()

    payload = load_2dgs_ply(args.ply.resolve())
    payload.update(
        {
            "step": int(args.iteration),
            "source_ply": str(args.ply.resolve()),
            "source_data": str(args.data.resolve()),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)

    # The 2DGS trainer consumes the saved Nerfstudio/OpenGL poses without
    # translating or scaling world coordinates. Keep AEGIR's existing camera
    # transform discovery contract by writing an identity sidecar next to the
    # exported checkpoint directory.
    transform_path = args.output.parent.parent / "dataparser_transforms.json"
    identity = {
        "transform": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
        "scale": 1.0,
        "source": "native_2dgs_aegir_nerfstudio_coordinates",
    }
    if transform_path.exists():
        existing = json.loads(transform_path.read_text(encoding="utf-8"))
        if existing != identity:
            raise ValueError(f"refusing to overwrite a different camera transform: {transform_path}")
    else:
        transform_path.write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")

    print(f"2DGS_GEOMETRY_EXPORT_OK gaussians={payload['gaussian_count']} iteration={args.iteration}")
    print(f"checkpoint: {args.output.resolve()}")
    print(f"camera_transform: {transform_path.resolve()}")


if __name__ == "__main__":
    main()
