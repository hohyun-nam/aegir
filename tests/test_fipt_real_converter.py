from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from aegir import fipt_real


def _write_counted(path: Path, values: np.ndarray) -> None:
    rows = [str(len(values))]
    for matrix in values:
        rows.extend(" ".join(f"{value:.9f}" for value in row) for row in matrix)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _make_scene(path: Path, corrupt_cam: bool = False) -> None:
    for name in ("Image", "MiDepth", "MiNormalGlobal", "ImMask"):
        (path / name).mkdir(parents=True, exist_ok=True)
    rotations = np.stack(
        (
            np.eye(3),
            np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]),
        )
    )
    origins = np.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]])
    rt = np.repeat(np.eye(4)[None], 2, axis=0)
    rt[:, :3, :3] = rotations
    rt[:, :3, 3] = origins
    intrinsics = np.repeat(
        np.array([[[10.0, 0.0, 5.5], [0.0, 10.0, 3.5], [0.0, 0.0, 1.0]]]),
        2,
        axis=0,
    )
    cam = np.stack((origins, origins + rotations[:, :, 2], -rotations[:, :, 1]), axis=1)
    if corrupt_cam:
        cam[0, 1] += 1.0
    _write_counted(path / "Rt.txt", rt)
    _write_counted(path / "K_list.txt", intrinsics)
    _write_counted(path / "cam.txt", cam)

    for index, rotation in enumerate(rotations):
        stem = f"{index:03d}_0001"
        Image.fromarray(np.full((8, 12, 3), 64 + index, dtype=np.uint8)).save(
            path / "Image" / f"{stem}.png"
        )
        (path / "Image" / f"{stem}.exr").write_bytes(b"test-exr")
        np.save(path / "MiDepth" / f"{stem}.npy", np.full((8, 12), 2.0))
        world_normal = rotation @ np.array([0.0, 0.0, -1.0])
        encoded = np.broadcast_to(
            np.rint((world_normal + 1.0) * 127.5).astype(np.uint8), (8, 12, 3)
        )
        Image.fromarray(encoded).save(path / "MiNormalGlobal" / f"{stem}.png")
        Image.fromarray(np.full((8, 12), 255, dtype=np.uint8)).save(
            path / "ImMask" / f"{stem}.png"
        )


def test_convert_fipt_real_scene(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "conference"
    output = tmp_path / "converted"
    _make_scene(source)
    monkeypatch.setattr(
        fipt_real,
        "_read_exr_rgb",
        lambda _: np.full((8, 12, 3), 3.0, dtype=np.float32),
    )

    summary = fipt_real.convert_fipt_real_scene(source, output)

    assert summary["frames"] == 2
    assert summary["z_depth_normal_dot"] > 0.99
    metadata = json.loads((output / "transforms.json").read_text(encoding="utf-8"))
    assert metadata["aegir_depth_mode"] == "z"
    assert metadata["depth_unit_scale_factor"] == 1.0
    assert metadata["normal_space"] == "world"
    assert len(metadata["frames"]) == 2
    first = metadata["frames"][0]
    assert np.allclose(
        np.asarray(first["transform_matrix"])[:3, :3],
        np.diag([1.0, -1.0, -1.0]),
    )
    assert first["radiance_hdr_file_path"] == "radiance_hdr/frame_0000.npy"
    assert np.load(output / first["radiance_hdr_file_path"]).dtype == np.float32
    assert np.load(output / first["depth_file_path"]).dtype == np.float32
    assert (output / first["normal_world_file_path"]).is_file()
    assert (output / first["mask_path"]).is_file()


def test_rejects_camera_direction_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "bad"
    _make_scene(source, corrupt_cam=True)

    with pytest.raises(ValueError, match="not an OpenCV camera-to-world"):
        fipt_real.convert_fipt_real_scene(source, tmp_path / "output")
