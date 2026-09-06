from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from aegir.conversion_common import (
    OPENCV_TO_OPENGL,
    independent_val_test_split,
    ray_distance_to_z,
)
from aegir.fipt_synthetic import (
    FIPT_TO_NERFSTUDIO_OPENGL,
    convert_fipt_synthetic_scene,
    fipt_pose_to_nerfstudio,
)
from aegir.hypersim import (
    _standard_projection,
    rectify_projection,
    rectify_world_camera,
)
from aegir.isaacsim import convert_isaacsim_scene


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_ray_distance_to_z() -> None:
    distance = np.full((3, 3), 2.0, dtype=np.float32)
    depth = ray_distance_to_z(distance, fl_x=1.0, fl_y=1.0, cx=1.0, cy=1.0)
    assert depth[1, 1] == pytest.approx(2.0)
    assert depth[0, 0] == pytest.approx(2.0 / np.sqrt(3.0))


def test_independent_val_test_split_preserves_train_and_interleaves_heldout() -> None:
    train = ["frame_0.png", "frame_1.png"]
    heldout = [f"frame_{index}.png" for index in range(2, 7)]
    val, test, metadata = independent_val_test_split(train, heldout)
    assert val == heldout[::2]
    assert test == heldout[1::2]
    assert metadata["original_shared_heldout_filenames"] == heldout
    assert not set(train).intersection(val + test)


def test_hypersim_standard_projection_rectification() -> None:
    intrinsic, rectification = rectify_projection(_standard_projection(1024, 768), 1024, 768)
    assert np.allclose(rectification, OPENCV_TO_OPENGL)
    assert intrinsic[0, 0] == pytest.approx(885.94395826)
    assert intrinsic[1, 1] == pytest.approx(885.65528314)
    assert intrinsic[0, 2] == pytest.approx(511.5)
    assert intrinsic[1, 2] == pytest.approx(383.5)


def test_hypersim_non_rigid_source_orientation_is_factored_into_intrinsics() -> None:
    source_orientation = np.diag([1.0, 1.0, 0.5])
    intrinsic, world_to_opencv = rectify_world_camera(
        _standard_projection(1024, 768),
        1024,
        768,
        source_orientation,
    )
    exported_rotation = world_to_opencv.T @ OPENCV_TO_OPENGL
    assert np.allclose(exported_rotation.T @ exported_rotation, np.eye(3))
    assert np.linalg.det(exported_rotation) == pytest.approx(1.0)
    assert intrinsic[0, 0] == pytest.approx(885.94395826 / 2.0)
    assert intrinsic[1, 1] == pytest.approx(885.65528314 / 2.0)


def test_convert_isaacsim_scene(tmp_path: Path) -> None:
    source = tmp_path / "isaac"
    for directory in ("images", "depth", "images_hdr_npy"):
        (source / directory).mkdir(parents=True, exist_ok=True)
    frames = []
    for index in range(2):
        stem = f"{index:04d}"
        Image.fromarray(np.full((4, 6, 3), 64 + index, dtype=np.uint8)).save(
            source / "images" / f"{stem}.png"
        )
        np.save(source / "depth" / f"{stem}.npy", np.full((4, 6), 2.0, dtype=np.float32))
        np.save(
            source / "images_hdr_npy" / f"{stem}.npy",
            np.full((4, 6, 3), 0.5, dtype=np.float32),
        )
        frames.append(
            {
                "file_path": f"images/{stem}",
                "depth_file_path": f"depth/{stem}",
                "hdr_npy_file_path": f"images_hdr_npy/{stem}.npy",
                "transform_matrix": np.eye(4).tolist(),
            }
        )
    metadata = {
        "fl_x": 5.0,
        "fl_y": 5.0,
        "cx": 3.0,
        "cy": 2.0,
        "w": 6,
        "h": 4,
        "frames": frames,
    }
    _write_json(source / "transforms.json", metadata)
    _write_json(source / "transforms_train.json", {**metadata, "frames": frames[:1]})
    _write_json(source / "transforms_test.json", {**metadata, "frames": frames[1:]})

    output = tmp_path / "converted"
    summary = convert_isaacsim_scene(source, output)
    converted = json.loads((output / "transforms.json").read_text())
    assert summary["frames"] == 2
    assert summary["train_frames"] == 1
    assert summary["eval_frames"] == 1
    assert converted["aegir_depth_mode"] == "z"
    assert converted["aegir_color"]["observation_domain"] == "linear_hdr"
    assert converted["train_filenames"] == ["images/frame_0000.png"]
    assert converted["val_filenames"] == ["images/frame_0001.png"]
    assert converted["test_filenames"] == []
    assert converted["aegir_evaluation_split"]["format"] == "aegir-heldout-split-v1"
    assert np.load(output / "depths/frame_0000.npy").dtype == np.float32


def test_convert_fipt_synthetic_scene(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "bathroom"
    frames_by_split = {}
    for split in ("train", "val"):
        for directory in ("Image", "Depth", "Normal", "DiffCol", "Roughness"):
            (source / split / directory).mkdir(parents=True, exist_ok=True)
        stem = "000_0001"
        for directory in ("Image", "Depth", "Normal", "DiffCol", "Roughness"):
            (source / split / directory / f"{stem}.exr").write_bytes(b"fake")
        frames_by_split[split] = [
            {
                "file_path": f"{split}/Image/{stem}",
                "transform_matrix": np.eye(4).tolist(),
            }
        ]
        _write_json(
            source / split / "transforms.json",
            {"camera_angle_x": np.pi / 2.0, "frames": frames_by_split[split]},
        )

    def fake_read(path: Path, *, rgb: bool) -> np.ndarray:
        if rgb:
            if path.parent.name == "Normal":
                value = np.zeros((4, 6, 3), dtype=np.float32)
                value[..., 2] = 1.0
                return value
            return np.full((4, 6, 3), 0.5, dtype=np.float32)
        return np.full((4, 6), 2.0, dtype=np.float32)

    monkeypatch.setattr("aegir.fipt_synthetic.read_exr", fake_read)
    output = tmp_path / "converted"
    summary = convert_fipt_synthetic_scene(source, output)
    converted = json.loads((output / "transforms.json").read_text())
    depth = np.load(output / "depths/frame_0000.npy")
    normal = np.asarray(Image.open(output / "normals_camera/frame_0000.png"))
    assert summary["frames"] == 2
    assert summary["generated_ldr_frames"] == 2
    assert converted["source_depth_mode"] == "ray"
    assert converted["aegir_depth_mode"] == "z"
    assert converted["aegir_source_camera_convention"] == "fipt_left_up_forward_c2w"
    assert converted["source_normal_space"] == "world"
    assert converted["normal_space"] == "camera"
    assert converted["aegir_color"]["observation_domain"] == "linear_hdr"
    assert converted["frames"][0]["ldr_exposure_scale"] > 0.0
    assert np.allclose(converted["frames"][0]["transform_matrix"], FIPT_TO_NERFSTUDIO_OPENGL)
    assert np.array_equal(normal[2, 3], np.asarray([128, 128, 0], dtype=np.uint8))
    assert len(converted["train_filenames"]) == 1
    assert len(converted["val_filenames"]) == 1
    assert len(converted["test_filenames"]) == 0
    assert depth[2, 3] == pytest.approx(2.0)
    assert depth[0, 0] < 2.0


def test_fipt_pose_conversion_preserves_center_and_rotates_camera_axes() -> None:
    angle = np.deg2rad(37.0)
    source = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0, 1.0],
            [np.sin(angle), np.cos(angle), 0.0, 2.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    converted = fipt_pose_to_nerfstudio(source)
    assert np.allclose(converted[:3, 3], source[:3, 3])
    assert np.allclose(converted, source @ FIPT_TO_NERFSTUDIO_OPENGL)
    assert np.linalg.det(converted[:3, :3]) == pytest.approx(1.0)
