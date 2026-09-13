import json
from pathlib import Path

import numpy as np

from scripts.blender_aegir.correct_capture_intrinsics import correct_dataset, correct_z_depth


def test_correct_z_depth_undoes_legacy_cycles_z_division() -> None:
    cycles_z = np.full((4, 4), 2.0, dtype=np.float64)
    y = (np.arange(4, dtype=np.float64) + 0.5 - 2.0)[:, None]
    x = (np.arange(4, dtype=np.float64) + 0.5 - 2.0)[None, :]
    old_factor = np.sqrt(1.0 + (x / 4.0) ** 2 + (y / 6.0) ** 2)
    legacy_depth = (cycles_z / old_factor).astype(np.float32)
    corrected = correct_z_depth(legacy_depth, 4.0, 6.0, 4.0, 4.0, 2.0, 2.0)
    np.testing.assert_allclose(corrected, cycles_z, rtol=1e-6)


def test_correct_dataset_preserves_source_and_excludes_derived_products(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "corrected"
    for name in ("images", "depths"):
        (source / name).mkdir(parents=True, exist_ok=True)
    (source / "images" / "frame.png").write_bytes(b"png")
    np.save(source / "depths" / "frame.npy", np.full((2, 2), 2.0, dtype=np.float32))
    (source / "colmap").mkdir()
    (source / "colmap" / "stale.txt").write_text("stale", encoding="utf-8")
    frame = {
        "file_path": "images/frame.png",
        "depth_file_path": "depths/frame.npy",
        "transform_matrix": np.eye(4).tolist(),
        "fl_x": 4.0,
        "fl_y": 6.0,
        "cx": 1.0,
        "cy": 1.0,
        "w": 2,
        "h": 2,
    }
    metadata = {
        "camera_model": "OPENCV",
        "fl_x": 4.0,
        "fl_y": 6.0,
        "cx": 1.0,
        "cy": 1.0,
        "w": 2,
        "h": 2,
        "ply_file_path": "colmap/sparse_pc.ply",
        "frames": [frame],
    }
    (source / "transforms.json").write_text(json.dumps(metadata), encoding="utf-8")

    report = correct_dataset(source, output)

    corrected_metadata = json.loads((output / "transforms.json").read_text(encoding="utf-8"))
    assert corrected_metadata["fl_x"] == corrected_metadata["fl_y"] == 4.0
    assert corrected_metadata["frames"][0]["fl_y"] == 4.0
    assert "ply_file_path" not in corrected_metadata
    assert (output / "images").is_symlink()
    assert not (output / "colmap").exists()
    assert report["corrected_depth_files"] == 1
    assert json.loads((source / "transforms.json").read_text())["fl_y"] == 6.0
