from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from aegir.config import AEGIRConfig
from aegir.data import MultiViewSceneDataset
from aegir.initialization import seed_emitters_from_predicted_depth
from aegir.provenance import (
    artifact_request,
    publish_artifact,
    validate_depth_directory,
    validate_predicted_depth,
    verify_cache,
)
from scripts.prepare_predicted_depth_view import metre_z_depth, prepare


class ConstantDepth:
    def infer(self, rgb):
        return torch.full((1, 1, *rgb.shape[-2:]), 2.5)


def make_source(root):
    root.mkdir()
    Image.fromarray(np.full((8, 8, 3), 128, np.uint8)).save(root / "rgb.png")
    Image.fromarray(np.zeros((8, 8), np.uint8)).save(root / "emitter.png")
    np.save(root / "hdr.npy", np.full((8, 8, 3), 4, np.float32))
    m = {
        "fl_x": 8.0,
        "fl_y": 8.0,
        "cx": 4.0,
        "cy": 4.0,
        "w": 8,
        "h": 8,
        "depth_unit_scale_factor": 0.001,
        "frames": [
            {
                "file_path": "rgb.png",
                "transform_matrix": np.eye(4).tolist(),
                "emitter_mask_path": "emitter.png",
                "depth_file_path": "missing_gt.npy",
                "normal_world_file_path": "missing_normal.npy",
                "metallic_gt_file_path": "missing_material.npy",
                "radiance_hdr_file_path": "hdr.npy",
            }
        ],
    }
    (root / "transforms.json").write_text(json.dumps(m))
    # Malformed GT emitter definitions must never be read in practical mode.
    (root / "emitters.json").write_text("malformed forbidden GT")
    return root


def test_predictions_are_metres_gt_free_and_cache_checks_content(tmp_path):
    source = make_source(tmp_path / "source")
    result = prepare(source, tmp_path / "pred", device="cpu", model=ConstantDepth())
    m = json.loads((result / "transforms.json").read_text())
    f = m["frames"][0]
    assert m["depth_unit_scale_factor"] == 1
    assert "normal_world_file_path" not in f and "metallic_gt_file_path" not in f
    assert not (result / "emitters.json").exists()
    assert np.load(result / f["depth_file_path"]).mean() == 2.5
    validate_predicted_depth(result)
    validate_depth_directory(result / "predicted_depth")
    prepare(source, result, device="cpu", model=ConstantDepth())
    np.save(result / f["depth_file_path"], np.full((8, 8), 2500.0))
    with pytest.raises(ValueError, match="content mismatch"):
        prepare(source, result, device="cpu", model=ConstantDepth())


def test_gt_directory_cannot_be_renamed_as_model_depth(tmp_path):
    source = make_source(tmp_path / "source")
    with pytest.raises(ValueError, match="RGB-model depth"):
        validate_predicted_depth(source)
    with pytest.raises((ValueError, FileNotFoundError)):
        validate_depth_directory(source)
    np.testing.assert_array_equal(metre_z_depth(torch.full((1, 1, 4, 4), 2.5)), np.full((4, 4), 2.5))
    with pytest.raises(ValueError):
        metre_z_depth(np.full((4, 4), np.nan))


def test_hdr_initialization_works_without_background_hits_and_matches_radiance():
    camera = {"fx": 8.0, "fy": 8.0, "cx": 4.0, "cy": 4.0, "c2w": torch.eye(4)}
    frame = SimpleNamespace(
        camera=camera,
        target=torch.zeros(8, 8, 3),
        radiance_hdr=torch.full((8, 8, 3), 7.0),
        metric_depth=torch.full((8, 8), 2.0),
        mask=torch.ones(8, 8, 1, dtype=torch.bool),
    )
    emitters = seed_emitters_from_predicted_depth(
        [frame],
        (torch.full((3,), -4.0), torch.full((3,), 4.0)),
        quantile=0.99,
        min_scale_fraction=0.0,
        persistence_min_views=1,
        persistence_min_fraction=0.0,
        dbscan_eps=1.0,
        min_samples=3,
    )
    assert emitters.num_emitters == 1
    assert emitters.centers[0, 2].item() == pytest.approx(-2.0)
    directions = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    assert torch.allclose(emitters.radiance(directions), torch.full((3, 1, 3), 7.0), atol=1e-5)
    frame.mask.zero_()
    with pytest.raises(ValueError, match="No persistent HDR"):
        seed_emitters_from_predicted_depth([frame], (torch.full((3,), -4.0), torch.full((3,), 4.0)))


def test_dataset_does_not_load_gt_emitters_by_default(tmp_path):
    source = make_source(tmp_path / "source")
    result = prepare(source, tmp_path / "pred", device="cpu", model=ConstantDepth())
    (result / "emitters.json").write_text("invalid GT JSON")
    checkpoint = tmp_path / "run/nerfstudio_models/step.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    (checkpoint.parent.parent / "dataparser_transforms.json").write_text(
        json.dumps({"transform": np.eye(4)[:3].tolist(), "scale": 1.0})
    )
    dataset = MultiViewSceneDataset(result, checkpoint, image_size=8)
    assert dataset.emitter_definitions == []
    assert AEGIRConfig().training.emitter_initialization == "hdr_multiview"


def test_seed_and_native_view_use_certified_depth_and_block_stale_masks(tmp_path):
    source = make_source(tmp_path / "source")
    result = prepare(source, tmp_path / "pred", device="cpu", model=ConstantDepth())
    seed = tmp_path / "seed.ply"
    root = Path(__file__).resolve().parents[1]

    def run(script, *args):
        subprocess.run(
            [sys.executable, str(root / "scripts" / script), *map(str, args)],
            cwd=root,
            check=True,
            capture_output=True,
        )

    run(
        "build_gt_depth_point_cloud.py",
        "--data",
        result,
        "--output",
        seed,
        "--pixel-stride",
        "1",
        "--voxel-size",
        ".1",
        "--statistical-nb-neighbors",
        "0",
    )
    run(
        "build_gt_depth_point_cloud.py",
        "--data",
        result,
        "--output",
        seed,
        "--pixel-stride",
        "1",
        "--voxel-size",
        ".1",
        "--statistical-nb-neighbors",
        "0",
    )
    view = tmp_path / "view"
    args = (
        "--source",
        result,
        "--output",
        view,
        "--mono-depth-source",
        result / "predicted_depth",
        "--point-cloud-source",
        seed,
        "--exclude-emitter-masks",
        "--encode-emitter-exemption",
    )
    run("prepare_rgb_pose_mask_view.py", *args)
    m = json.loads((view / "transforms.json").read_text())
    assert "depth_file_path" not in m["frames"][0]
    assert (view / "mono_depth/rgb.npy").resolve() == (result / "predicted_depth/rgb.npy").resolve()
    run("prepare_rgb_pose_mask_view.py", *args)
    # Exercise pending/completed cache guards without launching GPU training.
    external = tmp_path / "external"
    (external / "scene").mkdir(parents=True)
    (external / "train.py").write_text("# test trainer")
    (external / "scene/dataset_readers.py").write_text("# test reader")
    env = dict(os.environ, ALLOW_UNVERIFIED_DEPTH="0", DEPTH_LOSS_TYPE="pearson", TWO_DGS_ROOT=str(external))
    checkpoint = tmp_path / "native.pt"

    def stage(action, environment=env):
        return subprocess.run(
            [
                sys.executable,
                str(root / "scripts/check_training_provenance.py"),
                "--stage",
                "2dgs",
                "--action",
                action,
                "--data",
                str(view),
                "--output",
                str(checkpoint),
            ],
            env=environment,
            cwd=root,
            check=True,
            capture_output=True,
        )

    assert stage("begin").stdout.strip() == b"0"
    assert stage("begin").stdout.strip() == b"0"
    with pytest.raises(subprocess.CalledProcessError):
        stage("begin", dict(env, DEPTH_LOSS_TYPE="l1"))
    checkpoint.write_bytes(b"completed output")
    stage("complete")
    assert stage("begin").stdout.strip() == b"1"
    Image.fromarray(np.zeros((8, 8), np.uint8)).save(view / m["frames"][0]["mask_path"])
    with pytest.raises(subprocess.CalledProcessError):
        stage("begin")
    seed.write_bytes(b"changed seed")
    with pytest.raises(subprocess.CalledProcessError):
        run("prepare_rgb_pose_mask_view.py", *args)


def test_cache_rejects_wrong_configuration_and_unverified_outputs(tmp_path):
    output, manifest = tmp_path / "seed.ply", tmp_path / "manifest.json"
    output.write_bytes(b"cloud")
    request = artifact_request("depth_seed", {"voxel": 0.1}, {"origin": "rgb_model"})
    with pytest.raises(ValueError, match="Unverified cache"):
        verify_cache(manifest, request, output)
    publish_artifact(manifest, request, output)
    assert verify_cache(manifest, request, output)
    with pytest.raises(ValueError, match="mismatch"):
        verify_cache(
            manifest, artifact_request("depth_seed", {"voxel": 0.2}, {"origin": "rgb_model"}), output
        )


def test_gt_geometry_is_rejected_before_aegir_training(tmp_path):
    from aegir.trainer import train_shared_gaussians

    source = make_source(tmp_path / "source")
    result = prepare(source, tmp_path / "pred", device="cpu", model=ConstantDepth())
    geometry = tmp_path / "old_gt_geometry.pt"
    geometry.write_bytes(b"old GT-assisted geometry")
    with pytest.raises(ValueError, match="practical training provenance"):
        train_shared_gaussians(geometry, result, None, tmp_path / "out.pt", AEGIRConfig(), device="cpu")


def test_native_rgb_links_remap_splits_and_unverified_seeds_are_rejected(tmp_path):
    source = make_source(tmp_path / "source")
    m = json.loads((source / "transforms.json").read_text())
    m["train_filenames"] = ["rgb.png"]
    m["test_filenames"] = []
    (source / "transforms.json").write_text(json.dumps(m))
    result = prepare(source, tmp_path / "pred", device="cpu", model=ConstantDepth())
    root = Path(__file__).resolve().parents[1]

    def run(script, *args):
        return subprocess.run(
            [sys.executable, str(root / "scripts" / script), *map(str, args)],
            cwd=root,
            check=True,
            capture_output=True,
        )

    seed = tmp_path / "unverified.ply"
    seed.write_bytes(b"GT cloud")
    args = (
        "--source",
        result,
        "--output",
        tmp_path / "view",
        "--mono-depth-source",
        result / "predicted_depth",
        "--point-cloud-source",
        seed,
    )
    with pytest.raises(subprocess.CalledProcessError):
        run("prepare_rgb_pose_mask_view.py", *args)
    seed.unlink()
    run(
        "build_gt_depth_point_cloud.py",
        "--data",
        result,
        "--output",
        seed,
        "--pixel-stride",
        1,
        "--voxel-size",
        0.1,
        "--statistical-nb-neighbors",
        0,
    )
    run("prepare_rgb_pose_mask_view.py", *args)
    m = json.loads((tmp_path / "view/transforms.json").read_text())
    assert m["frames"][0]["file_path"] == "images/rgb.png"
    assert m["train_filenames"] == ["images/rgb.png"]
    assert m["test_filenames"] == []
    assert (tmp_path / "view/images/rgb.png").resolve() == source / "rgb.png"


def test_radius_persistence_crosses_voxel_boundaries_and_counts_views_once():
    from aegir.initialization import _persistent_candidate_mask, _radius_persistent_candidate_mask

    points = torch.tensor([[0.099, 0.0, 0.0], [0.101, 0.0, 0.0], [0.102, 0.0, 0.0]])
    views = torch.tensor([0, 1, 2])
    assert not _persistent_candidate_mask(points, views, torch.zeros(3), 0.1, 3, 3).any()
    assert _radius_persistent_candidate_mask(points, views, radius=0.02, num_views=3, required_views=3).all()
    repeated = torch.zeros(100, 3)
    assert not _radius_persistent_candidate_mask(
        repeated, torch.zeros(100, dtype=torch.long), radius=0.02, num_views=3, required_views=3
    ).any()
    far = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    assert not _radius_persistent_candidate_mask(far, views, radius=0.02, num_views=3, required_views=3).any()


def test_radius_persistence_does_not_fit_single_view_remainder(monkeypatch):
    from aegir import initialization

    points = torch.tensor([[[0.0, 0.0, -2.0], [0.1, 0.0, -2.0]], [[0.0, 0.1, -2.0], [0.1, 0.1, -2.0]]])
    sample = {"valid_mask": torch.ones(2, 2, 1, dtype=torch.bool), "radiance_hdr": torch.full((2, 2, 3), 7.0)}
    # Emulate a one-view remainder near candidates that were not retained from
    # other views: fitting it would defeat the minimum distinct-view condition.
    monkeypatch.setattr(
        initialization,
        "_radius_persistent_candidate_mask",
        lambda *args, **kwargs: torch.tensor([True] * 4 + [False] * 8),
    )
    with pytest.raises(ValueError, match="No persistent HDR"):
        initialization.seed_emitters_from_gbuffer_views(
            [torch.ones(2, 2, 3)] * 3,
            [{"points": points}] * 3,
            (torch.full((3,), -4.0), torch.full((3,), 4.0)),
            samples=[sample] * 3,
            score_mode="hdr",
            quantile=0.99,
            min_samples=3,
            persistence_min_views=3,
            persistence_radius_fraction=0.01,
            allow_fallback=False,
            dbscan_eps=1.0,
        )
