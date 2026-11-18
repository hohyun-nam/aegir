from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from aegir.config import LossWeights
from aegir.data import MultiViewSceneDataset
from aegir.losses import edge_aware_tv, focus_map, material_prior_loss, photometric_loss, ssim


def _write_scene(
    root: Path,
    *,
    with_splits: bool = True,
    missing_mask: bool = False,
) -> tuple[Path, Path]:
    root.mkdir(parents=True)
    (root / "images").mkdir()
    (root / "masks").mkdir()
    frames = []
    identity = torch.eye(4).tolist()
    for index in range(3):
        pixels = np.full((4, 4, 3), 32 + index, dtype=np.uint8)
        Image.fromarray(pixels).save(root / "images" / f"frame_{index:04d}.png")
        frame = {
            "file_path": f"images/frame_{index:04d}.png",
            "transform_matrix": identity,
        }
        if index == 0:
            frame["mask_path"] = "masks/missing.png" if missing_mask else "masks/frame_0000.png"
        frames.append(frame)

    if not missing_mask:
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[:, :2] = 255
        Image.fromarray(mask).save(root / "masks" / "frame_0000.png")

    metadata = {
        "fl_x": 4.0,
        "fl_y": 4.0,
        "cx": 2.0,
        "cy": 2.0,
        "w": 4,
        "h": 4,
        "frames": frames,
    }
    if with_splits:
        metadata.update(
            {
                "train_filenames": [
                    "images/frame_0000.png",
                    "images/frame_0002.png",
                ],
                "val_filenames": ["images/frame_0001.png"],
                "test_filenames": ["images/frame_0001.png"],
            }
        )
    (root / "transforms.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    run = root.parent / "run"
    checkpoint = run / "nerfstudio_models" / "step.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    (run / "dataparser_transforms.json").write_text(
        json.dumps(
            {
                "transform": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ],
                "scale": 1.0,
            }
        ),
        encoding="utf-8",
    )

    priors = root.parent / "priors"
    priors.mkdir()
    for index in range(3):
        torch.save(
            {
                "prior_albedo": torch.full((4, 4, 3), float(index) / 2.0),
                "prior_roughness": torch.full((4, 4, 1), 0.6),
                "prior_metallic": torch.full((4, 4, 1), 0.2),
                "prior_albedo_space": "linear_reflectance",
            },
            priors / f"frame_{index:04d}.pt",
        )
    return checkpoint, priors


def test_dataset_split_preserves_original_indices_and_prior_alignment(
    tmp_path: Path,
) -> None:
    scene = tmp_path / "scene"
    checkpoint, priors = _write_scene(scene)

    train = MultiViewSceneDataset(
        scene,
        checkpoint,
        priors,
        image_size=4,
        split="train",
    )
    val = MultiViewSceneDataset(scene, checkpoint, priors, image_size=4, split="val")
    test = MultiViewSceneDataset(scene, checkpoint, priors, image_size=4, split="test")
    all_frames = MultiViewSceneDataset(
        scene,
        checkpoint,
        priors,
        image_size=4,
        split="all",
    )

    assert [frame.index for frame in train.frames] == [0, 2]
    assert [frame.index for frame in val.frames] == [1]
    assert [frame.index for frame in test.frames] == [1]
    assert [frame.index for frame in all_frames.frames] == [0, 1, 2]
    assert torch.all(train.frames[1].priors["albedo"] == 1.0)
    assert int(train.frames[0].mask.sum()) == 8
    assert bool(val.frames[0].mask.all())

    limited = MultiViewSceneDataset(
        scene,
        checkpoint,
        priors,
        image_size=4,
        max_views=1,
        split="train",
    )
    assert [frame.index for frame in limited.frames] == [0]


def test_dataset_without_split_or_mask_falls_back_to_all_valid(
    tmp_path: Path,
) -> None:
    scene = tmp_path / "scene"
    checkpoint, _ = _write_scene(scene, with_splits=False)
    metadata_path = scene / "transforms.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["frames"][0].pop("mask_path")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    dataset = MultiViewSceneDataset(scene, checkpoint, image_size=4, split="train")
    assert [frame.index for frame in dataset.frames] == [0, 1, 2]
    assert all(bool(frame.mask.all()) for frame in dataset.frames)

    with pytest.raises(ValueError, match="no split metadata"):
        MultiViewSceneDataset(scene, checkpoint, image_size=4, split="val")


def test_ldr_dataset_does_not_load_hdr_copy_from_material_priors(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    checkpoint, priors = _write_scene(scene)
    prior_path = priors / "frame_0000.pt"
    payload = torch.load(prior_path, map_location="cpu", weights_only=False)
    payload["radiance_hdr"] = torch.full((4, 4, 3), 100.0)
    torch.save(payload, prior_path)

    dataset = MultiViewSceneDataset(scene, checkpoint, priors, image_size=4, split="train")

    assert dataset.observation_policy["domain"] == "ldr"
    assert dataset.frames[0].radiance_hdr is None


def test_referenced_missing_mask_is_reported(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    checkpoint, _ = _write_scene(scene, missing_mask=True)
    with pytest.raises(FileNotFoundError):
        MultiViewSceneDataset(scene, checkpoint, image_size=4, split="train")


def test_evaluation_split_rejects_train_overlap(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    checkpoint, _ = _write_scene(scene)
    metadata_path = scene / "transforms.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["test_filenames"].append("images/frame_0000.png")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="overlaps the train split"):
        MultiViewSceneDataset(scene, checkpoint, image_size=4, split="test")


def test_split_protocol_reports_shared_val_and_test(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    checkpoint, _ = _write_scene(scene)
    dataset = MultiViewSceneDataset(scene, checkpoint, image_size=4, split="test")

    assert dataset.split_protocol["held_out_from_train"] is True
    assert dataset.split_protocol["train_overlap_frames"] == 0
    assert dataset.split_protocol["val_test_identical"] is True
    assert dataset.split_protocol["val_test_overlap_frames"] == 1


def test_masked_losses_ignore_invalid_pixels() -> None:
    weights = LossWeights(focus_lambda=0.0, dssim_lambda=0.0)
    target = torch.zeros((4, 4, 3))
    pred = target.clone()
    pred[:, 2:] = 1.0
    mask = torch.zeros((4, 4, 1), dtype=torch.bool)
    mask[:, :2] = True

    assert photometric_loss(pred, target, weights, mask).item() == pytest.approx(0.0)
    assert photometric_loss(pred, target, weights).item() > 0.0

    rendered = {"albedo": pred}
    priors = {"albedo": target}
    assert material_prior_loss(rendered, priors, weights, mask).item() == pytest.approx(0.0)
    assert material_prior_loss(rendered, priors, weights).item() > 0.0


def test_full_view_focus_map_is_crop_invariant_and_respects_invalid_mask() -> None:
    generator = torch.Generator().manual_seed(17)
    target = torch.rand((13, 15, 3), generator=generator)
    mask = torch.ones((13, 15, 1), dtype=torch.bool)
    mask[4:9, 6:8] = False

    full_focus = focus_map(target, focus_lambda=5.0, kernel_size=7, mask=mask)
    crop_a = full_focus[2:11, 3:12]
    crop_b = full_focus[5:12, 5:14]

    # The overlapping source pixels retain the exact same full-view weights even
    # though they appear at different locations in differently sized crops.
    assert torch.equal(crop_a[3:9, 2:9], crop_b[:6, :7])

    changed_invalid = target.clone()
    changed_invalid[~mask.expand_as(target)] = 1.0e4
    changed_focus = focus_map(changed_invalid, focus_lambda=5.0, kernel_size=7, mask=mask)
    assert torch.equal(full_focus[mask], changed_focus[mask])


def test_photometric_loss_uses_precomputed_full_view_focus_crop() -> None:
    weights = LossWeights(focus_lambda=5.0, dssim_lambda=0.0)
    target = torch.linspace(0.0, 1.0, 12 * 14).view(12, 14, 1).expand(-1, -1, 3)
    mask = torch.ones((12, 14, 1), dtype=torch.bool)
    mask[3:7, 5] = False
    pred = target + 0.1
    full_focus = focus_map(target, weights.focus_lambda, mask=mask)
    ys, xs = slice(4, 10), slice(3, 11)

    loss = photometric_loss(
        pred[ys, xs],
        target[ys, xs],
        weights,
        mask[ys, xs],
        focus_weights=full_focus[ys, xs],
    )
    expanded_mask = mask[ys, xs].expand_as(pred[ys, xs])
    expected = (
        full_focus[ys, xs] * (pred[ys, xs] - target[ys, xs]).abs() * expanded_mask
    ).sum() / expanded_mask.sum()
    assert loss.item() == pytest.approx(expected.item())


def test_masked_ssim_and_tv_handle_fully_invalid_neighborhoods() -> None:
    weights = LossWeights()
    target = torch.zeros((4, 4, 3))
    pred = torch.ones((4, 4, 3))
    invalid = torch.zeros((4, 4, 1), dtype=torch.bool)

    assert ssim(pred, target, mask=invalid).item() == pytest.approx(1.0)
    assert photometric_loss(pred, target, weights, invalid).item() == pytest.approx(0.0)

    rendered = {
        "albedo": torch.arange(4, dtype=torch.float32).view(1, 4, 1).expand(4, 4, 3),
    }
    one_pixel = invalid.clone()
    one_pixel[1, 1] = True
    assert edge_aware_tv(rendered, target, weights, one_pixel).item() == pytest.approx(0.0)
    assert edge_aware_tv(rendered, target, weights).item() > 0.0
