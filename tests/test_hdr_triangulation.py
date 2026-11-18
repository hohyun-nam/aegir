from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

from aegir.hdr_triangulation import (
    _merge_collinear_fragments,
    detect_mask_observations,
    detect_observations,
    project,
    seed_emitters_from_hdr_multiview,
    seed_emitters_from_sam3_multiview,
)


def make_frames(ellipse=False, count=8):
    center = np.array([0.0, 0.0, 2.0])
    result = []
    for index in range(count):
        angle = index / count * 2 * np.pi
        origin = np.array([1.7 * np.cos(angle), 1.7 * np.sin(angle), 0.4 + 0.08 * index])
        back = origin - center
        back /= np.linalg.norm(back)
        right = np.cross([0.0, 0.0, 1.0], back)
        right /= np.linalg.norm(right)
        up = np.cross(back, right)
        c2w = np.eye(4)
        c2w[:3, :3] = np.stack([right, up, back], axis=1)
        c2w[:3, 3] = origin
        camera = {
            "c2w": torch.tensor(c2w, dtype=torch.float32),
            "fx": 100.0,
            "fy": 100.0,
            "cx": 64.0,
            "cy": 64.0,
        }
        if ellipse:
            theta = np.linspace(0, 2 * np.pi, 100, endpoint=False)
            cloud = center + np.stack(
                [0.4 * np.cos(theta), 0.25 * np.sin(theta), np.zeros_like(theta)], axis=-1
            )
        else:
            cloud = center + np.array(
                [[-0.4, -0.25, 0.0], [0.4, -0.25, 0.0], [0.4, 0.25, 0.0], [-0.4, 0.25, 0.0]]
            )
        uv, _ = project(cloud, camera)
        mask = Image.new("L", (128, 128))
        ImageDraw.Draw(mask).polygon(list(map(tuple, uv)), fill=255)
        bright = np.asarray(mask) > 0
        hdr = np.full((128, 128, 3), 0.05, np.float32)
        hdr[bright] = 10.0
        # No emitter mask, model depth, normal or GT metadata exists here.
        result.append(
            SimpleNamespace(
                index=index,
                camera=camera,
                radiance_hdr=torch.from_numpy(hdr),
                target=torch.from_numpy(np.clip(hdr / 10.0, 0.0, 1.0)),
                mask=torch.ones(128, 128, 1, dtype=torch.bool),
            )
        )
    return result


def test_hdr_triangulates_pose_support_without_depth_or_gt():
    frames = make_frames()
    emitters, diag = seed_emitters_from_hdr_multiview(
        frames, ([-3.0, -3.0, -1.0], [3.0, 3.0, 4.0]), representation="ellipsoid_3d"
    )
    assert emitters.num_emitters == 1
    assert np.linalg.norm(emitters.centers.detach().numpy()[0] - [0.0, 0.0, 2.0]) < 0.04
    assert abs(diag["emitters"][0]["normal"][2]) > 0.98
    assert diag["emitters"][0]["reprojection_max_px"] <= 3
    assert diag["emitters"][0]["views"] >= 3
    assert not diag["uses_depth_for_position"] and not diag["uses_gt_emitters"]
    assert emitters.scales.shape == (1, 3)
    scales = emitters.scales.detach()
    assert float(scales[0, 0] / scales[0, 1]) == pytest.approx(1.6, rel=0.15)
    assert np.pi * float(scales[0, 0] * scales[0, 1]) == pytest.approx(0.8 * 0.5, rel=0.2)
    assert torch.allclose(
        emitters.radiance(torch.tensor([[0.0, 0.0, 1.0]])), torch.full((1, 1, 3), 10.0), atol=1e-4
    )


def test_sam3_triangulates_pose_support_without_reading_depth():
    frames = make_frames()
    for frame in frames:
        frame.emitter_mask = (frame.radiance_hdr[..., :1] > 1.0)
        frame.metric_depth = object()
        frame.radiance_hdr = object()
    emitters, diag = seed_emitters_from_sam3_multiview(
        frames, ([-3.0, -3.0, -1.0], [3.0, 3.0, 4.0]), representation="ellipsoid_3d"
    )
    assert emitters.num_emitters == 1
    assert np.linalg.norm(emitters.centers.detach().numpy()[0] - [0.0, 0.0, 2.0]) < 0.04
    assert diag["method"] == "sam3_multiview"
    assert diag["high_threshold"] is None
    assert not diag["uses_depth_for_position"] and not diag["uses_gt_emitters"]
    assert not diag["uses_hdr_observations"]
    assert torch.allclose(
        emitters.radiance(torch.tensor([[0.0, 0.0, 1.0]])), torch.ones((1, 1, 3)), atol=1e-4
    )


def test_sam3_observations_require_explicit_masks():
    with pytest.raises(ValueError, match="requires emitter masks"):
        detect_mask_observations(make_frames())


def test_two_views_and_zero_baseline_cannot_produce_trusted_emitter():
    bounds = ([-3.0, -3.0, -1.0], [3.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="Insufficient complete HDR"):
        seed_emitters_from_hdr_multiview(make_frames(count=2), bounds)
    first = make_frames()[0]
    frames = [SimpleNamespace(**{**first.__dict__, "index": i}) for i in range(8)]
    with pytest.raises(ValueError, match="baseline/parallax"):
        seed_emitters_from_hdr_multiview(frames, bounds)


def test_hdr_region_growth_keeps_whole_panel_and_excludes_clipped_regions():
    frame = make_frames()[0]
    frame.radiance_hdr[:] = 0.1
    frame.radiance_hdr[20:40, 30:70] = 9.0
    frame.radiance_hdr[20:22, 30:35] = 10.0
    frame.radiance_hdr[0:15, 90:110] = 10.0
    obs, high, low = detect_observations([frame], quantile=0.995, grow_fraction=0.8)
    assert high == 10 and low == 8
    assert len(obs) == 1 and obs[0].pixels == 800
    assert obs[0].quad is not None


def test_ellipse_uses_multi_view_silhouettes_for_plane():
    emitters, diag = seed_emitters_from_hdr_multiview(
        make_frames(ellipse=True), ([-3.0, -3.0, -1.0], [3.0, 3.0, 4.0])
    )
    assert emitters.num_emitters == 1
    assert np.linalg.norm(emitters.centers.detach().numpy()[0] - [0.0, 0.0, 2.0]) < 0.05
    assert abs(diag["emitters"][0]["normal"][2]) > 0.97
    assert diag["emitters"][0]["method"] == "multi_view_silhouette_moments"


def test_collinear_hdr_fragments_merge_into_physical_strips_without_known_count():
    axis = np.array([1.0, 0.0, 0.0])
    cross = np.array([0.0, 0.0, 1.0])
    normal = np.array([0.0, 1.0, 0.0])
    basis = np.stack((axis, cross, normal), axis=1)
    models = []
    fragment = 0
    for offset in (-1.5, 1.5):
        for along in (-2.0, 0.0, 2.0):
            models.append(
                {
                    "center": np.array([along, 3.0, offset]),
                    "basis": basis.copy(),
                    "scales": np.array([1.05, 0.15, 0.03]),
                    "radiance": np.full(3, 10.0 + fragment),
                    "ids": np.array([fragment]),
                    "errors": np.array([0.5]),
                    "parallax": 20.0,
                    "quality": {},
                }
            )
            fragment += 1
    # Compact hotspots on the first strip are absorbed without widening it.
    models.append(
        {
            "center": np.array([1.0, 3.0, -1.75]),
            "basis": basis.copy(),
            "scales": np.array([0.12, 0.10, 0.03]),
            "radiance": np.full(3, 20.0),
            "ids": np.array([fragment]),
            "errors": np.array([0.4]),
            "parallax": 15.0,
            "quality": {},
        }
    )

    merged, records = _merge_collinear_fragments(models, diagonal=10.0)

    assert len(merged) == 2
    assert len(records) == 2
    assert sorted(model["quality"]["fragment_count"] for model in merged) == [3, 4]
    assert all(model["quality"]["method"] == "merged_collinear_fragments" for model in merged)
    assert all(model["scales"][0] == pytest.approx(3.05) for model in merged)
    assert all(model["scales"][1] == pytest.approx(0.15) for model in merged)


def test_compact_independent_emitters_are_not_fragment_merged():
    basis = np.eye(3)
    models = [
        {
            "center": np.array([float(index), 0.0, 2.0]),
            "basis": basis.copy(),
            "scales": np.array([0.3, 0.2]),
            "radiance": np.ones(3),
            "ids": np.array([index]),
            "errors": np.array([0.5]),
            "parallax": 10.0,
            "quality": {},
        }
        for index in range(6)
    ]

    merged, records = _merge_collinear_fragments(models, diagonal=10.0)

    assert merged is models
    assert records == []


def test_rejected_final_center_is_fitted_once_for_duplicate_correspondences(monkeypatch):
    from aegir import hdr_triangulation as h

    frames = make_frames()
    observations, _, _ = detect_observations(frames)
    ids = np.arange(len(observations))
    center = np.array([0.0, 0.0, 2.0])
    calls = {"score": 0, "shape": 0}
    monkeypatch.setattr(h, "_hypotheses", lambda *args: np.tile(center, (100, 1)))

    def score(points, *args):
        calls["score"] += 1
        return np.full(len(points), len(ids)), np.zeros(len(points))

    monkeypatch.setattr(h, "_support", score)
    monkeypatch.setattr(h, "_refine", lambda *args: (center, ids, np.zeros(len(ids)), 30.0))
    monkeypatch.setattr(h, "_corner_plane", lambda *args: None)

    def bad_shape(*args):
        calls["shape"] += 1
        # A fitted physical center with no matching observations is rejected.
        return np.array([100.0, 100.0, 2.0]), np.eye(3), np.ones(2), {}

    monkeypatch.setattr(h, "_silhouette_plane", bad_shape)
    with pytest.raises(ValueError, match="No HDR emitter passed"):
        seed_emitters_from_hdr_multiview(frames, ([-3.0, -3.0, -1.0], [3.0, 3.0, 4.0]))
    assert calls == {"score": 1, "shape": 1}


@pytest.mark.parametrize("mode", ["hdr_gbuffer", "hdr_multiview", "sam3_multiview"])
@pytest.mark.parametrize("initialize_only", [False, True])
def test_fresh_training_keeps_initialized_emitters_without_loading_legacy_checkpoint(
    tmp_path, monkeypatch, mode, initialize_only
):
    from aegir import trainer
    from aegir.config import AEGIRConfig

    dataset = SimpleNamespace(root=tmp_path, prior_dir=None)
    model = object()
    scene = SimpleNamespace(to=lambda device: scene, bounds=lambda quantile: (torch.zeros(3), torch.ones(3)))
    monkeypatch.setattr(trainer, "validate_predicted_depth", lambda *args: None)
    monkeypatch.setattr(trainer, "validate_sam3_ldr_view", lambda *args: None)
    monkeypatch.setattr(trainer, "validate_practical_geometry", lambda *args: None)
    monkeypatch.setattr(trainer, "validate_emitter_inclusive_geometry", lambda *args: None)
    monkeypatch.setattr(trainer, "training_inputs", lambda *args: {})
    monkeypatch.setattr(trainer, "MultiViewSceneDataset", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(trainer, "GaussianMaterialBackend", lambda *args, **kwargs: scene)
    monkeypatch.setattr(trainer, "initialize_emitters", lambda *args: model)
    monkeypatch.setattr(
        trainer,
        "associate_gaussians_with_emitters",
        lambda *args, **kwargs: {"tagged": 0, "counts": []},
    )

    def forbid_legacy(*args):
        pytest.fail("Fresh HDR initialization must not load a legacy checkpoint")

    monkeypatch.setattr(trainer, "load_legacy_emitters", forbid_legacy)
    fitted = []
    saved = []

    def make_trainer(scene, dataset, emitters, *args):
        assert emitters is model
        return SimpleNamespace(
            _save=lambda step, path: saved.append((step, Path(path).name)),
            fit=lambda: fitted.append(True),
        )

    monkeypatch.setattr(trainer, "SharedGaussianTrainer", make_trainer)
    config = AEGIRConfig()
    config.training.emitter_initialization = mode
    trainer.train_shared_gaussians(
        tmp_path / "geometry.pt",
        tmp_path,
        None,
        tmp_path / "out.pt",
        config,
        "cpu",
        initialize_only=initialize_only,
    )
    assert fitted == ([] if initialize_only else [True])
    assert saved == [(0, "aegir_initialization.pt")]
