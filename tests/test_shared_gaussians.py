from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from aegir.adaptive import adapt_emitters
from aegir.checkpoint import (
    CHECKPOINT_FORMAT,
    emitter_state_dict,
    emitters_from_state,
    load_checkpoint,
    save_checkpoint,
)
from aegir.config import AdaptiveConfig, AEGIRConfig, RenderConfig, config_from_dict
from aegir.emitter_inspection import match_emitter_centers
from aegir.emitters import AreaEmitterModel
from aegir.gaussian_scene import GaussianMaterialBackend, gaussian_surface_normals, planarize_gaussians
from aegir.initialization import _candidate_threshold
from aegir.render import DeferredRenderer
from aegir.trainer import _factor_patch_shape, curriculum_lrs


def make_export(path: Path, count: int = 4) -> None:
    torch.save(
        {
            "means": torch.tensor([[0.0, 0.0, -2.0], [0.1, 0.0, -2.0], [0.0, 0.1, -2.0], [0.1, 0.1, -2.0]])[
                :count
            ],
            "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(count, 1),
            "scales": torch.tensor([[0.1, 0.1, 0.01]]).repeat(count, 1),
            "opacities": torch.full((count, 1), 0.8),
            "features_dc_raw": torch.zeros((count, 1, 3)),
        },
        path,
    )


def test_backend_owns_one_material_per_gaussian(tmp_path: Path) -> None:
    checkpoint = tmp_path / "gaussians.pt"
    make_export(checkpoint)
    backend = GaussianMaterialBackend(checkpoint)
    assert backend.means.shape == (4, 3)
    assert backend.scales.shape == (4, 2)
    assert backend.albedo.shape == (4, 3)
    assert backend.roughness.shape == (4, 1)
    assert backend.metallic.shape == (4, 1)
    assert len(backend.geometry_parameters()) == 4
    assert len(backend.material_parameters()) == 3
    assert torch.allclose(backend.roughness, torch.full((4, 1), 0.6))
    assert torch.allclose(backend.metallic, torch.full((4, 1), 0.2))


def test_planarization_moves_smallest_axis_to_surface_normal() -> None:
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(3, 1)
    scales = torch.tensor([[0.01, 0.3, 0.2], [0.2, 0.01, 0.3], [0.3, 0.2, 0.01]])
    surfel_quats, surfel_scales = planarize_gaussians(quats, scales)
    normal = gaussian_surface_normals(surfel_quats)
    assert surfel_scales.shape == (3, 2)
    assert torch.allclose(normal, torch.eye(3), atol=1.0e-6)


def test_paper_curriculum() -> None:
    config = AEGIRConfig()
    assert curriculum_lrs(config, 0) == (1.0e-5, 1.0e-3, 1.0e-3)
    assert curriculum_lrs(config, 5_000) == (1.0e-5, 1.0e-3, 1.0e-3)
    geometry, material, emitter = curriculum_lrs(config, 10_000)
    assert geometry == 5.0e-6
    assert material == 5.0e-4
    assert emitter == 5.5e-4
    assert curriculum_lrs(config, 15_000) == (0.0, 0.0, 1.0e-4)
    assert curriculum_lrs(config, 19_999) == (0.0, 0.0, 1.0e-4)


def test_default_transport_batch_is_1024_primary_points() -> None:
    config = AEGIRConfig()
    assert config.training.rays_per_step == 1_024
    assert _factor_patch_shape(config.training.rays_per_step, 128, 128) == (32, 32)


def test_total_light_sampling_is_exact_and_balanced() -> None:
    emitters = AreaEmitterModel(torch.zeros((9, 3)), torch.ones((9, 3)))
    points = torch.zeros((2, 3))
    q, directions, distances, ids = emitters.sample_surface_total(
        points,
        128,
        generator=torch.Generator().manual_seed(4),
    )
    assert q.shape == directions.shape == (2, 128, 3)
    assert distances.shape == (2, 128, 1)
    assert ids.shape == (2, 128)
    for point_ids in ids:
        counts = torch.bincount(point_ids, minlength=9)
        assert int(counts.sum()) == 128
        assert int(counts.max() - counts.min()) <= 1


def test_rotated_anisotropic_emitter_intersection_uses_local_axes() -> None:
    half_angle = torch.tensor(torch.pi / 8.0)
    quaternion = torch.zeros((1, 4))
    quaternion[0, 0] = torch.cos(half_angle)
    quaternion[0, 3] = torch.sin(half_angle)
    emitters = AreaEmitterModel(
        centers=torch.zeros((1, 3)),
        scales=torch.tensor([[3.0, 1.0, 0.5]]),
        quaternions=quaternion,
    )
    rotation = emitters.rotations.detach()[0]
    directions = torch.stack((rotation[:, 0], rotation[:, 1], rotation[:, 2]))
    hit, distance, emitter_ids = emitters.intersect(torch.zeros((3, 3)), directions)

    assert hit.tolist() == [True, True, True]
    assert emitter_ids.tolist() == [0, 0, 0]
    assert torch.allclose(distance[:, 0], torch.tensor([3.0, 1.0, 0.5]), atol=1.0e-5)

    distance.sum().backward()
    assert emitters.raw_quaternions.grad is not None
    assert emitters.log_scales.grad is not None
    assert torch.isfinite(emitters.raw_quaternions.grad).all()
    assert torch.isfinite(emitters.log_scales.grad).all()


def test_emitter_radiance_apis_share_world_to_local_convention() -> None:
    half_angle = torch.tensor(torch.pi / 8.0)
    quaternion = torch.zeros((1, 4))
    quaternion[0, 0] = torch.cos(half_angle)
    quaternion[0, 3] = torch.sin(half_angle)
    emitters = AreaEmitterModel(
        centers=torch.zeros((1, 3)),
        scales=torch.tensor([[3.0, 1.0, 0.5]]),
        quaternions=quaternion,
        spreads=torch.tensor([[0.25, 1.0, 2.0]]),
    )
    rotation = emitters.rotations.detach()[0]
    directions = torch.stack((rotation[:, 0], rotation[:, 1], rotation[:, 2]))
    all_emitters = emitters.radiance(directions[:, None, :])[:, 0]
    selected_emitters = emitters.radiance_for_ids(
        directions,
        torch.zeros(3, dtype=torch.long),
    )

    assert torch.allclose(all_emitters, selected_emitters, atol=1.0e-7)


def test_legacy_geometry_prior_config_is_ignored() -> None:
    config = config_from_dict({"losses": {"prior_albedo": 0.75, "prior_normal": 0.05, "prior_depth": 0.05}})
    assert config.losses.prior_albedo == 0.75
    assert not hasattr(config.losses, "prior_normal")


def test_shared_checkpoint_round_trip(tmp_path: Path) -> None:
    source = tmp_path / "gaussians.pt"
    output = tmp_path / "aegir.pt"
    make_export(source)
    scene = GaussianMaterialBackend(source)
    emitters = AreaEmitterModel(torch.zeros((1, 3)), torch.full((1, 3), 0.1))
    initial_emitter_state = emitter_state_dict(emitters)
    save_checkpoint(
        output,
        scene,
        emitters,
        AEGIRConfig(),
        step=12,
        losses=[{"total": 1.0}],
        dataset_meta={"num_views": 2},
        initial_emitter_state=initial_emitter_state,
    )
    restored_scene, restored_emitters, payload = load_checkpoint(output, "cpu")
    assert payload["format"] == CHECKPOINT_FORMAT
    assert payload["step"] == 12
    assert restored_scene.gaussian_count == scene.gaussian_count
    assert restored_emitters.num_emitters == 1
    assert torch.allclose(restored_scene.albedo, scene.albedo)
    assert torch.allclose(
        payload["initial_emitter_state"]["centers"],
        initial_emitter_state["centers"],
    )


def test_v1_checkpoint_is_planarized_when_loaded(tmp_path: Path) -> None:
    source = tmp_path / "gaussians.pt"
    output = tmp_path / "legacy.pt"
    make_export(source)
    scene = GaussianMaterialBackend(source)
    emitters = AreaEmitterModel(torch.zeros((1, 3)), torch.full((1, 3), 0.1))
    save_checkpoint(output, scene, emitters, AEGIRConfig(), 0, [], {})
    payload = torch.load(output, map_location="cpu", weights_only=False)
    payload["format"] = "aegir_shared_gaussian_v1"
    payload["scene_state"]["raw_quats"] = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(scene.gaussian_count, 1)
    payload["scene_state"]["log_scales"] = (
        torch.tensor([[0.1, 0.01, 0.2]]).repeat(scene.gaussian_count, 1).log()
    )
    torch.save(payload, output)
    source.unlink()

    restored_scene, _, _ = load_checkpoint(output, "cpu")
    assert restored_scene.scales.shape == (scene.gaussian_count, 2)
    assert torch.allclose(restored_scene.normals, torch.tensor([[0.0, 1.0, 0.0]]).repeat(4, 1))


@pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("diff_surfel_rasterization") is None,
    reason="official 2DGS CUDA rasterizer is unavailable",
)
def test_2dgs_rasterizer_forward_backward(tmp_path: Path) -> None:
    source = tmp_path / "gaussians.pt"
    make_export(source, count=1)
    scene = GaussianMaterialBackend(source).cuda()
    camera = {
        "c2w": torch.eye(4, device="cuda"),
        "fx": 50.0,
        "fy": 50.0,
        "cx": 20.0,
        "cy": 24.0,
        "width": 64,
        "height": 64,
    }
    gbuffer = scene.rasterize(camera)
    loss = sum(gbuffer[key].sum() for key in ("albedo", "roughness", "metallic", "depth"))
    loss.backward()
    assert gbuffer["valid_mask"].any()
    peak_index = int(gbuffer["alpha"].argmax().item())
    assert divmod(peak_index, 64) == (24, 20)
    assert torch.allclose(gbuffer["depth"][24, 20], torch.tensor([2.0], device="cuda"))
    assert scene.scales.shape == (1, 2)
    assert all(parameter.grad is not None for parameter in scene.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in scene.parameters())


def test_emitter_matching_is_one_to_one_by_nearest_center() -> None:
    initial = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    final = torch.tensor([[9.5, 0.0, 0.0], [0.25, 0.0, 0.0]])
    matches = match_emitter_centers(initial, final)
    assert [(initial_id, final_id) for initial_id, final_id, _ in matches] == [
        (0, 1),
        (1, 0),
    ]
    assert abs(matches[0][2] - 0.25) < 1.0e-6


def test_candidate_threshold_subsamples_large_inputs() -> None:
    scores = torch.linspace(0.0, 1.0, 10_001)
    threshold = _candidate_threshold(scores, 0.995, max_quantile_samples=1_000)
    assert 0.98 < float(threshold) < 1.0


def test_emitter_split_replaces_parent_and_preserves_intensity() -> None:
    emitters = AreaEmitterModel(
        torch.zeros((1, 3)),
        torch.tensor([[1.0, 0.1, 0.1]]),
        torch.full((1, 3), 1000.0),
    )
    initial_color_sum = emitters.colors.sum(dim=0)

    changed = adapt_emitters(
        emitters,
        (torch.full((3,), -2.0), torch.full((3,), 2.0)),
        scene_scale=4.0,
        config=AdaptiveConfig(max_emitters=16),
    )

    assert changed
    assert emitters.num_emitters == 2
    assert torch.allclose(emitters.centers[:, 0].sort().values, torch.tensor([-0.5, 0.5]))
    assert torch.allclose(emitters.colors.sum(dim=0), initial_color_sum)


def test_ambient_is_absent_and_legacy_payload_is_ignored() -> None:
    emitters = AreaEmitterModel(
        torch.zeros((1, 3)),
        torch.ones((1, 3)),
    )
    assert not hasattr(emitters, "ambient_color")
    assert all("ambient" not in name for name, _ in emitters.named_parameters())

    state = emitter_state_dict(emitters)
    assert "ambient_color" not in state
    state["ambient_color"] = torch.full((3,), 100.0)
    restored = emitters_from_state(state, "cpu")
    assert not hasattr(restored, "ambient_color")
    assert "ambient_color" not in emitter_state_dict(restored)

    config = config_from_dict(
        {
            "losses": {"light_ambient": 1.0},
            "training": {"ambient_init": 10.0},
        }
    )
    assert not hasattr(config.losses, "light_ambient")
    assert not hasattr(config.training, "ambient_init")


def test_renderer_has_no_ambient_transport_or_outputs() -> None:
    renderer = DeferredRenderer(
        RenderConfig(
            num_brdf_samples=1,
            num_light_samples=1,
            point_visibility=False,
            indirect_bounce=False,
        )
    )
    emitters = AreaEmitterModel(
        torch.tensor([[0.0, 0.0, 2.0]]),
        torch.full((1, 3), 0.5),
    )
    gbuffer = {
        "points": torch.tensor([[0.0, 0.0, 0.0]]),
        "normals": torch.tensor([[0.0, 0.0, 1.0]]),
        "view_dirs": torch.tensor([[0.0, 0.0, 1.0]]),
        "albedo": torch.full((1, 3), 0.5),
        "roughness": torch.full((1, 1), 0.5),
        "metallic": torch.zeros((1, 1)),
    }
    rendered = renderer.render(
        gbuffer,
        emitters,
        generator=torch.Generator().manual_seed(7),
        tonemap=False,
    )

    assert all("ambient" not in key for key in rendered)
    assert torch.allclose(
        rendered["hdr"],
        rendered["direct_hdr"] + rendered["indirect_hdr"],
    )
