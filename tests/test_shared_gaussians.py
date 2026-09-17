from __future__ import annotations

import importlib.util
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from aegir import initialization
from aegir.adaptive import adapt_emitters, migrate_emitter_adam
from aegir.checkpoint import (
    CHECKPOINT_FORMAT,
    emitter_state_dict,
    emitters_from_state,
    load_checkpoint,
    save_checkpoint,
)
from aegir.config import (
    AdaptiveConfig,
    AEGIRConfig,
    RenderConfig,
    SceneCullingConfig,
    config_from_dict,
)
from aegir.emitter_inspection import match_emitter_centers
from aegir.emitters import AreaEmitterModel
from aegir.gaussian_scene import GaussianMaterialBackend, gaussian_surface_normals, planarize_gaussians
from aegir.initialization import _candidate_threshold
from aegir.losses import material_prior_loss
from aegir.render import DeferredRenderer
from aegir.scene_culling import (
    migrate_refined_adam,
    refine_scene_gaussians,
    should_refine_scene,
)
from aegir.trainer import (
    _factor_patch_shape,
    configure_emitter_optimization,
    configure_scene_geometry_optimization,
    constrain_emitter_scales,
    constrain_scene_scales,
    curriculum_lrs,
    full_view_material_prior_loss,
    scheduled_light_white_weight,
    training_sh_proxy_weight,
)


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
            "features_rest_raw": torch.zeros((count, 15, 3)),
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
    assert backend.training_sh_coeffs.shape == (4, 16, 3)
    assert backend.training_sh_degree == 3
    assert not backend.training_sh_coeffs.requires_grad


def test_scene_geometry_optimization_can_select_scale_only(tmp_path: Path) -> None:
    checkpoint = tmp_path / "gaussians.pt"
    make_export(checkpoint)
    scene = GaussianMaterialBackend(checkpoint)
    config = AEGIRConfig()
    config.training.optimize_scene_means = False
    config.training.optimize_scene_scales = True
    config.training.optimize_scene_rotations = False
    config.training.optimize_scene_opacities = False

    trainability = configure_scene_geometry_optimization(scene, config)

    assert not scene.means.requires_grad
    assert scene.log_scales.requires_grad
    assert not scene.raw_quats.requires_grad
    assert not scene.opacity_logits.requires_grad
    assert trainability == {
        id(scene.means): False,
        id(scene.log_scales): True,
        id(scene.raw_quats): False,
        id(scene.opacity_logits): False,
    }


def test_scene_scale_constraint_uses_initial_relative_bounds(tmp_path: Path) -> None:
    checkpoint = tmp_path / "gaussians.pt"
    make_export(checkpoint)
    scene = GaussianMaterialBackend(checkpoint)
    anchors = scene.scales.detach().clone()
    config = AEGIRConfig()
    config.training.constrain_scene_scales = True
    config.training.scene_scale_min_ratio = 0.9
    config.training.scene_scale_max_ratio = 1.1
    scene.log_scales.data[0] = (anchors[0] * 2.0).log()
    scene.log_scales.data[1] = (anchors[1] * 0.5).log()

    constrain_scene_scales(scene, anchors, config)

    assert torch.allclose(scene.scales[0], anchors[0] * 1.1)
    assert torch.allclose(scene.scales[1], anchors[1] * 0.9)


def test_neutral_albedo_initialization_ignores_dn_sh_dc(tmp_path: Path) -> None:
    checkpoint = tmp_path / "gaussians.pt"
    make_export(checkpoint)
    payload = torch.load(checkpoint, weights_only=False)
    payload["features_dc_raw"] = torch.full((4, 1, 3), 1.0)
    torch.save(payload, checkpoint)

    appearance_seeded = GaussianMaterialBackend(checkpoint)
    neutral_seeded = GaussianMaterialBackend(
        checkpoint,
        albedo_initialization="neutral",
        neutral_albedo=0.5,
    )

    assert not torch.allclose(appearance_seeded.albedo, torch.full((4, 3), 0.5))
    assert torch.allclose(neutral_seeded.albedo, torch.full((4, 3), 0.5))


def test_native_2dgs_checkpoint_skips_3d_planarization(tmp_path: Path) -> None:
    checkpoint = tmp_path / "native_2dgs.pt"
    scales = torch.tensor([[0.07, 0.13]]).repeat(4, 1)
    torch.save(
        {
            "representation": "2dgs",
            "means": torch.zeros((4, 3)),
            "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(4, 1),
            "scales": scales,
            "opacities_logits": torch.full((4, 1), 1.0),
            "features_dc_raw": torch.zeros((4, 1, 3)),
            "features_rest_raw": torch.zeros((4, 15, 3)),
        },
        checkpoint,
    )
    backend = GaussianMaterialBackend(checkpoint)
    assert torch.allclose(backend.scales, scales)
    assert torch.equal(backend.quats, torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(4, 1))


def test_representation_scale_mismatch_is_rejected(tmp_path: Path) -> None:
    checkpoint = tmp_path / "invalid_2dgs.pt"
    make_export(checkpoint)
    payload = torch.load(checkpoint, weights_only=False)
    payload["representation"] = "2dgs"
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="representation/scale mismatch"):
        GaussianMaterialBackend(checkpoint)


def test_dn_scene_refinement_splits_large_rows_and_preserves_survivor_adam(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "gaussians.pt"
    make_export(checkpoint)
    scene = GaussianMaterialBackend(checkpoint)
    scene.opacity_logits.data.copy_(
        torch.logit(torch.tensor([[0.05], [0.8], [0.8], [0.8]]))
    )
    scene.log_scales.data[1] = torch.tensor([1.2, 0.1]).log()
    geometry = torch.optim.Adam(scene.geometry_parameters(), lr=1.0e-4)
    material = torch.optim.Adam(
        [
            {"params": scene.albedo_parameters(), "lr": 6.0e-4},
            {"params": scene.non_albedo_material_parameters(), "lr": 2.0e-4},
        ]
    )
    loss = sum(parameter.square().mean() for parameter in scene.parameters())
    loss.backward()
    geometry.step()
    material.step()
    old_geometry_state = geometry.state[scene.means]["exp_avg"].detach().clone()
    old_material_state = material.state[scene.albedo_logits]["exp_avg"].detach().clone()

    result = refine_scene_gaussians(
        scene,
        completed_steps=3_100,
        average_xy_gradient=torch.tensor([0.0, 1.0, 1.0, 0.0]),
        max_2d_size=torch.tensor([0.0, 0.0, 0.2, 0.0]),
        config=SceneCullingConfig(
            opacity_threshold=0.1,
            densify_grad_threshold=0.5,
            densify_size_threshold=1.0,
            split_screen_size_threshold=0.15,
            max_gaussians=16,
        ),
        generator=torch.Generator().manual_seed(3),
    )
    geometry = migrate_refined_adam(
        geometry,
        result.old_geometry_parameters,
        result.new_geometry_parameters,
        result,
    )
    material = migrate_refined_adam(
        material,
        result.old_material_parameters,
        result.new_material_parameters,
        result,
    )

    assert material.param_groups[0]["params"] == scene.albedo_parameters()
    assert material.param_groups[1]["params"] == scene.non_albedo_material_parameters()

    assert result.changed
    assert result.pruned_below_opacity == 1
    assert result.split_parents == 2
    assert result.split_children == 4
    assert result.duplicated == 0
    assert result.survivor_old_indices.tolist() == [3]
    assert result.child_mask.tolist() == [False, True, True, True, True]
    assert scene.gaussian_count == 5
    assert scene.training_sh_coeffs.shape[0] == 5
    assert torch.equal(geometry.state[scene.means]["exp_avg"][0], old_geometry_state[3])
    assert torch.count_nonzero(geometry.state[scene.means]["exp_avg"][1:]) == 0
    assert torch.equal(
        material.state[scene.albedo_logits]["exp_avg"][0], old_material_state[3]
    )
    assert torch.count_nonzero(material.state[scene.albedo_logits]["exp_avg"][1:]) == 0


def test_large_scene_gaussian_without_high_gradient_is_not_deleted(tmp_path: Path) -> None:
    checkpoint = tmp_path / "gaussians.pt"
    make_export(checkpoint, count=1)
    scene = GaussianMaterialBackend(checkpoint)
    scene.log_scales.data[0] = torch.tensor([2.0, 1.0]).log()
    original = scene.means.detach().clone()

    result = refine_scene_gaussians(
        scene,
        completed_steps=3_100,
        average_xy_gradient=torch.zeros(1),
        max_2d_size=torch.ones(1),
        config=SceneCullingConfig(opacity_threshold=0.005),
    )

    assert not result.changed
    assert scene.gaussian_count == 1
    assert torch.equal(scene.means, original)


def test_scene_culling_schedule_and_legacy_config() -> None:
    config = SceneCullingConfig()
    assert not should_refine_scene(500, config)
    assert should_refine_scene(600, config)
    assert not should_refine_scene(650, config)
    assert not should_refine_scene(15_000, config)
    assert AEGIRConfig().scene_culling.enabled
    assert not config_from_dict({}).scene_culling.enabled


def test_planarization_moves_smallest_axis_to_surface_normal() -> None:
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(3, 1)
    scales = torch.tensor([[0.01, 0.3, 0.2], [0.2, 0.01, 0.3], [0.3, 0.2, 0.01]])
    surfel_quats, surfel_scales = planarize_gaussians(quats, scales)
    normal = gaussian_surface_normals(surfel_quats)
    assert surfel_scales.shape == (3, 2)
    assert torch.allclose(normal, torch.eye(3), atol=1.0e-6)


def test_paper_curriculum() -> None:
    config = AEGIRConfig()
    assert curriculum_lrs(config, 0) == (1.0e-5, 3.0e-3, 1.0e-3, 1.0e-3)
    assert curriculum_lrs(config, 5_000) == (1.0e-5, 3.0e-3, 1.0e-3, 1.0e-3)
    geometry, albedo, material, emitter = curriculum_lrs(config, 10_000)
    assert geometry == 5.0e-6
    assert albedo == 1.5e-3
    assert material == 5.0e-4
    assert emitter == 5.5e-4
    assert curriculum_lrs(config, 15_000) == (0.0, 0.0, 0.0, 1.0e-4)
    assert curriculum_lrs(config, 19_999) == (0.0, 0.0, 0.0, 1.0e-4)


def test_white_light_weight_is_disabled_by_default_and_can_ramp() -> None:
    config = AEGIRConfig()
    assert scheduled_light_white_weight(config, 0) == 0.0
    config.losses.light_white = 0.1
    config.training.light_white_ramp_end = 15_000
    assert scheduled_light_white_weight(config, 0) == 0.0
    assert scheduled_light_white_weight(config, 3_750) == 0.015625
    assert scheduled_light_white_weight(config, 7_500) == 0.05
    assert scheduled_light_white_weight(config, 15_000) == 0.1
    assert scheduled_light_white_weight(config, 19_999) == 0.1


def test_default_emitter_policy_fixes_pose_and_constrains_scale() -> None:
    config = AEGIRConfig()
    emitters = AreaEmitterModel(
        centers=torch.tensor([[1.0, 2.0, 3.0]]),
        scales=torch.tensor([[2.0, 1.0]]),
    )
    parameters = configure_emitter_optimization(emitters, config)

    assert not emitters.centers.requires_grad
    assert not emitters.raw_quaternions.requires_grad
    assert emitters.log_scales.requires_grad
    assert any(parameter is emitters.log_colors for parameter in parameters)
    assert len(parameters) == 5
    assert any(parameter is emitters.log_axis_gains for parameter in parameters)
    assert any(parameter is emitters.log_spreads for parameter in parameters)
    assert any(parameter is emitters.log_falloff_powers for parameter in parameters)
    assert not config.adaptive.enabled

    anchors = emitters.scales.detach().clone()
    emitters.log_scales.data.copy_((anchors * 2.0).log())
    constrain_emitter_scales(emitters, anchors, config)
    assert torch.allclose(emitters.scales, anchors)

    emitters.log_scales.data.copy_((anchors * 0.01).log())
    constrain_emitter_scales(emitters, anchors, config)
    assert torch.allclose(emitters.scales, anchors * 0.1)


def test_training_sh_proxy_fades_before_inference() -> None:
    config = AEGIRConfig()
    assert training_sh_proxy_weight(config, 0) == 1.0
    assert training_sh_proxy_weight(config, 5_000) == 1.0
    assert training_sh_proxy_weight(config, 7_500) == 0.5
    assert training_sh_proxy_weight(config, 10_000) == 0.0
    assert training_sh_proxy_weight(config, 15_000) == 0.0
    assert training_sh_proxy_weight(config, 19_999) == 0.0
    config.render.indirect_bounce = False
    assert training_sh_proxy_weight(config, 0) == 0.0


def test_training_transport_ablation_modes_are_exact_and_persisted() -> None:
    config = AEGIRConfig()
    config.training.training_transport_mode = "secondary_only"
    assert training_sh_proxy_weight(config, 0) == 0.0
    assert training_sh_proxy_weight(config, 19_999) == 0.0
    restored = config_from_dict(asdict(config))
    assert restored.training.training_transport_mode == "secondary_only"

    config.training.training_transport_mode = "sh_proxy_only"
    assert training_sh_proxy_weight(config, 0) == 1.0
    assert training_sh_proxy_weight(config, 19_999) == 1.0

    config.training.training_transport_mode = "unknown"
    with pytest.raises(ValueError, match="training transport mode"):
        training_sh_proxy_weight(config, 0)


def test_default_transport_budget_is_2_pow_16_rays() -> None:
    config = AEGIRConfig()
    primary_points = config.training.rays_per_step
    samples_per_point = config.render.num_brdf_samples + config.render.num_light_samples
    assert primary_points == 256
    assert samples_per_point == 256
    assert primary_points * samples_per_point == 65_536
    assert _factor_patch_shape(primary_points, 128, 128) == (16, 16)
    assert config.render.bounce_chunk_size == 65_536


def test_material_prior_supervises_pixels_outside_transport_patch() -> None:
    albedo = torch.zeros((8, 8, 3), requires_grad=True)
    prior = torch.zeros_like(albedo)
    prior[4:, 4:] = 1.0
    mask = torch.ones((8, 8, 1), dtype=torch.bool)

    # A hypothetical top-left transport patch contains no material error.
    patch_loss = material_prior_loss(
        {"albedo": albedo[:2, :2]},
        {"albedo": prior[:2, :2]},
        AEGIRConfig().losses,
        mask[:2, :2],
    )
    full_loss = full_view_material_prior_loss(
        {"albedo": albedo},
        {"albedo": prior},
        AEGIRConfig().losses,
        mask,
    )

    assert patch_loss.item() == 0.0
    assert full_loss.item() > 0.0
    full_loss.backward()
    assert torch.count_nonzero(albedo.grad[4:, 4:]) > 0


def test_thin_emitter_seed_floor_is_two_per_mille_of_scene_diagonal() -> None:
    assert AEGIRConfig().training.emitter_min_scale_fraction == pytest.approx(0.002)


def test_blender_bbox_fits_exactly_one_planar_ellipse() -> None:
    corners = torch.tensor(
        [
            [x, y, z]
            for x in (-2.0, 2.0)
            for y in (-1.0, 1.0)
            for z in (-0.01, 0.01)
        ]
    )
    center, scales, quaternion = initialization._fit_planar_bbox(
        corners,
        camera_positions=torch.tensor([[0.0, 0.0, -3.0]]),
    )
    emitter = AreaEmitterModel(center[None], scales[None], quaternions=quaternion[None])
    assert torch.allclose(center, torch.zeros(3))
    assert torch.allclose(scales, torch.tensor([2.0, 1.0]))
    assert float(emitter.normals[0, 2].detach()) < -0.999


def test_planar_mask_seed_can_restore_pre_v13_ellipsoid_thickness() -> None:
    planar = AreaEmitterModel(
        centers=torch.zeros((1, 3)),
        scales=torch.tensor([[0.30, 0.15]]),
    )
    corners = torch.tensor(
        [
            [x, y, z]
            for x in (-0.30, 0.30)
            for y in (-0.15, 0.15)
            for z in (-0.0125, 0.0125)
        ]
    )
    restored = initialization.restore_ellipsoid_support(
        planar,
        [{"model_bbox_corners": corners}],
        scene_scale=1.0,
        min_scale_fraction=0.002,
    )

    assert restored.representation == "ellipsoid_3d"
    torch.testing.assert_close(restored.scales, torch.tensor([[0.30, 0.15, 0.0125]]))
    hit, distance, _ = restored.intersect(
        torch.tensor([[0.0, 0.0, 1.0]]),
        torch.tensor([[0.0, 0.0, -1.0]]),
    )
    assert hit.item()
    assert distance.item() == pytest.approx(0.9875, abs=1.0e-5)

    points = torch.tensor([[0.0, 0.0, -1.0], [0.5, 0.0, 0.0]])
    q, directions, distances, ids = restored.sample_surface_total(
        points, 128, generator=torch.Generator().manual_seed(3)
    )
    assert q.shape == directions.shape == (2, 128, 3)
    assert distances.shape == (2, 128, 1)
    assert ids.shape == (2, 128)


def test_total_light_sampling_is_exact_and_balanced() -> None:
    emitters = AreaEmitterModel(torch.zeros((9, 3)), torch.ones((9, 2)))
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


def test_training_transport_crossfades_sh_to_explicit_nee(monkeypatch) -> None:
    calls = []

    def trace(origins, directions, distances, training_proxy):
        calls.append(training_proxy)
        shape = directions.shape[:-1] + (1,)
        return {
            "visibility": torch.full(shape, 0.25),
            "training_bounce": torch.full_like(directions, 1.5),
        }

    renderer = DeferredRenderer(
        RenderConfig(indirect_bounce=True),
        transport_fn=trace,
    )
    monkeypatch.setattr(
        renderer,
        "_explicit_secondary_bounce",
        lambda primary_trace, primary_dirs, emitters, generator: torch.full_like(
            primary_dirs, 2.0
        ),
    )
    emitters = AreaEmitterModel(torch.zeros((1, 3)), torch.ones((1, 2)))
    direct, indirect = renderer._incident_transport(
        points=torch.zeros((1, 3)),
        dirs=torch.tensor([[[0.0, 0.0, 1.0]]]),
        distances=torch.ones((1, 1, 1)),
        direct_radiance=torch.full((1, 1, 3), 4.0),
        emitters=emitters,
        occluder_points=None,
        generator=None,
        transport_mode="training",
    )

    assert calls == [True]
    assert torch.equal(direct, torch.ones_like(direct))
    assert torch.equal(indirect, torch.full_like(indirect, 1.5))
    assert not hasattr(renderer.config, "bounce_strength")

    calls.clear()
    _, faded_indirect = renderer._incident_transport(
        points=torch.zeros((1, 3)),
        dirs=torch.tensor([[[0.0, 0.0, 1.0]]]),
        distances=torch.ones((1, 1, 1)),
        direct_radiance=torch.full((1, 1, 3), 4.0),
        emitters=emitters,
        occluder_points=None,
        generator=None,
        transport_mode="training",
        training_proxy_weight=0.25,
    )
    assert calls == [True]
    assert torch.equal(faded_indirect, torch.full_like(faded_indirect, 1.875))

    calls.clear()
    _, explicit_indirect = renderer._incident_transport(
        points=torch.zeros((1, 3)),
        dirs=torch.tensor([[[0.0, 0.0, 1.0]]]),
        distances=torch.ones((1, 1, 1)),
        direct_radiance=torch.full((1, 1, 3), 4.0),
        emitters=emitters,
        occluder_points=None,
        generator=None,
        transport_mode="training",
        training_proxy_weight=0.0,
    )
    assert calls == [False]
    assert torch.equal(explicit_indirect, torch.full_like(explicit_indirect, 2.0))


def test_explicit_secondary_nee_is_nonzero_and_differentiable() -> None:
    def trace(origins, directions, distances, training_proxy):
        shape = directions.shape[:-1]
        return {
            "visibility": directions.new_zeros(shape + (1,)),
            "training_bounce": directions.new_zeros(shape + (3,)),
            "occlusion_weight": directions.new_ones(shape + (1,)),
            "hit_points": directions.new_zeros(shape + (3,)),
            "hit_normals": directions.new_tensor([0.0, 0.0, 1.0]).expand(shape + (3,)),
            "hit_materials": directions.new_tensor(
                [0.8, 0.7, 0.6, 0.5, 0.0]
            ).expand(shape + (5,)),
            "hit_alpha": directions.new_ones(shape + (1,)),
        }

    renderer = DeferredRenderer(
        RenderConfig(indirect_bounce=True, num_bounce_samples=4),
        visibility_fn=lambda origins, directions, distances: torch.ones_like(distances),
        transport_fn=trace,
    )
    emitters = AreaEmitterModel(
        centers=torch.tensor([[0.4, 0.1, 2.0]]),
        scales=torch.tensor([[0.2, 0.1]]),
        colors=torch.full((1, 3), 2.0),
        quaternions=torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
    )
    _, indirect = renderer._incident_transport(
        points=torch.zeros((1, 3)),
        dirs=torch.tensor([[[0.0, 0.0, -1.0]]]),
        distances=torch.ones((1, 1, 1)),
        direct_radiance=torch.zeros((1, 1, 3)),
        emitters=emitters,
        occluder_points=None,
        generator=torch.Generator().manual_seed(7),
        transport_mode="training",
        training_proxy_weight=0.0,
    )

    assert torch.isfinite(indirect).all()
    assert float(indirect.detach().sum()) > 0.0
    indirect.sum().backward()
    for parameter in (emitters.centers, emitters.log_scales, emitters.log_colors):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert float(parameter.grad.abs().sum()) > 0.0


def test_explicit_secondary_nee_is_chunk_size_invariant() -> None:
    def trace(origins, directions, distances, training_proxy):
        shape = directions.shape[:-1]
        return {
            "visibility": directions.new_zeros(shape + (1,)),
            "training_bounce": directions.new_zeros(shape + (3,)),
            "occlusion_weight": directions.new_ones(shape + (1,)),
            "hit_points": directions.new_zeros(shape + (3,)),
            "hit_normals": directions.new_tensor([0.0, 0.0, 1.0]).expand(shape + (3,)),
            "hit_materials": directions.new_tensor(
                [0.8, 0.7, 0.6, 0.5, 0.0]
            ).expand(shape + (5,)),
            "hit_alpha": directions.new_ones(shape + (1,)),
        }

    emitters = AreaEmitterModel(
        centers=torch.tensor([[0.4, 0.1, 2.0]]),
        scales=torch.tensor([[0.2, 0.1]]),
        colors=torch.full((1, 3), 2.0),
        quaternions=torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
    )
    outputs = []
    for chunk_size in (1, 65_536):
        renderer = DeferredRenderer(
            RenderConfig(
                indirect_bounce=True,
                num_bounce_samples=4,
                bounce_chunk_size=chunk_size,
            ),
            visibility_fn=lambda origins, directions, distances: torch.ones_like(distances),
            transport_fn=trace,
        )
        _, indirect = renderer._incident_transport(
            points=torch.zeros((2, 3)),
            dirs=torch.tensor([[[0.0, 0.0, -1.0]], [[0.0, 0.0, -1.0]]]),
            distances=torch.ones((2, 1, 1)),
            direct_radiance=torch.zeros((2, 1, 3)),
            emitters=emitters,
            occluder_points=None,
            generator=torch.Generator().manual_seed(17),
            transport_mode="training",
            training_proxy_weight=0.0,
        )
        outputs.append(indirect.detach())

    assert torch.equal(outputs[0], outputs[1])


def test_inference_transport_never_requests_training_sh_and_miss_is_zero() -> None:
    calls = []

    def trace(origins, directions, distances, training_proxy):
        calls.append(training_proxy)
        shape = directions.shape[:-1]
        return {
            "visibility": directions.new_zeros(shape + (1,)),
            "training_bounce": directions.new_full(shape + (3,), 1_000.0),
            "occlusion_weight": directions.new_ones(shape + (1,)),
            "hit_points": directions.new_zeros(shape + (3,)),
            "hit_normals": directions.new_tensor([0.0, 0.0, 1.0]).expand(shape + (3,)),
            "hit_materials": directions.new_tensor([0.5, 0.5, 0.5, 0.6, 0.2]).expand(shape + (5,)),
            "hit_alpha": directions.new_ones(shape + (1,)),
        }

    renderer = DeferredRenderer(
        RenderConfig(indirect_bounce=True, num_bounce_samples=1),
        visibility_fn=lambda origins, directions, distances: torch.ones_like(distances),
        transport_fn=trace,
    )
    # The emitter is below the secondary surface, so NEE samples it but the
    # clamped surface cosine is zero.
    emitters = AreaEmitterModel(
        centers=torch.tensor([[0.0, 0.0, -10.0]]),
        scales=torch.full((1, 2), 0.01),
    )
    direct, indirect = renderer._incident_transport(
        points=torch.zeros((1, 3)),
        dirs=torch.tensor([[[0.0, 0.0, 1.0]]]),
        distances=torch.ones((1, 1, 1)),
        direct_radiance=torch.full((1, 1, 3), 4.0),
        emitters=emitters,
        occluder_points=None,
        generator=torch.Generator().manual_seed(5),
        transport_mode="inference",
    )

    assert calls == [False]
    assert torch.count_nonzero(direct) == 0
    assert torch.count_nonzero(indirect) == 0

    enclosing_emitter = AreaEmitterModel(
        centers=torch.tensor([[0.0, 0.0, 2.0]]),
        scales=torch.full((1, 2), 0.1),
        colors=torch.full((1, 3), 2.0),
        quaternions=torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
    )
    _, explicit_bounce = renderer._incident_transport(
        points=torch.zeros((1, 3)),
        dirs=torch.tensor([[[0.0, 0.0, 1.0]]]),
        distances=torch.ones((1, 1, 1)),
        direct_radiance=torch.zeros((1, 1, 3)),
        emitters=enclosing_emitter,
        occluder_points=None,
        generator=torch.Generator().manual_seed(5),
        transport_mode="inference",
    )
    assert calls == [False, False]
    assert torch.isfinite(explicit_bounce).all()
    assert float(explicit_bounce.detach().sum()) > 0.0


def test_rotated_planar_emitter_intersection_uses_bounded_local_ellipse() -> None:
    half_angle = torch.tensor(torch.pi / 8.0)
    quaternion = torch.zeros((1, 4))
    quaternion[0, 0] = torch.cos(half_angle)
    quaternion[0, 3] = torch.sin(half_angle)
    emitters = AreaEmitterModel(
        centers=torch.zeros((1, 3)),
        scales=torch.tensor([[3.0, 1.0]]),
        quaternions=quaternion,
    )
    rotation = emitters.rotations.detach()[0]
    normal = rotation[:, 2]
    origins = torch.stack(
        (
            2.5 * rotation[:, 0] + 2.0 * normal,
            0.9 * rotation[:, 1] + 2.0 * normal,
            1.1 * rotation[:, 1] + 2.0 * normal,
            0.9 * rotation[:, 1] - 2.0 * normal,
        )
    )
    directions = torch.stack((-normal, -normal, -normal, normal))
    hit, distance, emitter_ids = emitters.intersect(origins, directions)

    assert hit.tolist() == [True, True, False, True]
    assert emitter_ids.tolist() == [0, 0, 0, 0]
    assert torch.allclose(distance[[0, 1, 3], 0], torch.full((3,), 2.0), atol=1.0e-5)
    assert torch.isinf(distance[2, 0])


def test_planar_emitter_directional_pdf_uses_both_sides() -> None:
    emitters = AreaEmitterModel(torch.zeros((1, 3)), torch.ones((1, 2)))
    distances = torch.full((2, 1), 2.0)
    ids = torch.zeros(2, dtype=torch.long)
    directions = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    pdf = emitters.light_pdf_surrogate(distances, ids, directions)
    assert torch.allclose(pdf[0], pdf[1])


def test_emitter_radiance_apis_share_world_to_local_convention() -> None:
    half_angle = torch.tensor(torch.pi / 8.0)
    quaternion = torch.zeros((1, 4))
    quaternion[0, 0] = torch.cos(half_angle)
    quaternion[0, 3] = torch.sin(half_angle)
    emitters = AreaEmitterModel(
        centers=torch.zeros((1, 3)),
        scales=torch.tensor([[3.0, 1.0]]),
        quaternions=quaternion,
    )
    rotation = emitters.rotations.detach()[0]
    directions = torch.stack((rotation[:, 2], -rotation[:, 2]))
    all_emitters = emitters.radiance(directions[:, None, :])[:, 0]
    selected_emitters = emitters.radiance_for_ids(
        directions,
        torch.zeros(2, dtype=torch.long),
    )

    assert torch.allclose(all_emitters, selected_emitters, atol=1.0e-7)


def test_planar_emission_learns_front_and_back_independently() -> None:
    emitters = AreaEmitterModel(
        centers=torch.zeros((1, 3)),
        scales=torch.ones((1, 2)),
        colors=torch.tensor([[3.0, 2.0, 1.0]]),
        axis_gains=torch.tensor([[1.0, 1.0, 1.0, 1.0, 4.0, 0.25]]),
        spreads=torch.tensor([[1.0, 1.0, 2.0]]),
    )
    directions = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    radiance = emitters.radiance_for_ids(directions, torch.zeros(2, dtype=torch.long))
    assert torch.all(radiance > 0.0)
    assert torch.all(radiance[0] > radiance[1])

    radiance.sum().backward()
    assert torch.count_nonzero(emitters.log_axis_gains.grad) > 0
    assert torch.count_nonzero(emitters.log_spreads.grad) > 0
    assert torch.count_nonzero(emitters.log_falloff_powers.grad) > 0


def test_planar_emission_receives_color_gradient() -> None:
    emitters = AreaEmitterModel(
        centers=torch.zeros((1, 3)),
        scales=torch.ones((1, 2)),
        colors=torch.ones((1, 3)),
    )
    radiance = emitters.radiance_for_ids(
        torch.tensor([[0.0, 0.0, 1.0]]),
        torch.zeros(1, dtype=torch.long),
    )
    radiance.sum().backward()

    assert emitters.log_colors.grad is not None
    assert torch.isfinite(emitters.log_colors.grad).all()
    assert torch.count_nonzero(emitters.log_colors.grad) > 0


def test_light_sampling_evaluates_emitter_to_surface_outgoing_direction(monkeypatch) -> None:
    renderer = DeferredRenderer(
        RenderConfig(
            num_brdf_samples=1,
            num_light_samples=1,
            point_visibility=False,
            indirect_bounce=False,
        )
    )
    emitters = AreaEmitterModel(
        centers=torch.tensor([[0.0, 0.0, 1.0]]),
        scales=torch.full((1, 2), 0.1),
        quaternions=torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
    )
    surface_to_emitter = torch.tensor([[[0.0, 0.0, 1.0]]])

    def fixed_sample(points, num_samples, generator=None):
        del generator
        assert num_samples == 1
        q = points[:, None, :] + surface_to_emitter
        return (
            q,
            surface_to_emitter.expand(points.shape[0], -1, -1),
            torch.ones((points.shape[0], 1, 1)),
            torch.zeros((points.shape[0], 1), dtype=torch.long),
        )

    outgoing_directions = []
    original_radiance = emitters.radiance_for_ids

    def capture_radiance(omega_out, emitter_ids, **kwargs):
        outgoing_directions.append(omega_out.detach().clone())
        return original_radiance(omega_out, emitter_ids, **kwargs)

    monkeypatch.setattr(emitters, "sample_surface_total", fixed_sample)
    monkeypatch.setattr(emitters, "radiance_for_ids", capture_radiance)
    renderer._light_samples(
        points=torch.zeros((1, 3)),
        normals=torch.tensor([[0.0, 0.0, 1.0]]),
        view_dirs=torch.tensor([[0.0, 0.0, 1.0]]),
        albedo=torch.full((1, 3), 0.5),
        roughness=torch.full((1, 1), 0.5),
        metallic=torch.zeros((1, 1)),
        emitters=emitters,
        occluder_points=None,
    )

    assert len(outgoing_directions) == 1
    assert torch.equal(outgoing_directions[0], -surface_to_emitter)


def test_legacy_geometry_prior_config_is_ignored() -> None:
    config = config_from_dict({"losses": {"prior_albedo": 0.75, "prior_normal": 0.05, "prior_depth": 0.05}})
    assert config.losses.prior_albedo == 0.75
    assert not hasattr(config.losses, "prior_normal")


def test_shared_checkpoint_round_trip(tmp_path: Path) -> None:
    source = tmp_path / "gaussians.pt"
    output = tmp_path / "aegir.pt"
    make_export(source)
    scene = GaussianMaterialBackend(source)
    emitters = AreaEmitterModel(torch.zeros((1, 3)), torch.full((1, 2), 0.1))
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
    assert restored_emitters.scales.shape == (1, 2)
    assert torch.allclose(restored_scene.albedo, scene.albedo)
    assert torch.allclose(
        payload["initial_emitter_state"]["centers"],
        initial_emitter_state["centers"],
    )


def test_v8_ellipsoid_checkpoint_preserves_directional_planar_emitter(tmp_path: Path) -> None:
    source = tmp_path / "gaussians.pt"
    output = tmp_path / "v8.pt"
    make_export(source)
    scene = GaussianMaterialBackend(source)
    emitters = AreaEmitterModel(torch.zeros((1, 3)), torch.full((1, 2), 0.1))
    save_checkpoint(output, scene, emitters, AEGIRConfig(), 0, [], {})
    payload = torch.load(output, map_location="cpu", weights_only=False)
    payload["format"] = "aegir_shared_gaussian_v8"
    payload["emitter_state"] = {
        "centers": torch.zeros((1, 3)),
        "quaternions": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        "scales": torch.tensor([[2.0, 1.0, 0.1]]),
        "colors": torch.tensor([[3.0, 2.0, 1.0]]),
        "spreads": torch.ones((1, 3)),
        "falloff_powers": torch.ones((1, 1)),
        "axis_gains": torch.tensor([[1.0, 1.0, 1.0, 1.0, 4.0, 0.25]]),
    }
    torch.save(payload, output)

    _, restored_emitters, restored_payload = load_checkpoint(output, "cpu")
    assert restored_payload["format"] == "aegir_shared_gaussian_v8"
    assert torch.allclose(restored_emitters.scales, torch.tensor([[2.0, 1.0]]))
    front = restored_emitters.normals
    front_radiance = restored_emitters.radiance_for_ids(front, torch.zeros(1, dtype=torch.long))
    back_radiance = restored_emitters.radiance_for_ids(-front, torch.zeros(1, dtype=torch.long))
    assert torch.all(front_radiance > 0.0)
    assert torch.all(front_radiance > back_radiance)


def test_v13_planar_checkpoint_preserves_front_radiance_with_neutral_directionality(
    tmp_path: Path,
) -> None:
    source = tmp_path / "gaussians.pt"
    output = tmp_path / "v13.pt"
    make_export(source)
    scene = GaussianMaterialBackend(source)
    emitters = AreaEmitterModel(
        torch.zeros((1, 3)),
        torch.full((1, 2), 0.1),
        colors=torch.tensor([[3.0, 2.0, 1.0]]),
    )
    save_checkpoint(output, scene, emitters, AEGIRConfig(), 0, [], {})
    payload = torch.load(output, map_location="cpu", weights_only=False)
    payload["format"] = "aegir_shared_gaussian_v13"
    for key in ("axis_gains", "spreads", "falloff_powers"):
        payload["emitter_state"].pop(key)
    payload.pop("emitter_raw_state")
    torch.save(payload, output)

    _, restored, _ = load_checkpoint(output, "cpu")
    ids = torch.zeros(1, dtype=torch.long)
    front = restored.radiance_for_ids(restored.normals, ids)
    back = restored.radiance_for_ids(-restored.normals, ids)
    assert torch.allclose(front, payload["emitter_state"]["colors"], atol=1.0e-5)
    assert torch.allclose(back, front, atol=1.0e-5)


def test_v1_checkpoint_is_planarized_when_loaded(tmp_path: Path) -> None:
    source = tmp_path / "gaussians.pt"
    output = tmp_path / "legacy.pt"
    make_export(source)
    scene = GaussianMaterialBackend(source)
    emitters = AreaEmitterModel(torch.zeros((1, 3)), torch.full((1, 2), 0.1))
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
    assert torch.allclose(
        gbuffer["camera_origins"][24, 20],
        torch.zeros(3, device="cuda"),
    )
    assert torch.allclose(
        gbuffer["camera_ray_dirs"][24, 20],
        torch.tensor([0.0, 0.0, -1.0], device="cuda"),
    )
    assert torch.allclose(
        gbuffer["surface_distances"][24, 20],
        torch.tensor([2.0], device="cuda"),
    )
    assert scene.scales.shape == (1, 2)
    assert gbuffer["means2d"].grad is not None
    assert torch.isfinite(gbuffer["means2d"].grad).all()
    assert float(gbuffer["means2d"].grad.abs().sum()) > 0.0
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


def test_dbscan_and_pca_use_only_persistent_candidates(monkeypatch) -> None:
    points = torch.tensor(
        [
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [100.0, 100.0, 100.0],
        ]
    ).reshape(2, 2, 3)
    # Keep display RGB deliberately different from scene-linear HDR so this
    # test detects regressions back to LDR-based emitter color initialization.
    target = torch.full((2, 2, 3), 0.25)
    radiance_hdr = torch.ones((2, 2, 3))
    monkeypatch.setattr(
        initialization,
        "_persistent_candidate_mask",
        lambda *args, **kwargs: torch.tensor([True, True, True, False]),
    )
    clustered = []

    def fake_cluster(value, eps, min_samples):
        clustered.append(value.clone())
        return torch.zeros(value.shape[0], dtype=torch.long)

    monkeypatch.setattr(initialization, "_cluster_points", fake_cluster)

    emitters = initialization.seed_emitters_from_gbuffer_views(
        [target],
        [{"points": points}],
        (torch.full((3,), -200.0), torch.full((3,), 200.0)),
        quantile=0.0,
        min_scale_fraction=0.0,
        min_samples=3,
        max_emitters=1,
        samples=[
            {
                "valid_mask": torch.ones((2, 2, 1), dtype=torch.bool),
                "radiance_hdr": radiance_hdr,
            }
        ],
        score_mode="hdr",
        persistence_min_views=1,
        persistence_min_fraction=0.0,
    )

    assert emitters.num_emitters == 1
    assert len(clustered) == 1
    assert torch.equal(clustered[0], points.reshape(-1, 3)[:3])
    assert torch.allclose(emitters.centers[0], torch.zeros(3), atol=1.0e-6)
    assert float(emitters.scales[0].detach().max()) == pytest.approx(2.0)
    assert float(emitters.scales[0].detach().min()) == pytest.approx(0.03)
    # Unit spreads attenuate by exp(-1); the amplitude compensates that kernel.
    assert torch.allclose(emitters.radiance(torch.tensor([[0., 0., 1.]]))[0, 0], torch.ones(3), atol=1.0e-6)


def test_emitter_split_replaces_parent_and_preserves_intensity() -> None:
    emitters = AreaEmitterModel(
        torch.zeros((1, 3)),
        torch.tensor([[1.0, 0.1]]),
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


def test_adaptive_split_uses_largest_scene_axis_instead_of_diagonal() -> None:
    emitters = AreaEmitterModel(
        centers=torch.zeros((1, 3)),
        scales=torch.tensor([[2.1, 0.1]]),
        colors=torch.ones((1, 3)),
    )
    bounds = (torch.full((3,), -5.0), torch.full((3,), 5.0))
    diagonal = float((bounds[1] - bounds[0]).norm())

    adaptation = adapt_emitters(
        emitters,
        bounds,
        scene_scale=diagonal,
        config=AdaptiveConfig(
            split_energy=1.0e6,
            split_scale_fraction=0.2,
            max_emitters=4,
        ),
    )

    # 2.1 exceeds 20% of the 10-unit largest extent, but not 20% of the
    # 17.32-unit diagonal deliberately passed through the legacy argument.
    assert adaptation
    assert emitters.num_emitters == 2


def _populate_emitter_adam(emitters: AreaEmitterModel) -> torch.optim.Adam:
    optimizer = torch.optim.Adam(
        emitters.parameters(),
        lr=3.0e-4,
        betas=(0.8, 0.95),
        eps=1.0e-7,
        weight_decay=1.0e-3,
    )
    weights = torch.arange(1, emitters.num_emitters + 1, dtype=emitters.centers.dtype)
    loss = sum((parameter * weights.to(parameter)[:, None]).sum() for parameter in emitters.parameters())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return optimizer


def _named_adam_state(
    emitters: AreaEmitterModel,
    optimizer: torch.optim.Adam,
) -> dict[str, dict[str, torch.Tensor]]:
    return {
        name: {
            key: value.detach().clone()
            for key, value in optimizer.state[parameter].items()
            if torch.is_tensor(value)
        }
        for name, parameter in emitters.named_parameters()
    }


def test_prune_preserves_every_survivor_adam_state_and_raw_parameter() -> None:
    emitters = AreaEmitterModel(
        centers=torch.tensor([[-0.5, 0.0, 0.0], [0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        scales=torch.full((3, 2), 0.1),
        colors=torch.tensor([[1.0, 1.0, 1.0], [1.0e-6, 1.0e-6, 1.0e-6], [2.0, 2.0, 2.0]]),
    )
    optimizer = _populate_emitter_adam(emitters)
    old_raw = {name: parameter.detach().clone() for name, parameter in emitters.named_parameters()}
    old_state = _named_adam_state(emitters, optimizer)

    adaptation = adapt_emitters(
        emitters,
        (torch.full((3,), -2.0), torch.full((3,), 2.0)),
        scene_scale=4.0,
        config=AdaptiveConfig(
            prune_energy=0.1,
            split_energy=1.0e6,
            split_scale_fraction=10.0,
        ),
    )
    migrated = migrate_emitter_adam(optimizer, emitters, adaptation)

    assert adaptation.new_to_old.tolist() == [0, 2]
    assert adaptation.child_mask.tolist() == [False, False]
    assert adaptation.pruned_old_indices.tolist() == [1]
    for (name, parameter), new_parameter in zip(
        emitters.named_parameters(), migrated.param_groups[0]["params"], strict=True
    ):
        assert torch.equal(parameter, old_raw[name][[0, 2]])
        assert torch.equal(migrated.state[new_parameter]["exp_avg"], old_state[name]["exp_avg"][[0, 2]])
        assert torch.equal(
            migrated.state[new_parameter]["exp_avg_sq"], old_state[name]["exp_avg_sq"][[0, 2]]
        )
        assert torch.equal(migrated.state[new_parameter]["step"], old_state[name]["step"])
    assert migrated.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]
    assert migrated.param_groups[0]["betas"] == optimizer.param_groups[0]["betas"]


def test_split_preserves_survivor_state_and_zero_initializes_children() -> None:
    emitters = AreaEmitterModel(
        centers=torch.tensor([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        scales=torch.tensor([[0.1, 0.1], [1.0, 0.1]]),
        colors=torch.ones((2, 3)),
    )
    optimizer = _populate_emitter_adam(emitters)
    old_raw = {name: parameter.detach().clone() for name, parameter in emitters.named_parameters()}
    old_state = _named_adam_state(emitters, optimizer)

    adaptation = adapt_emitters(
        emitters,
        (torch.full((3,), -2.0), torch.full((3,), 2.0)),
        scene_scale=4.0,
        config=AdaptiveConfig(split_energy=1.0e6, split_scale_fraction=0.2, max_emitters=4),
    )
    migrated = migrate_emitter_adam(optimizer, emitters, adaptation)

    assert adaptation.new_to_old.tolist() == [0, 1, 1]
    assert adaptation.child_mask.tolist() == [False, True, True]
    for (name, parameter), new_parameter in zip(
        emitters.named_parameters(), migrated.param_groups[0]["params"], strict=True
    ):
        state = migrated.state[new_parameter]
        assert torch.equal(parameter[0], old_raw[name][0])
        assert torch.equal(state["exp_avg"][0], old_state[name]["exp_avg"][0])
        assert torch.equal(state["exp_avg_sq"][0], old_state[name]["exp_avg_sq"][0])
        assert torch.count_nonzero(state["exp_avg"][1:]) == 0
        assert torch.count_nonzero(state["exp_avg_sq"][1:]) == 0
        assert torch.equal(state["step"], old_state[name]["step"])

    migrated.zero_grad(set_to_none=True)
    post_adaptation_loss = sum(parameter.square().mean() for parameter in emitters.parameters())
    post_adaptation_loss.backward()
    migrated.step()
    assert torch.isfinite(post_adaptation_loss)
    assert all(torch.isfinite(parameter).all() for parameter in emitters.parameters())
    assert all(
        torch.isfinite(value).all()
        for state in migrated.state.values()
        for value in state.values()
        if torch.is_tensor(value)
    )


def test_ambient_is_absent_and_legacy_payload_is_ignored() -> None:
    emitters = AreaEmitterModel(
        torch.zeros((1, 3)),
        torch.ones((1, 2)),
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
        torch.full((1, 2), 0.5),
        quaternions=torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
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


def test_renderer_masks_invalid_gaussians_and_composites_surface_alpha(monkeypatch) -> None:
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
        torch.full((1, 2), 0.5),
    )
    gbuffer = {
        "points": torch.zeros((2, 3)),
        "normals": torch.tensor([[0.0, 0.0, 1.0]]).expand(2, -1),
        "view_dirs": torch.tensor([[0.0, 0.0, 1.0]]).expand(2, -1),
        "albedo": torch.full((2, 3), 0.5),
        "roughness": torch.full((2, 1), 0.5),
        "metallic": torch.zeros((2, 1)),
        "valid_mask": torch.tensor([[True], [False]]),
        "alpha": torch.tensor([[0.25], [0.75]]),
    }

    def brdf_samples(points, *args, **kwargs):
        assert points.shape == (1, 3)
        return torch.ones((1, 3)), torch.full((1, 3), 2.0), 1

    def light_samples(points, *args, **kwargs):
        assert points.shape == (1, 3)
        return torch.full((1, 3), 3.0), torch.full((1, 3), 4.0), 1

    monkeypatch.setattr(renderer, "_brdf_samples", brdf_samples)
    monkeypatch.setattr(renderer, "_light_samples", light_samples)

    rendered = renderer.render(gbuffer, emitters, tonemap=False)

    assert torch.allclose(rendered["direct_hdr"][0], torch.full((3,), 0.5))
    assert torch.allclose(rendered["indirect_hdr"][0], torch.full((3,), 0.75))
    assert torch.count_nonzero(rendered["hdr"][1]) == 0


def _camera_emission_gbuffer(surface_distance: float) -> dict[str, torch.Tensor]:
    return {
        "points": torch.tensor([[0.0, 0.0, surface_distance]]),
        "normals": torch.tensor([[0.0, 0.0, -1.0]]),
        "view_dirs": torch.tensor([[0.0, 0.0, -1.0]]),
        "albedo": torch.full((1, 3), 0.5),
        "roughness": torch.full((1, 1), 0.5),
        "metallic": torch.zeros((1, 1)),
        "camera_origins": torch.zeros((1, 3)),
        "camera_ray_dirs": torch.tensor([[0.0, 0.0, 1.0]]),
        "surface_distances": torch.tensor([[surface_distance]]),
    }


def test_camera_visible_emitter_replaces_and_skips_surface_shading(monkeypatch) -> None:
    renderer = DeferredRenderer(
        RenderConfig(
            num_brdf_samples=1,
            num_light_samples=1,
            point_visibility=False,
            indirect_bounce=False,
        )
    )
    expected_radiance = torch.tensor([[4.0, 2.0, 1.0]])
    emitters = AreaEmitterModel(
        centers=torch.tensor([[0.0, 0.0, 1.0]]),
        scales=torch.full((1, 2), 0.2),
        colors=expected_radiance,
        quaternions=torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
    )
    monkeypatch.setattr(
        renderer,
        "_brdf_samples",
        lambda *args, **kwargs: pytest.fail("BRDF sampling ran for an emitter pixel"),
    )
    monkeypatch.setattr(
        renderer,
        "_light_samples",
        lambda *args, **kwargs: pytest.fail("light sampling ran for an emitter pixel"),
    )

    rendered = renderer.render(
        _camera_emission_gbuffer(surface_distance=3.0),
        emitters,
        generator=torch.Generator().manual_seed(11),
        tonemap=False,
    )

    assert torch.equal(rendered["emitter_mask"], torch.ones((1, 1)))
    expected_directional = emitters.radiance_for_ids(
        torch.tensor([[0.0, 0.0, -1.0]]), torch.zeros(1, dtype=torch.long)
    )
    assert torch.allclose(rendered["emission_hdr"], expected_directional, atol=1.0e-6)
    assert torch.count_nonzero(rendered["direct_hdr"]) == 0
    assert torch.count_nonzero(rendered["indirect_hdr"]) == 0
    assert torch.allclose(rendered["hdr"], expected_directional, atol=1.0e-6)


def test_camera_emitter_behind_gaussian_surface_is_occluded() -> None:
    renderer = DeferredRenderer(
        RenderConfig(
            num_brdf_samples=1,
            num_light_samples=1,
            point_visibility=False,
            indirect_bounce=False,
        )
    )
    emitters = AreaEmitterModel(
        centers=torch.tensor([[0.0, 0.0, 4.0]]),
        scales=torch.full((1, 2), 0.2),
        colors=torch.ones((1, 3)),
        quaternions=torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
    )

    rendered = renderer.render(
        _camera_emission_gbuffer(surface_distance=3.0),
        emitters,
        generator=torch.Generator().manual_seed(13),
        tonemap=False,
    )

    assert torch.count_nonzero(rendered["emitter_mask"]) == 0
    assert torch.count_nonzero(rendered["emission_hdr"]) == 0
    assert torch.allclose(
        rendered["hdr"],
        rendered["direct_hdr"] + rendered["indirect_hdr"],
    )
