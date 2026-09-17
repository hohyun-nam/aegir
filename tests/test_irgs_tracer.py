from __future__ import annotations

import pytest
import torch
from torch import nn

from aegir.config import RenderConfig
from aegir.emitters import AreaEmitterModel
from aegir.irgs_tracer import IRGSVisibilityTracer
from aegir.math_utils import safe_normalize
from aegir.render import DeferredRenderer

SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199


class _SingleSurfelScene(nn.Module):
    def __init__(self, coordinate_scale: float = 1.0) -> None:
        super().__init__()
        self.means = nn.Parameter(
            torch.tensor([[0.0, 0.0, coordinate_scale]], device="cuda")
        )
        self.log_scales = nn.Parameter(
            torch.tensor([[0.5, 0.5]], device="cuda").mul(coordinate_scale).log()
        )
        self.raw_quats = nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda"))
        self.opacity_logits = nn.Parameter(torch.logit(torch.tensor([[0.9]], device="cuda")))
        sh = torch.zeros((1, 16, 3), device="cuda")
        sh[:, 0] = 1.0
        sh[:, 3, 0] = 0.5
        self.register_buffer("training_sh_coeffs", sh)
        self.register_buffer("emitter_ids", torch.full((1,), -1, device="cuda", dtype=torch.long))

    @property
    def gaussian_count(self) -> int:
        return int(self.means.shape[0])

    @property
    def scales(self) -> torch.Tensor:
        return self.log_scales.exp()

    @property
    def quats(self) -> torch.Tensor:
        return safe_normalize(self.raw_quats, dim=-1)

    @property
    def opacities(self) -> torch.Tensor:
        return self.opacity_logits.sigmoid()

    @property
    def normals(self) -> torch.Tensor:
        return torch.tensor([[0.0, 0.0, -1.0]], device="cuda")

    @property
    def training_sh_degree(self) -> int:
        return 3

    @property
    def trace_material_features(self) -> torch.Tensor:
        return torch.tensor([[0.5, 0.4, 0.3, 0.6, 0.2]], device="cuda")

    @property
    def background_mask(self) -> torch.Tensor:
        return self.emitter_ids < 0


class _LayeredSurfelScene(_SingleSurfelScene):
    """Two coincident image-space surfels on opposite sides of a light."""

    def __init__(self) -> None:
        super().__init__()
        self.means = nn.Parameter(
            torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 3.0]], device="cuda")
        )
        self.log_scales = nn.Parameter(torch.full((2, 2), 0.5, device="cuda").log())
        self.raw_quats = nn.Parameter(
            torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2, device="cuda")
        )
        self.opacity_logits = nn.Parameter(
            torch.logit(torch.tensor([[0.2], [0.99]], device="cuda"))
        )
        self.training_sh_coeffs = self.training_sh_coeffs.detach().repeat(2, 1, 1)
        self.emitter_ids = self.emitter_ids.detach().repeat(2)

    @property
    def trace_material_features(self) -> torch.Tensor:
        return torch.tensor(
            [[0.5, 0.4, 0.3, 0.6, 0.2]] * 2,
            device="cuda",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="IRGS OptiX test requires CUDA")
def test_irgs_visibility_hits_misses_and_backpropagates() -> None:
    pytest.importorskip("surfel_tracer")
    scene = _SingleSurfelScene()
    config = RenderConfig(visibility_steepness=80.0)
    tracer = IRGSVisibilityTracer(scene, config, scene_scale=1.0)
    origins = torch.zeros((1, 2, 3), device="cuda")
    directions = torch.tensor([[[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]], device="cuda")
    distances = torch.full((1, 2, 1), 2.0, device="cuda")

    visibility = tracer.visibility(origins, directions, distances)

    assert visibility.shape == (1, 2, 1)
    assert float(visibility[0, 0, 0].detach()) < 0.3
    assert float(visibility[0, 1, 0].detach()) > 0.99
    visibility[0, 0, 0].backward()
    assert scene.opacity_logits.grad is not None
    assert torch.isfinite(scene.opacity_logits.grad).all()
    assert float(scene.opacity_logits.grad.abs().max()) > 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="IRGS OptiX test requires CUDA")
def test_irgs_bvh_excludes_emitter_associated_gaussians() -> None:
    pytest.importorskip("surfel_tracer")
    scene = _LayeredSurfelScene()
    scene.emitter_ids[0] = 0
    tracer = IRGSVisibilityTracer(scene, RenderConfig(), scene_scale=3.0)

    _, _, gaussian_ids = tracer._bounding_mesh()

    assert gaussian_ids.unique().tolist() == [1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="IRGS OptiX test requires CUDA")
@torch.inference_mode()
def test_irgs_visibility_runs_in_inspection_mode() -> None:
    pytest.importorskip("surfel_tracer")
    scene = _SingleSurfelScene()
    tracer = IRGSVisibilityTracer(scene, RenderConfig(), scene_scale=1.0)
    origins = torch.zeros((1, 1, 3), device="cuda")
    directions = torch.tensor([[[0.0, 0.0, 1.0]]], device="cuda")
    distances = torch.full((1, 1, 1), 2.0, device="cuda")

    visibility = tracer.visibility(origins, directions, distances)

    assert float(visibility[0, 0, 0]) < 0.3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="IRGS OptiX test requires CUDA")
def test_irgs_visibility_and_opacity_gradient_are_scale_invariant() -> None:
    pytest.importorskip("surfel_tracer")
    values = []
    gradients = []
    for coordinate_scale in (1.0, 10.0):
        scene = _SingleSurfelScene(coordinate_scale)
        tracer = IRGSVisibilityTracer(
            scene,
            RenderConfig(),
            scene_scale=coordinate_scale,
        )
        origins = torch.zeros((1, 1, 3), device="cuda")
        directions = torch.tensor([[[0.0, 0.0, 1.0]]], device="cuda")
        distances = torch.full((1, 1, 1), 2.0 * coordinate_scale, device="cuda")
        visibility = tracer.visibility(origins, directions, distances)
        visibility.sum().backward()
        values.append(visibility.detach())
        gradients.append(scene.opacity_logits.grad.detach())

    assert torch.allclose(values[0], values[1], atol=2.0e-5, rtol=2.0e-5)
    assert torch.allclose(gradients[0], gradients[1], atol=2.0e-5, rtol=2.0e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="IRGS OptiX test requires CUDA")
def test_training_full_sh_bounce_and_inference_proxy_disable() -> None:
    pytest.importorskip("surfel_tracer")
    scene = _SingleSurfelScene()
    tracer = IRGSVisibilityTracer(scene, RenderConfig(), scene_scale=1.0)
    origins = torch.zeros((1, 2, 3), device="cuda")
    directions = safe_normalize(
        torch.tensor([[[0.15, 0.0, 1.0], [-0.15, 0.0, 1.0]]], device="cuda"),
        dim=-1,
    )
    distances = torch.full((1, 2, 1), 2.0, device="cuda")

    training = tracer.trace_transport(origins, directions, distances, True)
    inference = tracer.trace_transport(origins, directions, distances, False)

    assert training["training_bounce"].shape == (1, 2, 3)
    assert float(training["training_bounce"].detach().abs().sum()) > 0.0
    assert not torch.allclose(
        training["training_bounce"][:, 0],
        training["training_bounce"][:, 1],
    )
    assert torch.count_nonzero(inference["training_bounce"]) == 0
    assert torch.allclose(
        training["hit_materials"],
        scene.trace_material_features.reshape(1, 1, 5).expand(1, 2, 5),
        atol=2.0e-5,
    )
    assert inference["hit_materials"].shape == (1, 2, 5)
    assert torch.isfinite(inference["hit_points"]).all()

    (training["training_bounce"].sum() + training["visibility"].sum()).backward()
    assert scene.opacity_logits.grad is not None
    assert torch.isfinite(scene.opacity_logits.grad).all()
    assert float(scene.opacity_logits.grad.abs().max()) > 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="IRGS OptiX test requires CUDA")
def test_explicit_secondary_nee_runs_through_irgs_and_backpropagates() -> None:
    pytest.importorskip("surfel_tracer")
    scene = _SingleSurfelScene()
    config = RenderConfig(indirect_bounce=True, num_bounce_samples=2)
    tracer = IRGSVisibilityTracer(scene, config, scene_scale=1.0)
    renderer = DeferredRenderer(
        config,
        visibility_fn=tracer.visibility,
        transport_fn=tracer.trace_transport,
    )
    emitters = AreaEmitterModel(
        centers=torch.tensor([[0.1, 0.0, -1.0]], device="cuda"),
        scales=torch.tensor([[0.2, 0.1]], device="cuda"),
        colors=torch.full((1, 3), 2.0, device="cuda"),
    )

    _, indirect = renderer._incident_transport(
        points=torch.zeros((1, 3), device="cuda"),
        dirs=torch.tensor([[[0.0, 0.0, 1.0]]], device="cuda"),
        distances=torch.full((1, 1, 1), 2.0, device="cuda"),
        direct_radiance=torch.zeros((1, 1, 3), device="cuda"),
        emitters=emitters,
        occluder_points=None,
        generator=torch.Generator(device="cuda").manual_seed(11),
        transport_mode="training",
        training_proxy_weight=0.0,
    )

    assert torch.isfinite(indirect).all()
    assert float(indirect.detach().sum()) > 0.0
    indirect.sum().backward()
    assert emitters.log_colors.grad is not None
    assert torch.isfinite(emitters.log_colors.grad).all()
    assert float(emitters.log_colors.grad.abs().sum()) > 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="IRGS OptiX test requires CUDA")
def test_irgs_refresh_resizes_dummy_sh_after_scene_densification() -> None:
    """A post-split Gaussian id must never index the pre-split dummy SH buffer."""

    pytest.importorskip("surfel_tracer")
    scene = _SingleSurfelScene()
    tracer = IRGSVisibilityTracer(scene, RenderConfig(), scene_scale=3.0)

    scene.means = nn.Parameter(
        torch.tensor([[0.0, 0.0, 1.0], [2.0, 0.0, 1.0]], device="cuda")
    )
    scene.log_scales = nn.Parameter(scene.log_scales.detach().repeat(2, 1))
    scene.raw_quats = nn.Parameter(scene.raw_quats.detach().repeat(2, 1))
    scene.opacity_logits = nn.Parameter(scene.opacity_logits.detach().repeat(2, 1))
    scene.training_sh_coeffs = scene.training_sh_coeffs.detach().repeat(2, 1, 1)
    tracer.refresh_bvh()

    assert tracer.dummy_sh.shape == (2, 1, 3)
    origins = torch.tensor([[[2.0, 0.0, 0.0]]], device="cuda")
    directions = torch.tensor([[[0.0, 0.0, 1.0]]], device="cuda")
    distances = torch.full((1, 1, 1), 2.0, device="cuda")
    visibility = tracer.visibility(origins, directions, distances)
    visibility.sum().backward()

    assert torch.isfinite(visibility).all()
    assert scene.opacity_logits.grad is not None
    assert torch.isfinite(scene.opacity_logits.grad).all()
    assert float(scene.opacity_logits.grad[1].abs().max()) > 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="IRGS OptiX test requires CUDA")
@torch.inference_mode()
def test_irgs_glue_matches_native_weighted_output_contract() -> None:
    """Lock the official IRGS composite/conditional conversion used by AEGIR."""

    pytest.importorskip("surfel_tracer")
    scene = _SingleSurfelScene()
    tracer = IRGSVisibilityTracer(scene, RenderConfig(), scene_scale=1.0)
    origins = torch.zeros((1, 1, 3), device="cuda")
    directions = safe_normalize(
        torch.tensor([[[0.1, 0.0, 1.0]]], device="cuda"),
        dim=-1,
    )

    traced = tracer._trace_raw(
        origins,
        directions,
        training_proxy=True,
        material_features=True,
    )

    intersection_x = float(directions[0, 0, 0] / directions[0, 0, 2])
    expected_alpha = 0.9 * torch.exp(
        torch.tensor(-0.5 * (intersection_x / 0.5) ** 2, device="cuda")
    )
    expected_color = torch.tensor(
        [
            0.5 + SH_C0 - SH_C1 * float(directions[0, 0, 0]) * 0.5,
            0.5 + SH_C0,
            0.5 + SH_C0,
        ],
        device="cuda",
    )
    expected_depth = 1.0 / directions[0, 0, 2]

    assert torch.allclose(traced["alpha"], expected_alpha.reshape(1, 1), atol=2.0e-5)
    # Native IRGS color remains opacity-composited for Equation 4.
    assert torch.allclose(
        traced["color"],
        expected_color.reshape(1, 1, 3) * expected_alpha,
        atol=2.0e-5,
    )
    # Geometry/material attributes are converted to conditional hit values.
    assert torch.allclose(traced["depth"], expected_depth.reshape(1, 1), atol=2.0e-5)
    assert torch.allclose(
        traced["normal"],
        torch.tensor([[[0.0, 0.0, -1.0]]], device="cuda"),
        atol=2.0e-5,
    )
    assert torch.allclose(
        traced["material"],
        scene.trace_material_features.reshape(1, 1, 5),
        atol=2.0e-5,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="IRGS OptiX test requires CUDA")
@torch.inference_mode()
def test_irgs_origin_offset_avoids_self_hit_and_ignores_geometry_behind_light() -> None:
    pytest.importorskip("surfel_tracer")
    scene = _SingleSurfelScene()
    tracer = IRGSVisibilityTracer(scene, RenderConfig(), scene_scale=1.0)
    origins = torch.tensor(
        [[[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]],
        device="cuda",
    )
    directions = torch.tensor(
        [[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]],
        device="cuda",
    )
    distances = torch.tensor([[[1.0], [0.5]]], device="cuda")

    visibility = tracer.visibility(origins, directions, distances)

    assert tracer.origin_offset == pytest.approx(0.05)
    assert tracer.proxy_thickness == pytest.approx(1.0e-6)
    assert torch.all(visibility > 0.999)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="IRGS OptiX test requires CUDA")
@torch.inference_mode()
def test_irgs_segment_trace_excludes_opaque_geometry_behind_light() -> None:
    """A rear opaque layer must not dilute a weak blocker before the emitter."""

    pytest.importorskip("surfel_tracer")
    scene = _LayeredSurfelScene()
    tracer = IRGSVisibilityTracer(scene, RenderConfig(), scene_scale=3.0)
    origins = torch.zeros((1, 1, 3), device="cuda")
    directions = torch.tensor([[[0.0, 0.0, 1.0]]], device="cuda")
    distances = torch.tensor([[[2.0]]], device="cuda")

    traced = tracer.trace_transport(origins, directions, distances, False)

    assert float(traced["visibility"][0, 0, 0]) == pytest.approx(0.8, abs=2.0e-3)
    assert float(traced["hit_points"][0, 0, 2]) == pytest.approx(1.0, abs=2.0e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="IRGS OptiX test requires CUDA")
def test_irgs_backward_reaches_rays_and_all_surfel_geometry_parameters() -> None:
    pytest.importorskip("surfel_tracer")
    scene = _SingleSurfelScene()
    tracer = IRGSVisibilityTracer(scene, RenderConfig(), scene_scale=1.0)
    origins = torch.tensor([[[0.02, -0.03, 0.0]]], device="cuda", requires_grad=True)
    directions = torch.tensor([[[0.13, 0.09, 1.0]]], device="cuda", requires_grad=True)
    distances = torch.full((1, 1, 1), 2.0, device="cuda")

    traced = tracer.trace_transport(origins, directions, distances, False)
    loss = (
        traced["visibility"].sum()
        + (traced["hit_points"] * traced["hit_points"].new_tensor([0.3, -0.2, 0.5])).sum()
        + (traced["hit_normals"] * traced["hit_normals"].new_tensor([0.4, 0.7, -0.1])).sum()
    )
    loss.backward()

    for value in (origins, directions):
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()
        assert float(value.grad.abs().max()) > 0.0
    for parameter in (
        scene.means,
        scene.log_scales,
        scene.raw_quats,
        scene.opacity_logits,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert float(parameter.grad.abs().max()) > 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="IRGS OptiX test requires CUDA")
def test_irgs_rejects_non_float32_before_native_binding() -> None:
    pytest.importorskip("surfel_tracer")
    scene = _SingleSurfelScene().double()

    with pytest.raises(TypeError, match="must be float32"):
        IRGSVisibilityTracer(scene, RenderConfig(), scene_scale=1.0)
