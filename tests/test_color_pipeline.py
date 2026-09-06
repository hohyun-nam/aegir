from __future__ import annotations

import pytest
import torch

from aegir.color import (
    apply_ldr_pipeline,
    configure_ldr_fallback,
    decode_prior_albedo,
    frame_exposure_scale,
    resolve_observation_policy,
)
from aegir.config import LossWeights
from aegir.losses import hdr_photometric_loss, photometric_loss
from aegir.math_utils import linear_to_srgb, srgb_to_linear


def test_srgb_linear_round_trip_and_reference_midpoint() -> None:
    srgb = torch.tensor([0.0, 0.04045, 0.5, 1.0])
    linear = srgb_to_linear(srgb)
    assert linear[2].item() == pytest.approx(0.21404114, abs=1.0e-7)
    assert torch.allclose(linear_to_srgb(linear), srgb, atol=2.0e-6)


def test_exact_ldr_pipeline_uses_declared_exposure_tone_map_and_transfer() -> None:
    hdr = torch.tensor([[[0.25, 1.0, 4.0]]])
    actual = apply_ldr_pipeline(
        hdr,
        exposure=2.0,
        tone_mapper="clip",
        transfer_function="gamma22",
    )
    expected = (hdr * 2.0).clamp(0.0, 1.0).pow(1.0 / 2.2)
    assert torch.allclose(actual, expected)


def test_observation_policy_prefers_complete_hdr() -> None:
    frames = [
        {"radiance_hdr_file_path": "radiance_hdr/a.npy"},
        {"radiance_hdr_file_path": "radiance_hdr/b.npy"},
    ]
    policy = resolve_observation_policy({}, frames)
    assert policy["domain"] == "linear_hdr"
    assert policy["loss_domain"] == "log1p_scene_linear_rgb"


def test_observation_policy_follows_exact_per_frame_ldr_metadata() -> None:
    metadata = {
        "aegir_color": {
            "format": "aegir-color-v1",
            "observation_domain": "ldr",
            "target_png": {
                "tone_mapper": "clip",
                "transfer_function": "gamma22",
                "exposure_policy": "fixed_per_frame",
            },
        }
    }
    frames = [{"ldr_exposure_scale": 0.25}, {"ldr_exposure_scale": 0.5}]
    policy = resolve_observation_policy(metadata, frames)
    assert policy["domain"] == "ldr"
    assert policy["tone_mapper"] == "clip"
    assert frame_exposure_scale(policy, frames[1]) == pytest.approx(0.5)


def test_uncharacterized_ldr_uses_explicit_learned_exposure_fallback() -> None:
    policy = resolve_observation_policy({}, [{"file_path": "images/a.png"}])
    assert policy["domain"] == "ldr"
    assert policy["exposure_policy"] == "learned_scene"
    assert policy["tone_mapper"] == "reinhard"
    assert policy["transfer_function"] == "srgb"
    configured = configure_ldr_fallback(
        policy,
        tone_mapper="clip",
        transfer_function="linear",
    )
    assert configured["tone_mapper"] == "clip"
    assert configured["target_encoding"] == "linear_ldr"


def test_mixed_hdr_and_ldr_scene_is_rejected() -> None:
    with pytest.raises(ValueError, match="mixes HDR and LDR-only"):
        resolve_observation_policy(
            {},
            [
                {"radiance_hdr_file_path": "radiance_hdr/a.npy"},
                {"file_path": "images/b.png"},
            ],
        )


def test_prior_albedo_is_always_linear_reflectance() -> None:
    encoded = torch.full((1, 1, 3), 0.5)
    decoded = decode_prior_albedo(encoded, {"basecolor_space": "model"})
    assert decoded.mean().item() == pytest.approx(0.21404114, abs=1.0e-7)
    declared = decode_prior_albedo(encoded, {"prior_albedo_space": "linear_reflectance"})
    assert torch.equal(declared, encoded)


def test_hdr_loss_is_symmetric_and_zero_for_identical_radiance() -> None:
    weights = LossWeights(dssim_lambda=0.2, focus_lambda=0.0)
    target = torch.tensor([[[0.0, 1.0, 8.0], [0.5, 2.0, 16.0]]])
    assert hdr_photometric_loss(target, target, weights).item() == pytest.approx(0.0)
    forward = hdr_photometric_loss(target * 2.0, target, weights)
    backward = hdr_photometric_loss(target, target * 2.0, weights)
    assert torch.allclose(forward, backward)


def test_saturated_ldr_l1_does_not_penalize_overprediction() -> None:
    weights = LossWeights(dssim_lambda=0.0, focus_lambda=0.0)
    target = torch.full((2, 2, 3), 1.0)
    over = photometric_loss(
        torch.full_like(target, 1.2),
        target,
        weights,
        saturation_threshold=0.98,
    )
    under = photometric_loss(
        torch.full_like(target, 0.8),
        target,
        weights,
        saturation_threshold=0.98,
    )
    assert over.item() == pytest.approx(0.0)
    assert under.item() == pytest.approx(0.2)
