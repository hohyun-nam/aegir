from copy import deepcopy
from dataclasses import asdict

import pytest

from aegir.cli import build_parser
from aegir.config import AEGIRConfig
from aegir.indirect import DIRECT_SH_MODE
from aegir.inspect import configure_inference_sampling
from aegir.sample_reduction import central_fraction_bounds


def learned_config():
    cfg = AEGIRConfig()
    cfg.render.lighting_mode = DIRECT_SH_MODE
    cfg.render.num_light_samples = 512
    return cfg


def test_inference_default_ignores_saved_sample_count_and_preserves_training():
    training = learned_config()
    original = asdict(training)
    cfg = deepcopy(training)
    configure_inference_sampling(cfg)
    assert cfg.render.num_light_samples == 128
    assert cfg.render.num_brdf_samples == 0
    assert cfg.render.direct_sample_reduction == "trimmed"
    assert cfg.render.direct_sample_keep_fraction == .9
    low, high = central_fraction_bounds(128, .9)
    assert (low, high, high - low) == (6, 122, 116)
    assert asdict(training) == original
    assert training.render.direct_sample_reduction == "mean"


@pytest.mark.parametrize("mode,fraction", [("mean", .9), ("trimmed80", .9), ("trimmed", .9)])
def test_explicit_mode_override(mode, fraction):
    cfg = learned_config()
    configure_inference_sampling(cfg, 512, mode)
    assert cfg.render.num_light_samples == 512
    assert cfg.render.direct_sample_reduction == mode
    assert cfg.render.direct_sample_keep_fraction == fraction


def test_custom_fraction_override():
    cfg = learned_config()
    configure_inference_sampling(cfg, direct_sample_keep_percent=75)
    assert cfg.render.direct_sample_reduction == "trimmed"
    assert cfg.render.direct_sample_keep_fraction == .75


def test_legacy_explicit_transport_keeps_supported_mean():
    cfg = AEGIRConfig()
    configure_inference_sampling(cfg)
    assert cfg.render.num_light_samples == cfg.render.num_brdf_samples == 128
    assert cfg.render.direct_sample_reduction == "mean"


@pytest.mark.parametrize("kwargs", [
    {"render_samples": 0}, {"render_samples": -1},
    {"direct_sample_keep_percent": 0}, {"direct_sample_keep_percent": 101},
    {"direct_sample_keep_percent": float("nan")},
    {"direct_sample_reduction": "mean", "direct_sample_keep_percent": 90},
])
def test_invalid_inference_overrides(kwargs):
    with pytest.raises(ValueError):
        configure_inference_sampling(learned_config(), **kwargs)


@pytest.mark.parametrize("command,inputs", [
    ("inspect", ["--nerfstudio-dir", "data"]),
    ("render-camera-path", ["--transforms", "transforms.json"]),
])
def test_cli_defaults_and_overrides(command, inputs):
    argv = [command, "--checkpoint", "a.pt", "--output-dir", "out"] + inputs
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_inference_sampling(cfg := learned_config(), args.render_samples,
                                 args.direct_sample_reduction, args.direct_sample_keep_percent)
    assert cfg.render.num_light_samples == 128
    assert cfg.render.direct_sample_keep_fraction == .9
    args = parser.parse_args(argv + ["--render-samples", "512", "--direct-sample-reduction", "mean"])
    configure_inference_sampling(cfg, args.render_samples, args.direct_sample_reduction,
                                 args.direct_sample_keep_percent)
    assert cfg.render.num_light_samples == 512 and cfg.render.direct_sample_reduction == "mean"
