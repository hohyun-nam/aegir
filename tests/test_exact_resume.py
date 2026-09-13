from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from aegir.adaptive import adapt_emitters, migrate_emitter_adam
from aegir.checkpoint import (
    CHECKPOINT_FORMAT,
    capture_rng_state,
    load_checkpoint,
    optimizer_state_dict_cpu,
    require_exact_resume_state,
    restore_rng_state,
    save_checkpoint,
)
from aegir.cli import build_parser
from aegir.config import AdaptiveConfig, AEGIRConfig
from aegir.emitters import AreaEmitterModel
from aegir.gaussian_scene import GaussianMaterialBackend


def _make_export(path: Path) -> None:
    torch.save(
        {
            "means": torch.tensor(
                [[0.0, 0.0, -2.0], [0.1, 0.0, -2.0], [0.0, 0.1, -2.0], [0.1, 0.1, -2.0]]
            ),
            "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(4, 1),
            "scales": torch.tensor([[0.1, 0.1, 0.01]]).repeat(4, 1),
            "opacities": torch.full((4, 1), 0.8),
            "features_dc_raw": torch.zeros((4, 1, 3)),
            "features_rest_raw": torch.zeros((4, 15, 3)),
        },
        path,
    )


def _new_state(source: Path):
    scene = GaussianMaterialBackend(source)
    emitters = AreaEmitterModel(
        centers=torch.zeros((1, 3)),
        scales=torch.tensor([[1.0, 0.1]]),
        colors=torch.ones((1, 3)),
    )
    optimizers = {
        "geometry": torch.optim.Adam(scene.geometry_parameters(), lr=1.0e-4),
        "material": torch.optim.Adam(scene.material_parameters(), lr=2.0e-4),
        "emitter": torch.optim.Adam(emitters.parameters(), lr=3.0e-4),
    }
    return scene, emitters, optimizers


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _toy_step(
    scene: GaussianMaterialBackend,
    emitters: AreaEmitterModel,
    optimizers: dict[str, torch.optim.Adam],
    step: int,
) -> tuple[dict[str, torch.optim.Adam], float, bool]:
    for optimizer in optimizers.values():
        optimizer.zero_grad(set_to_none=True)
    stochastic = torch.rand(()) + float(np.random.random()) + random.random()
    parameters = [*scene.parameters(), *emitters.parameters()]
    loss = sum(parameter.square().mean() for parameter in parameters)
    loss = loss + stochastic * sum(parameter.mean() for parameter in parameters) * 1.0e-4
    loss.backward()
    for optimizer in optimizers.values():
        optimizer.step()

    adapted = False
    if step == 2:
        adaptation = adapt_emitters(
            emitters,
            (torch.full((3,), -2.0), torch.full((3,), 2.0)),
            scene_scale=4.0,
            config=AdaptiveConfig(
                prune_energy=0.0,
                split_energy=1.0e6,
                split_scale_fraction=0.2,
                max_emitters=4,
            ),
        )
        if adaptation:
            optimizers["emitter"] = migrate_emitter_adam(
                optimizers["emitter"], emitters, adaptation
            )
            adapted = True
    return optimizers, float(loss.detach()), adapted


def _run_steps(scene, emitters, optimizers, start: int, end: int):
    losses = []
    adaptations = 0
    for step in range(start, end):
        optimizers, loss, adapted = _toy_step(scene, emitters, optimizers, step)
        losses.append(loss)
        adaptations += int(adapted)
    return optimizers, losses, adaptations


def _assert_tree_equal(first: Any, second: Any) -> None:
    if torch.is_tensor(first):
        assert torch.equal(first, second)
    elif isinstance(first, np.ndarray):
        assert np.array_equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            _assert_tree_equal(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert type(first) is type(second)
        assert len(first) == len(second)
        for left, right in zip(first, second, strict=True):
            _assert_tree_equal(left, right)
    else:
        assert first == second


def test_checkpoint_resume_matches_uninterrupted_training_across_split(tmp_path: Path) -> None:
    source = tmp_path / "gaussians.pt"
    checkpoint = tmp_path / "resume.pt"
    _make_export(source)
    total_steps = 7
    interruption_step = 4

    _seed_all(1234)
    full_scene, full_emitters, full_optimizers = _new_state(source)
    full_optimizers, full_losses, full_adaptations = _run_steps(
        full_scene, full_emitters, full_optimizers, 0, total_steps
    )
    full_rng = capture_rng_state("cpu")

    _seed_all(1234)
    interrupted_scene, interrupted_emitters, interrupted_optimizers = _new_state(source)
    interrupted_optimizers, first_losses, first_adaptations = _run_steps(
        interrupted_scene,
        interrupted_emitters,
        interrupted_optimizers,
        0,
        interruption_step,
    )
    training_state = {
        "next_step": interruption_step,
        "optimizers": {
            name: optimizer_state_dict_cpu(optimizer)
            for name, optimizer in interrupted_optimizers.items()
        },
        "rng": capture_rng_state("cpu"),
        "bounds": [torch.full((3,), -2.0), torch.full((3,), 2.0)],
        "surface_points": interrupted_scene.means.detach().cpu().clone(),
        "emitter_optimization": {
            "scale_anchors": interrupted_emitters.scales.detach().cpu().clone(),
        },
        "adaptation": {
            "checks": 1,
            "changes": first_adaptations,
            "last_check_step": 2,
            "last_change_step": 2,
        },
        "scene_culling": {
            "state": {
                "checks": 0,
                "changes": 0,
                "last_check_step": None,
                "last_change_step": None,
                "total_pruned": 0,
                "total_split_parents": 0,
                "total_split_children": 0,
                "total_duplicated": 0,
            },
            "max_2d_size": torch.zeros(interrupted_scene.gaussian_count),
            "xy_grad_norm": torch.zeros(interrupted_scene.gaussian_count),
            "xy_grad_vis_counts": torch.zeros(interrupted_scene.gaussian_count),
            "refinement_observations": 0,
        },
        "sampling": {
            "view": "global_step_mod_dataset_length",
            "patch": "cpu_generator_seeded_by_global_step",
            "transport": "global_torch_device_rng",
        },
    }
    raw_quaternions = interrupted_emitters.raw_quaternions.detach().clone()
    save_checkpoint(
        checkpoint,
        interrupted_scene,
        interrupted_emitters,
        AEGIRConfig(),
        step=interruption_step,
        losses=[{"step": float(index), "total": value} for index, value in enumerate(first_losses)],
        dataset_meta={"num_views": 1},
        training_state=training_state,
    )

    random.random()
    np.random.random()
    torch.rand(10)
    resumed_scene, resumed_emitters, payload = load_checkpoint(checkpoint, "cpu")
    restored_training = require_exact_resume_state(payload, checkpoint)
    assert payload["format"] == CHECKPOINT_FORMAT
    assert torch.equal(resumed_emitters.raw_quaternions, raw_quaternions)
    resumed_optimizers = {
        "geometry": torch.optim.Adam(resumed_scene.geometry_parameters(), lr=1.0e-4),
        "material": torch.optim.Adam(resumed_scene.material_parameters(), lr=2.0e-4),
        "emitter": torch.optim.Adam(resumed_emitters.parameters(), lr=3.0e-4),
    }
    for name, optimizer in resumed_optimizers.items():
        optimizer.load_state_dict(restored_training["optimizers"][name])
    restore_rng_state(restored_training["rng"], "cpu")
    resumed_optimizers, remaining_losses, resumed_adaptations = _run_steps(
        resumed_scene,
        resumed_emitters,
        resumed_optimizers,
        restored_training["next_step"],
        total_steps,
    )
    resumed_rng = capture_rng_state("cpu")

    assert first_losses + remaining_losses == full_losses
    assert first_adaptations + resumed_adaptations == full_adaptations == 1
    _assert_tree_equal(full_scene.state_dict(), resumed_scene.state_dict())
    _assert_tree_equal(full_emitters.state_dict(), resumed_emitters.state_dict())
    for name in full_optimizers:
        _assert_tree_equal(
            optimizer_state_dict_cpu(full_optimizers[name]),
            optimizer_state_dict_cpu(resumed_optimizers[name]),
        )
    _assert_tree_equal(full_rng, resumed_rng)


def test_legacy_checkpoint_is_rejected_for_exact_resume(tmp_path: Path) -> None:
    source = tmp_path / "gaussians.pt"
    checkpoint = tmp_path / "legacy.pt"
    _make_export(source)
    scene, emitters, _ = _new_state(source)
    save_checkpoint(checkpoint, scene, emitters, AEGIRConfig(), 3, [], {})
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["format"] = "aegir_shared_gaussian_v3"
    payload.pop("training_state", None)
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="only v13"):
        require_exact_resume_state(payload, checkpoint)
    loaded_scene, loaded_emitters, _ = load_checkpoint(checkpoint, "cpu")
    assert loaded_scene.gaussian_count == scene.gaussian_count
    assert loaded_emitters.num_emitters == emitters.num_emitters


def test_v6_checkpoint_is_loadable_but_not_exactly_resumable(tmp_path: Path) -> None:
    source = tmp_path / "gaussians.pt"
    checkpoint = tmp_path / "v6.pt"
    _make_export(source)
    scene, emitters, _ = _new_state(source)
    save_checkpoint(checkpoint, scene, emitters, AEGIRConfig(), 3, [], {})
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["format"] = "aegir_shared_gaussian_v6"
    payload["training_state"] = {"next_step": 3}
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="only v13"):
        require_exact_resume_state(payload, checkpoint)
    loaded_scene, loaded_emitters, _ = load_checkpoint(checkpoint, "cpu")
    assert loaded_scene.gaussian_count == scene.gaussian_count
    assert loaded_emitters.num_emitters == emitters.num_emitters


def test_resume_cli_is_explicit_and_exclusive() -> None:
    parser = build_parser()
    required = [
        "train-gaussian-scene",
        "--dnsplatter-checkpoint",
        "source.ckpt",
        "--nerfstudio-dir",
        "scene",
        "--output",
        "output.pt",
    ]
    args = parser.parse_args(
        required
        + [
            "--resume",
            "step10000.pt",
            "--steps",
            "20000",
            "--bounce-chunk-size",
            "65536",
        ]
    )
    assert args.resume == Path("step10000.pt")
    assert args.steps == 20_000
    assert args.bounce_chunk_size == 65_536
    with pytest.raises(SystemExit):
        parser.parse_args(
            required
            + ["--resume", "step10000.pt", "--emitter-checkpoint", "emitters.pt"]
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_rng_state_round_trip() -> None:
    original = capture_rng_state("cuda")
    expected = torch.rand((16,), device="cuda")
    restore_rng_state(original, "cuda")
    actual = torch.rand((16,), device="cuda")
    assert torch.equal(expected, actual)
