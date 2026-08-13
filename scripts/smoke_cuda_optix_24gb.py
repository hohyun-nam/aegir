#!/usr/bin/env python3
"""Run bounded 2DGS, IRGS/OptiX, and AEGIR CUDA smoke checks.

This intentionally uses a tiny scene subset. It validates CUDA execution and
gradient flow; it is not a quality or convergence experiment.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from aegir.checkpoint import load_checkpoint
from aegir.config import AEGIRConfig, RenderConfig
from aegir.data import MultiViewSceneDataset
from aegir.gaussian_scene import GaussianMaterialBackend
from aegir.irgs_tracer import IRGSVisibilityTracer
from aegir.math_utils import safe_normalize
from aegir.trainer import train_shared_gaussians


GIB = 1024**3
MIB = 1024**2


class _NvidiaSmiProcessMonitor:
    """Poll per-process framebuffer usage without requiring NVML bindings."""

    def __init__(self, interval_seconds: float = 0.1) -> None:
        self.interval_seconds = interval_seconds
        self.pid = os.getpid()
        self.peak_mib: int | None = None
        self.samples = 0
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                completed = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=pid,used_gpu_memory",
                        "--format=csv,noheader,nounits",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=5.0,
                )
                used_mib = 0
                found = False
                for line in completed.stdout.splitlines():
                    fields = [field.strip() for field in line.split(",")]
                    if len(fields) != 2 or fields[0] != str(self.pid):
                        continue
                    used_mib += int(fields[1])
                    found = True
                if found:
                    self.peak_mib = max(self.peak_mib or 0, used_mib)
                    self.samples += 1
            except Exception as exc:  # Monitoring must not hide the CUDA test result.
                if self.error is None:
                    self.error = f"{type(exc).__name__}: {exc}"
            self._stop.wait(self.interval_seconds)


class _SingleSurfelScene(nn.Module):
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.means = nn.Parameter(torch.tensor([[0.0, 0.0, 1.0]], device=device))
        self.log_scales = nn.Parameter(torch.tensor([[0.5, 0.5]], device=device).log())
        self.raw_quats = nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device))
        self.opacity_logits = nn.Parameter(torch.logit(torch.tensor([[0.9]], device=device)))

    @property
    def gaussian_count(self) -> int:
        return 1

    @property
    def scales(self) -> torch.Tensor:
        return self.log_scales.exp()

    @property
    def quats(self) -> torch.Tensor:
        return safe_normalize(self.raw_quats, dim=-1)

    @property
    def opacities(self) -> torch.Tensor:
        return self.opacity_logits.sigmoid()


def _memory_snapshot() -> dict[str, float]:
    torch.cuda.synchronize()
    return {
        "allocated_gib": torch.cuda.memory_allocated() / GIB,
        "reserved_gib": torch.cuda.memory_reserved() / GIB,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / GIB,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / GIB,
    }


def _all_parameter_gradients_finite(module: nn.Module) -> tuple[bool, list[str]]:
    failures = []
    for name, parameter in module.named_parameters():
        if parameter.grad is None:
            failures.append(f"{name}:missing")
        elif not bool(torch.isfinite(parameter.grad).all()):
            failures.append(f"{name}:nonfinite")
    return not failures, failures


def _run_2dgs(
    checkpoint: Path,
    dataset_dir: Path,
    prior_dir: Path,
    image_size: int,
    max_gaussians: int,
) -> dict[str, Any]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    dataset = MultiViewSceneDataset(
        dataset_dir,
        checkpoint,
        prior_dir,
        image_size=image_size,
        max_views=1,
        split="train",
    )
    scene = GaussianMaterialBackend(checkpoint, max_gaussians=max_gaussians).cuda()
    camera = {
        key: value.cuda() if torch.is_tensor(value) else value
        for key, value in dataset.frames[0].camera.items()
    }
    gbuffer = scene.rasterize(camera)
    valid = gbuffer["valid_mask"].bool()
    if not bool(valid.any()):
        raise RuntimeError("2DGS rasterizer produced no valid pixels")
    loss = sum(gbuffer[key][valid.expand_as(gbuffer[key])].mean() for key in ("albedo", "depth"))
    loss = loss + gbuffer["roughness"][valid].mean() + gbuffer["metallic"][valid].mean()
    loss.backward()
    gradients_finite, gradient_failures = _all_parameter_gradients_finite(scene)
    if not gradients_finite:
        raise RuntimeError(f"2DGS gradient validation failed: {gradient_failures}")
    result = {
        "valid_pixels": int(valid.sum().item()),
        "gaussians": scene.gaussian_count,
        "loss": float(loss.detach().cpu()),
        "gradients_finite": gradients_finite,
        "memory": _memory_snapshot(),
    }
    del loss, gbuffer, scene, dataset
    torch.cuda.empty_cache()
    return result


def _run_irgs() -> dict[str, Any]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda")
    scene = _SingleSurfelScene(device)
    tracer = IRGSVisibilityTracer(scene, RenderConfig(), scene_scale=1.0)
    origins = torch.zeros((1, 2, 3), device=device)
    directions = torch.tensor([[[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]], device=device)
    distances = torch.full((1, 2, 1), 2.0, device=device)
    visibility = tracer.visibility(origins, directions, distances)
    hit_visibility = float(visibility[0, 0, 0].detach().cpu())
    miss_visibility = float(visibility[0, 1, 0].detach().cpu())
    if not hit_visibility < 0.3 or not miss_visibility > 0.99:
        raise RuntimeError(
            f"unexpected IRGS visibility values: hit={hit_visibility}, miss={miss_visibility}"
        )
    visibility[0, 0, 0].backward()
    gradient = scene.opacity_logits.grad
    gradient_finite = gradient is not None and bool(torch.isfinite(gradient).all())
    gradient_nonzero = gradient is not None and float(gradient.abs().max()) > 0.0
    if not gradient_finite or not gradient_nonzero:
        raise RuntimeError("IRGS opacity gradient is missing, non-finite, or zero")
    result = {
        "hit_visibility": hit_visibility,
        "miss_visibility": miss_visibility,
        "opacity_gradient_finite": gradient_finite,
        "opacity_gradient_nonzero": gradient_nonzero,
        "memory": _memory_snapshot(),
    }
    del visibility, tracer, scene
    torch.cuda.empty_cache()
    return result


def _validate_smoke_checkpoint(path: Path, expected_steps: int) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"smoke checkpoint was not created: {path}")
    scene, emitters, payload = load_checkpoint(path, "cpu")
    if int(payload["step"]) != expected_steps:
        raise RuntimeError(f"smoke checkpoint step is {payload['step']}, expected {expected_steps}")
    nonfinite = []
    for prefix, state in (
        ("scene", scene.state_dict()),
        ("emitter", emitters.state_dict()),
    ):
        for name, value in state.items():
            if torch.is_tensor(value) and not bool(torch.isfinite(value).all()):
                nonfinite.append(f"{prefix}.{name}")
    for record in payload.get("losses", []):
        for key, value in record.items():
            if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                nonfinite.append(f"losses.{key}")
    if nonfinite:
        raise RuntimeError(f"non-finite smoke checkpoint values: {nonfinite[:8]}")
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "step": int(payload["step"]),
        "gaussians": scene.gaussian_count,
        "emitters": emitters.num_emitters,
        "loss_records": len(payload.get("losses", [])),
        "all_saved_values_finite": True,
    }


def _run_training(
    checkpoint: Path,
    dataset_dir: Path,
    prior_dir: Path,
    output: Path,
    steps: int,
    image_size: int,
    max_gaussians: int,
) -> dict[str, Any]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    config = AEGIRConfig()
    config.raster.image_size = image_size
    config.training.steps = steps
    config.training.rays_per_step = 64
    config.training.surface_anchor_points = 128
    config.training.freeze_start = steps + 1
    config.training.freeze_end = steps + 2
    config.training.adaptive_every = 100
    config.training.checkpoint_every = max(1, steps // 2)
    config.render.num_brdf_samples = 2
    config.render.num_light_samples = 2
    config.render.num_bounce_samples = 1
    config.render.bounce_proxy_points = 128
    config.render.bounce_chunk_size = 64
    config.adaptive.max_emitters = 2
    train_shared_gaussians(
        checkpoint,
        dataset_dir,
        prior_dir,
        output,
        config,
        "cuda",
        max_views=1,
        max_gaussians=max_gaussians,
    )
    memory = _memory_snapshot()
    checkpoint_result = _validate_smoke_checkpoint(output, steps)
    torch.cuda.empty_cache()
    return {
        "config": {
            "steps": steps,
            "image_size": image_size,
            "primary_rays_per_step": config.training.rays_per_step,
            "brdf_samples_per_ray": config.render.num_brdf_samples,
            "light_samples_per_ray": config.render.num_light_samples,
            "bounce_samples_per_ray": config.render.num_bounce_samples,
            "max_views": 1,
            "max_gaussians": max_gaussians,
        },
        "checkpoint": checkpoint_result,
        "memory": memory,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dnsplatter-checkpoint", type=Path, required=True)
    parser.add_argument("--nerfstudio-dir", type=Path, required=True)
    parser.add_argument("--prior-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--max-gaussians", type=int, default=512)
    parser.add_argument("--vram-limit-gib", type=float, default=24.0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    for path in (args.dnsplatter_checkpoint, args.nerfstudio_dir, args.prior_dir):
        if not path.exists():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    smoke_checkpoint = args.output_dir / "aegir_smoke.pt"

    monitor = _NvidiaSmiProcessMonitor()
    monitor.start()
    started = time.time()
    status = "failed"
    error: str | None = None
    stages: dict[str, Any] = {}
    try:
        stages["2dgs"] = _run_2dgs(
            args.dnsplatter_checkpoint,
            args.nerfstudio_dir,
            args.prior_dir,
            args.image_size,
            args.max_gaussians,
        )
        stages["irgs_optix"] = _run_irgs()
        stages["training"] = _run_training(
            args.dnsplatter_checkpoint,
            args.nerfstudio_dir,
            args.prior_dir,
            smoke_checkpoint,
            args.steps,
            args.image_size,
            args.max_gaussians,
        )
        peak_reserved = max(stage["memory"]["peak_reserved_gib"] for stage in stages.values())
        if peak_reserved >= args.vram_limit_gib:
            raise RuntimeError(
                f"PyTorch peak reserved VRAM {peak_reserved:.3f} GiB reached/exceeded "
                f"the {args.vram_limit_gib:.3f} GiB limit"
            )
        status = "passed"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        monitor.stop()
        result = {
            "format": "aegir-cuda-optix-smoke-v1",
            "status": status,
            "error": error,
            "elapsed_seconds": time.time() - started,
            "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "vram_limit_gib": args.vram_limit_gib,
            "nvidia_smi_process_peak_mib": monitor.peak_mib,
            "nvidia_smi_samples": monitor.samples,
            "nvidia_smi_monitor_error": monitor.error,
            "stages": stages,
        }
        result_path = args.output_dir / "smoke_result.json"
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
