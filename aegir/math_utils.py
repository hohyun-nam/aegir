from __future__ import annotations

from functools import lru_cache

import torch
from torch import Tensor


def safe_normalize(x: Tensor, dim: int = -1, eps: float = 1.0e-8) -> Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


def quat_to_matrix(quat: Tensor) -> Tensor:
    """Convert quaternions in wxyz order to rotation matrices."""

    q = safe_normalize(quat, dim=-1)
    w, x, y, z = q.unbind(-1)
    two = 2.0
    return torch.stack(
        [
            1 - two * (y * y + z * z),
            two * (x * y - z * w),
            two * (x * z + y * w),
            two * (x * y + z * w),
            1 - two * (x * x + z * z),
            two * (y * z - x * w),
            two * (x * z - y * w),
            two * (y * z + x * w),
            1 - two * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def matrix_to_quat(matrix: Tensor) -> Tensor:
    """Convert rotation matrices to quaternions in wxyz order."""

    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"expected [..., 3, 3] rotation matrices, got {tuple(matrix.shape)}")
    m00, m01, m02 = matrix[..., 0, :].unbind(-1)
    m10, m11, m12 = matrix[..., 1, :].unbind(-1)
    m20, m21, m22 = matrix[..., 2, :].unbind(-1)
    q_abs = (
        torch.stack(
            (
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ),
            dim=-1,
        )
        .clamp_min(0.0)
        .sqrt()
    )
    candidates = torch.stack(
        (
            torch.stack((q_abs[..., 0].square(), m21 - m12, m02 - m20, m10 - m01), dim=-1),
            torch.stack((m21 - m12, q_abs[..., 1].square(), m10 + m01, m02 + m20), dim=-1),
            torch.stack((m02 - m20, m10 + m01, q_abs[..., 2].square(), m12 + m21), dim=-1),
            torch.stack((m10 - m01, m02 + m20, m12 + m21, q_abs[..., 3].square()), dim=-1),
        ),
        dim=-2,
    )
    candidates = candidates / (2.0 * q_abs[..., None].clamp_min(0.1))
    choice = q_abs.argmax(dim=-1)
    gather = choice[..., None, None].expand(choice.shape + (1, 4))
    return safe_normalize(candidates.gather(-2, gather).squeeze(-2), dim=-1)


def cosine_sample_hemisphere(num: int, normals: Tensor, generator=None) -> Tensor:
    """Cosine-weighted hemisphere samples around each normal."""

    device = normals.device
    dtype = normals.dtype
    n = normals.reshape(-1, 3)
    u1 = torch.rand((n.shape[0], num), device=device, dtype=dtype, generator=generator)
    u2 = torch.rand((n.shape[0], num), device=device, dtype=dtype, generator=generator)
    r = torch.sqrt(u1.clamp_min(1.0e-8))
    phi = 2.0 * torch.pi * u2
    local = torch.stack(
        [r * torch.cos(phi), r * torch.sin(phi), torch.sqrt((1.0 - u1).clamp_min(0.0))],
        dim=-1,
    )

    up = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype).expand_as(n)
    alt = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype).expand_as(n)
    tangent_seed = torch.where((n[..., 2:3].abs() > 0.99), alt, up)
    tangent = safe_normalize(torch.cross(tangent_seed, n, dim=-1), dim=-1)
    bitangent = torch.cross(n, tangent, dim=-1)
    world = (
        local[..., 0:1] * tangent[:, None]
        + local[..., 1:2] * bitangent[:, None]
        + local[..., 2:3] * n[:, None]
    )
    return safe_normalize(world, dim=-1).reshape(normals.shape[:-1] + (num, 3))


@lru_cache(maxsize=128)
def _balanced_hammersley_pattern(num_samples: int, emitters: int, device, dtype) -> Tensor:
    """Interleaved Hammersley sets, each sized to its emitter's actual budget."""
    def radical_inverse(index: int) -> float:
        result, weight = 0.0, 0.5
        while index:
            result += (index & 1) * weight
            index >>= 1
            weight *= 0.5
        return result

    quotient, remainder = divmod(num_samples, emitters)
    uv = []
    for slot in range(num_samples):
        count = quotient + int(slot % emitters < remainder)
        ordinal = slot // emitters
        uv.append(((ordinal + 0.5) / count, radical_inverse(ordinal)))
    return torch.tensor(uv, device=device, dtype=dtype)


def balanced_rqmc_uv(emitter_ids: Tensor, num_emitters: int, dtype, generator=None) -> Tensor:
    """Randomized low-discrepancy samples without changing emitter allocation.

    Each pixel/emitter receives an independent uniform toroidal (Cranley-Patterson)
    shift. Every sample therefore retains the original uniform UV marginal;
    samples within an emitter are deliberately correlated to improve coverage.
    The ID order must be the balanced cyclic order of sample_surface_total.
    This improves placement, NOT the historical ellipsoid PDF surrogate.
    """
    batch, count = emitter_ids.shape
    base = _balanced_hammersley_pattern(count, num_emitters, emitter_ids.device, dtype)
    shifts = torch.rand((batch, num_emitters, 2), device=emitter_ids.device,
                        dtype=dtype, generator=generator)
    selected = shifts.gather(1, emitter_ids[..., None].expand(-1, -1, 2))
    return (base[None] + selected).remainder(1.0)


def uniform_sample_sphere(shape: tuple[int, ...], device, dtype, generator=None) -> Tensor:
    z = torch.rand(shape, device=device, dtype=dtype, generator=generator) * 2.0 - 1.0
    phi = torch.rand(shape, device=device, dtype=dtype, generator=generator) * 2.0 * torch.pi
    r = torch.sqrt((1.0 - z * z).clamp_min(0.0))
    return torch.stack([r * torch.cos(phi), r * torch.sin(phi), z], dim=-1)


def tonemap_reinhard(rgb: Tensor, exposure: float = 1.0) -> Tensor:
    hdr = rgb.clamp_min(0.0) * exposure
    return hdr / (1.0 + hdr)


def linear_to_srgb(rgb: Tensor) -> Tensor:
    rgb = rgb.clamp(0.0, 1.0)
    low = rgb * 12.92
    high = 1.055 * rgb.clamp_min(1.0e-8).pow(1.0 / 2.4) - 0.055
    return torch.where(rgb <= 0.0031308, low, high)


def srgb_to_linear(rgb: Tensor) -> Tensor:
    rgb = rgb.clamp(0.0, 1.0)
    low = rgb / 12.92
    high = ((rgb + 0.055) / 1.055).clamp_min(0.0).pow(2.4)
    return torch.where(rgb <= 0.04045, low, high)
