from __future__ import annotations

import torch
from torch import Tensor

from .math_utils import safe_normalize


class MicrofacetBRDF:
    """Diffuse plus GGX microfacet BRDF from Appendix A.1."""

    def __init__(self, eps: float = 1.0e-6) -> None:
        self.eps = eps

    def __call__(
        self,
        view_dir: Tensor,
        light_dir: Tensor,
        normal: Tensor,
        albedo: Tensor,
        roughness: Tensor,
        metallic: Tensor,
    ) -> Tensor:
        v = safe_normalize(view_dir, dim=-1, eps=self.eps)
        w = safe_normalize(light_dir, dim=-1, eps=self.eps)
        n = safe_normalize(normal, dim=-1, eps=self.eps)

        while v.ndim < w.ndim:
            v = v.unsqueeze(-2)
            n = n.unsqueeze(-2)
            albedo = albedo.unsqueeze(-2)
            roughness = roughness.unsqueeze(-2)
            metallic = metallic.unsqueeze(-2)

        ndotv = (n * v).sum(-1, keepdim=True).clamp_min(0.0)
        ndotw = (n * w).sum(-1, keepdim=True).clamp_min(0.0)
        h = safe_normalize(v + w, dim=-1, eps=self.eps)
        vdoth = (v * h).sum(-1, keepdim=True).clamp(0.0, 1.0)
        ndoth = (n * h).sum(-1, keepdim=True).clamp_min(0.0)

        alpha = roughness.clamp(0.02, 1.0).pow(2.0)
        diffuse = albedo * (1.0 - metallic).clamp(0.0, 1.0) / torch.pi
        d = self._ggx_distribution(ndoth, alpha)
        g = self._smith_geometry(ndotv, ndotw, alpha)
        f0 = 0.04 * (1.0 - metallic) + albedo * metallic
        f = f0 + (1.0 - f0) * (1.0 - vdoth).pow(5.0)
        specular = d * g * f / (4.0 * ndotv * ndotw + self.eps)
        return diffuse + specular

    def _ggx_distribution(self, ndoth: Tensor, alpha: Tensor) -> Tensor:
        a2 = alpha.pow(2.0)
        denom = ndoth.pow(2.0) * (a2 - 1.0) + 1.0
        return a2 / (torch.pi * denom.pow(2.0) + self.eps)

    def _smith_geometry(self, ndotv: Tensor, ndotw: Tensor, alpha: Tensor) -> Tensor:
        return self._smith_g1(ndotv, alpha) * self._smith_g1(ndotw, alpha)

    def _smith_g1(self, ndotx: Tensor, alpha: Tensor) -> Tensor:
        a = alpha.clamp_min(self.eps)
        return 2.0 * ndotx / (ndotx + torch.sqrt(a.pow(2.0) + (1.0 - a.pow(2.0)) * ndotx.pow(2.0)) + self.eps)
