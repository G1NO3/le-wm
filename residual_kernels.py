"""Common conditional residual-kernel implementations for latent dynamics."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class ResidualKernel(nn.Module):
    """Interface shared by deployable normalized-residual distributions."""

    nfe: int = 0

    def sample(
        self,
        condition: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
        uniform_noise: torch.Tensor | None = None,
        steps: int | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def loss(self, target: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class GlobalDiagonalGaussian(ResidualKernel):
    """Standard normal in normalized residual space.

    The global location and scale are buffers so all uncertainty heads can use
    the exact same frozen residual targets and normalization statistics.
    """

    def __init__(self, residual_dim: int):
        super().__init__()
        self.location = nn.Parameter(torch.zeros(residual_dim))
        self.raw_scale = nn.Parameter(torch.full((residual_dim,), 0.5413248546))

    @property
    def scale(self):
        return F.softplus(self.raw_scale).clamp_min(1e-4)

    def sample(self, condition, *, noise=None, uniform_noise=None, steps=None):
        del uniform_noise, steps
        if noise is None:
            noise = torch.randn(
                *condition.shape[:-1],
                self.location.numel(),
                device=condition.device,
                dtype=condition.dtype,
            )
        return self.location.to(noise) + self.scale.to(noise) * noise

    def loss(self, target, condition):
        del condition
        return 0.5 * (
            ((target - self.location) / self.scale).square()
            + 2 * self.scale.log()
            + math.log(2 * math.pi)
        ).mean()


class _ConditionalHead(ResidualKernel):
    def __init__(self, condition_dim: int, residual_dim: int, hidden_dim=512, depth=3):
        super().__init__()
        layers: list[nn.Module] = []
        for index in range(depth):
            layers.extend(
                [
                    nn.Linear(condition_dim if index == 0 else hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                ]
            )
        self.backbone = nn.Sequential(*layers)
        self.residual_dim = residual_dim


class ConditionalDiagonalGaussian(_ConditionalHead):
    def __init__(self, condition_dim, residual_dim, hidden_dim=512, depth=3):
        super().__init__(condition_dim, residual_dim, hidden_dim, depth)
        self.out = nn.Linear(hidden_dim, 2 * residual_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def parameters_for(self, condition):
        mean, raw_scale = self.out(self.backbone(condition)).chunk(2, dim=-1)
        scale = F.softplus(raw_scale + 0.5413248546).clamp_min(1e-4)
        return mean, scale

    def sample(self, condition, *, noise=None, uniform_noise=None, steps=None):
        del uniform_noise, steps
        mean, scale = self.parameters_for(condition)
        noise = torch.randn_like(mean) if noise is None else noise
        return mean + scale * noise

    def loss(self, target, condition):
        mean, scale = self.parameters_for(condition)
        return (0.5 * ((target - mean) / scale).square() + scale.log()).mean()


class ConditionalGaussianMixture(_ConditionalHead):
    def __init__(
        self,
        condition_dim,
        residual_dim,
        hidden_dim=512,
        depth=3,
        components=2,
    ):
        super().__init__(condition_dim, residual_dim, hidden_dim, depth)
        self.components = components
        self.out = nn.Linear(hidden_dim, components * (1 + 2 * residual_dim))
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def parameters_for(self, condition):
        shape = (*condition.shape[:-1], self.components, 1 + 2 * self.residual_dim)
        raw = self.out(self.backbone(condition)).reshape(shape)
        logits = raw[..., 0]
        mean = raw[..., 1 : 1 + self.residual_dim]
        scale = F.softplus(raw[..., 1 + self.residual_dim :] + 0.5413248546)
        return logits, mean, scale.clamp_min(1e-4)

    def sample(self, condition, *, noise=None, uniform_noise=None, steps=None):
        del steps
        logits, mean, scale = self.parameters_for(condition)
        if noise is None:
            component_noise = torch.randn_like(mean[..., 0, :])
        else:
            component_noise = noise
        if uniform_noise is None:
            uniforms = torch.rand_like(logits[..., 0])
        else:
            uniforms = uniform_noise
            if uniforms.shape == (*logits.shape[:-1], 1):
                uniforms = uniforms.squeeze(-1)
            if uniforms.shape != logits.shape[:-1]:
                raise ValueError(
                    "GMM component uniforms must match condition leading axes; "
                    f"got {tuple(uniforms.shape)} for {tuple(logits.shape[:-1])}"
                )
        probs = logits.softmax(dim=-1)
        component = (uniforms.unsqueeze(-1) > probs.cumsum(dim=-1)).sum(-1)
        component = component.clamp_max(self.components - 1)
        gather = component[..., None, None].expand(*component.shape, 1, self.residual_dim)
        chosen_mean = mean.gather(-2, gather).squeeze(-2)
        chosen_scale = scale.gather(-2, gather).squeeze(-2)
        return chosen_mean + chosen_scale * component_noise

    def loss(self, target, condition):
        logits, mean, scale = self.parameters_for(condition)
        target = target.unsqueeze(-2)
        component_log_prob = -0.5 * ((target - mean) / scale).square()
        component_log_prob -= scale.log() + 0.5 * math.log(2 * math.pi)
        component_log_prob = component_log_prob.sum(dim=-1)
        return -torch.logsumexp(logits.log_softmax(-1) + component_log_prob, -1).mean()


def build_residual_kernel(kind: str, *, residual_dim: int, condition_dim: int, **kwargs):
    """Construct a normalized-residual kernel from a stable config name."""

    names = {
        "global_gaussian": GlobalDiagonalGaussian,
        "conditional_gaussian": ConditionalDiagonalGaussian,
        "gmm": ConditionalGaussianMixture,
    }
    if kind not in names:
        raise ValueError(f"Unknown residual kernel {kind!r}; expected one of {sorted(names)}")
    cls = names[kind]
    if cls is GlobalDiagonalGaussian:
        return cls(residual_dim=residual_dim)
    return cls(condition_dim=condition_dim, residual_dim=residual_dim, **kwargs)
