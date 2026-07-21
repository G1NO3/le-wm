"""Frozen linear physical-state probes for latent distribution evaluation."""

from __future__ import annotations

import torch


class RidgeProbe:
    def __init__(self, regularization=1e-3):
        self.regularization = regularization
        self.weight = None
        self.mean_x = None
        self.mean_y = None

    def fit(self, latent, target):
        latent = torch.as_tensor(latent).float()
        target = torch.as_tensor(target).float()
        self.mean_x = latent.mean(0, keepdim=True)
        self.mean_y = target.mean(0, keepdim=True)
        x = latent - self.mean_x
        y = target - self.mean_y
        identity = torch.eye(x.size(1), device=x.device, dtype=x.dtype)
        self.weight = torch.linalg.solve(
            x.T @ x + self.regularization * identity, x.T @ y
        )
        return self

    def __call__(self, latent):
        if self.weight is None:
            raise RuntimeError("Probe has not been fit")
        latent = torch.as_tensor(latent).float()
        return (latent - self.mean_x.to(latent)) @ self.weight.to(latent) + self.mean_y.to(latent)

    def state_dict(self):
        return {
            "regularization": self.regularization,
            "weight": self.weight,
            "mean_x": self.mean_x,
            "mean_y": self.mean_y,
        }

    @classmethod
    def from_state_dict(cls, state):
        probe = cls(state["regularization"])
        probe.weight = state["weight"]
        probe.mean_x = state["mean_x"]
        probe.mean_y = state["mean_y"]
        return probe
