"""Sample-based proper scores, calibration, and risk metrics."""

from __future__ import annotations

import math

import torch


def _as_context_samples(x):
    if x.ndim < 3:
        raise ValueError("Expected [context, sample, feature...] tensors")
    return x.flatten(2)


def energy_score(samples, observations):
    """Multivariate energy score per context (lower is better)."""

    x = _as_context_samples(samples)
    y = _as_context_samples(observations)
    xy = torch.cdist(x, y).mean(dim=(1, 2))
    xx = torch.cdist(x, x).mean(dim=(1, 2))
    return xy - 0.5 * xx


def energy_skill_score(samples, observations, baseline_samples, eps=1e-8):
    """Aggregate proper-score skill relative to a paired baseline.

    ``1 - ES(model) / ES(baseline)`` is positive when the model explains the
    observed error distribution better than the baseline, zero for no change,
    and negative when it hurts. Scores are summed across paired contexts before
    taking the ratio so near-perfect individual baseline contexts do not cause
    unstable per-context divisions.
    """

    model_score = energy_score(samples, observations).sum()
    baseline_score = energy_score(baseline_samples, observations).sum()
    return 1.0 - model_score / baseline_score.clamp_min(eps)


def mean_error_explained_fraction(samples, observations, baseline_samples, eps=1e-8):
    """Fraction of paired baseline squared error removed by predictive means."""

    model_mean = _as_context_samples(samples).mean(dim=1)
    observation_mean = _as_context_samples(observations).mean(dim=1)
    baseline_mean = _as_context_samples(baseline_samples).mean(dim=1)
    model_error = (model_mean - observation_mean).square().sum()
    baseline_error = (baseline_mean - observation_mean).square().sum()
    return 1.0 - model_error / baseline_error.clamp_min(eps)


def trajectory_energy_score(samples, observations):
    """Energy score over complete flattened trajectories, per context."""

    if samples.ndim < 4 or observations.ndim < 4:
        raise ValueError(
            "Trajectory scores require [context, sample, time, feature...] tensors"
        )
    return energy_score(samples, observations)


def lag1_autocorrelation(samples, eps=1e-8):
    """Lag-one correlation of sampled trajectories, per context."""

    if samples.ndim < 4 or samples.size(2) < 2:
        raise ValueError("Lag-one correlation requires at least two time steps")
    values = samples.flatten(3)
    previous = values[:, :, :-1].flatten(1)
    following = values[:, :, 1:].flatten(1)
    previous = previous - previous.mean(1, keepdim=True)
    following = following - following.mean(1, keepdim=True)
    numerator = (previous * following).sum(1)
    denominator = previous.square().sum(1).sqrt() * following.square().sum(1).sqrt()
    return numerator / denominator.clamp_min(eps)


def lag1_autocorrelation_error(samples, observations):
    """Absolute simulator-versus-model lag-one correlation error."""

    return (lag1_autocorrelation(samples) - lag1_autocorrelation(observations)).abs()


def cross_time_covariance_error(samples, observations):
    """Frobenius error of adjacent-step cross covariance, per context."""

    def adjacent_covariance(values):
        values = values.flatten(3)
        previous = values[:, :, :-1].flatten(1, 2)
        following = values[:, :, 1:].flatten(1, 2)
        previous = previous - previous.mean(1, keepdim=True)
        following = following - following.mean(1, keepdim=True)
        return previous.transpose(-1, -2) @ following / max(previous.size(1) - 1, 1)

    difference = adjacent_covariance(samples) - adjacent_covariance(observations)
    return difference.square().sum((-1, -2)).sqrt()


def mmd_rbf(samples, observations, bandwidth=None):
    x, y = _as_context_samples(samples), _as_context_samples(observations)
    joined = torch.cat([x, y], dim=1)
    distances = torch.cdist(joined, joined).square()
    if bandwidth is None:
        positive = distances[distances > 0]
        bandwidth = positive.median().clamp_min(1e-8) if positive.numel() else 1.0
    kxx = torch.exp(-torch.cdist(x, x).square() / (2 * bandwidth)).mean((1, 2))
    kyy = torch.exp(-torch.cdist(y, y).square() / (2 * bandwidth)).mean((1, 2))
    kxy = torch.exp(-torch.cdist(x, y).square() / (2 * bandwidth)).mean((1, 2))
    return kxx + kyy - 2 * kxy


def sliced_wasserstein(samples, observations, *, projections=128, generator=None):
    x, y = _as_context_samples(samples), _as_context_samples(observations)
    directions = torch.randn(
        projections, x.size(-1), device=x.device, dtype=x.dtype, generator=generator
    )
    directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    xp = (x @ directions.T).sort(dim=1).values
    yp = (y @ directions.T).sort(dim=1).values
    quantiles = max(xp.size(1), yp.size(1))
    grid = torch.linspace(0, 1, quantiles, device=x.device, dtype=x.dtype)

    def resample(values):
        if values.size(1) == quantiles:
            return values
        position = grid * (values.size(1) - 1)
        lo, hi = position.floor().long(), position.ceil().long()
        weight = (position - lo).view(1, -1, 1)
        return values[:, lo] * (1 - weight) + values[:, hi] * weight

    return (resample(xp) - resample(yp)).abs().mean((1, 2))


def randomized_pit(samples, observations, *, generator=None):
    """Randomized empirical PIT values, retaining context/sample/feature axes."""

    x, y = _as_context_samples(samples), _as_context_samples(observations)
    less = (x[:, :, None] < y[:, None]).sum(dim=1)
    equal = (x[:, :, None] == y[:, None]).sum(dim=1)
    uniform = torch.rand(equal.shape, device=x.device, dtype=x.dtype, generator=generator)
    return (less + uniform * (equal + 1)) / (x.size(1) + 1)


def interval_metrics(samples, observations, level):
    if not 0 < level < 1:
        raise ValueError("level must lie in (0, 1)")
    x, y = _as_context_samples(samples), _as_context_samples(observations)
    tail = (1 - level) / 2
    lo, hi = torch.quantile(x, torch.tensor([tail, 1 - tail], device=x.device), dim=1)
    coverage = ((y >= lo[:, None]) & (y <= hi[:, None])).float().mean((1, 2))
    width = (hi - lo).mean(1)
    return coverage, width


def aggregate_particle_cost(cost, objective="mean", tail_fraction=0.25):
    """Aggregate [batch, candidate, particle] costs for particle MPC."""

    if cost.ndim != 3:
        raise ValueError("Particle costs must have [batch, candidate, particle] axes")
    if objective == "mean":
        return cost.mean(dim=-1)
    if objective != "cvar":
        raise ValueError("objective must be 'mean' or 'cvar'")
    count = max(1, math.ceil(cost.size(-1) * tail_fraction))
    return cost.topk(count, dim=-1, largest=True).values.mean(dim=-1)


def paired_bootstrap_interval(delta, *, samples=10000, level=0.95, generator=None):
    delta = torch.as_tensor(delta).flatten()
    indices = torch.randint(
        delta.numel(), (samples, delta.numel()), device=delta.device, generator=generator
    )
    means = delta[indices].mean(1)
    alpha = (1 - level) / 2
    return torch.quantile(means, torch.tensor([alpha, 1 - alpha], device=delta.device))


def pooled_mode_separation(values, labels, groups=None):
    """Distance between two mode means in pooled RMS standard deviations.

    When multiple simulator contexts are pooled, ``groups`` identifies the
    context for each realization. The effect and within-mode variance are then
    computed after pairing within context, so pose differences between
    contexts cannot masquerade as a hidden-physics effect.
    """

    values = torch.as_tensor(values).float().flatten(1)
    labels = torch.as_tensor(labels).bool().flatten()
    if values.size(0) != labels.numel():
        raise ValueError("values and labels must have the same realization axis")
    if labels.all() or (~labels).all():
        raise ValueError("Both hidden-mode groups are required")
    if groups is not None:
        groups = torch.as_tensor(groups).flatten()
        if groups.numel() != labels.numel():
            raise ValueError("groups must identify every realization")
        effects = []
        residuals = []
        for group in groups.unique():
            selected = groups == group
            if not (labels[selected].any() and (~labels[selected]).any()):
                continue
            low = values[selected & ~labels]
            high = values[selected & labels]
            effects.append(high.mean(0) - low.mean(0))
            residuals.extend([low - low.mean(0), high - high.mean(0)])
        if not effects:
            raise ValueError("No context contains both hidden-mode groups")
        mean_distance = torch.stack(effects).mean(0).norm()
        centered = torch.cat(residuals)
        degrees = max(centered.numel() - 2 * len(effects), 1)
        return mean_distance / (centered.square().sum() / degrees).sqrt().clamp_min(1e-8)
    first, second = values[~labels], values[labels]
    mean_distance = (first.mean(0) - second.mean(0)).norm()
    pooled_variance = (
        (first - first.mean(0)).square().sum()
        + (second - second.mean(0)).square().sum()
    ) / max(first.numel() + second.numel() - 2, 1)
    return mean_distance / pooled_variance.sqrt().clamp_min(1e-8)
