#!/usr/bin/env python
"""Score fixed-mode posterior forks and uncertainty contraction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch

from scripts.eval.evaluate_fork_samples import score_distribution, summarize
from stochastic_metrics import (
    energy_score,
    energy_skill_score,
    paired_bootstrap_interval,
)


def sample_spread(samples):
    values = samples.flatten(2)
    centered = values - values.mean(1, keepdim=True)
    # RMS radius is exactly zero for repeated deterministic samples and avoids
    # the cancellation noise of high-dimensional pairwise cdist(x, x).
    return centered.square().sum(-1).mean(1).sqrt()


def mode_contrast(samples, truth, eps=1e-8):
    if samples.size(0) % 2:
        raise ValueError("Fixed-mode contexts must be ordered in low/high pairs")
    predicted = samples.flatten(2).mean(1)
    observed = truth.flatten(2).mean(1)
    predicted_delta = predicted[1::2] - predicted[0::2]
    observed_delta = observed[1::2] - observed[0::2]
    error = (predicted_delta - observed_delta).square().sum()
    target = observed_delta.square().sum().clamp_min(eps)
    cosine = torch.nn.functional.cosine_similarity(
        predicted_delta, observed_delta, dim=-1, eps=eps
    )
    return {
        "error_explained": float(1.0 - error / target),
        "cosine": float(cosine.mean()),
        "magnitude_ratio": float(
            predicted_delta.norm(dim=-1).sum()
            / observed_delta.norm(dim=-1).sum().clamp_min(eps)
        ),
    }


def mode_contrast_squared_error(samples, truth):
    predicted = samples.flatten(2).mean(1)
    observed = truth.flatten(2).mean(1)
    predicted_delta = predicted[1::2] - predicted[0::2]
    observed_delta = observed[1::2] - observed[0::2]
    return (predicted_delta - observed_delta).square().sum(-1)


def ci(values, seed):
    interval = paired_bootstrap_interval(
        values, generator=torch.Generator().manual_seed(seed)
    )
    return [float(value) for value in interval]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--posterior-input", type=Path, required=True)
    parser.add_argument("--prior-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--primary-horizon", type=int, default=8)
    parser.add_argument("--seed", type=int, default=33072)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite immutable result: {args.output}")
    posterior = torch.load(args.posterior_input, map_location="cpu", weights_only=True)
    prior = torch.load(args.prior_input, map_location="cpu", weights_only=True)
    metadata = posterior["metadata"]
    if metadata.get("regime") != "post_contact_fixed_mode":
        raise ValueError("Posterior payload has the wrong regime")
    if metadata.get("scene_seeds") != prior["metadata"].get("context_seeds"):
        raise ValueError("Prior and posterior payloads do not use the same scenes")
    observation_steps = int(metadata["observation_model_steps"])
    posterior_truth = posterior["simulator_futures"]
    posterior_models = posterior["model_samples"]
    prior_models = prior["model_samples"]
    horizons = posterior["horizons"]
    if max(horizons) > prior["simulator_futures"].size(2):
        raise ValueError("Prior rollout is too short for same-horizon contraction scoring")

    result = {
        "posterior_input": str(args.posterior_input.resolve()),
        "prior_input": str(args.prior_input.resolve()),
        "metadata": metadata,
        "horizons": horizons,
        "models": {},
        "paired_primary_bootstrap": {},
    }
    per_model = {}
    for model_name, samples in posterior_models.items():
        if model_name not in prior_models:
            raise ValueError(f"Prior payload is missing model {model_name}")
        model_result = {}
        per_model[model_name] = {}
        for horizon in horizons:
            posterior_index = horizon - 1
            # Compare equal numbers of forecast transitions.  Aligning by
            # absolute task time would give the posterior fewer stochastic
            # residual draws and create mechanical "contraction".
            prior_index = horizon - 1
            model_samples = samples[:, :, posterior_index]
            truth = posterior_truth[:, :, posterior_index]
            mean_samples = model_samples.mean(1, keepdim=True)
            values = score_distribution(model_samples, truth, seed=args.seed)
            metrics = summarize(values)
            model_score = values["energy_score"]
            mean_score = energy_score(mean_samples, truth)
            post_spread = sample_spread(model_samples)
            prior_samples = prior_models[model_name][:, :, prior_index]
            prior_samples = prior_samples.repeat_interleave(2, dim=0)
            prior_spread = sample_spread(prior_samples)
            metrics.update(
                {
                    "mean_energy_score": float(mean_score.mean()),
                    "stochastic_energy_skill_vs_mean": float(
                        energy_skill_score(model_samples, truth, mean_samples)
                    ),
                    "sample_spread": float(post_spread.mean()),
                    "same_horizon_prior_sample_spread": float(prior_spread.mean()),
                    "spread_contraction": float(
                        1.0
                        - post_spread.sum()
                        / prior_spread.sum().clamp_min(1e-8)
                    ),
                    "mode_contrast": mode_contrast(model_samples, truth),
                    "energy_score_low": float(model_score[0::2].mean()),
                    "energy_score_high": float(model_score[1::2].mean()),
                }
            )
            model_result[str(horizon)] = metrics
            if horizon == args.primary_horizon:
                per_model[model_name] = {
                    "score": model_score,
                    "mean_score": mean_score,
                    "spread_delta": prior_spread - post_spread,
                    "contrast_error": mode_contrast_squared_error(
                        model_samples, truth
                    ),
                }
        result["models"][model_name] = model_result

    deterministic = per_model["deterministic"]
    result["primary_horizon"] = args.primary_horizon
    result["verdicts"] = {}
    for offset, (model_name, values) in enumerate(per_model.items()):
        if model_name == "deterministic":
            continue
        score_delta = deterministic["score"] - values["score"]
        mean_delta = deterministic["score"] - values["mean_score"]
        stochastic_delta = values["mean_score"] - values["score"]
        spread_delta = values["spread_delta"]
        contrast_delta = (
            deterministic["contrast_error"] - values["contrast_error"]
        )
        bootstrap = {
            "versus_deterministic": {
                "mean_improvement": float(score_delta.mean()),
                "ci95": ci(score_delta, args.seed + offset),
            },
            "mean_versus_deterministic": {
                "mean_improvement": float(mean_delta.mean()),
                "ci95": ci(mean_delta, args.seed + 10 + offset),
            },
            "sampling_versus_mean": {
                "mean_improvement": float(stochastic_delta.mean()),
                "ci95": ci(stochastic_delta, args.seed + 20 + offset),
            },
            "spread_contraction": {
                "mean_improvement": float(spread_delta.mean()),
                "ci95": ci(spread_delta, args.seed + 30 + offset),
            },
            "mode_contrast_error": {
                "mean_improvement": float(contrast_delta.mean()),
                "ci95": ci(contrast_delta, args.seed + 40 + offset),
            },
        }
        result["paired_primary_bootstrap"][model_name] = bootstrap
        contrast = result["models"][model_name][str(args.primary_horizon)][
            "mode_contrast"
        ]
        gates = {
            "posterior_beats_deterministic_ci95": (
                bootstrap["versus_deterministic"]["ci95"][0] > 0
            ),
            "posterior_mean_beats_deterministic_ci95": (
                bootstrap["mean_versus_deterministic"]["ci95"][0] > 0
            ),
            "spread_contracts_ci95": (
                bootstrap["spread_contraction"]["ci95"][0] > 0
            ),
            "mode_contrast_improves_over_deterministic_ci95": (
                bootstrap["mode_contrast_error"]["ci95"][0] > 0
            ),
            "mode_contrast_direction_positive": contrast["cosine"] > 0,
        }
        result["verdicts"][model_name] = {
            "gates": gates,
            "fixed_mode_adaptation": all(gates.values()),
            "sampling_beats_mean": (
                bootstrap["sampling_versus_mean"]["ci95"][0] > 0
            ),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
