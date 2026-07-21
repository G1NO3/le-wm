#!/usr/bin/env python
"""Evaluate model samples against exact simulator-fork future distributions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch

from stochastic_metrics import (
    cross_time_covariance_error,
    energy_score,
    energy_skill_score,
    interval_metrics,
    lag1_autocorrelation_error,
    mean_error_explained_fraction,
    mmd_rbf,
    paired_bootstrap_interval,
    randomized_pit,
    sliced_wasserstein,
    trajectory_energy_score,
)


def covariance(samples):
    centered = samples - samples.mean(1, keepdim=True)
    return centered.transpose(-1, -2) @ centered / max(samples.size(1) - 1, 1)


def score_distribution(samples, observations, *, seed=0):
    generator = torch.Generator(device=samples.device).manual_seed(seed)
    sample_flat = samples.flatten(2)
    observation_flat = observations.flatten(2)
    sample_mean = sample_flat.mean(1)
    observation_mean = observation_flat.mean(1)
    mean_error = (sample_mean - observation_mean).norm(dim=-1)
    covariance_error = (
        covariance(sample_flat) - covariance(observation_flat)
    ).square().sum((-1, -2)).sqrt()
    result = {
        "energy_score": energy_score(samples, observations),
        "sliced_wasserstein": sliced_wasserstein(
            samples, observations, generator=generator
        ),
        "mmd": mmd_rbf(samples, observations),
        "mean_error": mean_error,
        "covariance_error": covariance_error,
    }
    pit = randomized_pit(samples, observations, generator=generator)
    result["pit_ece"] = (
        torch.histc(pit, bins=10, min=0, max=1) / pit.numel() - 0.1
    ).abs().mean().expand(samples.size(0))
    for level in (0.5, 0.8, 0.9):
        coverage, width = interval_metrics(samples, observations, level)
        key = int(level * 100)
        result[f"coverage_{key}"] = coverage
        result[f"interval_width_{key}"] = width
    return result


def summarize(values):
    return {name: float(value.mean()) for name, value in values.items()}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--primary-horizon", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    payload = torch.load(args.input, map_location="cpu", weights_only=True)
    horizons = payload.get("horizons", [1, 5, 10, 20])
    observations = payload["simulator_futures"]
    predictions = payload["model_samples"]
    result = {
        "input": str(args.input),
        "horizons": horizons,
        "metadata": payload.get("metadata", {}),
        "models": {},
    }
    per_context_primary = {}
    for model_name, samples in predictions.items():
        model_result = {}
        for horizon in horizons:
            horizon_index = horizon - 1
            values = score_distribution(
                samples[:, :, horizon_index],
                observations[:, :, horizon_index],
                seed=args.seed,
            )
            model_result[str(horizon)] = summarize(values)
            if horizon == args.primary_horizon:
                per_context_primary[model_name] = values["energy_score"]
            if horizon >= 2:
                trajectory_samples = samples[:, :, :horizon]
                trajectory_truth = observations[:, :, :horizon]
                model_result[str(horizon)]["trajectory_energy_score"] = float(
                    trajectory_energy_score(
                        trajectory_samples, trajectory_truth
                    ).mean()
                )
                model_result[str(horizon)]["lag1_autocorrelation_error"] = float(
                    lag1_autocorrelation_error(
                        trajectory_samples, trajectory_truth
                    ).mean()
                )
                model_result[str(horizon)]["cross_time_covariance_error"] = float(
                    cross_time_covariance_error(
                        trajectory_samples, trajectory_truth
                    ).mean()
                )
        result["models"][model_name] = model_result

    if "deterministic" in predictions:
        nominal = predictions["deterministic"]
        for model_name, samples in predictions.items():
            for horizon in horizons:
                horizon_index = horizon - 1
                model_samples = samples[:, :, horizon_index]
                truth = observations[:, :, horizon_index]
                nominal_samples = nominal[:, :, horizon_index]
                residual_mean = model_samples.mean(dim=1, keepdim=True)
                metrics = result["models"][model_name][str(horizon)]
                metrics["residual_energy_skill"] = float(
                    energy_skill_score(model_samples, truth, nominal_samples)
                )
                metrics["residual_mean_error_explained"] = float(
                    mean_error_explained_fraction(
                        model_samples, truth, nominal_samples
                    )
                )
                metrics["stochastic_energy_skill_vs_residual_mean"] = float(
                    energy_skill_score(model_samples, truth, residual_mean)
                )

    memory_model = "flow_gru_memory"
    memoryless_model = "flow"
    if memory_model in predictions and memoryless_model in predictions:
        for horizon in horizons:
            index = horizon - 1
            memory_samples = predictions[memory_model][:, :, index]
            memoryless_samples = predictions[memoryless_model][:, :, index]
            truth = observations[:, :, index]
            result["models"][memory_model][str(horizon)]["memory_energy_skill"] = float(
                energy_skill_score(memory_samples, truth, memoryless_samples)
            )

    if "physical_simulator_futures" in payload:
        result["physical_state"] = {}
        for state_name, state_truth in payload["physical_simulator_futures"].items():
            result["physical_state"][state_name] = {}
            for model_name, state_samples in payload["physical_model_samples"][state_name].items():
                result["physical_state"][state_name][model_name] = {}
                for horizon in horizons:
                    values = score_distribution(
                        state_samples[:, :, horizon - 1],
                        state_truth[:, :, horizon - 1],
                        seed=args.seed,
                    )
                    result["physical_state"][state_name][model_name][str(horizon)] = summarize(values)
            physical_predictions = payload["physical_model_samples"][state_name]
            if "deterministic" in physical_predictions:
                nominal = physical_predictions["deterministic"]
                for model_name, state_samples in physical_predictions.items():
                    for horizon in horizons:
                        model_samples = state_samples[:, :, horizon - 1]
                        truth = state_truth[:, :, horizon - 1]
                        nominal_samples = nominal[:, :, horizon - 1]
                        residual_mean = model_samples.mean(dim=1, keepdim=True)
                        metrics = result["physical_state"][state_name][model_name][str(horizon)]
                        metrics["residual_energy_skill"] = float(
                            energy_skill_score(model_samples, truth, nominal_samples)
                        )
                        metrics["residual_mean_error_explained"] = float(
                            mean_error_explained_fraction(
                                model_samples, truth, nominal_samples
                            )
                        )
                        metrics["stochastic_energy_skill_vs_residual_mean"] = float(
                            energy_skill_score(model_samples, truth, residual_mean)
                        )

    if "mode_labels" in payload and "model_mode_labels" in payload:
        truth = payload["mode_labels"].bool()
        result["slip_mode"] = {}
        for model_name, labels in payload["model_mode_labels"].items():
            labels = labels.bool()
            true_rate = truth.float().mean(1)
            predicted_rate = labels.float().mean(1)
            # Distributional mode recall/precision compare represented mass,
            # rather than pairing independent simulator and model samples.
            overlap = torch.minimum(true_rate, predicted_rate).sum()
            result["slip_mode"][model_name] = {
                "precision": float(overlap / predicted_rate.sum().clamp_min(1e-8)),
                "recall": float(overlap / true_rate.sum().clamp_min(1e-8)),
                "prevalence_mae": float((true_rate - predicted_rate).abs().mean()),
            }

    if "flow" in per_context_primary:
        result["paired_energy_bootstrap"] = {}
        for baseline in ("deterministic", "conditional_gaussian", "gmm"):
            if baseline not in per_context_primary:
                continue
            # Positive means the flow improves (has a lower proper score).
            delta = per_context_primary[baseline] - per_context_primary["flow"]
            interval = paired_bootstrap_interval(
                delta,
                generator=torch.Generator().manual_seed(args.seed),
            )
            result["paired_energy_bootstrap"][baseline] = {
                "mean_improvement": float(delta.mean()),
                "ci95": [float(x) for x in interval],
            }

    if memory_model in per_context_primary:
        result["paired_memory_bootstrap"] = {}
        for baseline in (memoryless_model, "flow_gru_memory_reset", "flow_gru_memory_shuffled"):
            if baseline not in per_context_primary:
                continue
            delta = per_context_primary[baseline] - per_context_primary[memory_model]
            interval = paired_bootstrap_interval(
                delta,
                generator=torch.Generator().manual_seed(args.seed),
            )
            result["paired_memory_bootstrap"][baseline] = {
                "mean_improvement": float(delta.mean()),
                "ci95": [float(x) for x in interval],
            }

    stages = payload.get("context_stage")
    if stages is not None and memory_model in per_context_primary:
        stages = [str(value) for value in stages]
        result["memory_improvement_by_stage"] = {}
        for stage in sorted(set(stages)):
            mask = torch.tensor([value == stage for value in stages])
            if not mask.any() or memoryless_model not in per_context_primary:
                continue
            delta = (
                per_context_primary[memoryless_model][mask]
                - per_context_primary[memory_model][mask]
            )
            result["memory_improvement_by_stage"][stage] = {
                "contexts": int(mask.sum()),
                "mean_improvement": float(delta.mean()),
            }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
