#!/usr/bin/env python
"""Aggregate the confirmatory FetchPush gate across model seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_named_path(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use SEED=/path/to/result-directory")
    seed, path = value.split("=", 1)
    return seed, Path(path)


def interval(values):
    quantiles = torch.quantile(values, torch.tensor([0.025, 0.975]))
    return [float(value) for value in quantiles]


def crossed_bootstrap(model, deterministic, residual_mean, *, draws, seed):
    """Resample model seeds and shared contexts as crossed random effects."""
    if model.shape != deterministic.shape or model.shape != residual_mean.shape:
        raise ValueError("Primary score arrays must have identical seed/context shape")
    num_seeds, num_contexts = model.shape
    generator = torch.Generator().manual_seed(seed)
    residual_delta = torch.empty(draws)
    stochastic_delta = torch.empty(draws)
    residual_skill = torch.empty(draws)
    stochastic_skill = torch.empty(draws)
    for draw in range(draws):
        seed_indices = torch.randint(
            num_seeds, (num_seeds,), generator=generator
        )
        context_indices = torch.randint(
            num_contexts, (num_contexts,), generator=generator
        )
        sampled_model = model[seed_indices][:, context_indices].mean()
        sampled_deterministic = deterministic[seed_indices][:, context_indices].mean()
        sampled_mean = residual_mean[seed_indices][:, context_indices].mean()
        residual_delta[draw] = sampled_deterministic - sampled_model
        stochastic_delta[draw] = sampled_mean - sampled_model
        residual_skill[draw] = 1.0 - sampled_model / sampled_deterministic
        stochastic_skill[draw] = 1.0 - sampled_model / sampled_mean
    return {
        "versus_deterministic": {
            "mean_improvement": float((deterministic - model).mean()),
            "ci95": interval(residual_delta),
            "energy_skill": float(1.0 - model.mean() / deterministic.mean()),
            "energy_skill_ci95": interval(residual_skill),
        },
        "versus_residual_mean": {
            "mean_improvement": float((residual_mean - model).mean()),
            "ci95": interval(stochastic_delta),
            "energy_skill": float(1.0 - model.mean() / residual_mean.mean()),
            "energy_skill_ci95": interval(stochastic_skill),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=parse_named_path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=23072)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite immutable summary: {args.output}")

    runs = {}
    for seed_name, directory in args.run:
        runs[seed_name] = {
            "verdict": json.loads((directory / "verdict.json").read_text()),
            "fork": json.loads((directory / "fork_metrics.json").read_text()),
            "directory": directory,
        }
    if len(runs) < 2:
        raise SystemExit("At least two model seeds are required")

    first = next(iter(runs.values()))
    horizon = int(first["verdict"]["horizon"])
    context_seeds = first["fork"]["metadata"].get("context_seeds")
    models = set(first["verdict"]["models"])
    for seed_name, run in runs.items():
        if int(run["verdict"]["horizon"]) != horizon:
            raise ValueError(f"Horizon mismatch for seed {seed_name}")
        if run["fork"]["metadata"].get("context_seeds") != context_seeds:
            raise ValueError(f"Exact-fork context mismatch for seed {seed_name}")
        models &= set(run["verdict"]["models"])

    model_results = {}
    for model_name in sorted(models):
        model_scores = []
        deterministic_scores = []
        mean_scores = []
        per_seed = {}
        for seed_name, run in runs.items():
            fork = run["fork"]
            scores = fork["primary_context_energy_scores"]
            mean = fork["primary_context_mean_energy_scores"]
            model_scores.append(torch.tensor(scores[model_name]))
            deterministic_scores.append(torch.tensor(scores["deterministic"]))
            mean_scores.append(torch.tensor(mean[model_name]))
            verdict = run["verdict"]["models"][model_name]
            per_seed[seed_name] = {
                "strict_stochastic_help": verdict["strict_stochastic_help"],
                "heldout_residual_skill": verdict["heldout"][
                    "residual_energy_skill"
                ],
                "heldout_stochastic_skill_vs_residual_mean": verdict["heldout"][
                    "stochastic_energy_skill_vs_residual_mean"
                ],
                "fork_residual_skill": verdict["exact_fork"][
                    "residual_energy_skill"
                ],
                "fork_stochastic_skill_vs_residual_mean": verdict["exact_fork"][
                    "stochastic_energy_skill_vs_residual_mean"
                ],
            }
        bootstrap = crossed_bootstrap(
            torch.stack(model_scores),
            torch.stack(deterministic_scores),
            torch.stack(mean_scores),
            draws=args.bootstrap_draws,
            seed=args.seed,
        )
        gates = {
            "all_seed_verdicts_pass": all(
                values["strict_stochastic_help"] for values in per_seed.values()
            ),
            "all_heldout_residual_skills_positive": all(
                values["heldout_residual_skill"] > 0 for values in per_seed.values()
            ),
            "all_heldout_stochastic_skills_positive": all(
                values["heldout_stochastic_skill_vs_residual_mean"] > 0
                for values in per_seed.values()
            ),
            "multiseed_fork_beats_deterministic_ci95": (
                bootstrap["versus_deterministic"]["ci95"][0] > 0
            ),
            "multiseed_fork_beats_residual_mean_ci95": (
                bootstrap["versus_residual_mean"]["ci95"][0] > 0
            ),
        }
        model_results[model_name] = {
            "per_seed": per_seed,
            "crossed_seed_context_bootstrap": bootstrap,
            "gates": gates,
            "strict_multiseed_stochastic_help": all(gates.values()),
        }

    passing = [
        name
        for name, result in model_results.items()
        if result["strict_multiseed_stochastic_help"]
    ]
    result = {
        "question": "Does stochastic residual spread replicate across model seeds?",
        "horizon": horizon,
        "model_seeds": list(runs),
        "contexts_per_seed": len(context_seeds) if context_seeds is not None else None,
        "shared_exact_fork_contexts": True,
        "bootstrap": {
            "design": "crossed resampling of model seeds and shared simulator contexts",
            "draws": args.bootstrap_draws,
            "seed": args.seed,
        },
        "models": model_results,
        "strictly_passing_models": passing,
        "verdict": "stochastic_residual_replicates" if passing else "not_replicated",
        "runs": {
            seed_name: str(run["directory"].resolve())
            for seed_name, run in runs.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
