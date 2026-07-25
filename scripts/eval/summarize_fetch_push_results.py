#!/usr/bin/env python
"""Combine held-out and exact-fork metrics into an attribution verdict."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_named_path(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use NAME=/path/to/evaluation.json")
    name, path = value.split("=", 1)
    return name, Path(path)


def model_verdict(name, heldout, fork, horizon):
    heldout_metrics = heldout["flow"]
    fork_metrics = fork["models"][name][str(horizon)]
    paired = fork["paired_model_bootstrap"][name]
    versus_nominal = paired["versus_deterministic"]
    versus_mean = paired["versus_residual_mean"]
    gates = {
        "heldout_residual_skill_positive": (
            heldout_metrics["residual_energy_skill"] > 0
        ),
        "heldout_stochastic_skill_positive": (
            heldout_metrics["stochastic_energy_skill_vs_residual_mean"] > 0
        ),
        "fork_residual_skill_positive": fork_metrics["residual_energy_skill"] > 0,
        "fork_stochastic_skill_positive": (
            fork_metrics["stochastic_energy_skill_vs_residual_mean"] > 0
        ),
        "fork_beats_deterministic_ci95": versus_nominal["ci95"][0] > 0,
        "fork_beats_residual_mean_ci95": versus_mean["ci95"][0] > 0,
    }
    return {
        "heldout": {
            "energy_score": heldout_metrics["energy_score"],
            "residual_energy_skill": heldout_metrics["residual_energy_skill"],
            "stochastic_energy_skill_vs_residual_mean": heldout_metrics[
                "stochastic_energy_skill_vs_residual_mean"
            ],
        },
        "exact_fork": {
            "horizon": horizon,
            "energy_score": fork_metrics["energy_score"],
            "residual_energy_skill": fork_metrics["residual_energy_skill"],
            "stochastic_energy_skill_vs_residual_mean": fork_metrics[
                "stochastic_energy_skill_vs_residual_mean"
            ],
            "versus_deterministic": versus_nominal,
            "versus_residual_mean": versus_mean,
        },
        "gates": gates,
        "strict_stochastic_help": all(gates.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heldout", action="append", type=parse_named_path, required=True)
    parser.add_argument("--fork", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=10)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite immutable verdict: {args.output}")

    fork = json.loads(args.fork.read_text())
    heldout = {
        name: json.loads(path.read_text()) for name, path in args.heldout
    }
    models = {
        name: model_verdict(name, metrics, fork, args.horizon)
        for name, metrics in heldout.items()
    }
    strict = [name for name, metrics in models.items() if metrics["strict_stochastic_help"]]
    result = {
        "question": "Does sampled residual spread help beyond deterministic LeWM and residual-mean correction?",
        "horizon": args.horizon,
        "models": models,
        "strictly_passing_models": strict,
        "verdict": "stochastic_residual_helps" if strict else "not_demonstrated",
        "one_seed_gate_only": True,
        "inputs": {
            "fork": str(args.fork.resolve()),
            "heldout": {name: str(path.resolve()) for name, path in args.heldout},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
