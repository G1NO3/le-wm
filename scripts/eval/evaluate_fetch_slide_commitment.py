#!/usr/bin/env python
"""Evaluate learned stochastic residual decisions on FetchSlide commitment."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import numpy as np
import stable_worldmodel  # noqa: F401 - registers swm environments
import torch

from experiment_data import sha256_file
from scripts.data.probe_fetch_slide_friction import (
    collect_strike_context,
    comparison_summary,
    discrete_distribution_index,
    exact_terminal_positions,
    fixed_profile,
    mean_state_index,
    smooth_distribution_index,
    strike_candidates,
    terminal_distances,
)
from stochastic_physics import FetchSlideHiddenFriction


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contexts", type=int, default=64)
    parser.add_argument("--seed", type=int, default=703072)
    parser.add_argument("--raw-horizon", type=int, default=100)
    parser.add_argument("--success-threshold", type=float, default=0.05)
    parser.add_argument("--success-temperature", type=float, default=0.01)
    parser.add_argument("--bootstrap-draws", type=int, default=20_000)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


@torch.no_grad()
def model_terminal_positions(model, initial_state, candidates, device):
    if candidates.shape[1] % model.action_block:
        raise ValueError("Candidate horizon must divide into model action blocks")
    raw = torch.as_tensor(candidates, device=device).float()
    action_blocks = raw.reshape(
        raw.size(0), raw.size(1) // model.action_block, -1
    )
    count = action_blocks.size(0)
    state = torch.as_tensor(initial_state, device=device).float()
    mean_state = state[None].expand(count, -1).clone()
    mode_state = state[None, None].expand(count, 2, -1).clone()
    mode = torch.arange(2, device=device)[None].expand(count, -1)
    for step in range(action_blocks.size(1)):
        block = action_blocks[:, step]
        mean_state = model.transition_mode_mixture(mean_state, block, mode=None)
        mode_state = model.transition_mode_mixture(
            mode_state,
            block[:, None].expand(-1, 2, -1),
            mode=mode,
        )
    return (
        mean_state[:, 3:6].cpu().numpy(),
        mode_state[:, :, 3:6].cpu().numpy(),
    )


def main():
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite immutable result: {args.output}")
    if args.raw_horizon <= 0 or args.raw_horizon % 2:
        raise SystemExit("--raw-horizon must be a positive multiple of two")

    modes = (0.2, 1.0)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = torch.load(
        args.checkpoint, map_location=device, weights_only=False
    ).to(device).eval()
    env = FetchSlideHiddenFriction(
        gym.make(
            "swm/FetchSlide-v3",
            max_episode_steps=args.raw_horizon + 80,
            render_mode="rgb_array",
        ),
        profile=fixed_profile(1.0),
        terminate_at_goal=False,
    )

    model_mean_successes = []
    model_distribution_successes = []
    oracle_mean_successes = []
    oracle_smooth_successes = []
    oracle_discrete_successes = []
    records = []
    candidate_seed = int(args.seed)
    try:
        while len(records) < args.contexts:
            context = collect_strike_context(env, candidate_seed)
            candidate_seed += 1
            if context is None:
                continue
            candidates, parameters = strike_candidates(
                context["state"],
                horizon=args.raw_horizon,
                speed_min=0.4,
                speed_max=1.0,
                speed_count=13,
                duration_min=1,
                duration_max=4,
                angle_max_deg=0.0,
                angle_count=1,
            )
            goal = context["state"][-3:]
            exact = exact_terminal_positions(
                env, context["fork_state"], candidates, modes
            )
            exact_distances = terminal_distances(exact, goal)
            exact_successes = exact_distances <= args.success_threshold
            oracle_mean_index = mean_state_index(exact, goal)
            oracle_smooth_index = smooth_distribution_index(
                exact_distances,
                args.success_threshold,
                args.success_temperature,
            )
            oracle_discrete_index = discrete_distribution_index(
                exact_distances, args.success_threshold
            )

            predicted_mean, predicted_modes = model_terminal_positions(
                model, context["state"], candidates, device
            )
            model_mean_index = int(
                np.argmin(terminal_distances(predicted_mean, goal))
            )
            predicted_mode_distances = terminal_distances(predicted_modes, goal)
            model_distribution_index = smooth_distribution_index(
                predicted_mode_distances,
                args.success_threshold,
                args.success_temperature,
            )

            model_mean_selected = exact_successes[model_mean_index]
            model_distribution_selected = exact_successes[
                model_distribution_index
            ]
            oracle_mean_selected = exact_successes[oracle_mean_index]
            oracle_smooth_selected = exact_successes[oracle_smooth_index]
            oracle_discrete_selected = exact_successes[oracle_discrete_index]
            model_mean_successes.append(model_mean_selected)
            model_distribution_successes.append(model_distribution_selected)
            oracle_mean_successes.append(oracle_mean_selected)
            oracle_smooth_successes.append(oracle_smooth_selected)
            oracle_discrete_successes.append(oracle_discrete_selected)
            records.append(
                {
                    "seed": context["seed"],
                    "indices": {
                        "model_mean": model_mean_index,
                        "model_distribution": model_distribution_index,
                        "oracle_mean": oracle_mean_index,
                        "oracle_smooth": oracle_smooth_index,
                        "oracle_discrete": oracle_discrete_index,
                    },
                    "parameters": {
                        "model_mean": parameters[model_mean_index],
                        "model_distribution": parameters[
                            model_distribution_index
                        ],
                        "oracle_mean": parameters[oracle_mean_index],
                        "oracle_smooth": parameters[oracle_smooth_index],
                        "oracle_discrete": parameters[oracle_discrete_index],
                    },
                    "model_mean_success": model_mean_selected.tolist(),
                    "model_distribution_success": (
                        model_distribution_selected.tolist()
                    ),
                    "oracle_mean_success": oracle_mean_selected.tolist(),
                    "oracle_smooth_success": oracle_smooth_selected.tolist(),
                    "oracle_discrete_success": oracle_discrete_selected.tolist(),
                    "model_mean_exact_terminal_distance_m": exact_distances[
                        model_mean_index
                    ].tolist(),
                    "model_distribution_exact_terminal_distance_m": (
                        exact_distances[model_distribution_index].tolist()
                    ),
                }
            )
            print(
                f"slide commitment {len(records)}/{args.contexts} "
                f"seed={context['seed']}",
                flush=True,
            )
    finally:
        env.close()

    model_comparison = comparison_summary(
        "learned_distribution_vs_learned_residual_mean",
        model_mean_successes,
        model_distribution_successes,
        draws=args.bootstrap_draws,
        seed=args.seed,
    )
    oracle_smooth_comparison = comparison_summary(
        "exact_smooth_distribution_vs_exact_mean_state",
        oracle_mean_successes,
        oracle_smooth_successes,
        draws=args.bootstrap_draws,
        seed=args.seed + 1,
    )
    oracle_discrete_comparison = comparison_summary(
        "exact_discrete_distribution_vs_exact_mean_state",
        oracle_mean_successes,
        oracle_discrete_successes,
        draws=args.bootstrap_draws,
        seed=args.seed + 2,
    )
    result = {
        "task": "fetch_slide_hidden_friction_commitment",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "contexts": args.contexts,
        "episodes": 2 * args.contexts,
        "seed": args.seed,
        "friction_modes": list(modes),
        "raw_horizon": args.raw_horizon,
        "success_threshold_m": args.success_threshold,
        "comparisons": {
            "learned": model_comparison,
            "oracle_smooth": oracle_smooth_comparison,
            "oracle_discrete": oracle_discrete_comparison,
        },
        "gates": {
            "exact_task_value": oracle_smooth_comparison[
                "strict_improvement_gate"
            ],
            "learned_stochastic_value": model_comparison[
                "strict_improvement_gate"
            ],
        },
        "records": records,
    }
    result["gates"]["pass"] = all(result["gates"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
