#!/usr/bin/env python
"""Evaluate deterministic vanilla LeWM on FetchSlide strike commitment."""

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
    comparison_summary,
    exact_terminal_positions,
    fixed_profile,
    mean_state_index,
    smooth_distribution_index,
    strike_candidates,
    terminal_distances,
)
from scripts.eval.generate_fetch_push_fork_samples import (
    action_statistics,
    pack_actions,
    preprocess_images,
    rollout_model,
)
from scripts.eval.generate_fetch_slide_lewm_fork_samples import (
    collect_lewm_strike_context,
    require_vanilla_lewm,
)
from stochastic_physics import FetchSlideHiddenFriction


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contexts", type=int, default=64)
    parser.add_argument("--seed", type=int, default=703072)
    parser.add_argument("--raw-horizon", type=int, default=100)
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--frameskip", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--success-threshold", type=float, default=0.05)
    parser.add_argument("--success-temperature", type=float, default=0.01)
    parser.add_argument("--bootstrap-draws", type=int, default=20_000)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


@torch.no_grad()
def vanilla_lewm_candidate_index(
    model,
    context,
    candidates,
    action_mean,
    action_std,
    *,
    history_size,
    frameskip,
    image_size,
    device,
):
    count = len(candidates)
    model_horizon = candidates.shape[1] // frameskip
    context_pixels = preprocess_images(
        context["context_frames"][None],
        image_size,
    ).to(device)
    context_pixels = context_pixels.expand(count, *context_pixels.shape[1:])
    prefix = np.broadcast_to(
        context["past_actions"][None],
        (count, *context["past_actions"].shape),
    )
    raw_actions = np.concatenate([prefix, candidates], axis=1)
    action_chunks = pack_actions(
        raw_actions,
        action_mean,
        action_std,
        frameskip,
    )
    expected_chunks = history_size + model_horizon - 1
    if action_chunks.size(1) != expected_chunks:
        raise RuntimeError(
            f"Expected {expected_chunks} action chunks, got {action_chunks.size(1)}"
        )
    rollout = rollout_model(
        model,
        context_pixels,
        action_chunks,
        samples=1,
        history_size=history_size,
        horizon=model_horizon,
        flow_steps=1,
        device=device,
        seed=context["seed"],
    )[:, 0]
    goal_pixels = preprocess_images(
        context["goal_image"][None],
        image_size,
    ).to(device)[:, None]
    # rollout_model intentionally returns CPU tensors so large candidate
    # batches do not retain GPU memory between contexts.
    goal_embedding = model.encode({"pixels": goal_pixels})["emb"][:, -1].cpu()
    terminal_cost = (rollout[:, -1] - goal_embedding).square().sum(dim=-1)
    return int(terminal_cost.argmin()), terminal_cost.cpu().numpy()


def success_summary(values, *, draws, seed):
    values = np.asarray(values, dtype=bool)
    scene_rate = values.mean(axis=1)
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(scene_rate), size=(draws, len(scene_rate)))
    bootstrap = scene_rate[indices].mean(axis=1)
    return {
        "success_rate": float(values.mean()),
        "success_rate_low": float(values[:, 0].mean()),
        "success_rate_high": float(values[:, 1].mean()),
        "paired_both_success_rate": float(values.all(axis=1).mean()),
        "scene_cluster_bootstrap_ci95": [
            float(value) for value in np.quantile(bootstrap, [0.025, 0.975])
        ],
    }


def main():
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite immutable result: {args.output}")
    if args.raw_horizon <= 0 or args.raw_horizon % args.frameskip:
        raise SystemExit("--raw-horizon must be divisible by --frameskip")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    ).to(device).eval()
    require_vanilla_lewm(model)
    action_mean, action_std = action_statistics(args.dataset)
    modes = (0.2, 1.0)
    env = FetchSlideHiddenFriction(
        gym.make(
            "swm/FetchSlide-v3",
            max_episode_steps=args.raw_horizon + 80,
            render_mode="rgb_array",
        ),
        profile=fixed_profile(1.0),
        terminate_at_goal=False,
    )

    lewm_successes = []
    exact_mean_successes = []
    exact_distribution_successes = []
    records = []
    candidate_seed = int(args.seed)
    try:
        while len(records) < args.contexts:
            context = collect_lewm_strike_context(
                env,
                candidate_seed,
                history_size=args.history_size,
                frameskip=args.frameskip,
            )
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
                env,
                context["fork_state"],
                candidates,
                modes,
            )
            distances = terminal_distances(exact, goal)
            successes = distances <= args.success_threshold
            lewm_index, model_cost = vanilla_lewm_candidate_index(
                model,
                context,
                candidates,
                action_mean,
                action_std,
                history_size=args.history_size,
                frameskip=args.frameskip,
                image_size=args.image_size,
                device=device,
            )
            exact_mean_index = mean_state_index(exact, goal)
            exact_distribution_index = smooth_distribution_index(
                distances,
                args.success_threshold,
                args.success_temperature,
            )
            lewm_successes.append(successes[lewm_index])
            exact_mean_successes.append(successes[exact_mean_index])
            exact_distribution_successes.append(
                successes[exact_distribution_index]
            )
            records.append(
                {
                    "seed": context["seed"],
                    "indices": {
                        "vanilla_lewm": lewm_index,
                        "exact_mean_state": exact_mean_index,
                        "exact_distribution": exact_distribution_index,
                    },
                    "parameters": {
                        "vanilla_lewm": parameters[lewm_index],
                        "exact_mean_state": parameters[exact_mean_index],
                        "exact_distribution": parameters[
                            exact_distribution_index
                        ],
                    },
                    "vanilla_lewm_success": successes[lewm_index].tolist(),
                    "vanilla_lewm_terminal_distance_m": distances[
                        lewm_index
                    ].tolist(),
                    "vanilla_lewm_model_cost": float(model_cost[lewm_index]),
                }
            )
            print(
                f"vanilla LeWM commitment {len(records)}/{args.contexts} "
                f"seed={context['seed']}",
                flush=True,
            )
    finally:
        env.close()

    lewm_successes = np.asarray(lewm_successes, dtype=bool)
    exact_mean_successes = np.asarray(exact_mean_successes, dtype=bool)
    exact_distribution_successes = np.asarray(
        exact_distribution_successes,
        dtype=bool,
    )
    oracle = comparison_summary(
        "exact_distribution_vs_exact_mean_state",
        exact_mean_successes,
        exact_distribution_successes,
        draws=args.bootstrap_draws,
        seed=args.seed,
    )
    result = {
        "task": "fetch_slide_hidden_friction_vanilla_lewm_commitment",
        "model": "vanilla_lewm",
        "residual_enabled": False,
        "selection_cost": "terminal latent distance to native goal-image embedding",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256_file(args.dataset),
        "contexts": args.contexts,
        "episodes": 2 * args.contexts,
        "seed": args.seed,
        "friction_modes": list(modes),
        "raw_horizon": args.raw_horizon,
        "success_threshold_m": args.success_threshold,
        "vanilla_lewm": success_summary(
            lewm_successes,
            draws=args.bootstrap_draws,
            seed=args.seed,
        ),
        "exact_oracle": oracle,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "records"},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
