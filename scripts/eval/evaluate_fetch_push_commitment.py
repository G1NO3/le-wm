#!/usr/bin/env python
"""Evaluate an uncertainty-sensitive open-loop FetchPush commitment task.

The ordinary-observation expert performs only the pre-contact approach.  At
push onset, a controller must choose one finite push pulse before the hidden
low/high friction outcome unfolds.  Deterministic and stochastic selectors
use the same learned residual model and candidate library; the sole difference
is scoring the residual-mode mean versus both persistent outcome modes.

Exact simulator forks also provide an oracle decision-value screen.  A task is
not suitable for a stochastic-control claim unless even the exact two-mode
distribution beats its exact deterministic mean on episode success.
"""

from __future__ import annotations

import argparse
import copy
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

from fetch_push_expert import FetchPushExpertPolicy
from scripts.eval.generate_fetch_push_fork_samples import fixed_profile, set_mode
from stochastic_physics import FetchPushHiddenFriction


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contexts", type=int, default=64)
    parser.add_argument("--seed", type=int, default=903072)
    parser.add_argument("--raw-horizon", type=int, default=20)
    parser.add_argument("--speed-min", type=float, default=0.1)
    parser.add_argument("--speed-max", type=float, default=1.0)
    parser.add_argument("--speed-count", type=int, default=10)
    parser.add_argument("--duration-min", type=int, default=2)
    parser.add_argument("--duration-stride", type=int, default=2)
    parser.add_argument("--success-threshold", type=float, default=0.05)
    parser.add_argument("--success-temperature", type=float, default=0.01)
    parser.add_argument("--bootstrap-draws", type=int, default=20000)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def collect_push_context(env, policy, seed, max_steps=50):
    state, _ = env.reset(seed=seed)
    phase = 0
    for _ in range(max_steps):
        action, phase = policy._action(state, phase)
        state, _, _, _, _ = env.step(action)
        if phase == 3:
            return {
                "seed": int(seed),
                "state": np.asarray(state, dtype=np.float32).copy(),
                "fork_state": env.get_fork_state(),
            }
    return None


def push_pulse_candidates(
    state,
    *,
    raw_horizon,
    speed_min,
    speed_max,
    speed_count,
    duration_min,
    duration_stride,
):
    """Construct observation-only constant-direction push pulses."""

    if raw_horizon <= 0 or raw_horizon % 2:
        raise ValueError("raw_horizon must be a positive multiple of two")
    object_position = np.asarray(state[3:6], dtype=np.float32)
    goal = np.asarray(state[-3:], dtype=np.float32)
    direction = goal[:2] - object_position[:2]
    norm = float(np.linalg.norm(direction))
    direction = direction / max(norm, 1e-8)
    speeds = np.linspace(speed_min, speed_max, speed_count, dtype=np.float32)
    durations = np.arange(
        duration_min, raw_horizon + 1, duration_stride, dtype=np.int64
    )
    if len(durations) == 0:
        raise ValueError("Candidate duration grid is empty")

    candidates = np.zeros(
        (len(speeds) * len(durations), raw_horizon, 4), dtype=np.float32
    )
    parameters = []
    index = 0
    for speed in speeds:
        for duration in durations:
            candidates[index, :duration, :2] = speed * direction
            parameters.append({"speed": float(speed), "duration": int(duration)})
            index += 1
    return candidates, parameters


def exact_candidate_futures(env, context, candidates, modes=(0.2, 3.0)):
    futures = np.empty(
        (len(candidates), len(modes), candidates.shape[1], 3), dtype=np.float32
    )
    for candidate_index, actions in enumerate(candidates):
        for mode_index, multiplier in enumerate(modes):
            env.set_fork_state(set_mode(context, multiplier))
            for step, action in enumerate(actions):
                env.step(action)
                futures[candidate_index, mode_index, step] = env._object_position()
    return futures


@torch.no_grad()
def model_candidate_futures(model, initial_state, candidates, device):
    """Roll the learned residual mean and two persistent residual modes."""

    state = torch.as_tensor(initial_state, device=device).float()
    raw = torch.as_tensor(candidates, device=device).float()
    action_blocks = raw.reshape(raw.size(0), raw.size(1) // 2, 8)
    count, horizon = action_blocks.shape[:2]

    mean_state = state[None].expand(count, -1).clone()
    mode_state = state[None, None].expand(count, 2, -1).clone()
    mode = torch.arange(2, device=device)[None].expand(count, -1)
    mean_positions = []
    mode_positions = []
    for step in range(horizon):
        block = action_blocks[:, step]
        mean_state = model.transition_mode_mixture(mean_state, block, mode=None)
        expanded_block = block[:, None].expand(-1, 2, -1)
        mode_state = model.transition_mode_mixture(
            mode_state, expanded_block, mode=mode
        )
        mean_positions.append(mean_state[:, 3:6])
        mode_positions.append(mode_state[:, :, 3:6])
    return (
        torch.stack(mean_positions, dim=1).cpu().numpy(),
        torch.stack(mode_positions, dim=2).cpu().numpy(),
    )


def minimum_distances(futures, goal):
    return np.linalg.norm(futures - np.asarray(goal), axis=-1).min(axis=-1)


def deterministic_index(mean_futures, goal):
    return int(np.argmin(minimum_distances(mean_futures, goal)))


def stochastic_index(mode_futures, goal, threshold, temperature):
    distances = minimum_distances(mode_futures, goal)
    failure = 1.0 / (1.0 + np.exp(-(distances - threshold) / temperature))
    expected_failure = failure.mean(axis=-1)
    # A tiny distance tie-break makes selection reproducible without changing
    # any non-tied predicted success probability.
    return int(np.argmin(expected_failure + 1e-7 * distances.mean(axis=-1)))


def selected_success(exact_futures, index, goal, threshold):
    distances = minimum_distances(exact_futures[index], goal)
    return distances <= threshold, distances


def paired_interval(deterministic, stochastic, draws, seed):
    deterministic = np.asarray(deterministic, dtype=float)
    stochastic = np.asarray(stochastic, dtype=float)
    scene_delta = stochastic.mean(axis=1) - deterministic.mean(axis=1)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(scene_delta), size=(draws, len(scene_delta)))
    samples = scene_delta[indices].mean(axis=1)
    return float(scene_delta.mean()), np.quantile(samples, [0.025, 0.975])


def summarize_pair(name, deterministic, stochastic, *, draws, seed):
    deterministic = np.asarray(deterministic, dtype=bool)
    stochastic = np.asarray(stochastic, dtype=bool)
    improvement, interval = paired_interval(
        deterministic, stochastic, draws=draws, seed=seed
    )
    return {
        "name": name,
        "deterministic_success_rate": float(deterministic.mean()),
        "stochastic_success_rate": float(stochastic.mean()),
        "absolute_success_improvement": improvement,
        "scene_cluster_bootstrap_ci95": [float(value) for value in interval],
        "stochastic_only_successes": int(np.sum(stochastic & ~deterministic)),
        "deterministic_only_successes": int(np.sum(deterministic & ~stochastic)),
        "strict_improvement_gate": bool(interval[0] > 0),
    }


def main():
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite commitment result: {args.output}")
    if args.duration_stride <= 0:
        raise SystemExit("--duration-stride must be positive")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = torch.load(
        args.checkpoint, map_location=device, weights_only=False
    ).to(device).eval()
    env = FetchPushHiddenFriction(
        gym.make(
            "swm/FetchPush-v3",
            max_episode_steps=100,
            render_mode="rgb_array",
        ),
        profile=fixed_profile(1.0),
        terminate_at_goal=False,
    )
    policy = FetchPushExpertPolicy(seed=args.seed)

    records = []
    candidate_seed = int(args.seed)
    while len(records) < args.contexts:
        context = collect_push_context(env, policy, candidate_seed)
        candidate_seed += 1
        if context is None:
            continue
        candidates, parameters = push_pulse_candidates(
            context["state"],
            raw_horizon=args.raw_horizon,
            speed_min=args.speed_min,
            speed_max=args.speed_max,
            speed_count=args.speed_count,
            duration_min=args.duration_min,
            duration_stride=args.duration_stride,
        )
        exact = exact_candidate_futures(env, context["fork_state"], candidates)
        goal = context["state"][-3:]
        exact_mean = exact.mean(axis=1)
        oracle_det_index = deterministic_index(exact_mean, goal)
        oracle_stoch_index = stochastic_index(
            exact, goal, args.success_threshold, args.success_temperature
        )

        model_mean, model_modes = model_candidate_futures(
            model, context["state"], candidates, device
        )
        model_det_index = deterministic_index(model_mean, goal)
        model_stoch_index = stochastic_index(
            model_modes, goal, args.success_threshold, args.success_temperature
        )

        record = {
            "seed": context["seed"],
            "oracle_deterministic_index": oracle_det_index,
            "oracle_stochastic_index": oracle_stoch_index,
            "model_deterministic_index": model_det_index,
            "model_stochastic_index": model_stoch_index,
            "parameters": {
                "oracle_deterministic": parameters[oracle_det_index],
                "oracle_stochastic": parameters[oracle_stoch_index],
                "model_deterministic": parameters[model_det_index],
                "model_stochastic": parameters[model_stoch_index],
            },
        }
        for prefix, index in (
            ("oracle_deterministic", oracle_det_index),
            ("oracle_stochastic", oracle_stoch_index),
            ("model_deterministic", model_det_index),
            ("model_stochastic", model_stoch_index),
        ):
            success, distance = selected_success(
                exact, index, goal, args.success_threshold
            )
            record[f"{prefix}_success"] = success.tolist()
            record[f"{prefix}_minimum_distance"] = distance.tolist()
        records.append(record)
        print(f"context {len(records)}/{args.contexts} seed={context['seed']}", flush=True)
    env.close()

    def outcomes(key):
        return np.asarray([record[key] for record in records], dtype=bool)

    oracle = summarize_pair(
        "exact distribution versus exact mean",
        outcomes("oracle_deterministic_success"),
        outcomes("oracle_stochastic_success"),
        draws=args.bootstrap_draws,
        seed=args.seed,
    )
    learned = summarize_pair(
        "learned persistent modes versus learned residual mean",
        outcomes("model_deterministic_success"),
        outcomes("model_stochastic_success"),
        draws=args.bootstrap_draws,
        seed=args.seed + 1,
    )
    result = {
        "task": "fetch_push_hidden_friction_commitment",
        "checkpoint": str(args.checkpoint.resolve()),
        "contexts": args.contexts,
        "episodes": 2 * args.contexts,
        "context_seeds": [record["seed"] for record in records],
        "friction_modes": [0.2, 3.0],
        "candidate_grid": {
            "raw_horizon": args.raw_horizon,
            "speed_min": args.speed_min,
            "speed_max": args.speed_max,
            "speed_count": args.speed_count,
            "duration_min": args.duration_min,
            "duration_stride": args.duration_stride,
            "candidate_count": len(parameters),
        },
        "success_threshold": args.success_threshold,
        "success_temperature": args.success_temperature,
        "oracle_decision_value": oracle,
        "learned_decision_value": learned,
        "task_gate": bool(
            oracle["strict_improvement_gate"]
            and learned["absolute_success_improvement"] > 0
        ),
        "one_validation_screen_only": True,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
