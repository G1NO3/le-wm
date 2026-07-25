#!/usr/bin/env python
"""Calibrate FetchPush friction modes before collecting a training dataset."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import gymnasium as gym
import numpy as np
import stable_worldmodel  # noqa: F401 - registers swm environments

from fetch_push_expert import FetchPushExpertPolicy
from stochastic_physics import (
    FETCH_PUSH_PROFILES,
    FetchPushHiddenFriction,
    PhysicsProfile,
)


def fixed_profile(multiplier):
    return PhysicsProfile(
        f"fixed_{multiplier:g}",
        (1.0, 1.0),
        (float(multiplier), float(multiplier)),
        (1.0, 1.0),
        0.0,
    )


def make_env(profile, *, terminate_at_goal):
    base = gym.make("swm/FetchPush-v3", max_episode_steps=100)
    return FetchPushHiddenFriction(
        base,
        profile=profile,
        terminate_at_goal=terminate_at_goal,
    )


def run_task_episodes(multiplier, episodes, seed):
    env = make_env(fixed_profile(multiplier), terminate_at_goal=True)
    policy = FetchPushExpertPolicy(seed=seed)
    successes = []
    steps = []
    final_distances = []
    for episode in range(episodes):
        state, info = env.reset(seed=seed + episode)
        phase = 0
        for step in range(100):
            action, phase = policy._action(state, phase)
            state, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        successes.append(bool(info["success"]))
        steps.append(step + 1)
        final_distances.append(float(info["metrics/final_goal_distance"]))
    env.close()
    return {
        "episodes": episodes,
        "success_rate": float(np.mean(successes)),
        "mean_steps": float(np.mean(steps)),
        "mean_final_goal_distance_m": float(np.mean(final_distances)),
    }


def rollout_mode(env, context, actions, multiplier):
    state = copy.deepcopy(context)
    state["mode"]["surface_friction_multiplier"] = float(multiplier)
    env.set_fork_state(state)
    trajectory = []
    for action in actions:
        env.step(action)
        trajectory.append(env._object_position())
    return np.asarray(trajectory)


def collect_fork_separation(low, high, contexts, horizon, seed):
    env = make_env(fixed_profile(1.0), terminate_at_goal=False)
    policy = FetchPushExpertPolicy(seed=seed)
    pushed = []
    passive = []
    kept_seeds = []
    for episode in range(contexts):
        state, _ = env.reset(seed=seed + episode)
        phase = 0
        # Stop at the first state whose next controller command is a push.
        for _ in range(40):
            action, phase = policy._action(state, phase)
            state, _, _, _, _ = env.step(action)
            if phase == 3:
                break
        if phase != 3:
            continue
        context = env.get_fork_state()

        # Generate one nominal, closed-loop sequence, then hold that exact
        # action sequence fixed for both simulator modes.
        actions = []
        nominal_state = state
        nominal_phase = phase
        for _ in range(horizon):
            action, nominal_phase = policy._action(nominal_state, nominal_phase)
            actions.append(action.copy())
            nominal_state, _, _, _, _ = env.step(action)

        low_future = rollout_mode(env, context, actions, low)
        high_future = rollout_mode(env, context, actions, high)
        pushed.append(np.stack([low_future, high_future]))
        zeros = np.zeros((horizon, env.action_space.shape[0]), dtype=np.float32)
        passive.append(
            np.stack(
                [
                    rollout_mode(env, context, zeros, low),
                    rollout_mode(env, context, zeros, high),
                ]
            )
        )
        kept_seeds.append(seed + episode)
    env.close()

    pushed = np.asarray(pushed)
    passive = np.asarray(passive)
    pushed_sep = np.linalg.norm(pushed[:, 0] - pushed[:, 1], axis=-1)
    passive_sep = np.linalg.norm(passive[:, 0] - passive[:, 1], axis=-1)
    horizons = sorted(set([1, min(5, horizon), min(10, horizon), horizon]))
    by_horizon = {
        str(step): {
            "mean_separation_m": float(pushed_sep[:, step - 1].mean()),
            "median_separation_m": float(np.median(pushed_sep[:, step - 1])),
            "mean_passive_separation_m": float(passive_sep[:, step - 1].mean()),
        }
        for step in horizons
    }
    final_push = pushed_sep[:, -1]
    final_passive = passive_sep[:, -1]
    return {
        "contexts": len(pushed),
        "context_seeds": kept_seeds,
        "horizon": horizon,
        "by_horizon": by_horizon,
        "median_final_separation_m": float(np.median(final_push)),
        "median_passive_final_separation_m": float(np.median(final_passive)),
        "median_action_dependence_ratio": float(
            np.median((final_push + 1e-9) / (final_passive + 1e-9))
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=FETCH_PUSH_PROFILES, default="push_strong")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--contexts", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    profile = FETCH_PUSH_PROFILES[args.profile]
    low, high = profile.surface_friction
    task = {
        "low": run_task_episodes(low, args.episodes, args.seed),
        "high": run_task_episodes(high, args.episodes, args.seed),
    }
    mixture_success = 0.5 * (
        task["low"]["success_rate"] + task["high"]["success_rate"]
    )
    forks = collect_fork_separation(
        low, high, args.contexts, args.horizon, args.seed + 100_000
    )
    final_sep = forks["median_final_separation_m"]
    passive_sep = forks["median_passive_final_separation_m"]
    result = {
        "profile": profile.__dict__,
        "seed": args.seed,
        "task": task,
        "balanced_mixture_success_rate": mixture_success,
        "forks": forks,
        "gates": {
            "nontrivial_task": 0.4 <= mixture_success <= 0.8,
            "persistent_friction_separation": final_sep >= 0.02,
            "action_dependent": final_sep >= 5 * max(passive_sep, 1e-6),
        },
    }
    result["gates"]["pass"] = all(result["gates"].values())
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        if args.output.exists():
            raise SystemExit(f"Refusing to overwrite immutable probe: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
