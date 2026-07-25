#!/usr/bin/env python
"""Screen FetchSlide for decision value under hidden persistent friction.

At a pre-contact strike state, every candidate impulse is replayed from the
same exact simulator fork under low and high puck/table friction.  The screen
compares an action selected from the mean terminal state with actions selected
from the full two-outcome distribution.  No learned model is involved: this is
the cheap oracle gate that must pass before data collection or training.
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

from fetch_push_expert import FetchPushExpertPolicy
from stochastic_physics import (
    FETCH_SLIDE_PROFILES,
    FetchSlideHiddenFriction,
    PhysicsProfile,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", choices=FETCH_SLIDE_PROFILES, default="slide_strong"
    )
    parser.add_argument("--contexts", type=int, default=64)
    parser.add_argument("--seed", type=int, default=403072)
    parser.add_argument(
        "--horizon",
        type=int,
        default=100,
        help="Post-strike raw steps; 100 leaves both calibrated modes settled.",
    )
    parser.add_argument("--speed-min", type=float, default=0.4)
    parser.add_argument("--speed-max", type=float, default=1.0)
    parser.add_argument("--speed-count", type=int, default=13)
    parser.add_argument("--duration-min", type=int, default=1)
    parser.add_argument("--duration-max", type=int, default=4)
    parser.add_argument("--angle-max-deg", type=float, default=0.0)
    parser.add_argument("--angle-count", type=int, default=1)
    parser.add_argument("--success-threshold", type=float, default=0.05)
    parser.add_argument("--success-temperature", type=float, default=0.01)
    parser.add_argument("--bootstrap-draws", type=int, default=20_000)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def fixed_profile(multiplier):
    return PhysicsProfile(
        f"fixed_{multiplier:g}",
        (1.0, 1.0),
        (float(multiplier), float(multiplier)),
        (1.0, 1.0),
        0.0,
    )


def collect_strike_context(env, seed, max_steps=50):
    state, _ = env.reset(seed=seed)
    policy = FetchPushExpertPolicy(
        seed=seed,
        behind_offset=0.075,
        clearance=0.08,
    )
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


def strike_candidates(
    state,
    *,
    horizon,
    speed_min,
    speed_max,
    speed_count,
    duration_min,
    duration_max,
    angle_max_deg,
    angle_count,
):
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if speed_count <= 0 or angle_count <= 0:
        raise ValueError("speed-count and angle-count must be positive")
    if not 1 <= duration_min <= duration_max <= horizon:
        raise ValueError("durations must satisfy 1 <= min <= max <= horizon")

    object_position = np.asarray(state[3:6], dtype=np.float32)
    goal = np.asarray(state[-3:], dtype=np.float32)
    direction = goal[:2] - object_position[:2]
    direction /= max(float(np.linalg.norm(direction)), 1e-8)
    speeds = np.linspace(speed_min, speed_max, speed_count, dtype=np.float32)
    durations = np.arange(duration_min, duration_max + 1, dtype=np.int64)
    angles = np.deg2rad(
        np.linspace(-angle_max_deg, angle_max_deg, angle_count, dtype=np.float32)
    )

    candidates = np.zeros(
        (len(speeds) * len(durations) * len(angles), horizon, 4),
        dtype=np.float32,
    )
    parameters = []
    candidate_index = 0
    for angle in angles:
        cosine, sine = float(np.cos(angle)), float(np.sin(angle))
        rotated = np.array(
            [
                cosine * direction[0] - sine * direction[1],
                sine * direction[0] + cosine * direction[1],
            ],
            dtype=np.float32,
        )
        for speed in speeds:
            for duration in durations:
                candidates[candidate_index, :duration, :2] = speed * rotated
                parameters.append(
                    {
                        "angle_deg": float(np.rad2deg(angle)),
                        "speed": float(speed),
                        "duration": int(duration),
                    }
                )
                candidate_index += 1
    return candidates, parameters


def exact_terminal_positions(env, fork_state, candidates, modes):
    terminal = np.empty((len(candidates), len(modes), 3), dtype=np.float32)
    for candidate_index, actions in enumerate(candidates):
        for mode_index, multiplier in enumerate(modes):
            state = copy.deepcopy(fork_state)
            state["mode"]["surface_friction_multiplier"] = float(multiplier)
            env.set_fork_state(state)
            for action in actions:
                env.step(action)
            terminal[candidate_index, mode_index] = env._object_position()
    return terminal


def terminal_distances(positions, goal):
    return np.linalg.norm(positions - np.asarray(goal), axis=-1)


def mean_state_index(positions, goal):
    mean_positions = positions.mean(axis=1)
    return int(np.argmin(terminal_distances(mean_positions, goal)))


def smooth_distribution_index(distances, threshold, temperature):
    logits = np.clip((distances - threshold) / temperature, -60.0, 60.0)
    failure = 1.0 / (1.0 + np.exp(-logits))
    score = failure.mean(axis=1) + 1e-7 * distances.mean(axis=1)
    return int(np.argmin(score))


def discrete_distribution_index(distances, threshold):
    failure = (distances > threshold).mean(axis=1)
    score = failure + 1e-7 * distances.mean(axis=1)
    return int(np.argmin(score))


def paired_interval(deterministic, stochastic, draws, seed):
    deterministic = np.asarray(deterministic, dtype=float)
    stochastic = np.asarray(stochastic, dtype=float)
    scene_delta = stochastic.mean(axis=1) - deterministic.mean(axis=1)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(scene_delta), size=(draws, len(scene_delta)))
    bootstrap = scene_delta[indices].mean(axis=1)
    return float(scene_delta.mean()), [
        float(value) for value in np.quantile(bootstrap, [0.025, 0.975])
    ]


def comparison_summary(name, baseline, candidate, *, draws, seed):
    baseline = np.asarray(baseline, dtype=bool)
    candidate = np.asarray(candidate, dtype=bool)
    improvement, interval = paired_interval(
        baseline, candidate, draws=draws, seed=seed
    )
    return {
        "name": name,
        "mean_state_success_rate": float(baseline.mean()),
        "distribution_success_rate": float(candidate.mean()),
        "absolute_success_improvement": improvement,
        "scene_cluster_bootstrap_ci95": interval,
        "distribution_only_successes": int(np.sum(candidate & ~baseline)),
        "mean_state_only_successes": int(np.sum(baseline & ~candidate)),
        "strict_improvement_gate": bool(interval[0] > 0),
    }


def main():
    args = parse_args()
    if args.contexts <= 0:
        raise SystemExit("--contexts must be positive")
    if args.output is not None and args.output.exists():
        raise SystemExit(f"Refusing to overwrite immutable probe: {args.output}")

    profile = FETCH_SLIDE_PROFILES[args.profile]
    modes = tuple(float(value) for value in profile.surface_friction)
    env = FetchSlideHiddenFriction(
        gym.make(
            "swm/FetchSlide-v3",
            max_episode_steps=max(200, args.horizon + 60),
            render_mode="rgb_array",
        ),
        profile=fixed_profile(1.0),
        terminate_at_goal=False,
    )

    records = []
    mean_successes = []
    smooth_successes = []
    discrete_successes = []
    mode_reachable = []
    shared_reachable = []
    preferred_action_disagreement = []
    terminal_mode_separation = []
    attempts = 0
    candidate_seed = int(args.seed)
    try:
        while len(records) < args.contexts and attempts < 3 * args.contexts:
            context = collect_strike_context(env, candidate_seed)
            candidate_seed += 1
            attempts += 1
            if context is None:
                continue
            candidates, parameters = strike_candidates(
                context["state"],
                horizon=args.horizon,
                speed_min=args.speed_min,
                speed_max=args.speed_max,
                speed_count=args.speed_count,
                duration_min=args.duration_min,
                duration_max=args.duration_max,
                angle_max_deg=args.angle_max_deg,
                angle_count=args.angle_count,
            )
            terminal = exact_terminal_positions(
                env, context["fork_state"], candidates, modes
            )
            goal = context["state"][-3:]
            distances = terminal_distances(terminal, goal)
            mean_index = mean_state_index(terminal, goal)
            smooth_index = smooth_distribution_index(
                distances, args.success_threshold, args.success_temperature
            )
            discrete_index = discrete_distribution_index(
                distances, args.success_threshold
            )
            successes = distances <= args.success_threshold
            mean_selected = successes[mean_index]
            smooth_selected = successes[smooth_index]
            discrete_selected = successes[discrete_index]

            mean_successes.append(mean_selected)
            smooth_successes.append(smooth_selected)
            discrete_successes.append(discrete_selected)
            mode_reachable.append(successes.any(axis=0))
            shared_reachable.append(bool(successes.all(axis=1).any()))
            preferred = np.argmin(distances, axis=0)
            preferred_action_disagreement.append(bool(preferred[0] != preferred[1]))
            terminal_mode_separation.append(
                float(np.linalg.norm(terminal[mean_index, 0] - terminal[mean_index, 1]))
            )
            records.append(
                {
                    "seed": context["seed"],
                    "mean_state_index": mean_index,
                    "smooth_distribution_index": smooth_index,
                    "discrete_distribution_index": discrete_index,
                    "parameters": {
                        "mean_state": parameters[mean_index],
                        "smooth_distribution": parameters[smooth_index],
                        "discrete_distribution": parameters[discrete_index],
                    },
                    "mean_state_success": mean_selected.tolist(),
                    "smooth_distribution_success": smooth_selected.tolist(),
                    "discrete_distribution_success": discrete_selected.tolist(),
                    "mean_state_terminal_distance_m": distances[mean_index].tolist(),
                    "smooth_distribution_terminal_distance_m": distances[
                        smooth_index
                    ].tolist(),
                    "discrete_distribution_terminal_distance_m": distances[
                        discrete_index
                    ].tolist(),
                    "mode_reachable": successes.any(axis=0).tolist(),
                    "shared_candidate_reachable": bool(
                        successes.all(axis=1).any()
                    ),
                }
            )
    finally:
        env.close()

    if len(records) != args.contexts:
        raise RuntimeError(
            f"Collected only {len(records)} strike contexts after {attempts} attempts"
        )

    mean_successes = np.asarray(mean_successes, dtype=bool)
    smooth_successes = np.asarray(smooth_successes, dtype=bool)
    discrete_successes = np.asarray(discrete_successes, dtype=bool)
    mode_reachable = np.asarray(mode_reachable, dtype=bool)
    smooth_comparison = comparison_summary(
        "smooth_distribution_vs_mean_state",
        mean_successes,
        smooth_successes,
        draws=args.bootstrap_draws,
        seed=args.seed,
    )
    discrete_comparison = comparison_summary(
        "discrete_distribution_vs_mean_state",
        mean_successes,
        discrete_successes,
        draws=args.bootstrap_draws,
        seed=args.seed + 1,
    )
    result = {
        "task": "fetch_slide_hidden_friction_commitment_oracle_screen",
        "profile": profile.__dict__,
        "seed": args.seed,
        "contexts": len(records),
        "attempts": attempts,
        "friction_modes": list(modes),
        "success_definition": {
            "metric": "terminal puck-to-goal Euclidean distance",
            "threshold_m": args.success_threshold,
            "horizon_raw_steps": args.horizon,
        },
        "candidate_grid": {
            "count": len(candidates),
            "speed": [args.speed_min, args.speed_max, args.speed_count],
            "duration": [args.duration_min, args.duration_max],
            "angle": [-args.angle_max_deg, args.angle_max_deg, args.angle_count],
        },
        "mean_state_success_rate_by_mode": mean_successes.mean(axis=0).tolist(),
        "smooth_distribution_success_rate_by_mode": smooth_successes.mean(
            axis=0
        ).tolist(),
        "discrete_distribution_success_rate_by_mode": discrete_successes.mean(
            axis=0
        ).tolist(),
        "per_mode_candidate_reachability": mode_reachable.mean(axis=0).tolist(),
        "shared_candidate_reachability": float(np.mean(shared_reachable)),
        "preferred_action_disagreement_rate": float(
            np.mean(preferred_action_disagreement)
        ),
        "median_terminal_mode_separation_m": float(
            np.median(terminal_mode_separation)
        ),
        "comparisons": {
            "smooth": smooth_comparison,
            "discrete": discrete_comparison,
        },
        "gates": {
            "context_yield": bool(len(records) / attempts >= 0.8),
            "both_modes_controllable": bool(np.all(mode_reachable.mean(axis=0) >= 0.5)),
            "different_action_rankings": bool(
                np.mean(preferred_action_disagreement) >= 0.25
            ),
            "action_dependent_mode_separation": bool(
                np.median(terminal_mode_separation) >= args.success_threshold
            ),
            "smooth_distribution_value": smooth_comparison[
                "strict_improvement_gate"
            ],
            "discrete_distribution_value": discrete_comparison[
                "strict_improvement_gate"
            ],
        },
        "records": records,
    }
    result["gates"]["pass"] = all(result["gates"].values())
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
