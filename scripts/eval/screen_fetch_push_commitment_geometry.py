#!/usr/bin/env python
"""Exact-fork geometry screen for the FetchPush commitment special case.

No learned model and no training are involved.  For each push-onset context we
roll a dense constant-direction push-pulse library through exact MuJoCo forks
under two hidden friction multipliers, then measure how much an oracle that
knows the *distribution* of outcomes beats an oracle that plans on the *mean*
trajectory.  The goal is to find the task geometry (friction gap, success
tolerance, candidate grid) that maximizes this ceiling and, in particular,
maximizes the number of "clean-win" contexts where a robust hedge action wins
both friction modes while the mean-optimal action fails both.

A clean-win context is the maximal demonstration of stochastic-over-deterministic
control: the distribution oracle succeeds on both episodes, the mean oracle
fails on both, and the gap is structural (the mean lands in the valley between
the two modes), not a matter of estimation noise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import stable_worldmodel  # noqa: F401 - registers swm environments

from fetch_push_expert import FetchPushExpertPolicy
from scripts.eval.evaluate_fetch_push_commitment import (
    collect_push_context,
    exact_candidate_futures,
    minimum_distances,
)
from scripts.eval.generate_fetch_push_fork_samples import fixed_profile
from stochastic_physics import FetchPushHiddenFriction


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contexts", type=int, default=24)
    parser.add_argument("--seed", type=int, default=903072)
    parser.add_argument("--raw-horizon", type=int, default=20)
    parser.add_argument("--speed-min", type=float, default=0.05)
    parser.add_argument("--speed-max", type=float, default=1.2)
    parser.add_argument("--speed-count", type=int, default=12)
    parser.add_argument("--duration-min", type=int, default=4)
    parser.add_argument("--duration-stride", type=int, default=2)
    parser.add_argument(
        "--friction-pairs",
        type=str,
        default="0.2,3.0;0.15,4.0;0.1,5.0;0.25,2.5",
        help="Semicolon-separated low,high friction multiplier pairs.",
    )
    parser.add_argument(
        "--tolerances",
        type=str,
        default="0.03,0.05,0.07",
        help="Comma-separated success tolerances (metres) to evaluate.",
    )
    return parser.parse_args()


def dense_push_candidates(state, *, raw_horizon, speed_min, speed_max, speed_count,
                          duration_min, duration_stride):
    """Constant-direction push pulses on a dense speed x duration grid."""
    if raw_horizon <= 0 or raw_horizon % 2:
        raise ValueError("raw_horizon must be a positive multiple of two")
    object_position = np.asarray(state[3:6], dtype=np.float32)
    goal = np.asarray(state[-3:], dtype=np.float32)
    direction = goal[:2] - object_position[:2]
    direction = direction / max(float(np.linalg.norm(direction)), 1e-8)
    speeds = np.linspace(speed_min, speed_max, speed_count, dtype=np.float32)
    durations = np.arange(duration_min, raw_horizon + 1, duration_stride, dtype=np.int64)
    candidates = np.zeros((len(speeds) * len(durations), raw_horizon, 4), dtype=np.float32)
    parameters = []
    index = 0
    for speed in speeds:
        for duration in durations:
            candidates[index, :duration, :2] = speed * direction
            parameters.append({"speed": float(speed), "duration": int(duration)})
            index += 1
    return candidates, parameters


def reduce_context(exact_futures, goal):
    """Reduce one context's exact rollouts to the two distance arrays we need.

    ``mode_min_dist`` (n_cand, 2) is each real mode trajectory's closest
    approach to goal -- used for success and hedge detection.  ``mean_traj_dist``
    (n_cand,) is the closest approach of the *averaged-position* trajectory --
    what a mean/deterministic planner optimizes.  The averaging happens in
    position space, so it creates the valley pathology when the two modes
    straddle the goal.
    """
    goal = np.asarray(goal)
    mode_min_dist = minimum_distances(exact_futures, goal)  # (n_cand, n_modes)
    mean_traj = exact_futures.mean(axis=1)  # (n_cand, horizon, 3)
    mean_traj_dist = np.linalg.norm(mean_traj - goal, axis=-1).min(axis=-1)  # (n_cand,)
    return mode_min_dist, mean_traj_dist


def evaluate_setting(mode_min_dist, mean_traj_dist, tolerance):
    """Score both oracles from the reduced per-context distance arrays.

    Mean oracle: pick the candidate whose *averaged-position* trajectory gets
    closest to goal (valley-prone, distribution-blind).  Distribution oracle:
    pick the candidate with the fewest failing modes (ties broken by mean-traj
    distance), i.e. it selects a hedge when one exists.
    """
    low = mode_min_dist[:, :, 0]
    high = mode_min_dist[:, :, 1]

    # Mean oracle: minimizes distance of the averaged-position trajectory.
    det_idx = mean_traj_dist.argmin(axis=1)

    # Distribution oracle: minimize expected 0/1 failure, tie-break on mean-traj.
    failure = (mode_min_dist > tolerance).sum(axis=2)  # 0, 1, or 2 failing modes
    score = failure + 1e-6 * mean_traj_dist
    sto_idx = score.argmin(axis=1)

    ctx = np.arange(mode_min_dist.shape[0])
    det_low_ok = low[ctx, det_idx] <= tolerance
    det_high_ok = high[ctx, det_idx] <= tolerance
    sto_low_ok = low[ctx, sto_idx] <= tolerance
    sto_high_ok = high[ctx, sto_idx] <= tolerance

    # Episode-level success (each context = one low + one high episode).
    det_success = np.stack([det_low_ok, det_high_ok], axis=1)
    sto_success = np.stack([sto_low_ok, sto_high_ok], axis=1)

    hedge_exists = ((low <= tolerance) & (high <= tolerance)).any(axis=1)
    det_fails_both = (~det_low_ok) & (~det_high_ok)
    clean_win = hedge_exists & det_fails_both

    return {
        "det_success": det_success,
        "sto_success": sto_success,
        "hedge_exists": hedge_exists,
        "clean_win": clean_win,
    }


def paired_bootstrap(det_scene, sto_scene, draws, seed):
    """Bootstrap the paired per-scene success difference (any scalar per scene)."""
    scene_delta = np.asarray(sto_scene, dtype=float) - np.asarray(det_scene, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(scene_delta), size=(draws, len(scene_delta)))
    samples = scene_delta[idx].mean(axis=1)
    return float(scene_delta.mean()), [float(q) for q in np.quantile(samples, [0.025, 0.975])]


def main():
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite screen result: {args.output}")

    friction_pairs = []
    for chunk in args.friction_pairs.split(";"):
        low, high = (float(x) for x in chunk.split(","))
        friction_pairs.append((low, high))
    tolerances = [float(x) for x in args.tolerances.split(",")]

    env = FetchPushHiddenFriction(
        gym.make("swm/FetchPush-v3", max_episode_steps=100, render_mode="rgb_array"),
        profile=fixed_profile(1.0),
        terminate_at_goal=False,
    )
    policy = FetchPushExpertPolicy(seed=args.seed)

    # Collect a fixed pool of push-onset contexts once; reuse across settings.
    contexts = []
    candidate_seed = int(args.seed)
    while len(contexts) < args.contexts:
        context = collect_push_context(env, policy, candidate_seed)
        candidate_seed += 1
        if context is not None:
            contexts.append(context)
        print(f"collected context {len(contexts)}/{args.contexts}", flush=True)

    settings = []
    for (low, high) in friction_pairs:
        # Roll every candidate under both modes once for this friction pair.
        per_ctx_mode_dist = []
        per_ctx_mean_dist = []
        for ci, context in enumerate(contexts):
            candidates, _ = dense_push_candidates(
                context["state"],
                raw_horizon=args.raw_horizon,
                speed_min=args.speed_min,
                speed_max=args.speed_max,
                speed_count=args.speed_count,
                duration_min=args.duration_min,
                duration_stride=args.duration_stride,
            )
            exact = exact_candidate_futures(
                env, context["fork_state"], candidates, modes=(low, high)
            )
            goal = context["state"][-3:]
            mode_min_dist, mean_traj_dist = reduce_context(exact, goal)
            per_ctx_mode_dist.append(mode_min_dist)
            per_ctx_mean_dist.append(mean_traj_dist)
            print(f"  friction {low}/{high}: context {ci + 1}/{len(contexts)}", flush=True)
        mode_min_dist = np.stack(per_ctx_mode_dist, axis=0)  # (n_ctx, n_cand, 2)
        mean_traj_dist = np.stack(per_ctx_mean_dist, axis=0)  # (n_ctx, n_cand)

        for tolerance in tolerances:
            ev = evaluate_setting(mode_min_dist, mean_traj_dist, tolerance)
            # Episode-level: each mode outcome graded independently.
            improvement, ci95 = paired_bootstrap(
                ev["det_success"].mean(axis=1), ev["sto_success"].mean(axis=1),
                draws=20000, seed=args.seed,
            )
            # Context-level: one committed action must work under BOTH frictions.
            det_scene = ev["det_success"].all(axis=1)
            sto_scene = ev["sto_success"].all(axis=1)
            ctx_improvement, ctx_ci95 = paired_bootstrap(
                det_scene, sto_scene, draws=20000, seed=args.seed + 7
            )
            n_clean = int(ev["clean_win"].sum())
            settings.append({
                "friction_low": low,
                "friction_high": high,
                "tolerance": tolerance,
                "oracle_mean_success_rate": float(ev["det_success"].mean()),
                "oracle_distribution_success_rate": float(ev["sto_success"].mean()),
                "absolute_success_improvement": improvement,
                "scene_cluster_bootstrap_ci95": ci95,
                "context_oracle_mean_success_rate": float(det_scene.mean()),
                "context_oracle_distribution_success_rate": float(sto_scene.mean()),
                "context_absolute_success_improvement": ctx_improvement,
                "context_scene_cluster_bootstrap_ci95": ctx_ci95,
                "hedge_exists_fraction": float(ev["hedge_exists"].mean()),
                "clean_win_contexts": n_clean,
                "clean_win_fraction": float(ev["clean_win"].mean()),
                "clean_win_context_indices": [int(i) for i in np.nonzero(ev["clean_win"])[0]],
                "strict_gate": bool(ci95[0] > 0),
            })
            print(
                f"friction {low}/{high} tol {tolerance}: "
                f"episode dist {ev['sto_success'].mean():.3f} vs mean {ev['det_success'].mean():.3f} "
                f"(+{improvement:.3f}) | context dist {sto_scene.mean():.3f} vs mean "
                f"{det_scene.mean():.3f} (+{ctx_improvement:.3f}, CI {ctx_ci95})",
                flush=True,
            )
    env.close()

    settings.sort(key=lambda s: s["absolute_success_improvement"], reverse=True)
    result = {
        "task": "fetch_push_commitment_geometry_screen",
        "exact_forks_only": True,
        "contexts": len(contexts),
        "context_seeds": [c["seed"] for c in contexts],
        "candidate_grid": {
            "raw_horizon": args.raw_horizon,
            "speed_min": args.speed_min,
            "speed_max": args.speed_max,
            "speed_count": args.speed_count,
            "duration_min": args.duration_min,
            "duration_stride": args.duration_stride,
        },
        "friction_pairs": friction_pairs,
        "tolerances": tolerances,
        "settings_sorted_by_improvement": settings,
        "best_setting": settings[0] if settings else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("\nBEST:", json.dumps(result["best_setting"], indent=2))


if __name__ == "__main__":
    main()
