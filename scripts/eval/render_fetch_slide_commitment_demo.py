#!/usr/bin/env python
"""Render a qualitative FetchSlide commitment comparison from exact forks.

FetchSlide is the transfer test of the same hidden-friction commitment scheme:
one strike must be committed before the hidden surface friction is revealed, and
the block then slides ballistically to rest. On this task the exact mean-state
oracle is essentially always wrong (it strikes for the average of two very
different slide distances), while the exact distribution oracle can pick a
strike that lands within tolerance under both frictions.

This script searches for such a scene, replays the mean-oracle and
distribution-oracle strikes through exact simulator forks under both frictions,
and composes the same 2x2 panel layout as the FetchPush demo. No learned model
is involved; selection uses the true physics.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import numpy as np
import stable_worldmodel  # noqa: F401 - registers swm environments

import scripts.eval.render_fetch_push_commitment_demo as R
from scripts.data.probe_fetch_slide_friction import (
    collect_strike_context,
    discrete_distribution_index,
    exact_terminal_positions,
    fixed_profile,
    mean_state_index,
    strike_candidates,
    terminal_distances,
)
from stochastic_physics import FetchSlideHiddenFriction

MODES = (("LOW FRICTION", 0.2), ("HIGH FRICTION", 1.0))
THRESHOLD = 0.05


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Composite 2x2 mp4 (optional).")
    parser.add_argument("--poster", type=Path)
    parser.add_argument("--individual-dir", type=Path,
                        help="If set, write one mp4 per (method, friction) cell here.")
    parser.add_argument("--individual-prefix", type=str, default="slide_panel")
    parser.add_argument("--seed", type=int, default=703072)
    parser.add_argument("--max-search", type=int, default=60)
    parser.add_argument("--raw-horizon", type=int, default=100)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--initial-hold", type=int, default=8)
    parser.add_argument("--final-hold", type=int, default=22)
    return parser.parse_args()


def slide_replay(env, fork_state, actions, multiplier, *, initial_hold, final_hold):
    state = copy.deepcopy(fork_state)
    state["mode"]["surface_friction_multiplier"] = float(multiplier)
    env.set_fork_state(state)
    frames = [R.render(env)]
    distances = [float(np.linalg.norm(env._object_position() - env._goal_position()))]
    for action in actions:
        env.step(action)
        frames.append(R.render(env))
        distances.append(
            float(np.linalg.norm(env._object_position() - env._goal_position()))
        )
    frames = [frames[0]] * initial_hold + frames + [frames[-1]] * final_hold
    distances = [distances[0]] * initial_hold + distances + [distances[-1]] * final_hold
    distances = np.asarray(distances)
    running_minimum = np.minimum.accumulate(distances)
    # FetchSlide success is a settled (terminal) outcome; show per-moment
    # closeness during motion, and let the final frame reflect where it rested.
    successes = distances <= THRESHOLD
    return frames, distances, running_minimum, successes


def main():
    args = parse_args()
    for path in (args.output, args.poster):
        if path is not None and path.exists():
            raise SystemExit(f"Refusing to overwrite existing artifact: {path}")
    if args.output is None and args.individual_dir is None:
        raise SystemExit("Provide --output (composite) and/or --individual-dir (panels)")

    env = FetchSlideHiddenFriction(
        gym.make(
            "swm/FetchSlide-v3",
            max_episode_steps=args.raw_horizon + 80,
            render_mode="rgb_array",
        ),
        profile=fixed_profile(1.0),
        terminate_at_goal=False,
    )
    modes = tuple(m for _, m in MODES)
    chosen = None
    seed = int(args.seed)
    try:
        for _ in range(args.max_search):
            context = collect_strike_context(env, seed)
            seed += 1
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
            exact = exact_terminal_positions(env, context["fork_state"], candidates, modes)
            distances = terminal_distances(exact, goal)  # (ncand, 2)
            det = mean_state_index(exact, goal)
            sto = discrete_distribution_index(distances, THRESHOLD)
            det_ok = distances[det] <= THRESHOLD
            sto_ok = distances[sto] <= THRESHOLD
            # On FetchSlide no single strike hedges both very different slide
            # distances, so the honest demo is commitment: the mean strike
            # averages the two outcomes and misses BOTH (the valley), while the
            # distribution strike commits to the recoverable (slippery) mode and
            # lands it. Require mean-wins-0, distribution-wins the low mode.
            if det_ok.sum() == 0 and sto_ok[0] and det != sto:
                chosen = dict(
                    context=context, candidates=candidates, parameters=parameters,
                    det=det, sto=sto, det_d=distances[det], sto_d=distances[sto],
                )
                print(
                    f"scene seed={context['seed']} mean_idx={det} dist_idx={sto} "
                    f"mean_d={np.round(distances[det]*100,1)}cm "
                    f"dist_d={np.round(distances[sto]*100,1)}cm",
                    flush=True,
                )
                break
        if chosen is None:
            raise SystemExit("No clean FetchSlide commitment scene found in search budget")

        R.MODES = MODES
        R.METHODS = ("Mean-oracle strike", "Distribution-oracle strike")
        R.TITLE = "FetchSlide commitment — exact-fork oracle"
        R.SUBTITLE = "Mean strike averages the two slides and misses both; distribution strike commits"
        R.CAPTION = (
            "Transfer task: mean-state oracle ~0% vs distribution oracle 47.7% "
            "(episode, 64 scenes) — no strike hedges both, so the win is commitment"
        )

        selected = (chosen["det"], chosen["sto"])
        trajectories = {}
        for column, candidate_index in enumerate(selected):
            for row, (_, multiplier) in enumerate(MODES):
                trajectories[(column, row)] = slide_replay(
                    env,
                    chosen["context"]["fork_state"],
                    chosen["candidates"][candidate_index],
                    multiplier,
                    initial_hold=args.initial_hold,
                    final_hold=args.final_hold,
                )
        length = len(trajectories[(0, 0)][0])
        if args.output is not None:
            composed = [
                R.compose(
                    index,
                    trajectories,
                    (chosen["parameters"][selected[0]], chosen["parameters"][selected[1]]),
                    THRESHOLD,
                    chosen["context"]["seed"],
                )
                for index in range(length)
            ]
            args.output.parent.mkdir(parents=True, exist_ok=True)
            R.encode_video(composed, args.output, args.fps)
            if args.poster is not None:
                args.poster.parent.mkdir(parents=True, exist_ok=True)
                composed[-1].save(args.poster)
            print(f"video={args.output}")
            if args.poster is not None:
                print(f"poster={args.poster}")
        if args.individual_dir is not None:
            R.render_individual_panels(
                trajectories,
                (chosen["parameters"][selected[0]], chosen["parameters"][selected[1]]),
                THRESHOLD,
                args.individual_dir,
                args.individual_prefix,
                args.fps,
                moving_label="SLIDING",
                show_closest=False,
            )
        for column, method in enumerate(R.METHODS):
            for row, (mode_name, multiplier) in enumerate(MODES):
                minimum = trajectories[(column, row)][1][-1]
                status = "success" if minimum <= THRESHOLD else "miss"
                print(f"{method} | {mode_name} ({multiplier:g}x): {status}, terminal={minimum*100:.2f} cm")
    finally:
        env.close()


if __name__ == "__main__":
    main()
