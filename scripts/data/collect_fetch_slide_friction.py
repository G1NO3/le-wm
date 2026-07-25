#!/usr/bin/env python
"""Collect paired low/high-friction FetchSlide strike trajectories."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from functools import partial
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")

import stable_worldmodel as swm

from fetch_slide_expert import FetchSlideStrikePolicy
from scripts.data.collect_stochastic_ogbench import CommandedActionWriter
from stochastic_physics import FETCH_SLIDE_PROFILES, FetchSlidePairedFriction


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", choices=FETCH_SLIDE_PROFILES, default="slide_strong"
    )
    parser.add_argument("--pairs", type=int, default=100)
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Must remain 1 so collection cannot stop between paired resets.",
    )
    parser.add_argument("--max-episode-steps", type=int, default=130)
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--seed", type=int, default=4072)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--confirm-large",
        action="store_true",
        help="Required for collections above the 100-pair pilot.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.pairs <= 0:
        raise SystemExit("--pairs must be positive")
    if args.num_envs != 1:
        raise SystemExit(
            "Paired collection requires --num-envs 1; asynchronous vector "
            "completion can leave the final counterfactual pair incomplete"
        )
    if args.pairs > 100 and not args.confirm_large:
        raise SystemExit("Refusing a post-pilot collection without --confirm-large")
    if args.output.exists():
        raise SystemExit(f"Refusing to append to immutable dataset: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    world = swm.World(
        env_name="swm/FetchSlide-v3",
        num_envs=1,
        image_shape=(args.image_size, args.image_size),
        max_episode_steps=args.max_episode_steps,
        pre_wrappers=[
            partial(
                FetchSlidePairedFriction,
                profile=args.profile,
                collection_seed_origin=args.seed,
                scene_seed_origin=args.seed,
                terminate_at_goal=False,
                inner_max_episode_steps=args.max_episode_steps,
            )
        ],
    )
    world.set_policy(FetchSlideStrikePolicy(seed=args.seed))
    episodes = 2 * args.pairs
    world.collect(
        writer=CommandedActionWriter(args.output),
        episodes=episodes,
        seed=args.seed,
    )
    world.close()

    metadata = {
        "task": "fetch_slide_hidden_friction",
        "pairs": args.pairs,
        "episodes": episodes,
        "collection_seed_origin": args.seed,
        "scene_seed_range": [args.seed, args.seed + args.pairs - 1],
        "pairing": (
            "consecutive logical episodes share scene and open-loop strike; "
            "low then high friction"
        ),
        "collection_policy": {
            "name": "FetchSlideStrikePolicy",
            "uses_realized_friction": False,
            "speed": [0.4, 1.0, 13],
            "duration": [1, 4],
            "angle_deg": [-3.0, 3.0],
        },
        "hidden_physics": FETCH_SLIDE_PROFILES[args.profile].__dict__,
        "image_size": args.image_size,
        "max_episode_steps": args.max_episode_steps,
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "output_path": str(args.output.resolve()),
    }
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
