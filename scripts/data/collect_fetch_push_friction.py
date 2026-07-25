#!/usr/bin/env python
"""Collect paired low/high-friction FetchPush demonstrations."""

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

from fetch_push_expert import FetchPushExpertPolicy
from scripts.data.collect_stochastic_ogbench import CommandedActionWriter
from stochastic_physics import FETCH_PUSH_PROFILES, FetchPushPairedFriction


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=FETCH_PUSH_PROFILES, default="push_strong")
    parser.add_argument("--pairs", type=int, default=500)
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Keep at 1 so the episode budget cannot stop between paired resets.",
    )
    parser.add_argument("--max-episode-steps", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--confirm-large",
        action="store_true",
        help="Required for collections above the 500-pair pilot.",
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
    if args.pairs > 500 and not args.confirm_large:
        raise SystemExit("Refusing a post-pilot collection without --confirm-large")
    if args.output.exists():
        raise SystemExit(f"Refusing to append to immutable dataset: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    world = swm.World(
        env_name="swm/FetchPush-v3",
        num_envs=args.num_envs,
        image_shape=(args.image_size, args.image_size),
        max_episode_steps=args.max_episode_steps,
        pre_wrappers=[
            partial(
                FetchPushPairedFriction,
                profile=args.profile,
                collection_seed_origin=args.seed,
                scene_seed_origin=args.seed,
            )
        ],
    )
    world.set_policy(FetchPushExpertPolicy(seed=args.seed))
    episodes = 2 * args.pairs
    world.collect(
        writer=CommandedActionWriter(args.output),
        episodes=episodes,
        seed=args.seed,
    )
    world.close()

    metadata = {
        "task": "fetch_push_hidden_friction",
        "pairs": args.pairs,
        "episodes": episodes,
        "collection_seed_origin": args.seed,
        "scene_seed_range": [args.seed, args.seed + args.pairs - 1],
        "pairing": "consecutive logical episodes share the same scene seed; low then high",
        "hidden_physics": FETCH_PUSH_PROFILES[args.profile].__dict__,
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
