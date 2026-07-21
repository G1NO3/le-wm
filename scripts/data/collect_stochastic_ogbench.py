#!/usr/bin/env python
"""Collect gated stochastic multi-cube demonstrations with commanded actions."""

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
from stable_worldmodel.data.format import get_format

from fixed_task_expert import FixedTaskCubeExpertPolicy
from stochastic_physics import OGBenchHiddenPhysics, PHYSICS_PROFILES


TASKS = {
    "double_stack": {"env_type": "double", "reward_task_id": 5},
    "triple_cycle": {"env_type": "triple", "reward_task_id": 4},
    "quadruple_stack": {"env_type": "quadruple", "reward_task_id": 5},
    "octuple_stack2": {"env_type": "octuple", "reward_task_id": 5},
}


def align_commanded_actions(episodes):
    """Match stable-worldmodel's transition alignment for audit actions.

    ``World.collect`` rotates its model-facing ``action`` column after a
    trajectory is buffered. The wrapper-provided audit column must undergo the
    same rotation so each saved row contains the action that produced it.
    """

    for episode in episodes:
        commanded = episode.get("commanded_action")
        if commanded:
            commanded.append(commanded.pop(0))
            if "action" in episode:
                for action, recorded in zip(episode["action"], commanded, strict=True):
                    if not (action == recorded).all():
                        raise RuntimeError("commanded-action alignment invariant failed")
        yield episode


class CommandedActionWriter:
    """Writer adapter that aligns commanded-action audit rows."""

    def __init__(self, path):
        self._context = get_format("hdf5").open_writer(path)

    def __enter__(self):
        self._writer = self._context.__enter__()
        return self

    def __exit__(self, *args):
        return self._context.__exit__(*args)

    def write_episodes(self, episodes):
        return self._writer.write_episodes(align_commanded_actions(episodes))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--profile", choices=PHYSICS_PROFILES, default="strong")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--confirm-large",
        action="store_true",
        help="Required for collections above the 1,000-episode pilot gate.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.episodes > 1000 and not args.confirm_large:
        raise SystemExit("Refusing a post-pilot collection without --confirm-large")
    if args.output.exists():
        raise SystemExit(f"Refusing to append to immutable dataset: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    task = TASKS[args.task]
    world = swm.World(
        env_name="swm/OGBCube-v0",
        num_envs=args.num_envs,
        image_shape=(224, 224),
        max_episode_steps=args.max_episode_steps,
        env_type=task["env_type"],
        reward_task_id=task["reward_task_id"],
        ob_type="states",
        terminate_at_goal=True,
        pre_wrappers=[
            partial(OGBenchHiddenPhysics, profile=args.profile, seed=args.seed)
        ],
    )
    policy = FixedTaskCubeExpertPolicy(seed=args.seed)
    world.set_policy(policy)
    world.collect(
        writer=CommandedActionWriter(args.output),
        episodes=args.episodes,
        seed=args.seed,
    )
    metadata = {
        "task": args.task,
        "episodes": args.episodes,
        "environment_seeds": list(range(args.seed, args.seed + args.num_envs)),
        "hidden_physics": PHYSICS_PROFILES[args.profile].__dict__,
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "output_path": str(args.output.resolve()),
    }
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    world.close()


if __name__ == "__main__":
    main()
