#!/usr/bin/env python
"""Generate exact FetchSlide friction forks and state-model predictions."""

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
import torch

from scripts.data.probe_fetch_slide_friction import (
    collect_strike_context,
    fixed_profile,
    strike_candidates,
)
from scripts.eval.generate_fetch_state_fork_samples import rollout
from stochastic_physics import FetchSlideHiddenFriction


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contexts", type=int, default=64)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--raw-horizon", type=int, default=100)
    parser.add_argument("--flow-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=603072)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--kernel", choices=("flow", "mode_mixture"), default="mode_mixture"
    )
    parser.add_argument("--temporal-common-noise", action="store_true")
    return parser.parse_args()


def exact_futures(env, fork_state, actions, modes, frameskip=2):
    futures = []
    for multiplier in modes:
        state = copy.deepcopy(fork_state)
        state["mode"]["surface_friction_multiplier"] = float(multiplier)
        env.set_fork_state(state)
        positions = []
        for index, action in enumerate(actions):
            env.step(action)
            if (index + 1) % frameskip == 0:
                positions.append(env._object_position().astype(np.float32))
        futures.append(np.stack(positions))
    return np.stack(futures)


def main():
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite immutable payload: {args.output}")
    if args.samples % 2:
        raise SystemExit("--samples must be even for balanced low/high truth")
    if args.raw_horizon <= 0 or args.raw_horizon % 2:
        raise SystemExit("--raw-horizon must be a positive multiple of two")

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
    contexts = []
    actions = []
    futures = []
    candidate_seed = int(args.seed)
    try:
        while len(contexts) < args.contexts:
            context = collect_strike_context(env, candidate_seed)
            candidate_seed += 1
            if context is None:
                continue
            library, _ = strike_candidates(
                context["state"],
                horizon=args.raw_horizon,
                speed_min=0.4,
                speed_max=1.0,
                speed_count=13,
                duration_min=1,
                duration_max=4,
                angle_max_deg=3.0,
                angle_count=7,
            )
            rng = np.random.default_rng(args.seed + context["seed"])
            selected = library[int(rng.integers(0, len(library)))]
            contexts.append(context)
            actions.append(selected)
            futures.append(
                exact_futures(env, context["fork_state"], selected, modes)
            )
            print(
                f"exact slide fork {len(contexts)}/{args.contexts} "
                f"seed={context['seed']}",
                flush=True,
            )
    finally:
        env.close()

    initial_state = np.stack([value["state"] for value in contexts])
    raw_actions = np.stack(actions)
    model_horizon = args.raw_horizon // 2
    action_blocks = raw_actions.reshape(args.contexts, model_horizon, -1)
    mode_positions = torch.from_numpy(np.stack(futures))
    truth = mode_positions.repeat_interleave(args.samples // 2, dim=1)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = torch.load(args.checkpoint, map_location=device, weights_only=False)
    deterministic_kernel = "flow" if args.kernel == "mode_mixture" else args.kernel
    predictions = {
        "deterministic": rollout(
            model,
            initial_state,
            action_blocks,
            samples=args.samples,
            stochastic=False,
            flow_steps=args.flow_steps,
            seed=args.seed,
            device=device,
            kernel=deterministic_kernel,
        ),
        args.kernel: rollout(
            model,
            initial_state,
            action_blocks,
            samples=args.samples,
            stochastic=True,
            flow_steps=args.flow_steps,
            seed=args.seed + 1,
            device=device,
            temporal_common_noise=args.temporal_common_noise,
            kernel=args.kernel,
        ),
    }
    payload = {
        "simulator_futures": truth,
        "model_samples": predictions,
        "horizons": [1, 5, 10, model_horizon],
        "context_stage": ["pre_strike_prior_mixture"] * args.contexts,
        "metadata": {
            "task": "fetch_slide_hidden_friction",
            "representation": "ordinary_state_object_position",
            "regime": "pre_strike_prior_mixture",
            "context_seeds": [value["seed"] for value in contexts],
            "friction_modes": list(modes),
            "frameskip": 2,
            "raw_horizon": args.raw_horizon,
            "checkpoint": str(args.checkpoint.resolve()),
            "temporal_common_noise": args.temporal_common_noise,
            "kernel": args.kernel,
            "deterministic_kernel": deterministic_kernel,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(
        f"saved {args.contexts} contexts x {args.samples} samples x "
        f"{model_horizon} model steps to {args.output}"
    )


if __name__ == "__main__":
    main()
