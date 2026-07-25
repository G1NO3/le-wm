#!/usr/bin/env python
"""Score ordinary-state residual dynamics on exact low/high-friction forks."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import numpy as np
import stable_worldmodel  # noqa: F401
import torch

from fetch_push_expert import FetchPushExpertPolicy
from scripts.eval.generate_fetch_push_fork_samples import collect_context, fixed_profile
from stochastic_physics import FetchPushHiddenFriction


@torch.no_grad()
def rollout(
    model,
    initial_state,
    action_blocks,
    *,
    samples,
    stochastic,
    flow_steps,
    seed,
    device,
    temporal_common_noise=False,
    kernel="flow",
):
    model = model.to(device).eval()
    state = torch.as_tensor(initial_state, device=device).float()[:, None].expand(-1, samples, -1).clone()
    actions = torch.as_tensor(action_blocks, device=device).float()[:, None].expand(-1, samples, -1, -1)
    generator = torch.Generator(device=device).manual_seed(seed)
    persistent_noise = (
        torch.randn(
            *state.shape[:-1], model.dynamic_dim, device=device, generator=generator
        )
        if stochastic and temporal_common_noise
        else None
    )
    persistent_mode = None
    if kernel == "mode_mixture" and stochastic:
        persistent_mode = (
            torch.arange(samples, device=device) >= samples // 2
        ).long()[None].expand(state.size(0), -1)
    trajectory = []
    for index in range(actions.size(2)):
        if kernel == "mode_mixture":
            state = model.transition_mode_mixture(
                state, actions[:, :, index], mode=persistent_mode
            )
            trajectory.append(state[..., 3:6].cpu())
            continue
        noise = persistent_noise if persistent_noise is not None else (
            torch.randn(*state.shape[:-1], model.dynamic_dim, device=device, generator=generator)
            if stochastic
            else None
        )
        state = model.transition(
            state, actions[:, :, index], noise=noise, flow_steps=flow_steps
        )
        trajectory.append(state[..., 3:6].cpu())
    return torch.stack(trajectory, dim=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contexts", type=int, default=64)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--frameskip", type=int, default=2)
    parser.add_argument("--flow-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=403072)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--temporal-common-noise", action="store_true")
    parser.add_argument("--kernel", choices=("flow", "mode_mixture"), default="flow")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite immutable payload: {args.output}")
    if args.samples % 2:
        raise SystemExit("--samples must be even for balanced low/high truth")

    env = FetchPushHiddenFriction(
        gym.make("swm/FetchPush-v3", max_episode_steps=100, render_mode="rgb_array"),
        profile=fixed_profile(1.0),
        terminate_at_goal=False,
    )
    policy = FetchPushExpertPolicy(seed=args.seed)
    contexts = []
    candidate = args.seed
    while len(contexts) < args.contexts:
        value = collect_context(
            env,
            policy,
            candidate,
            history_size=args.history_size,
            frameskip=args.frameskip,
            horizon=args.horizon,
            modes=(0.2, 3.0),
        )
        if value is not None:
            contexts.append(value)
        candidate += 1
    env.close()

    initial_state = np.stack([value["context_states"][-1] for value in contexts])
    raw_actions = np.stack([value["actions"] for value in contexts])
    future_start = (args.history_size - 1) * args.frameskip
    future_actions = raw_actions[:, future_start:]
    action_blocks = future_actions.reshape(args.contexts, args.horizon, -1)
    mode_positions = torch.from_numpy(
        np.stack([value["future_positions"] for value in contexts])
    )
    truth = mode_positions.repeat_interleave(args.samples // 2, dim=1)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = torch.load(args.checkpoint, map_location=device, weights_only=False)
    # A mode-mixture payload must retain the pure frozen nominal as its
    # deterministic baseline. The evaluator separately computes each sampled
    # model's predictive-mean score, so this provides both required
    # attributions: residual mixture versus nominal, and stochastic spread
    # versus the same mixture's mean.
    deterministic_kernel = "flow" if args.kernel == "mode_mixture" else args.kernel
    predictions = {
        "deterministic": rollout(
            model, initial_state, action_blocks,
            samples=args.samples, stochastic=False, flow_steps=args.flow_steps,
            seed=args.seed, device=device,
            kernel=deterministic_kernel,
        ),
        "flow": rollout(
            model, initial_state, action_blocks,
            samples=args.samples, stochastic=True, flow_steps=args.flow_steps,
            seed=args.seed + 1, device=device,
            temporal_common_noise=args.temporal_common_noise,
            kernel=args.kernel,
        ),
    }
    payload = {
        "simulator_futures": truth,
        "model_samples": predictions,
        "horizons": sorted({1, min(5, args.horizon), args.horizon}),
        "context_stage": ["pre_contact_prior_mixture"] * args.contexts,
        "metadata": {
            "task": "fetch_push_hidden_friction",
            "representation": "ordinary_state_object_position",
            "regime": "pre_contact_prior_mixture",
            "context_seeds": [value["seed"] for value in contexts],
            "friction_modes": [0.2, 3.0],
            "frameskip": args.frameskip,
            "checkpoint": str(args.checkpoint.resolve()),
            "temporal_common_noise": args.temporal_common_noise,
            "kernel": args.kernel,
            "deterministic_kernel": deterministic_kernel,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"saved {args.contexts} contexts x {args.samples} samples x {args.horizon} steps to {args.output}")


if __name__ == "__main__":
    main()
