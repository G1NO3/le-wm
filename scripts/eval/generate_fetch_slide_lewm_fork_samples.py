#!/usr/bin/env python
"""Generate exact FetchSlide forks and deterministic vanilla-LeWM rollouts."""

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

from fetch_push_expert import FetchPushExpertPolicy
from scripts.data.probe_fetch_slide_friction import (
    fixed_profile,
    strike_candidates,
)
from scripts.eval.generate_fetch_push_fork_samples import (
    action_statistics,
    encode_futures,
    pack_actions,
    preprocess_images,
    render,
    rollout_model,
)
from stochastic_physics import FetchSlideHiddenFriction


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contexts", type=int, default=64)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--raw-horizon", type=int, default=100)
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--frameskip", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=603072)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def collect_lewm_strike_context(
    env,
    seed,
    *,
    history_size=3,
    frameskip=2,
    max_steps=50,
):
    """Approach the puck and retain the exact pixel/action LeWM context."""

    state, info = env.reset(seed=seed)
    policy = FetchPushExpertPolicy(
        seed=seed,
        behind_offset=0.075,
        clearance=0.08,
    )
    phase = 0
    frames = [render(env)]
    actions = []
    for _ in range(max_steps):
        action, phase = policy._action(state, phase)
        actions.append(np.asarray(action, dtype=np.float32).copy())
        state, _, _, _, info = env.step(action)
        frames.append(render(env))
        if phase == 3 and len(actions) >= (history_size - 1) * frameskip:
            break
    if phase != 3:
        return None

    context_step = len(actions)
    indices = [
        context_step - (history_size - 1 - index) * frameskip
        for index in range(history_size)
    ]
    return {
        "seed": int(seed),
        "state": np.asarray(state, dtype=np.float32).copy(),
        "fork_state": env.get_fork_state(),
        "context_frames": np.stack([frames[index] for index in indices]),
        "past_actions": np.stack(actions[indices[0] : context_step]),
        "goal_image": np.asarray(info["goal"]).copy(),
    }


def exact_future_images(env, fork_state, actions, modes, *, frameskip=2):
    frames = []
    positions = []
    for multiplier in modes:
        state = copy.deepcopy(fork_state)
        state["mode"]["surface_friction_multiplier"] = float(multiplier)
        env.set_fork_state(state)
        mode_frames = []
        mode_positions = []
        for index, action in enumerate(actions):
            env.step(action)
            if (index + 1) % frameskip == 0:
                mode_frames.append(render(env))
                mode_positions.append(env._object_position().astype(np.float32))
        frames.append(np.stack(mode_frames))
        positions.append(np.stack(mode_positions))
    return np.stack(frames), np.stack(positions)


def require_vanilla_lewm(model):
    if getattr(model, "residual_flow", None) is not None:
        raise RuntimeError("Checkpoint contains a residual flow; vanilla LeWM required")
    if getattr(model, "residual_kernel", None) is not None:
        raise RuntimeError("Checkpoint contains a residual kernel; vanilla LeWM required")


def main():
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite immutable payload: {args.output}")
    if args.samples <= 0 or args.samples % 2:
        raise SystemExit("--samples must be a positive even number")
    if args.raw_horizon <= 0 or args.raw_horizon % args.frameskip:
        raise SystemExit("--raw-horizon must be divisible by --frameskip")

    modes = (0.2, 1.0)
    model_horizon = args.raw_horizon // args.frameskip
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
    future_actions = []
    mode_frames = []
    mode_positions = []
    candidate_seed = int(args.seed)
    try:
        while len(contexts) < args.contexts:
            context = collect_lewm_strike_context(
                env,
                candidate_seed,
                history_size=args.history_size,
                frameskip=args.frameskip,
            )
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
            generator = np.random.default_rng(args.seed + context["seed"])
            selected = library[int(generator.integers(0, len(library)))]
            exact_frames, exact_positions = exact_future_images(
                env,
                context["fork_state"],
                selected,
                modes,
                frameskip=args.frameskip,
            )
            contexts.append(context)
            future_actions.append(selected)
            mode_frames.append(exact_frames)
            mode_positions.append(exact_positions)
            print(
                f"vanilla LeWM exact fork {len(contexts)}/{args.contexts} "
                f"seed={context['seed']}",
                flush=True,
            )
    finally:
        env.close()

    context_pixels = preprocess_images(
        np.stack([value["context_frames"] for value in contexts]),
        args.image_size,
    )
    exact_pixels = preprocess_images(np.stack(mode_frames), args.image_size)
    raw_actions = np.stack(
        [
            np.concatenate([context["past_actions"], future], axis=0)
            for context, future in zip(contexts, future_actions, strict=True)
        ]
    )
    action_mean, action_std = action_statistics(args.dataset)
    action_chunks = pack_actions(
        raw_actions,
        action_mean,
        action_std,
        args.frameskip,
    )
    expected_chunks = args.history_size + model_horizon - 1
    if action_chunks.size(1) != expected_chunks:
        raise RuntimeError(
            f"Expected {expected_chunks} action chunks, got {action_chunks.size(1)}"
        )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    ).to(device).eval()
    require_vanilla_lewm(model)
    exact_latents = encode_futures(model, exact_pixels, device)
    repeats = args.samples // 2
    truth = exact_latents.repeat_interleave(repeats, dim=1)
    physical_truth = torch.from_numpy(np.stack(mode_positions)).repeat_interleave(
        repeats,
        dim=1,
    )
    prediction = rollout_model(
        model,
        context_pixels,
        action_chunks,
        samples=args.samples,
        history_size=args.history_size,
        horizon=model_horizon,
        flow_steps=1,
        device=device,
        seed=args.seed,
    )
    payload = {
        "simulator_futures": truth,
        "model_samples": {"deterministic": prediction},
        "simulator_object_positions": physical_truth,
        "horizons": [1, 5, 10, model_horizon],
        "context_stage": ["pre_strike_prior_mixture"] * args.contexts,
        "metadata": {
            "task": "fetch_slide_hidden_friction",
            "model": "vanilla_lewm",
            "residual_enabled": False,
            "representation": "vanilla_lewm_latent",
            "regime": "pre_strike_prior_mixture",
            "context_seeds": [value["seed"] for value in contexts],
            "friction_modes": list(modes),
            "frameskip": args.frameskip,
            "raw_horizon": args.raw_horizon,
            "checkpoint": str(args.checkpoint.resolve()),
            "dataset": str(args.dataset.resolve()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(
        f"saved vanilla LeWM {args.contexts} contexts x {args.samples} samples x "
        f"{model_horizon} model steps to {args.output}"
    )


if __name__ == "__main__":
    main()
