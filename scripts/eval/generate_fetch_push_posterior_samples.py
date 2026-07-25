#!/usr/bin/env python
"""Generate fixed-friction posterior forks after observed pushing motion."""

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
from scripts.eval.generate_fetch_push_fork_samples import (
    action_statistics,
    encode_futures,
    fixed_profile,
    pack_actions,
    parse_checkpoint,
    preprocess_images,
    render,
    rollout_model,
    set_mode,
    state_statistics,
    normalize_states,
)
from stochastic_physics import FetchPushHiddenFriction


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nominal-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", action="append", type=parse_checkpoint, default=[]
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contexts", type=int, default=128, help="Number of scenes")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--observation-model-steps", type=int, default=2)
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--frameskip", type=int, default=2)
    parser.add_argument("--flow-steps", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=203072)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def collect_scene(
    env,
    policy,
    seed,
    *,
    history_size,
    frameskip,
    observation_model_steps,
    horizon,
    modes,
):
    if observation_model_steps < history_size - 1:
        raise ValueError("Observation steps must fill the complete model history")
    state, _ = env.reset(seed=seed)
    phase = 0
    for _ in range(50):
        action, phase = policy._action(state, phase)
        state, _, _, _, _ = env.step(action)
        if phase == 3:
            break
    if phase != 3:
        return None

    onset = env.get_fork_state()
    raw_observation_steps = observation_model_steps * frameskip
    raw_future_steps = horizon * frameskip
    shared_actions = []
    planning_state = state
    planning_phase = phase
    for _ in range(raw_observation_steps + raw_future_steps):
        action, planning_phase = policy._action(planning_state, planning_phase)
        shared_actions.append(action.copy())
        planning_state, _, _, _, _ = env.step(action)

    observation_actions = shared_actions[:raw_observation_steps]
    future_actions = shared_actions[raw_observation_steps:]
    history_raw_steps = (history_size - 1) * frameskip
    action_history = observation_actions[-history_raw_steps:]
    contexts = []
    for multiplier in modes:
        env.set_fork_state(set_mode(copy.deepcopy(onset), multiplier))
        model_frames = [render(env)]
        model_positions = [env._object_position().astype(np.float32)]
        model_states = [np.asarray(state, dtype=np.float32).copy()]
        for index, action in enumerate(observation_actions):
            mode_state, _, _, _, _ = env.step(action)
            if (index + 1) % frameskip == 0:
                model_frames.append(render(env))
                model_positions.append(env._object_position().astype(np.float32))
                model_states.append(np.asarray(mode_state, dtype=np.float32).copy())
        if len(model_frames) < history_size:
            raise RuntimeError("Observed trajectory did not fill model history")
        context_frames = np.stack(model_frames[-history_size:])
        future_frames = []
        future_positions = []
        for index, action in enumerate(future_actions):
            env.step(action)
            if (index + 1) % frameskip == 0:
                future_frames.append(render(env))
                future_positions.append(env._object_position().astype(np.float32))
        contexts.append(
            {
                "context_frames": context_frames,
                "memory_frames": np.stack(model_frames),
                "observed_object_positions": np.stack(model_positions),
                "observed_states": np.stack(model_states),
                "actions": np.asarray(
                    action_history + future_actions, dtype=np.float32
                ),
                "memory_actions": np.asarray(
                    observation_actions + future_actions, dtype=np.float32
                ),
                "future_frames": np.stack(future_frames),
                "future_positions": np.stack(future_positions),
                "mode": float(multiplier),
                "seed": seed,
            }
        )
    return contexts


def rollout_paired_modes(
    model,
    context_pixels,
    action_chunks,
    *,
    memory_pixels=None,
    memory_action_chunks=None,
    memory_states=None,
    **kwargs,
):
    """Use common residual noise for each low/high context pair."""
    if context_pixels.size(0) % 2:
        raise ValueError("Paired-mode rollout requires an even context count")
    low = rollout_model(
        model,
        context_pixels[0::2],
        action_chunks[0::2],
        memory_pixels=None if memory_pixels is None else memory_pixels[0::2],
        memory_action_chunks=(
            None if memory_action_chunks is None else memory_action_chunks[0::2]
        ),
        memory_states=None if memory_states is None else memory_states[0::2],
        **kwargs,
    )
    high = rollout_model(
        model,
        context_pixels[1::2],
        action_chunks[1::2],
        memory_pixels=None if memory_pixels is None else memory_pixels[1::2],
        memory_action_chunks=(
            None if memory_action_chunks is None else memory_action_chunks[1::2]
        ),
        memory_states=None if memory_states is None else memory_states[1::2],
        **kwargs,
    )
    return torch.stack((low, high), dim=1).flatten(0, 1)


@torch.no_grad()
def observed_diagnostics(model, pixels, actions, *, history_size, device):
    pixels = pixels.to(device)
    actions = actions.to(device)
    length = pixels.size(1)
    encoded = model.encode({"pixels": pixels, "action": actions[:, :length]})
    embeddings = encoded["emb"]
    action_embeddings = encoded["act_emb"]
    conditions = []
    residuals = []
    for target in range(1, length):
        start = max(0, target - history_size)
        prediction = model.predict(
            embeddings[:, start:target], action_embeddings[:, start:target]
        )[:, -1:]
        conditions.append(
            model.residual_condition(
                embeddings[:, target - 1 : target],
                action_embeddings[:, target - 1 : target],
                prediction,
            ).squeeze(1)
        )
        residuals.append((embeddings[:, target : target + 1] - prediction).squeeze(1))
    return {
        "observation_embeddings": embeddings.cpu(),
        "residual_conditions": torch.stack(conditions, dim=1).cpu(),
        "raw_residuals": torch.stack(residuals, dim=1).cpu(),
    }


@torch.no_grad()
def observed_memory_states(model, diagnostics, *, observed_states=None, device):
    if getattr(model, "residual_memory", None) is None:
        return None
    conditions = diagnostics["residual_conditions"].to(device)
    residuals = diagnostics["raw_residuals"].to(device)
    memory = model.init_residual_memory(
        (conditions.size(0),), device=device, dtype=conditions.dtype
    )
    states = []
    for index in range(conditions.size(1)):
        memory = model.update_residual_memory(
            memory,
            conditions[:, index],
            model.normalize_residual(residuals[:, index]),
            (
                None
                if observed_states is None
                else observed_states[:, index + 1].to(device)
                - observed_states[:, index].to(device)
            ),
        )
        states.append(memory)
    return torch.stack(states, dim=1).cpu()


def main():
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite immutable payload: {args.output}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    modes = (0.2, 3.0)
    env = FetchPushHiddenFriction(
        gym.make(
            "swm/FetchPush-v3",
            max_episode_steps=100,
            render_mode="rgb_array",
        ),
        profile=fixed_profile(1.0),
        terminate_at_goal=False,
    )
    policy = FetchPushExpertPolicy(seed=args.seed)
    scenes = []
    candidate = args.seed
    while len(scenes) < args.contexts:
        values = collect_scene(
            env,
            policy,
            candidate,
            history_size=args.history_size,
            frameskip=args.frameskip,
            observation_model_steps=args.observation_model_steps,
            horizon=args.horizon,
            modes=modes,
        )
        if values is not None:
            scenes.append(values)
        candidate += 1
    env.close()
    contexts = [value for scene in scenes for value in scene]

    context_images = np.stack([value["context_frames"] for value in contexts])
    memory_images = np.stack([value["memory_frames"] for value in contexts])
    observed_positions = np.stack(
        [value["observed_object_positions"] for value in contexts]
    )
    observed_states = np.stack([value["observed_states"] for value in contexts])
    future_images = np.stack([value["future_frames"] for value in contexts])
    raw_actions = np.stack([value["actions"] for value in contexts])
    raw_memory_actions = np.stack(
        [value["memory_actions"] for value in contexts]
    )
    context_pixels = preprocess_images(context_images, args.image_size)
    memory_pixels = preprocess_images(memory_images, args.image_size)
    # encode_futures expects [context, realization, horizon, H, W, C].
    future_pixels = preprocess_images(future_images[:, None], args.image_size)
    action_mean, action_std = action_statistics(args.dataset)
    state_mean, state_std = state_statistics(args.dataset)
    normalized_observed_states = normalize_states(
        observed_states, state_mean, state_std
    )
    action_chunks = pack_actions(
        raw_actions, action_mean, action_std, args.frameskip
    )
    memory_action_chunks = pack_actions(
        raw_memory_actions, action_mean, action_std, args.frameskip
    )
    expected_chunks = args.history_size + args.horizon - 1
    if action_chunks.size(1) != expected_chunks:
        raise RuntimeError(
            f"Expected {expected_chunks} action chunks, got {action_chunks.size(1)}"
        )

    nominal = torch.load(
        args.nominal_checkpoint, map_location=device, weights_only=False
    ).to(device).eval()
    diagnostics = observed_diagnostics(
        nominal,
        memory_pixels,
        memory_action_chunks,
        history_size=args.history_size,
        device=device,
    )
    diagnostics["observed_object_positions"] = torch.from_numpy(observed_positions)
    diagnostics["observed_states"] = torch.from_numpy(observed_states)
    diagnostics["normalized_observed_states"] = normalized_observed_states
    diagnostic_memory = {}
    fixed_latents = encode_futures(nominal, future_pixels, device)
    simulator_futures = fixed_latents.repeat_interleave(args.samples, dim=1)
    fixed_positions = torch.from_numpy(
        np.stack([value["future_positions"] for value in contexts])[:, None]
    ).repeat_interleave(args.samples, dim=1)

    models = {
        "deterministic": rollout_paired_modes(
            nominal,
            context_pixels,
            action_chunks,
            samples=args.samples,
            history_size=args.history_size,
            horizon=args.horizon,
            flow_steps=args.flow_steps,
            device=device,
            seed=args.seed,
            memory_pixels=memory_pixels,
            memory_action_chunks=memory_action_chunks,
        )
    }
    for offset, (name, path) in enumerate(args.checkpoint, start=1):
        model = torch.load(path, map_location=device, weights_only=False).to(device).eval()
        needs_observations = bool(
            int(getattr(getattr(model, "residual_memory", None), "observation_dim", 0))
        )
        memory_states = observed_memory_states(
            model,
            diagnostics,
            observed_states=(normalized_observed_states if needs_observations else None),
            device=device,
        )
        if memory_states is not None:
            diagnostic_memory[name] = memory_states
        models[name] = rollout_paired_modes(
            model,
            context_pixels,
            action_chunks,
            samples=args.samples,
            history_size=args.history_size,
            horizon=args.horizon,
            flow_steps=args.flow_steps,
            device=device,
            seed=args.seed + offset,
            memory_pixels=memory_pixels,
            memory_action_chunks=memory_action_chunks,
            memory_states=(normalized_observed_states if needs_observations else None),
        )

    payload = {
        "simulator_futures": simulator_futures,
        "model_samples": models,
        "simulator_object_positions": fixed_positions,
        "horizons": sorted(set([1, min(5, args.horizon), args.horizon])),
        "context_stage": ["post_contact_fixed_mode"] * len(contexts),
        "context_mode": [value["mode"] for value in contexts],
        "diagnostics": {
            **diagnostics,
            "residual_memory_states": diagnostic_memory,
        },
        "metadata": {
            "task": "fetch_push_hidden_friction",
            "regime": "post_contact_fixed_mode",
            "scene_seeds": [scene[0]["seed"] for scene in scenes],
            "context_seeds": [value["seed"] for value in contexts],
            "friction_modes": list(modes),
            "observation_model_steps": args.observation_model_steps,
            "frameskip": args.frameskip,
            "raw_action_horizon": args.frameskip * args.horizon,
            "nominal_checkpoint": str(args.nominal_checkpoint.resolve()),
            "checkpoints": {name: str(path.resolve()) for name, path in args.checkpoint},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(
        f"saved {args.contexts} scenes x {len(modes)} fixed modes x "
        f"{args.samples} samples x {args.horizon} steps to {args.output}"
    )


if __name__ == "__main__":
    main()
