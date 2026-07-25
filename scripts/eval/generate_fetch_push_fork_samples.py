#!/usr/bin/env python
"""Generate exact prior-mixture forks and matching LeWM latent rollouts."""

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
import h5py
import numpy as np
import stable_worldmodel  # noqa: F401 - registers swm environments
import torch

from fetch_push_expert import FetchPushExpertPolicy
from stochastic_physics import FetchPushHiddenFriction, PhysicsProfile
from utils import get_img_preprocessor


def fixed_profile(multiplier):
    return PhysicsProfile(
        f"fixed_{multiplier:g}",
        (1.0, 1.0),
        (float(multiplier), float(multiplier)),
        (1.0, 1.0),
        0.0,
    )


def parse_checkpoint(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use NAME=/path/to/object.ckpt")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("Use NAME=/path/to/object.ckpt")
    return name, Path(path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nominal-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=parse_checkpoint,
        default=[],
        help="Residual model as NAME=/path/to/object.ckpt; repeat as needed.",
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contexts", type=int, default=32)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--frameskip", type=int, default=2)
    parser.add_argument("--flow-steps", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=103072)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def render(env):
    image = env.render()
    if image is None:
        raise RuntimeError("FetchPush renderer returned no RGB frame")
    return np.asarray(image).copy()


def set_mode(context, multiplier):
    state = copy.deepcopy(context)
    state["mode"]["surface_friction_multiplier"] = float(multiplier)
    return state


def collect_context(env, policy, seed, *, history_size, frameskip, horizon, modes):
    state, _ = env.reset(seed=seed)
    phase = 0
    frames = [render(env)]
    states = [np.asarray(state, dtype=np.float32).copy()]
    actions = []
    for _ in range(50):
        action, phase = policy._action(state, phase)
        actions.append(action.copy())
        state, _, _, _, _ = env.step(action)
        frames.append(render(env))
        states.append(np.asarray(state, dtype=np.float32).copy())
        if phase == 3 and len(actions) >= (history_size - 1) * frameskip:
            break
    if phase != 3:
        return None

    context = env.get_fork_state()
    context_step = len(actions)
    indices = [
        context_step - (history_size - 1 - index) * frameskip
        for index in range(history_size)
    ]
    context_frames = np.stack([frames[index] for index in indices])
    context_states = np.stack([states[index] for index in indices])
    past_actions = actions[indices[0] : context_step]

    future_actions = []
    nominal_state = state
    nominal_phase = phase
    for _ in range(frameskip * horizon):
        action, nominal_phase = policy._action(nominal_state, nominal_phase)
        future_actions.append(action.copy())
        nominal_state, _, _, _, _ = env.step(action)
    all_actions = np.asarray(past_actions + future_actions, dtype=np.float32)

    future_frames = []
    future_positions = []
    for multiplier in modes:
        env.set_fork_state(set_mode(context, multiplier))
        mode_frames = []
        mode_positions = []
        for index, action in enumerate(future_actions):
            env.step(action)
            if (index + 1) % frameskip == 0:
                mode_frames.append(render(env))
                mode_positions.append(env._object_position().astype(np.float32))
        future_frames.append(np.stack(mode_frames))
        future_positions.append(np.stack(mode_positions))
    return {
        "context_frames": context_frames,
        "context_states": context_states,
        "actions": all_actions,
        "future_frames": np.stack(future_frames),
        "future_positions": np.stack(future_positions),
        "seed": seed,
    }


def action_statistics(dataset_path):
    with h5py.File(dataset_path, "r") as dataset:
        action = torch.from_numpy(np.asarray(dataset["action"], dtype=np.float32))
    return action.mean(0), action.std(0)


def state_statistics(dataset_path):
    with h5py.File(dataset_path, "r") as dataset:
        state = torch.from_numpy(np.asarray(dataset["state"], dtype=np.float32))
    valid = state[~torch.isnan(state).any(dim=1)]
    return valid.mean(0), valid.std(0)


def normalize_states(raw_states, mean, std):
    states = torch.as_tensor(raw_states, dtype=torch.float32)
    return torch.nan_to_num((states - mean) / std, 0.0)


def pack_actions(raw_actions, mean, std, frameskip):
    actions = torch.as_tensor(raw_actions, dtype=torch.float32)
    # The scripted policy never moves the gripper actuator, so that column has
    # zero empirical variance. Training applies the same normalizer and then
    # replaces its NaN with zero before encoding actions.
    actions = torch.nan_to_num((actions - mean) / std, 0.0)
    if actions.size(-2) % frameskip:
        raise ValueError("Raw action sequence is not divisible by frameskip")
    return actions.reshape(*actions.shape[:-2], actions.size(-2) // frameskip, -1)


def preprocess_images(images, image_size):
    shape = images.shape
    flat = images.reshape(-1, *shape[-3:])
    transformed = get_img_preprocessor("pixels", "pixels", image_size)(
        {"pixels": flat}
    )["pixels"]
    return transformed.reshape(*shape[:-3], *transformed.shape[-3:])


@torch.no_grad()
def encode_futures(model, images, device, batch_size=128):
    flat = images.flatten(0, 2)
    chunks = []
    for start in range(0, len(flat), batch_size):
        pixels = flat[start : start + batch_size].to(device)[:, None]
        chunks.append(model.encode({"pixels": pixels})["emb"][:, 0].cpu())
    encoded = torch.cat(chunks)
    return encoded.reshape(*images.shape[:3], -1)


@torch.no_grad()
def rollout_model(
    model,
    context_pixels,
    action_chunks,
    *,
    samples,
    history_size,
    horizon,
    flow_steps,
    device,
    seed,
    memory_pixels=None,
    memory_action_chunks=None,
    memory_states=None,
):
    model = model.to(device).eval()
    contexts = context_pixels.to(device)
    actions = action_chunks.to(device)
    batch = contexts.size(0)
    info = {
        "pixels": contexts[:, None].expand(-1, samples, -1, -1, -1, -1).clone()
    }
    action_sequence = actions[:, None].expand(-1, samples, -1, -1).clone()

    if getattr(model, "residual_memory", None) is not None:
        memory_contexts = (
            context_pixels if memory_pixels is None else memory_pixels
        ).to(device)
        memory_actions = (
            action_chunks if memory_action_chunks is None else memory_action_chunks
        ).to(device)
        memory_length = memory_contexts.size(1)
        encoded = model.encode(
            {
                "pixels": memory_contexts,
                "action": memory_actions[:, :memory_length],
            }
        )
        info["residual_memory"] = model.update_residual_memory_from_history(
            None,
            encoded["emb"],
            encoded["act_emb"],
            observations=(
                None if memory_states is None else memory_states.to(device)
            ),
            start_target=1,
            history_size=history_size,
        )

    stochastic = model.residual_flow is not None or getattr(
        model, "residual_kernel", None
    ) is not None
    generator = torch.Generator(device=device).manual_seed(seed)
    output = model.rollout(
        info,
        action_sequence,
        history_size=history_size,
        stochastic=stochastic,
        flow_steps=flow_steps,
        sampling_generator=generator,
    )["predicted_emb"]
    future = output[:, :, history_size : history_size + horizon]
    if future.shape[:3] != (batch, samples, horizon):
        raise RuntimeError(f"Unexpected rollout shape: {tuple(future.shape)}")
    return future.cpu()


def main():
    args = parse_args()
    if args.samples % 2:
        raise SystemExit("--samples must be even for a balanced two-mode truth")
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite immutable fork payload: {args.output}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

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
    modes = (0.2, 3.0)
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
            modes=modes,
        )
        if value is not None:
            contexts.append(value)
        candidate += 1
    env.close()

    context_images = np.stack([value["context_frames"] for value in contexts])
    raw_context_states = np.stack([value["context_states"] for value in contexts])
    raw_actions = np.stack([value["actions"] for value in contexts])
    mode_images = np.stack([value["future_frames"] for value in contexts])
    mode_positions = np.stack([value["future_positions"] for value in contexts])
    context_pixels = preprocess_images(context_images, args.image_size)
    mode_pixels = preprocess_images(mode_images, args.image_size)
    action_mean, action_std = action_statistics(args.dataset)
    state_mean, state_std = state_statistics(args.dataset)
    context_states = normalize_states(raw_context_states, state_mean, state_std)
    action_chunks = pack_actions(
        raw_actions, action_mean, action_std, args.frameskip
    )
    expected_chunks = args.history_size + args.horizon - 1
    if action_chunks.size(1) != expected_chunks:
        raise RuntimeError(
            f"Expected {expected_chunks} action chunks, got {action_chunks.size(1)}"
        )

    nominal = torch.load(
        args.nominal_checkpoint, map_location=device, weights_only=False
    ).to(device).eval()
    mode_latents = encode_futures(nominal, mode_pixels, device)
    repeats = args.samples // 2
    simulator_futures = mode_latents.repeat_interleave(repeats, dim=1)
    physical_futures = torch.from_numpy(mode_positions).repeat_interleave(
        repeats, dim=1
    )

    models = {
        "deterministic": rollout_model(
            nominal,
            context_pixels,
            action_chunks,
            samples=args.samples,
            history_size=args.history_size,
            horizon=args.horizon,
            flow_steps=args.flow_steps,
            device=device,
            seed=args.seed,
        )
    }
    for offset, (name, path) in enumerate(args.checkpoint, start=1):
        model = torch.load(path, map_location=device, weights_only=False)
        needs_observations = bool(
            int(getattr(getattr(model, "residual_memory", None), "observation_dim", 0))
        )
        models[name] = rollout_model(
            model,
            context_pixels,
            action_chunks,
            samples=args.samples,
            history_size=args.history_size,
            horizon=args.horizon,
            flow_steps=args.flow_steps,
            device=device,
            seed=args.seed + offset,
            memory_states=(context_states if needs_observations else None),
        )

    payload = {
        "simulator_futures": simulator_futures,
        "model_samples": models,
        "simulator_object_positions": physical_futures,
        "horizons": sorted(set([1, min(5, args.horizon), args.horizon])),
        "context_stage": ["pre_contact_prior_mixture"] * args.contexts,
        "metadata": {
            "task": "fetch_push_hidden_friction",
            "regime": "pre_contact_prior_mixture",
            "context_seeds": [value["seed"] for value in contexts],
            "friction_modes": list(modes),
            "frameskip": args.frameskip,
            "raw_action_horizon": args.frameskip * args.horizon,
            "nominal_checkpoint": str(args.nominal_checkpoint.resolve()),
            "checkpoints": {name: str(path.resolve()) for name, path in args.checkpoint},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(
        f"saved {args.contexts} contexts x {args.samples} samples x "
        f"{args.horizon} steps to {args.output}"
    )


if __name__ == "__main__":
    main()
