#!/usr/bin/env python
"""Replay an OGBench collection and save compact exact-fork pilot evidence."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
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

from experiment_data import sha256_file
from scripts.data.collect_stochastic_ogbench import TASKS
from stochastic_physics import OGBenchHiddenPhysics, rollout_simulator_forks


def _episode_ranges(lengths):
    ends = np.cumsum(lengths)
    starts = np.concatenate([[0], ends[:-1]])
    return zip(starts, ends, strict=True)


def _choose_context(
    actions, start, end, horizon, contact_index=None, target_contact=None
):
    """Prefer a grasp onset, then a closing action, then the episode midpoint."""

    last = end - horizon
    if last < start:
        return None
    if contact_index is not None and target_contact is not None:
        stages = np.asarray(contact_index[start:end])
        previous = np.concatenate([[0], stages[:-1]])
        onset = (stages == target_contact) & (previous < target_contact)
        for relative in np.flatnonzero(onset):
            index = start + int(relative)
            if index <= last:
                return index
    gripper = actions[start:end, -1]
    closing = gripper <= -0.05
    onset = closing & ~np.concatenate([[False], closing[:-1]])
    for relative in np.flatnonzero(onset):
        index = start + int(relative)
        if index <= last:
            return index
    for relative in np.flatnonzero(closing):
        index = start + int(relative)
        if index <= last:
            return index
    return start + (last - start) // 2


def _physical_state(raw):
    data = raw._data
    return np.concatenate([data.qpos.copy(), data.qvel.copy()]).astype(np.float32)


def _stage_balanced_contexts(lengths, contact_index, horizon, count):
    """Select unique grasp-onset contexts, balancing later contacts when possible."""

    candidates = {}
    for episode_index, (start, end) in enumerate(_episode_ranges(lengths)):
        stages = np.asarray(contact_index[start:end])
        previous = np.concatenate([[0], stages[:-1]])
        for relative in np.flatnonzero(stages > previous):
            row = int(start + relative)
            if row + horizon <= end:
                stage = int(stages[relative])
                candidates.setdefault(stage, []).append((episode_index, row, stage))
    if not candidates:
        return []
    selected = []
    quota = max(1, count // len(candidates))
    remaining = []
    for stage in sorted(candidates):
        values = candidates[stage]
        take = min(quota, len(values))
        indices = np.linspace(0, len(values) - 1, take, dtype=int)
        chosen = {int(index) for index in indices}
        selected.extend(values[index] for index in sorted(chosen))
        remaining.extend(value for index, value in enumerate(values) if index not in chosen)
    if len(selected) < count and remaining:
        take = min(count - len(selected), len(remaining))
        indices = np.linspace(0, len(remaining) - 1, take, dtype=int)
        selected.extend(remaining[int(index)] for index in indices)
    return selected[:count]


def _restore_recorded_context(wrapper, dataset, row):
    state = wrapper.get_fork_state()
    state["qpos"] = np.asarray(dataset["prev_qpos"][row]).copy()
    state["qvel"] = np.asarray(dataset["prev_qvel"][row]).copy()
    state["mode"] = {
        key: float(value)
        for key, value in zip(
            (
                "mass_multiplier",
                "surface_friction_multiplier",
                "pad_friction_multiplier",
            ),
            np.asarray(dataset["privileged/physics_mode"][row]),
            strict=True,
        )
    }
    raw = wrapper.env.unwrapped
    state["environment"]["_prev_qpos"] = state["qpos"].copy()
    state["environment"]["_prev_qvel"] = state["qvel"].copy()
    if "control" in dataset and "ctrl" in state:
        control = np.asarray(dataset["control"][row])
        if control.shape == state["ctrl"].shape:
            state["ctrl"] = control.copy()
    wrapper.set_fork_state(state)
    # The risk reference does not affect dynamics, but keeping it at the fork
    # state avoids counting pre-context motion as collateral displacement.
    wrapper._initial_body_positions = wrapper._data.xpos[wrapper._cube_bodies].copy()
    return wrapper.get_fork_state()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--profile", choices=("strong", "medium", "mild"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contexts", type=int, default=64)
    parser.add_argument("--realizations", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--seed", type=int, default=9182)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite immutable evidence: {args.output}")
    validation_path = args.input.with_suffix(".validation.json")
    if not validation_path.exists():
        raise SystemExit(f"Validate the collection first: missing {validation_path}")
    validation = json.loads(validation_path.read_text())
    dataset_hash = sha256_file(args.input)
    if validation.get("dataset_sha256") != dataset_hash:
        raise RuntimeError("Collection hash does not match its validation manifest")

    task = TASKS[args.task]
    env = gym.make(
        "swm/OGBCube-v0",
        env_type=task["env_type"],
        reward_task_id=task["reward_task_id"],
        ob_type="states",
        terminate_at_goal=False,
        max_episode_steps=1000,
    )
    wrapper = OGBenchHiddenPhysics(env, profile=args.profile, seed=args.seed)
    futures, labels, modes, context_rows, contact_stages = [], [], [], [], []
    episode_success = None
    with h5py.File(args.input, "r") as dataset:
        required = {"action", "prev_qpos", "prev_qvel", "ep_len", "success"}
        missing = sorted(required - set(dataset.keys()))
        if missing:
            raise RuntimeError(f"Collection cannot be forked; missing {missing}")
        lengths = np.asarray(dataset["ep_len"])
        ends = np.cumsum(lengths) - 1
        episode_success = np.asarray(dataset["success"])[ends].astype(np.bool_)
        ranges = list(_episode_ranges(lengths))
        if not ranges:
            raise RuntimeError("Collection has no episodes")
        contact_column = dataset.get("privileged/contact_index")
        selected_contexts = (
            _stage_balanced_contexts(
                lengths, contact_column, args.horizon, args.contexts
            )
            if contact_column is not None
            else []
        )
        if not selected_contexts:
            selected_episodes = np.linspace(
                0, len(ranges) - 1, min(args.contexts, len(ranges)), dtype=int
            )
            selected_contexts = []
            for episode_index in selected_episodes:
                start, end = ranges[int(episode_index)]
                row = _choose_context(
                    dataset["action"], int(start), int(end), args.horizon
                )
                if row is not None:
                    selected_contexts.append((int(episode_index), row, 1))
        for episode_index, row, target_contact in selected_contexts:
            if row is None:
                continue
            wrapper.reset(seed=args.seed + int(episode_index))
            context = _restore_recorded_context(wrapper, dataset, row)
            action_window = np.asarray(dataset["action"][row : row + args.horizon])
            fork_futures, privileged = rollout_simulator_forks(
                wrapper,
                context,
                action_window,
                realizations=args.realizations,
                observe=_physical_state,
            )
            if any(len(future) != args.horizon for future in fork_futures):
                raise RuntimeError("A simulator fork terminated before the requested horizon")
            futures.extend(fork_futures)
            mode_array = np.asarray(
                [
                    [
                        item["physics_mode"]["mass_multiplier"],
                        item["physics_mode"]["surface_friction_multiplier"],
                        item["physics_mode"]["pad_friction_multiplier"],
                    ]
                    for item in privileged
                ],
                dtype=np.float32,
            )
            modes.extend(mode_array)
            labels.extend(mode_array[:, 0] > np.mean(wrapper.profile.mass))
            context_rows.extend([row] * args.realizations)
            observed_stage = (
                int(contact_column[row]) if contact_column is not None else target_contact
            )
            contact_stages.extend([observed_stage] * args.realizations)

    wrapper.close()
    if not futures or len(set(labels)) != 2:
        raise RuntimeError("Fork evidence requires futures from both mass-mode groups")
    payload = {
        "episode_success": torch.from_numpy(episode_success),
        "h5_physical_state": torch.from_numpy(np.stack(futures)),
        "mode_label": torch.as_tensor(labels, dtype=torch.bool),
        "physics_mode": torch.as_tensor(np.asarray(modes), dtype=torch.float32),
        "context_row": torch.as_tensor(context_rows, dtype=torch.int64),
        "contact_stage": torch.as_tensor(contact_stages, dtype=torch.int64),
        "dataset_sha256": dataset_hash,
        "task": args.task,
        "profile": args.profile,
        "horizon": args.horizon,
        "realizations": args.realizations,
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    result = {
        "dataset_sha256": dataset_hash,
        "evidence_sha256": sha256_file(args.output),
        "episodes": int(len(episode_success)),
        "contexts": int(len(futures) // args.realizations),
        "realizations": args.realizations,
        "horizon": args.horizon,
        "physical_state_dim": int(np.asarray(futures).shape[-1]),
        "contact_stages": sorted(set(int(value) for value in contact_stages)),
        "output_path": str(args.output.resolve()),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
