#!/usr/bin/env python
"""Train label-free, episode-consistent residual modes on frozen Fetch dynamics.

Only ``ep_len``, ordinary ``state``, and commanded ``action`` arrays are read
from the dataset. The supervised diagnostic heads in the base checkpoint are
discarded and reinitialized. A balanced complete-episode winner-take-all loss
then discovers two persistent residual outcomes without friction labels.
"""

from __future__ import annotations

import argparse
import copy
import json
import platform
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import h5py
import numpy as np
import torch
from torch import nn

from experiment_data import sha256_file, sha256_json


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=25041)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--assignment",
        choices=("balanced_episode", "paired_object_motion"),
        default="balanced_episode",
    )
    parser.add_argument(
        "--center-mode-residuals",
        action="store_true",
        help=(
            "Force the balanced residual outcomes to average to zero at every "
            "transition, preserving the frozen nominal as the mixture mean."
        ),
    )
    parser.add_argument("--object-motion-threshold", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_unlabeled_transitions(dataset_path, manifest_path, action_block=2):
    manifest = json.loads(manifest_path.read_text())
    # Deliberately do not access any privileged dataset key here.
    with h5py.File(dataset_path, "r") as dataset:
        lengths = np.asarray(dataset["ep_len"], dtype=np.int64)
        state = np.asarray(dataset["state"], dtype=np.float32)
        action = np.asarray(dataset["action"], dtype=np.float32)
    ends = np.cumsum(lengths)
    starts = np.concatenate([[0], ends[:-1]])
    result = {}
    for split, episodes in manifest["episodes"].items():
        rows = []
        episode_ids = []
        for episode in episodes:
            start, end = int(starts[episode]), int(ends[episode])
            episode_rows = np.arange(start, end - action_block, dtype=np.int64)
            rows.append(episode_rows)
            episode_ids.append(
                np.full(len(episode_rows), int(episode), dtype=np.int64)
            )
        rows = np.concatenate(rows)
        episode_ids = np.concatenate(episode_ids)
        action_blocks = np.concatenate(
            [action[rows + offset] for offset in range(1, action_block + 1)],
            axis=-1,
        )
        next_state = state[rows + action_block]
        result[split] = (
            state[rows],
            action_blocks,
            next_state[:, :25] - state[rows, :25],
            episode_ids,
        )
    return result, manifest


def reset_mode_heads(model):
    for module in model.mode_residual_heads.modules():
        if isinstance(module, (nn.Linear, nn.LayerNorm)):
            module.reset_parameters()


def to_device(split, device):
    return tuple(torch.from_numpy(value).to(device) for value in split)


def mode_loss(model, tensors, assignment, threshold, return_assignments=False):
    if assignment == "balanced_episode":
        return model.episode_wta_residual_loss(
            *tensors, return_assignments=return_assignments
        )
    if assignment == "paired_object_motion":
        return model.paired_object_motion_wta_residual_loss(
            *tensors,
            motion_threshold=threshold,
            return_assignments=return_assignments,
        )
    raise ValueError(f"Unknown assignment objective: {assignment}")


def assignment_summary(model, tensors, assignment, threshold):
    model.eval()
    with torch.no_grad():
        loss, unique_ids, episode_loss, head_assignment = mode_loss(
            model,
            tensors,
            assignment,
            threshold,
            return_assignments=True,
        )
    return {
        "loss": float(loss),
        "episodes": int(len(unique_ids)),
        "head_counts": torch.bincount(
            head_assignment, minlength=2
        ).cpu().tolist(),
        "mean_assigned_episode_loss": float(
            episode_loss.gather(-1, head_assignment[:, None]).mean()
        ),
    }


def main():
    args = parse_args()
    metadata_path = args.output.with_suffix(".metadata.json")
    if args.output.exists() or metadata_path.exists():
        raise SystemExit(f"Refusing to overwrite immutable output: {args.output}")
    if not args.output.name.endswith("_object.ckpt"):
        raise SystemExit("--output must end in _object.ckpt")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    splits, manifest = load_unlabeled_transitions(
        args.dataset, args.split_manifest
    )
    model = torch.load(
        args.base_checkpoint, map_location=device, weights_only=False
    ).to(device)
    model.requires_grad_(False)
    reset_mode_heads(model)
    model.center_mode_residuals = bool(args.center_mode_residuals)
    model.mode_residual_heads.requires_grad_(True)
    train_tensors = to_device(splits["train"], device)
    val_tensors = to_device(splits["val"], device)

    optimizer = torch.optim.AdamW(
        model.mode_residual_heads.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-5,
    )
    history = []
    best_loss = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss = mode_loss(
            model,
            train_tensors,
            args.assignment,
            args.object_motion_threshold,
        )
        train_loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = mode_loss(
                model,
                val_tensors,
                args.assignment,
                args.object_motion_threshold,
            )
        history.append(
            {
                "epoch": epoch,
                "train": float(train_loss.detach()),
                "val": float(val_loss),
            }
        )
        if float(val_loss) < best_loss:
            best_loss = float(val_loss)
            best_state = copy.deepcopy(model.mode_residual_heads.state_dict())
        print(
            f"episode-wta epoch={epoch} train={float(train_loss.detach()):.7f} "
            f"val={float(val_loss):.7f}",
            flush=True,
        )
    model.mode_residual_heads.load_state_dict(best_state)
    summaries = {
        "train": assignment_summary(
            model, train_tensors, args.assignment, args.object_motion_threshold
        ),
        "val": assignment_summary(
            model, val_tensors, args.assignment, args.object_motion_threshold
        ),
    }
    model.eval().cpu()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, args.output)
    metadata = {
        "model": "FetchStateResidualDynamics",
        "residual_mode_training": args.assignment,
        "uses_privileged_mode_labels": False,
        "dataset_keys_read": ["ep_len", "state", "action"],
        "balanced_component_prior": [0.5, 0.5],
        "center_mode_residuals": bool(args.center_mode_residuals),
        "mixture_mean": (
            "frozen_nominal"
            if args.center_mode_residuals
            else "frozen_nominal_plus_learned_mean_residual"
        ),
        "scene_pairing": "consecutive episode IDs grouped by floor(id / 2)",
        "object_motion_threshold": args.object_motion_threshold,
        "base_checkpoint": str(args.base_checkpoint.resolve()),
        "base_checkpoint_sha256": sha256_file(args.base_checkpoint),
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256_file(args.dataset),
        "split_manifest": str(args.split_manifest.resolve()),
        "split_manifest_sha256": sha256_json(
            {key: value for key, value in manifest.items() if key != "sha256"}
        ),
        "seed": args.seed,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "best_val_loss": best_loss,
        "assignment_summary": summaries,
        "transitions": {key: len(value[0]) for key, value in splits.items()},
        "action_alignment": "state[t] + action[t+1:t+3] -> state[t+2]",
        "history": history,
        "host": platform.node(),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "output": str(args.output.resolve()),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {key: value for key, value in metadata.items() if key != "history"},
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
