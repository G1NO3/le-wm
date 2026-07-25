#!/usr/bin/env python
"""Diagnose frozen label-free episode modes against withheld friction labels.

This evaluator is strictly post-training: it reads the privileged label only
to determine what already-frozen head assignments represent. It reports
permutation-invariant assignment accuracy for full residuals and several
ordinary-observation object-motion views. No checkpoint is selected or changed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import h5py
import numpy as np
import torch

from scripts.train_fetch_episode_modes import load_unlabeled_transitions


OBJECT_DIMS = [3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def episode_truth(dataset_path):
    with h5py.File(dataset_path, "r") as dataset:
        lengths = np.asarray(dataset["ep_len"], dtype=np.int64)
        mode = np.asarray(
            dataset["privileged/friction_mode"], dtype=np.int64
        ).reshape(-1)
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    truth = mode[starts]
    for episode, (start, length) in enumerate(zip(starts, lengths, strict=True)):
        values = mode[start : start + length]
        if not np.all(values == truth[episode]):
            raise ValueError(f"Friction mode changes within episode {episode}")
    return truth


def assignment_metrics(model, transition_loss, episode_ids, truth_by_episode):
    unique, _, episode_loss, balanced = model.balanced_episode_assignments(
        transition_loss, episode_ids
    )
    natural = episode_loss.argmin(dim=-1)
    truth = torch.as_tensor(
        truth_by_episode[unique.cpu().numpy()], device=unique.device
    ).long()

    def summarize(assignment):
        direct = float((assignment == truth).float().mean())
        accuracy = max(direct, 1.0 - direct)
        margin = (episode_loss[:, 0] - episode_loss[:, 1]).abs()
        return {
            "permutation_invariant_accuracy": accuracy,
            "direct_accuracy": direct,
            "head_counts": torch.bincount(assignment, minlength=2).cpu().tolist(),
            "episodes": int(len(unique)),
            "mean_absolute_loss_margin": float(margin.mean()),
        }

    return {"natural": summarize(natural), "balanced": summarize(balanced)}


@torch.no_grad()
def diagnose_split(model, split, truth_by_episode, device):
    state, action, delta, episode_ids = (
        torch.from_numpy(value).to(device) for value in split
    )
    nominal = model.nominal_delta(state, action)
    target = (delta - nominal) / model._safe_scale(model.residual_scale)
    condition = model.residual_condition(state, action, nominal)
    predictions = torch.stack(
        [head(condition) for head in model.mode_residual_heads], dim=-2
    )
    squared = (predictions - target.unsqueeze(-2)).square()
    result = {
        "all_transitions_all_dims": assignment_metrics(
            model, squared.mean(dim=-1), episode_ids, truth_by_episode
        ),
        "all_transitions_object_dims": assignment_metrics(
            model,
            squared[..., OBJECT_DIMS].mean(dim=-1),
            episode_ids,
            truth_by_episode,
        ),
        "object_displacement_quantiles": [
            float(value)
            for value in torch.quantile(
                delta[:, 3:6].norm(dim=-1),
                torch.tensor([0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0], device=device),
            )
        ],
    }
    displacement = delta[:, 3:6].norm(dim=-1)
    for threshold in (1e-5, 1e-4, 5e-4, 1e-3):
        mask = displacement > threshold
        name = f"moving_object_dims_gt_{threshold:g}"
        result[name] = {
            "transitions": int(mask.sum()),
            **assignment_metrics(
                model,
                squared[mask][..., OBJECT_DIMS].mean(dim=-1),
                episode_ids[mask],
                truth_by_episode,
            ),
        }
    return result


def main():
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite assignment diagnostic: {args.output}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = torch.load(
        args.checkpoint, map_location=device, weights_only=False
    ).to(device).eval()
    splits, _ = load_unlabeled_transitions(args.dataset, args.split_manifest)
    truth = episode_truth(args.dataset)
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "diagnostic_only_after_checkpoint_freeze": True,
        "privileged_label_used_for_training_or_selection": False,
        "object_dims": OBJECT_DIMS,
        "splits": {
            name: diagnose_split(model, split, truth, device)
            for name, split in splits.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
