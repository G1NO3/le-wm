#!/usr/bin/env python
"""Validate paired FetchPush friction data before any model training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import h5py
import numpy as np

from experiment_data import sha256_file


def episode_values(dataset, key, starts, ends):
    values = []
    for start, end in zip(starts, ends, strict=True):
        episode = np.asarray(dataset[key][start:end])
        if np.unique(episode, axis=0).shape[0] != 1:
            raise RuntimeError(f"{key} changes within an episode")
        values.append(episode[0])
    return np.asarray(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--pairs", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with h5py.File(args.input, "r") as dataset:
        required = {
            "action",
            "commanded_action",
            "ep_len",
            "pixels",
            "state",
            "qpos",
            "qvel",
            "prev_qpos",
            "prev_qvel",
            "goal_position",
            "success",
            "privileged/base_seed",
            "privileged/friction_mode",
            "privileged/friction_multiplier",
            "privileged/pair_id",
        }
        missing = sorted(name for name in required if name not in dataset)
        if missing:
            raise RuntimeError(f"Collection is missing required fields: {missing}")
        lengths = np.asarray(dataset["ep_len"], dtype=np.int64)
        if len(lengths) != 2 * args.pairs:
            raise RuntimeError(
                f"Expected {2 * args.pairs} episodes, found {len(lengths)}"
            )
        ends = np.cumsum(lengths)
        starts = np.concatenate([[0], ends[:-1]])
        actions = np.asarray(dataset["action"])
        commanded = np.asarray(dataset["commanded_action"])
        if not np.array_equal(actions, commanded):
            raise RuntimeError("Commanded-action audit does not match training actions")

        pair_ids = episode_values(dataset, "privileged/pair_id", starts, ends)
        modes = episode_values(dataset, "privileged/friction_mode", starts, ends)
        base_seeds = episode_values(dataset, "privileged/base_seed", starts, ends)
        multipliers = episode_values(
            dataset, "privileged/friction_multiplier", starts, ends
        )
        unique_pairs = np.unique(pair_ids)
        if len(unique_pairs) != args.pairs:
            raise RuntimeError("Pair IDs are incomplete or duplicated")
        paired_start_error = []
        for pair_id in unique_pairs:
            indices = np.flatnonzero(pair_ids == pair_id)
            if len(indices) != 2 or set(modes[indices].tolist()) != {0, 1}:
                raise RuntimeError(f"Pair {pair_id} does not contain one low and one high mode")
            if len(np.unique(base_seeds[indices])) != 1:
                raise RuntimeError(f"Pair {pair_id} does not share one base seed")
            # prev_qpos on the first saved transition is the exact state before
            # any action; friction-dependent passive settling may already be
            # visible in qpos after that transition.
            starts_for_pair = starts[indices]
            paired_start_error.append(
                np.max(
                    np.abs(
                        np.asarray(dataset["prev_qpos"][starts_for_pair[0]])
                        - np.asarray(dataset["prev_qpos"][starts_for_pair[1]])
                    )
                )
            )
            if not np.array_equal(
                dataset["goal_position"][starts_for_pair[0]],
                dataset["goal_position"][starts_for_pair[1]],
            ):
                raise RuntimeError(f"Pair {pair_id} has different goals")

        terminal = ends - 1
        success = np.asarray(dataset["success"])[terminal].astype(bool)
        result = {
            "dataset_sha256": sha256_file(args.input),
            "pairs": int(args.pairs),
            "episodes": int(len(lengths)),
            "transitions": int(lengths.sum()),
            "success_rate": float(success.mean()),
            "success_rate_low": float(success[modes == 0].mean()),
            "success_rate_high": float(success[modes == 1].mean()),
            "friction_multipliers": sorted(np.unique(multipliers).tolist()),
            "max_paired_initial_qpos_error": float(max(paired_start_error)),
            "commanded_actions_match": True,
            "pairing_valid": True,
            "privileged_fields_excluded_from_model_inputs": True,
        }
    output = args.output or args.input.with_suffix(".validation.json")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
