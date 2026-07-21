#!/usr/bin/env python
"""Validate an immutable stochastic OGBench collection before downstream use."""

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with h5py.File(args.input, "r") as dataset:
        required = {
            "action",
            "commanded_action",
            "ep_len",
            "success",
            "privileged/physics_mode",
            "privileged/slip_event",
        }
        missing = sorted(name for name in required if name not in dataset)
        if missing:
            raise RuntimeError(f"Collection is missing required fields: {missing}")
        episode_lengths = np.asarray(dataset["ep_len"])
        if len(episode_lengths) != args.episodes:
            raise RuntimeError(
                f"Expected {args.episodes} episodes, found {len(episode_lengths)}"
            )
        actions = np.asarray(dataset["action"])
        commanded = np.asarray(dataset["commanded_action"])
        if not np.array_equal(actions, commanded):
            raise RuntimeError("Commanded-action audit does not match model-facing actions")
        modes = np.asarray(dataset["privileged/physics_mode"])
        result = {
            "dataset_sha256": sha256_file(args.input),
            "episodes": int(len(episode_lengths)),
            "transitions": int(episode_lengths.sum()),
            "success_rate": float(np.asarray(dataset["success"])[
                np.cumsum(episode_lengths) - 1
            ].mean()),
            "unique_hidden_modes": int(np.unique(modes, axis=0).shape[0]),
            "slip_events": int(np.asarray(dataset["privileged/slip_event"]).sum()),
            "commanded_actions_match": True,
            "privileged_fields_excluded_from_model_inputs": True,
        }
        if "privileged/contact_index" in dataset:
            contacts = np.asarray(dataset["privileged/contact_index"])
            result["contact_onsets"] = int(
                sum(
                    np.diff(
                        np.concatenate([[0], contacts[start:end]])
                    ).clip(min=0).sum()
                    for start, end in zip(
                        np.concatenate([[0], np.cumsum(episode_lengths)[:-1]]),
                        np.cumsum(episode_lengths),
                        strict=True,
                    )
                )
            )
    output = args.output or args.input.with_suffix(".validation.json")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
