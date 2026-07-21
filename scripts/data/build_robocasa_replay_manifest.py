#!/usr/bin/env python
"""Build an immutable, stage-annotated RoboCasa physics-replay manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


STAGE_PATTERNS = {
    "approach_grasp": ("approach", "grasp", "pick"),
    "lift": ("lift",),
    "transport": ("transport", "navigate", "move"),
    "cabinet_placement": ("place", "cabinet", "insert"),
}


def decode(values):
    return [x.decode() if isinstance(x, bytes) else str(x) for x in values]


def find_annotation(group, requested=None):
    candidates = [requested] if requested else []
    candidates += ["subtask_stage", "atomic_skill", "stage", "subtask_name"]
    for key in candidates:
        if key and key in group:
            return key, decode(group[key][:])
    raise KeyError(f"No per-frame subtask annotation in {group.name}")


def stage_windows(labels, window=25):
    result = {}
    lowered = [label.lower() for label in labels]
    for stage, patterns in STAGE_PATTERNS.items():
        indices = [i for i, label in enumerate(lowered) if any(x in label for x in patterns)]
        if indices:
            center = indices[len(indices) // 2]
            start = max(0, min(center - window // 2, len(labels) - window))
            result[stage] = {"start": start, "end": min(len(labels), start + window)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--demonstrations", type=int, default=200)
    parser.add_argument("--realizations", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--annotation-key")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite immutable manifest: {args.output}")
    rng = np.random.default_rng(args.seed)
    with h5py.File(args.input, "r") as handle:
        root = handle["data"] if "data" in handle else handle
        demo_names = sorted(name for name, value in root.items() if isinstance(value, h5py.Group))
        if len(demo_names) < args.demonstrations:
            raise SystemExit(f"Requested {args.demonstrations} demos, found {len(demo_names)}")
        selected = sorted(rng.choice(demo_names, args.demonstrations, replace=False).tolist())
        demos = []
        for demo_index, name in enumerate(selected):
            annotation_key, labels = find_annotation(root[name], args.annotation_key)
            windows = stage_windows(labels)
            if set(windows) != set(STAGE_PATTERNS):
                raise RuntimeError(f"{name} lacks all four required stages: {sorted(windows)}")
            demos.append(
                {
                    "demo": name,
                    "annotation_key": annotation_key,
                    "windows": windows,
                    "physics_seeds": [
                        args.seed + demo_index * args.realizations + i
                        for i in range(args.realizations)
                    ],
                }
            )
    manifest = {
        "version": 1,
        "task": "PickPlaceCounterToCabinet",
        "source": str(args.input.resolve()),
        "demonstrations": demos,
        "realizations_per_demo": args.realizations,
        "control": {"goal_offset": 25, "budget": 50},
    }
    manifest["sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
