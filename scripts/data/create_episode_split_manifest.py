#!/usr/bin/env python
"""Create an immutable episode-disjoint split tied to a validated HDF5 hash."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import h5py
import numpy as np

from experiment_data import (
    build_collection_manifest,
    build_grouped_collection_manifest,
    save_or_verify_manifest,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument(
        "--group-key",
        help="Optional per-transition HDF5 field whose episode-constant values stay together.",
    )
    args = parser.parse_args()
    validation_path = args.input.with_suffix(".validation.json")
    if not validation_path.exists():
        raise SystemExit(f"Validate the collection first: missing {validation_path}")
    validation = json.loads(validation_path.read_text())
    with h5py.File(args.input, "r") as dataset:
        lengths = np.asarray(dataset["ep_len"], dtype=np.int64)
        group_ids = None
        if args.group_key:
            if args.group_key not in dataset:
                raise RuntimeError(f"Missing requested grouping field: {args.group_key}")
            ends = np.cumsum(lengths)
            starts = np.concatenate([[0], ends[:-1]])
            transition_groups = np.asarray(dataset[args.group_key])
            group_ids = []
            for start, end in zip(starts, ends, strict=True):
                values = np.unique(transition_groups[start:end])
                if len(values) != 1:
                    raise RuntimeError(
                        f"Grouping field {args.group_key} changes within an episode"
                    )
                group_ids.append(int(values[0]))
    if len(lengths) != validation["episodes"]:
        raise RuntimeError("Validation and HDF5 episode counts differ")
    if group_ids is None:
        manifest = build_collection_manifest(
            lengths, validation["dataset_sha256"], seed=args.seed
        )
    else:
        manifest = build_grouped_collection_manifest(
            lengths,
            validation["dataset_sha256"],
            group_ids,
            seed=args.seed,
            group_key=args.group_key,
        )
    save_or_verify_manifest(args.output, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
