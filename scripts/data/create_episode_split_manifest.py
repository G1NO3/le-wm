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

from experiment_data import build_collection_manifest, save_or_verify_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3072)
    args = parser.parse_args()
    validation_path = args.input.with_suffix(".validation.json")
    if not validation_path.exists():
        raise SystemExit(f"Validate the collection first: missing {validation_path}")
    validation = json.loads(validation_path.read_text())
    with h5py.File(args.input, "r") as dataset:
        lengths = np.asarray(dataset["ep_len"], dtype=np.int64)
    if len(lengths) != validation["episodes"]:
        raise RuntimeError("Validation and HDF5 episode counts differ")
    manifest = build_collection_manifest(
        lengths, validation["dataset_sha256"], seed=args.seed
    )
    save_or_verify_manifest(args.output, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
