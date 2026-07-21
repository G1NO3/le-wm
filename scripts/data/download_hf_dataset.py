#!/usr/bin/env python
"""Download a pinned HF dataset revision and verify collection hashes."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from huggingface_hub import snapshot_download

from experiment_data import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=os.environ.get("HF_DATASET_REPO"))
    parser.add_argument("--revision", required=True)
    parser.add_argument("--include", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.repo:
        raise SystemExit("Pass --repo or set HF_DATASET_REPO")
    if args.revision in {"main", "master", "refs/heads/main"}:
        raise SystemExit("Use an immutable Hub commit or tag, not a moving branch")
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(
        snapshot_download(
            repo_id=args.repo,
            repo_type="dataset",
            revision=args.revision,
            allow_patterns=args.include,
            local_dir=args.output,
        )
    )
    datasets = list(root.rglob("*.h5"))
    if not datasets:
        raise RuntimeError("Pinned download did not contain an HDF5 collection")
    result = {}
    for dataset in datasets:
        manifest = dataset.with_suffix(".validation.json")
        if not manifest.exists():
            raise RuntimeError(f"Missing validation manifest for {dataset}")
        expected = json.loads(manifest.read_text())["dataset_sha256"]
        actual = sha256_file(dataset)
        if actual != expected:
            raise RuntimeError(f"Hash mismatch for {dataset}")
        result[str(dataset.relative_to(root))] = actual
    print(json.dumps({"revision": args.revision, "verified": result}, indent=2))


if __name__ == "__main__":
    main()
