#!/usr/bin/env python
"""Publish validated immutable experiment artifacts to a HF dataset repo."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from huggingface_hub import CommitOperationAdd, HfApi

from experiment_data import sha256_file


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=os.environ.get("HF_DATASET_REPO"))
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--private", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("artifacts", type=Path, nargs="+")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.repo:
        raise SystemExit("Pass --repo or set HF_DATASET_REPO")
    artifacts = [path.resolve() for path in args.artifacts]
    missing = [str(path) for path in artifacts if not path.is_file()]
    if missing:
        raise SystemExit(f"Missing artifacts: {missing}")
    h5_files = [path for path in artifacts if path.suffix == ".h5"]
    for dataset in h5_files:
        manifest = dataset.with_suffix(".validation.json")
        if manifest not in artifacts:
            raise RuntimeError(f"Upload must include validation manifest {manifest}")
        expected = json.loads(manifest.read_text())["dataset_sha256"]
        if sha256_file(dataset) != expected:
            raise RuntimeError(f"Hash mismatch for {dataset}")
    inventory = {
        path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in artifacts
    }
    if args.dry_run:
        print(json.dumps({"repo": args.repo, "prefix": args.prefix, "files": inventory}, indent=2))
        return
    api = HfApi()
    api.create_repo(
        args.repo, repo_type="dataset", private=args.private, exist_ok=True
    )
    operations = [
        CommitOperationAdd(
            path_in_repo=f"{args.prefix.strip('/')}/{path.name}",
            path_or_fileobj=str(path),
        )
        for path in artifacts
    ]
    commit = api.create_commit(
        repo_id=args.repo,
        repo_type="dataset",
        operations=operations,
        commit_message=f"Add validated LeWM artifacts at {args.prefix}",
        commit_description=json.dumps(inventory, indent=2, sort_keys=True),
    )
    print(json.dumps({"repo": args.repo, "commit": commit.oid, "files": inventory}, indent=2))


if __name__ == "__main__":
    main()
