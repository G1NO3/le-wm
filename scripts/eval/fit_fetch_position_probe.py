#!/usr/bin/env python
"""Fit a train-only frozen ridge probe from nominal latents to Fetch object XYZ."""

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

from experiment_data import sha256_file
from physical_probes import RidgeProbe
from scripts.eval.generate_fetch_push_fork_samples import preprocess_images


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-train", type=int, default=20000)
    parser.add_argument("--max-val", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--regularization", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=53072)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def episode_rows(offsets, lengths, episodes):
    return np.concatenate(
        [np.arange(offsets[index], offsets[index] + lengths[index]) for index in episodes]
    )


@torch.no_grad()
def encode_rows(model, dataset, rows, *, batch_size, device):
    latents, positions = [], []
    for start in range(0, len(rows), batch_size):
        selected = np.sort(rows[start : start + batch_size])
        images = np.asarray(dataset["pixels"][selected])
        pixels = preprocess_images(images, 224).to(device)[:, None]
        latents.append(model.encode({"pixels": pixels})["emb"][:, 0].cpu())
        positions.append(torch.from_numpy(np.asarray(dataset["object_position"][selected])))
    return torch.cat(latents), torch.cat(positions).float()


def error_summary(prediction, target):
    distance = (prediction - target).norm(dim=-1)
    return {
        "rmse_xyz": float((prediction - target).square().mean().sqrt()),
        "mean_distance": float(distance.mean()),
        "median_distance": float(distance.median()),
        "within_5cm": float((distance <= 0.05).float().mean()),
    }


def main():
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite probe: {args.output}")
    manifest = json.loads(args.split_manifest.read_text())
    generator = np.random.default_rng(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = torch.load(args.checkpoint, map_location=device, weights_only=False).to(device).eval()
    with h5py.File(args.dataset, "r") as dataset:
        lengths = np.asarray(dataset["ep_len"], dtype=np.int64)
        offsets = np.asarray(dataset["ep_offset"], dtype=np.int64)
        train_rows = episode_rows(offsets, lengths, manifest["episodes"]["train"])
        val_rows = episode_rows(offsets, lengths, manifest["episodes"]["val"])
        generator.shuffle(train_rows)
        generator.shuffle(val_rows)
        train_latent, train_position = encode_rows(
            model, dataset, train_rows[: args.max_train], batch_size=args.batch_size, device=device
        )
        val_latent, val_position = encode_rows(
            model, dataset, val_rows[: args.max_val], batch_size=args.batch_size, device=device
        )
    probe = RidgeProbe(args.regularization).fit(train_latent, train_position)
    metrics = {
        "train": error_summary(probe(train_latent), train_position),
        "val": error_summary(probe(val_latent), val_position),
    }
    payload = {
        "probes": {"object_position": probe.state_dict()},
        "metrics": metrics,
        "metadata": {
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "dataset": str(args.dataset.resolve()),
            "dataset_sha256": manifest["dataset_sha256"],
            "split_sha256": manifest["sha256"],
            "train_rows": len(train_latent),
            "val_rows": len(val_latent),
            "seed": args.seed,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps({"metrics": metrics, "metadata": payload["metadata"]}, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
