#!/usr/bin/env python
"""Compute one immutable residual scale from a frozen nominal LeWM checkpoint."""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf

from experiment_data import episode_disjoint_split, sha256_file, sha256_json
from utils import get_column_normalizer, get_img_preprocessor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite immutable scale: {args.output}")

    cfg = OmegaConf.load(args.config)
    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    transforms = [get_img_preprocessor("pixels", "pixels", cfg.img_size)]
    for column in cfg.data.dataset.keys_to_load:
        if not column.startswith("pixels"):
            transforms.append(get_column_normalizer(dataset, column, column))
    dataset.transform = spt.data.transforms.Compose(*transforms)
    subsets, manifest = episode_disjoint_split(
        dataset,
        args.split_manifest,
        seed=cfg.seed,
        fractions=tuple(cfg.data.get("split_fractions", [0.8, 0.1, 0.1])),
    )
    loader = torch.utils.data.DataLoader(
        subsets["train"],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = torch.load(args.checkpoint, map_location=device, weights_only=False).to(device).eval()
    count = 0
    total = None
    square_total = None
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
            batch["action"] = torch.nan_to_num(batch["action"], 0)
            encoded = model.encode(batch)
            context = encoded["emb"][:, : cfg.wm.history_size]
            actions = encoded["act_emb"][:, : cfg.wm.history_size]
            target = encoded["emb"][:, cfg.wm.num_preds :]
            prediction = model.predict(context, actions)
            residual = (target[:, -1] - prediction[:, -1]).double()
            count += residual.size(0)
            total = residual.sum(0) if total is None else total + residual.sum(0)
            square_total = residual.square().sum(0) if square_total is None else square_total + residual.square().sum(0)
    mean = (total / count).float()
    scale = (square_total / count - mean.double().square()).clamp_min(0).sqrt().float()
    payload = {
        "version": 2,
        "residual_mean": mean.cpu(),
        "residual_scale": scale.cpu(),
        "count": count,
        "split_sha256": manifest["sha256"],
        "dataset_sha256": manifest["dataset_sha256"],
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
    }
    payload["mean_sha256"] = sha256_json(mean.tolist())
    payload["scale_sha256"] = sha256_json(scale.tolist())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    metadata = {
        key: value
        for key, value in payload.items()
        if key not in {"residual_mean", "residual_scale"}
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
