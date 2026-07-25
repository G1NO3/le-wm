#!/usr/bin/env python
"""Train Fetch nominal/flow dynamics and diagnostic supervised mode heads."""

from __future__ import annotations

import argparse
import copy
import json
import platform
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from experiment_data import sha256_file, sha256_json
from state_residual_dynamics import FetchStateResidualDynamics


def transition_rows(lengths, episodes, action_block=2):
    ends = np.cumsum(np.asarray(lengths, dtype=np.int64))
    starts = np.concatenate([[0], ends[:-1]])
    rows = []
    for episode in episodes:
        start, end = int(starts[episode]), int(ends[episode])
        rows.extend(range(start, end - action_block))
    return np.asarray(rows, dtype=np.int64)


def load_transitions(dataset_path, manifest_path):
    manifest = json.loads(manifest_path.read_text())
    with h5py.File(dataset_path, "r") as dataset:
        lengths = np.asarray(dataset["ep_len"], dtype=np.int64)
        state = np.asarray(dataset["state"], dtype=np.float32)
        action = np.asarray(dataset["action"], dtype=np.float32)
        friction_mode = np.asarray(
            dataset["privileged/friction_mode"], dtype=np.int64
        ).reshape(-1)
    result = {}
    for split, episodes in manifest["episodes"].items():
        rows = transition_rows(lengths, episodes)
        action_block = np.concatenate([action[rows + 1], action[rows + 2]], axis=-1)
        next_state = state[rows + 2]
        result[split] = (
            state[rows],
            action_block,
            next_state[:, :25] - state[rows, :25],
            friction_mode[rows],
        )
    return result, state, action, manifest


def stats(values, minimum=1e-6):
    return values.mean(axis=0), np.maximum(values.std(axis=0), minimum)


def loader(split, batch_size, shuffle):
    tensors = [torch.from_numpy(value) for value in split]
    return DataLoader(
        TensorDataset(*tensors),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )


def nominal_epoch(model, batches, optimizer, device):
    training = optimizer is not None
    model.train(training)
    total = count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in batches:
            state, action, delta = batch[:3]
            state, action, delta = (x.to(device, non_blocking=True) for x in (state, action, delta))
            prediction = model.nominal_delta(state, action)
            loss = ((prediction - delta) / model.delta_scale).square().mean()
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total += float(loss) * len(state)
            count += len(state)
    return total / count


def flow_epoch(model, batches, optimizer, device, seed=None):
    training = optimizer is not None
    model.train(training)
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    total = count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in batches:
            state, action, delta = batch[:3]
            state, action, delta = (x.to(device, non_blocking=True) for x in (state, action, delta))
            loss = model.flow_matching_loss(state, action, delta)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total += float(loss) * len(state)
            count += len(state)
    return total / count


def mode_epoch(model, batches, optimizer, device):
    training = optimizer is not None
    model.train(training)
    total = count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for state, action, delta, mode in batches:
            state, action, delta, mode = (
                x.to(device, non_blocking=True)
                for x in (state, action, delta, mode)
            )
            loss = model.mode_residual_loss(state, action, delta, mode)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total += float(loss) * len(state)
            count += len(state)
    return total / count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Object checkpoint ending in _object.ckpt")
    parser.add_argument("--seed", type=int, default=23041)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--nominal-epochs", type=int, default=80)
    parser.add_argument("--flow-epochs", type=int, default=80)
    parser.add_argument("--mode-epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".metadata.json").exists():
        raise SystemExit(f"Refusing to overwrite immutable output: {args.output}")
    if not args.output.name.endswith("_object.ckpt"):
        raise SystemExit("--output must end in _object.ckpt for AutoCostModel")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    splits, all_state, all_action, manifest = load_transitions(args.dataset, args.split_manifest)
    train_state, _, train_delta, _ = splits["train"]
    state_mean, state_scale = stats(train_state)
    action_mean, action_scale = stats(all_action)
    delta_mean, delta_scale = stats(train_delta)
    model = FetchStateResidualDynamics(
        state_mean=state_mean,
        state_scale=state_scale,
        action_mean=action_mean,
        action_scale=action_scale,
        delta_mean=delta_mean,
        delta_scale=delta_scale,
        residual_scale=np.ones(25, dtype=np.float32),
    ).to(device)
    train_loader = loader(splits["train"], args.batch_size, True)
    val_loader = loader(splits["val"], args.batch_size, False)

    optimizer = torch.optim.AdamW(model.nominal.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    best_loss, best_state = float("inf"), None
    history = {"nominal": [], "flow": [], "mode": []}
    for epoch in range(1, args.nominal_epochs + 1):
        train_loss = nominal_epoch(model, train_loader, optimizer, device)
        val_loss = nominal_epoch(model, val_loader, None, device)
        history["nominal"].append({"epoch": epoch, "train": train_loss, "val": val_loss})
        if val_loss < best_loss:
            best_loss, best_state = val_loss, copy.deepcopy(model.nominal.state_dict())
        print(f"nominal epoch={epoch} train={train_loss:.7f} val={val_loss:.7f}", flush=True)
    model.nominal.load_state_dict(best_state)
    model.nominal.requires_grad_(False)

    with torch.no_grad():
        residuals = []
        for batch in loader(splits["train"], args.batch_size, False):
            state, action, delta = batch[:3]
            state, action, delta = (x.to(device, non_blocking=True) for x in (state, action, delta))
            residuals.append((delta - model.nominal_delta(state, action)).cpu())
        residual_scale = torch.cat(residuals).std(dim=0).clamp_min(1e-4)
        model.residual_scale.copy_(residual_scale.to(device))

    optimizer = torch.optim.AdamW(model.residual_flow.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    best_loss, best_state = float("inf"), None
    for epoch in range(1, args.flow_epochs + 1):
        train_loss = flow_epoch(model, train_loader, optimizer, device)
        val_loss = flow_epoch(model, val_loader, None, device, seed=args.seed + 100000)
        history["flow"].append({"epoch": epoch, "train": train_loss, "val": val_loss})
        if val_loss < best_loss:
            best_loss, best_state = val_loss, copy.deepcopy(model.residual_flow.state_dict())
        print(f"flow epoch={epoch} train={train_loss:.7f} val={val_loss:.7f}", flush=True)
    model.residual_flow.load_state_dict(best_state)

    optimizer = torch.optim.AdamW(
        model.mode_residual_heads.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-5,
    )
    best_loss, best_state = float("inf"), None
    for epoch in range(1, args.mode_epochs + 1):
        train_loss = mode_epoch(model, train_loader, optimizer, device)
        val_loss = mode_epoch(model, val_loader, None, device)
        history["mode"].append(
            {"epoch": epoch, "train": train_loss, "val": val_loss}
        )
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.mode_residual_heads.state_dict())
        print(
            f"mode epoch={epoch} train={train_loss:.7f} val={val_loss:.7f}",
            flush=True,
        )
    model.mode_residual_heads.load_state_dict(best_state)
    model.eval().cpu()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, args.output)
    metadata = {
        "model": "FetchStateResidualDynamics",
        "seed": args.seed,
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256_file(args.dataset),
        "split_manifest": str(args.split_manifest.resolve()),
        "split_manifest_sha256": sha256_json({k: v for k, v in manifest.items() if k != "sha256"}),
        "transitions": {key: len(value[0]) for key, value in splits.items()},
        "action_alignment": "state[t] + action[t+1:t+3] -> state[t+2]",
        "nominal_best_val": min(row["val"] for row in history["nominal"]),
        "flow_best_val": min(row["val"] for row in history["flow"]),
        "mode_best_val": min(row["val"] for row in history["mode"]),
        "history": history,
        "host": platform.node(),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "output": str(args.output.resolve()),
    }
    args.output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in metadata.items() if k != "history"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
