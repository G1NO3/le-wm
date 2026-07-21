#!/usr/bin/env python
"""Diagnose the residual GRU memory as a hidden-mode filter over long replays.

The memory head trains on at most ``history_size - 1`` teacher-forced updates
per window, but deployment replays entire episodes (hundreds of model steps)
into the same GRU. This diagnostic replays held-out episodes step by step,
exactly as ``ResidualWorldModelPolicy`` does, and reports as a function of the
number of observed transitions:

- linear-probe accuracy from the memory state to the hidden physics mode,
- memory-state norm and unit-saturation statistics,
- residual energy score with the accumulated memory versus a reset memory,
  with a paired bootstrap interval on the per-episode difference.

A stable filter should sharpen mode identification and keep (or improve) the
memory-vs-reset energy skill as updates accumulate. Degradation past the
trained update count is the preregistered trigger for longer-horizon
(burn-in/TBPTT) memory training before any control experiment.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import h5py
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf

from experiment_data import episode_disjoint_split, sha256_file
from physical_probes import RidgeProbe
from stochastic_metrics import energy_score, energy_skill_score, paired_bootstrap_interval
from utils import get_column_normalizer, get_img_preprocessor


def parse_checkpoint_grid(text):
    """Parse an update-count grid like ``1,2,5,10`` into sorted unique ints."""

    values = sorted({int(part) for part in str(text).split(",") if part.strip()})
    if not values or values[0] < 1:
        raise ValueError("Update-count checkpoints must be positive integers")
    return values


def axis_thresholds(modes):
    """Midpoint decision threshold per physics axis from observed multipliers."""

    modes = torch.as_tensor(modes, dtype=torch.float32)
    if modes.ndim != 2 or modes.size(-1) != 3:
        raise ValueError("Physics modes must be [episode, 3] multipliers")
    thresholds = []
    for axis in range(modes.size(-1)):
        unique = modes[:, axis].unique()
        if unique.numel() < 2:
            raise ValueError(
                f"Physics axis {axis} has a single observed value; cannot label modes"
            )
        thresholds.append(float((unique.min() + unique.max()) / 2))
    return thresholds


def mode_bits(modes, thresholds):
    """Binary mode label per axis: multiplier above the axis threshold."""

    modes = torch.as_tensor(modes, dtype=torch.float32)
    thresholds = torch.as_tensor(thresholds, dtype=torch.float32)
    return modes > thresholds


def split_probe_episodes(episode_ids, seed):
    """Deterministic disjoint fit/eval halves for the post-hoc probe."""

    episode_ids = sorted(int(episode) for episode in episode_ids)
    generator = np.random.default_rng(seed)
    order = generator.permutation(len(episode_ids))
    half = len(episode_ids) // 2
    fit = sorted(episode_ids[index] for index in order[:half])
    evaluate = sorted(episode_ids[index] for index in order[half:])
    return fit, evaluate


def fit_mode_probe(states, bits, regularization):
    """Closed-form ridge from memory states to signed mode bits."""

    targets = torch.as_tensor(bits).float() * 2.0 - 1.0
    return RidgeProbe(regularization).fit(torch.as_tensor(states).float(), targets)


def probe_accuracy(probe, states, bits):
    """Per-axis and exact-mode accuracy of a fitted bit probe."""

    bits = torch.as_tensor(bits).bool()
    predicted = probe(torch.as_tensor(states).float()) > 0
    per_axis = (predicted == bits).float().mean(dim=0)
    exact = (predicted == bits).all(dim=-1).float().mean()
    return {
        "axis_accuracy": [float(value) for value in per_axis],
        "exact_accuracy": float(exact),
    }


def majority_exact_accuracy(fit_bits, eval_bits):
    """Exact-mode accuracy of predicting the most frequent fit-split mode."""

    fit_bits = torch.as_tensor(fit_bits).bool()
    eval_bits = torch.as_tensor(eval_bits).bool()
    classes = (fit_bits.long() * torch.tensor([4, 2, 1])).sum(dim=-1)
    majority = classes.bincount(minlength=8).argmax()
    eval_classes = (eval_bits.long() * torch.tensor([4, 2, 1])).sum(dim=-1)
    return float((eval_classes == majority).float().mean())


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="GRU-flow object checkpoint")
    parser.add_argument("--config", default=None, help="Defaults to config.yaml beside it")
    parser.add_argument("--output", type=Path, required=True, help="JSON output path")
    parser.add_argument(
        "--state-output",
        type=Path,
        default=None,
        help="Optional torch payload with raw memory states for figures",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-episodes", type=int, default=100)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--flow-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--probe-ridge", type=float, default=1e-2)
    parser.add_argument(
        "--update-checkpoints",
        default="1,2,3,5,8,12,20,30,50,75,100,150,200",
        help="Comma-separated update counts at which to evaluate the memory",
    )
    return parser.parse_args()


def cache_dir():
    return Path(os.environ.get("STABLEWM_HOME", swm.data.utils.get_cache_dir()))


def resolve_path(path_like):
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path
    for candidate in (Path.cwd() / path, cache_dir() / path):
        if candidate.exists():
            return candidate
    return Path.cwd() / path


def load_model_and_config(args, device):
    checkpoint = resolve_path(args.checkpoint)
    config_path = (
        resolve_path(args.config) if args.config else checkpoint.parent / "config.yaml"
    )
    if not config_path.exists():
        raise FileNotFoundError(f"Training config not found: {config_path}")
    cfg = OmegaConf.load(config_path)
    model = torch.load(checkpoint, map_location=device, weights_only=False)
    model = model.to(device).eval()
    if getattr(model, "residual_memory", None) is None:
        raise SystemExit("The checkpoint has no residual memory; nothing to diagnose")
    model.freeze_residual_scale()
    return model, cfg, checkpoint, config_path


def build_val_dataset(cfg, checkpoint):
    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    transforms = [
        get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.get("img_size", 224))
    ]
    for column in cfg.data.dataset.keys_to_load:
        if not column.startswith("pixels"):
            transforms.append(get_column_normalizer(dataset, column, column))
    dataset.transform = spt.data.transforms.Compose(*transforms)
    manifest_path = cfg.data.get("split_manifest") or checkpoint.parent / "split_manifest.json"
    if not Path(manifest_path).exists():
        raise FileNotFoundError(f"Episode split manifest required: {manifest_path}")
    _, manifest = episode_disjoint_split(
        dataset,
        manifest_path,
        seed=cfg.seed,
        fractions=tuple(cfg.data.get("split_fractions", [0.8, 0.1, 0.1])),
    )
    return dataset, manifest


def episode_clip_order(dataset, episodes):
    """Ordered clip indices per requested episode, following dataset order."""

    grouped = defaultdict(list)
    for clip_index, (episode, offset) in enumerate(dataset.clip_indices):
        if int(episode) in episodes:
            grouped[int(episode)].append((int(offset), clip_index))
    ordered = {}
    for episode, entries in grouped.items():
        entries.sort()
        offsets = [offset for offset, _ in entries]
        ordered[episode] = {
            "clips": [clip for _, clip in entries],
            "stride": int(offsets[1] - offsets[0]) if len(offsets) > 1 else None,
        }
    return ordered


def episode_physics_modes(dataset, episodes):
    """Episode-constant hidden physics multipliers from the raw collection."""

    h5_path = getattr(dataset, "h5_path", None)
    if h5_path is None:
        raise SystemExit("Dataset does not expose h5_path; cannot read physics modes")
    with h5py.File(h5_path, "r") as handle:
        if "privileged/physics_mode" not in handle:
            raise SystemExit(
                "Collection lacks privileged/physics_mode; recollect with hidden physics"
            )
        lengths = np.asarray(handle["ep_len"])
        starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
        modes = {
            episode: np.asarray(handle["privileged/physics_mode"][int(starts[episode])])
            for episode in episodes
        }
    return modes


@torch.no_grad()
def replay_episode(model, dataset, clips, cfg, args, device, generator):
    """Replay one episode's windows into the memory; score at checkpoint depths.

    Returns per-checkpoint memory states and (memory, reset) energy scores.
    """

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    checkpoints = parse_checkpoint_grid(args.update_checkpoints)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, clips),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
    )

    conditions, residuals = [], []
    for batch in loader:
        batch["action"] = torch.nan_to_num(batch["action"], 0.0)
        batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        output = model.encode(batch)
        emb, act_emb = output["emb"], output["act_emb"]
        ctx_emb, ctx_act = emb[:, :ctx_len], act_emb[:, :ctx_len]
        pred_emb = model.predict(ctx_emb, ctx_act)
        base_condition = model.residual_condition(ctx_emb, ctx_act, pred_emb)
        target = emb[:, n_preds:][:, -1]
        normalized = model.normalize_residual(target - pred_emb[:, -1])
        conditions.append(base_condition[:, -1])
        residuals.append(normalized)
    conditions = torch.cat(conditions, dim=0)
    residuals = torch.cat(residuals, dim=0)

    memory = model.init_residual_memory((1,), device=device, dtype=conditions.dtype)
    reset = torch.zeros_like(memory)
    states, scores = {}, {}
    for step in range(conditions.size(0)):
        if step + 1 > checkpoints[-1]:
            break
        memory = model.update_residual_memory(
            memory, conditions[step : step + 1], residuals[step : step + 1]
        )
        updates = step + 1
        if updates not in checkpoints or step + 1 >= conditions.size(0):
            continue
        states[updates] = memory.squeeze(0).detach().cpu()
        condition = conditions[step + 1 : step + 2]
        observation = residuals[step + 1 : step + 2]
        pair = {}
        for name, state in (("memory", memory), ("reset", reset)):
            expanded_condition = condition.expand(args.num_samples, -1)
            expanded_memory = state.expand(args.num_samples, -1)
            noise = torch.randn(
                args.num_samples,
                observation.size(-1),
                device=device,
                dtype=observation.dtype,
                generator=generator,
            )
            sampled = model.sample_residual(
                torch.zeros_like(noise),
                expanded_condition,
                steps=args.flow_steps,
                noise=noise,
                memory=expanded_memory,
            )
            normalized = model.normalize_residual(sampled)
            pair[name] = float(
                energy_score(normalized[None], observation[None, None]).squeeze(0)
            )
        scores[updates] = pair
    return states, scores


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    model, cfg, checkpoint, config_path = load_model_and_config(args, device)
    dataset, manifest = build_val_dataset(cfg, checkpoint)
    val_episodes = sorted(manifest["episodes"]["val"])[: args.max_episodes]
    clip_order = episode_clip_order(dataset, set(val_episodes))
    val_episodes = [episode for episode in val_episodes if episode in clip_order]
    modes = episode_physics_modes(dataset, val_episodes)

    mode_matrix = torch.as_tensor(
        np.stack([modes[episode] for episode in val_episodes]), dtype=torch.float32
    )
    thresholds = axis_thresholds(mode_matrix)
    bits_by_episode = {
        episode: mode_bits(mode_matrix[index : index + 1], thresholds).squeeze(0)
        for index, episode in enumerate(val_episodes)
    }

    states_by_checkpoint = defaultdict(dict)
    scores_by_checkpoint = defaultdict(dict)
    strides = set()
    for episode in val_episodes:
        entry = clip_order[episode]
        if len(entry["clips"]) < 2:
            continue
        if entry["stride"] is not None:
            strides.add(entry["stride"])
        states, scores = replay_episode(
            model, dataset, entry["clips"], cfg, args, device, generator
        )
        for updates, state in states.items():
            states_by_checkpoint[updates][episode] = state
        for updates, pair in scores.items():
            scores_by_checkpoint[updates][episode] = pair

    fit_episodes, eval_episodes = split_probe_episodes(val_episodes, args.seed)
    fit_states, fit_bits = [], []
    for updates, per_episode in states_by_checkpoint.items():
        for episode, state in per_episode.items():
            if episode in set(fit_episodes):
                fit_states.append(state)
                fit_bits.append(bits_by_episode[episode])
    if not fit_states:
        raise SystemExit("No memory states were collected; check the episode split")
    probe = fit_mode_probe(torch.stack(fit_states), torch.stack(fit_bits), args.probe_ridge)

    curves = {}
    bootstrap_generator = torch.Generator().manual_seed(args.seed)
    for updates in sorted(states_by_checkpoint):
        per_episode = states_by_checkpoint[updates]
        eval_ids = [episode for episode in eval_episodes if episode in per_episode]
        entry = {"episodes": len(per_episode)}
        stacked = torch.stack([per_episode[episode] for episode in sorted(per_episode)])
        norms = stacked.norm(dim=-1)
        entry["memory_norm_mean"] = float(norms.mean())
        entry["memory_norm_max"] = float(norms.max())
        entry["saturated_unit_fraction"] = float((stacked.abs() > 0.95).float().mean())
        if eval_ids:
            eval_states = torch.stack([per_episode[episode] for episode in eval_ids])
            eval_bits = torch.stack([bits_by_episode[episode] for episode in eval_ids])
            entry["probe"] = probe_accuracy(probe, eval_states, eval_bits)
            entry["probe"]["majority_exact_accuracy"] = majority_exact_accuracy(
                torch.stack([bits_by_episode[episode] for episode in fit_episodes]),
                eval_bits,
            )
        pairs = scores_by_checkpoint.get(updates, {})
        if pairs:
            memory_scores = torch.tensor([pairs[episode]["memory"] for episode in sorted(pairs)])
            reset_scores = torch.tensor([pairs[episode]["reset"] for episode in sorted(pairs)])
            entry["energy_score_memory"] = float(memory_scores.mean())
            entry["energy_score_reset"] = float(reset_scores.mean())
            entry["memory_energy_skill_vs_reset"] = float(
                1.0 - memory_scores.sum() / reset_scores.sum().clamp_min(1e-8)
            )
            interval = paired_bootstrap_interval(
                reset_scores - memory_scores, generator=bootstrap_generator
            )
            entry["paired_delta_ci95"] = [float(interval[0]), float(interval[1])]
        curves[str(updates)] = entry

    result = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "config": str(config_path.resolve()),
        "dataset_sha256": manifest["dataset_sha256"],
        "split_sha256": manifest["sha256"],
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "seed": args.seed,
        "num_samples": args.num_samples,
        "flow_steps": args.flow_steps,
        "probe_ridge": args.probe_ridge,
        "probe_fit_episodes": fit_episodes,
        "probe_eval_episodes": eval_episodes,
        "trained_max_history_updates": int(
            cfg.loss.residual_flow.get("memory", {}).get("max_history_updates", 5)
        ),
        "window_history_size": int(cfg.wm.history_size),
        "window_stride": sorted(strides),
        "axis_thresholds": thresholds,
        "memory_hidden_dim": int(model.residual_memory.hidden_dim),
        "curves": curves,
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text + "\n")
    if args.state_output is not None:
        payload = {
            "states": {
                str(updates): {
                    str(episode): state for episode, state in per_episode.items()
                }
                for updates, per_episode in states_by_checkpoint.items()
            },
            "mode_bits": {
                str(episode): bits for episode, bits in bits_by_episode.items()
            },
            "axis_thresholds": thresholds,
            "checkpoint_sha256": result["checkpoint_sha256"],
            "dataset_sha256": result["dataset_sha256"],
        }
        args.state_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, args.state_output)


if __name__ == "__main__":
    main()
