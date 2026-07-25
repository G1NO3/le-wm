#!/usr/bin/env python
"""Compare deterministic and stochastic Fetch success on paired scenes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_summary(path):
    payload = json.loads(path.read_text())
    return payload, np.asarray(payload["metrics"]["episode_successes"], dtype=bool)


def load_summary_shards(paths):
    """Join immutable control shards while retaining their episode seeds."""

    payloads = []
    successes = []
    seeds = []
    for path in paths:
        payload, shard_successes = load_summary(path)
        shard_seeds = np.asarray(payload["metrics"]["seeds"], dtype=np.int64)
        if len(shard_successes) != len(shard_seeds):
            raise ValueError(f"Success/seed length mismatch in {path}")
        payloads.append(payload)
        successes.append(shard_successes)
        seeds.append(shard_seeds)
    joined_successes = np.concatenate(successes)
    joined_seeds = np.concatenate(seeds)
    if len(np.unique(joined_seeds)) != len(joined_seeds):
        raise ValueError("Control shards contain duplicate episode seeds")
    return payloads, joined_successes, joined_seeds


def scene_cluster_interval(deterministic, stochastic, seeds, *, seed=3072, resamples=20000):
    seeds = np.asarray(seeds, dtype=np.int64)
    if len(seeds) % 2 or np.any(seeds[1::2] != seeds[::2] + 1):
        raise ValueError("Expected consecutive low/high episode pairs")
    scene_delta = (stochastic.astype(float) - deterministic.astype(float)).reshape(-1, 2).mean(1)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(scene_delta), size=(resamples, len(scene_delta)))
    bootstrap = scene_delta[indices].mean(1)
    return scene_delta, np.quantile(bootstrap, [0.025, 0.975])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--deterministic", action="append", type=Path, required=True,
        help="Deterministic summary shard; repeat for disjoint seed blocks.",
    )
    parser.add_argument(
        "--stochastic", action="append", type=Path, required=True,
        help="Stochastic summary shard; repeat for disjoint seed blocks.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3072)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite comparison: {args.output}")
    _, det, det_seeds = load_summary_shards(args.deterministic)
    _, stoch, stoch_seeds = load_summary_shards(args.stochastic)
    if set(det_seeds.tolist()) != set(stoch_seeds.tolist()):
        raise ValueError("Control arms did not evaluate identical episode seeds")
    det_order = np.argsort(det_seeds)
    stoch_order = np.argsort(stoch_seeds)
    det, det_seeds = det[det_order], det_seeds[det_order]
    stoch, stoch_seeds = stoch[stoch_order], stoch_seeds[stoch_order]
    scene_delta, interval = scene_cluster_interval(
        det, stoch, det_seeds, seed=args.seed
    )
    result = {
        "deterministic_summary": [str(path.resolve()) for path in args.deterministic],
        "stochastic_summary": [str(path.resolve()) for path in args.stochastic],
        "episodes": int(len(det)),
        "paired_scenes": int(len(det) // 2),
        "deterministic_success_rate": float(det.mean()),
        "stochastic_success_rate": float(stoch.mean()),
        "absolute_success_improvement": float(stoch.mean() - det.mean()),
        "scene_cluster_bootstrap_ci95": [float(value) for value in interval],
        "stochastic_only_successes": int(np.sum(stoch & ~det)),
        "deterministic_only_successes": int(np.sum(det & ~stoch)),
        "shared_successes": int(np.sum(stoch & det)),
        "shared_failures": int(np.sum(~stoch & ~det)),
        "scenes_improved": int(np.sum(scene_delta > 0)),
        "scenes_harmed": int(np.sum(scene_delta < 0)),
        "strict_improvement_gate": bool(interval[0] > 0),
        "episode_seeds": det_seeds.tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
