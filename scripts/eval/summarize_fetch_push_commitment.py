#!/usr/bin/env python
"""Combine immutable FetchPush commitment shards and recompute paired gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

BASE_CONSISTENT_FIELDS = (
    "task",
    "checkpoint",
    "friction_modes",
    "candidate_grid",
    "success_threshold",
    "success_temperature",
)


def paired_interval(deterministic, stochastic, draws, seed):
    deterministic = np.asarray(deterministic, dtype=float)
    stochastic = np.asarray(stochastic, dtype=float)
    scene_delta = stochastic.mean(axis=1) - deterministic.mean(axis=1)
    rng = np.random.default_rng(seed)
    # Preserve the exact ordinary scene bootstrap while avoiding a potentially
    # huge draws-by-scenes integer matrix on memory-capped login nodes.
    samples = np.empty(draws, dtype=np.float64)
    batch_size = 512
    for start in range(0, draws, batch_size):
        stop = min(start + batch_size, draws)
        indices = rng.integers(
            0, len(scene_delta), size=(stop - start, len(scene_delta))
        )
        samples[start:stop] = scene_delta[indices].mean(axis=1)
    return float(scene_delta.mean()), np.quantile(samples, [0.025, 0.975])


def summarize_pair(name, deterministic, stochastic, *, draws, seed):
    deterministic = np.asarray(deterministic, dtype=bool)
    stochastic = np.asarray(stochastic, dtype=bool)
    improvement, interval = paired_interval(
        deterministic, stochastic, draws=draws, seed=seed
    )
    return {
        "name": name,
        "deterministic_success_rate": float(deterministic.mean()),
        "stochastic_success_rate": float(stochastic.mean()),
        "deterministic_success_rate_by_friction_mode": [
            float(value) for value in deterministic.mean(axis=0)
        ],
        "stochastic_success_rate_by_friction_mode": [
            float(value) for value in stochastic.mean(axis=0)
        ],
        "absolute_success_improvement": improvement,
        "scene_cluster_bootstrap_ci95": [float(value) for value in interval],
        "stochastic_only_successes": int(np.sum(stochastic & ~deterministic)),
        "deterministic_only_successes": int(np.sum(deterministic & ~stochastic)),
        "strict_improvement_gate": bool(interval[0] > 0),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=50000)
    parser.add_argument("--bootstrap-seed", type=int, default=1003072)
    parser.add_argument("--selector", choices=("mode", "flow"), default="mode")
    return parser.parse_args()


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def combine_commitment_shards(shards, *, sources, draws, seed, selector="mode"):
    if not shards:
        raise ValueError("At least one commitment shard is required")
    if len(shards) != len(sources):
        raise ValueError("sources and shards must have equal length")

    if selector == "mode":
        deterministic_key = "model_deterministic_success"
        stochastic_key = "model_stochastic_success"
        learned_name = "learned persistent modes versus learned residual mean"
        consistent_fields = BASE_CONSISTENT_FIELDS
    elif selector == "flow":
        deterministic_key = "flow_deterministic_success"
        stochastic_key = "flow_stochastic_success"
        learned_name = (
            "label-free persistent flow samples versus their predictive mean"
        )
        consistent_fields = BASE_CONSISTENT_FIELDS + ("flow_sampling",)
    else:
        raise ValueError(f"Unknown selector: {selector}")

    reference = shards[0]
    records = []
    seen_seeds = set()
    shard_metadata = []
    for source, shard in zip(sources, shards, strict=True):
        for field in consistent_fields:
            if shard.get(field) != reference.get(field):
                raise ValueError(f"Shard {source} disagrees on {field}")
        shard_records = shard.get("records", [])
        if shard.get("contexts") != len(shard_records):
            raise ValueError(f"Shard {source} has an inconsistent context count")
        if shard.get("episodes") != 2 * len(shard_records):
            raise ValueError(f"Shard {source} has an inconsistent episode count")
        record_seeds = [int(record["seed"]) for record in shard_records]
        if shard.get("context_seeds") != record_seeds:
            raise ValueError(f"Shard {source} context_seeds do not match its records")
        duplicates = seen_seeds.intersection(record_seeds)
        if duplicates:
            raise ValueError(
                f"Duplicate context seeds across shards: {sorted(duplicates)[:5]}"
            )
        if len(set(record_seeds)) != len(record_seeds):
            raise ValueError(f"Shard {source} contains duplicate context seeds")
        seen_seeds.update(record_seeds)
        records.extend(shard_records)
        shard_metadata.append(
            {
                "source": str(source),
                "contexts": len(shard_records),
                "first_context_seed": min(record_seeds),
                "last_context_seed": max(record_seeds),
            }
        )

    def outcomes(key):
        values = np.asarray([record[key] for record in records], dtype=bool)
        if values.shape != (len(records), 2):
            raise ValueError(f"Outcome {key} must have shape [contexts, 2]")
        return values

    oracle = summarize_pair(
        "exact distribution versus exact mean",
        outcomes("oracle_deterministic_success"),
        outcomes("oracle_stochastic_success"),
        draws=draws,
        seed=seed,
    )
    learned = summarize_pair(
        learned_name,
        outcomes(deterministic_key),
        outcomes(stochastic_key),
        draws=draws,
        seed=seed + 1,
    )
    return {
        **{field: reference[field] for field in consistent_fields},
        "selector": selector,
        "contexts": len(records),
        "episodes": 2 * len(records),
        "context_seeds": [int(record["seed"]) for record in records],
        "shards": shard_metadata,
        "bootstrap_draws": int(draws),
        "bootstrap_seed": int(seed),
        "oracle_decision_value": oracle,
        "learned_decision_value": learned,
        "primary_learned_improvement_gate": learned["strict_improvement_gate"],
        "confirmation_gate": bool(
            oracle["strict_improvement_gate"]
            and learned["strict_improvement_gate"]
        ),
        "fresh_confirmation": True,
    }


def main():
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite summary: {args.output}")
    shards = [json.loads(path.read_text()) for path in args.input]
    result = combine_commitment_shards(
        shards,
        sources=args.input,
        draws=args.bootstrap_draws,
        seed=args.bootstrap_seed,
        selector=args.selector,
    )
    for metadata, path in zip(result["shards"], args.input, strict=True):
        metadata["sha256"] = file_sha256(path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
