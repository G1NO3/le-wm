#!/usr/bin/env python
"""Probe where hidden friction becomes decodable in posterior observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def grouped_probe(features, labels, groups, *, seed):
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    groups = np.asarray(groups, dtype=np.int64)
    predictions = np.empty_like(labels)
    fold_accuracy = []
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for train, test in splitter.split(features, labels, groups):
        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=2000, random_state=seed),
        )
        classifier.fit(features[train], labels[train])
        predictions[test] = classifier.predict(features[test])
        fold_accuracy.append(float((predictions[test] == labels[test]).mean()))
    return {
        "accuracy": float((predictions == labels).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "fold_accuracy": fold_accuracy,
        "feature_dim": int(features.shape[1]),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=43072)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite immutable result: {args.output}")
    payload = torch.load(args.input, map_location="cpu", weights_only=True)
    diagnostics = payload["diagnostics"]
    modes = np.asarray(payload["context_mode"])
    labels = (modes > np.median(modes)).astype(np.int64)
    context_seeds = np.asarray(payload["metadata"]["context_seeds"])
    unique_seeds = {seed: index for index, seed in enumerate(dict.fromkeys(context_seeds))}
    groups = np.asarray([unique_seeds[seed] for seed in context_seeds])

    positions = diagnostics["observed_object_positions"].numpy()
    observed_states = diagnostics.get("normalized_observed_states")
    embeddings = diagnostics["observation_embeddings"].numpy()
    conditions = diagnostics["residual_conditions"].numpy()
    residuals = diagnostics["raw_residuals"].numpy()
    memory_states = diagnostics.get("residual_memory_states", {})
    observation_steps = residuals.shape[1]
    result = {
        "input": str(args.input.resolve()),
        "contexts": int(len(labels)),
        "scenes": int(len(np.unique(groups))),
        "class_balance": float(labels.mean()),
        "split": "five-fold grouped by paired simulator scene",
        "probes": {},
    }
    for depth in range(1, observation_steps + 1):
        displacement = positions[:, 1 : depth + 1] - positions[:, :1]
        embedding_feature = np.concatenate(
            [embeddings[:, depth], embeddings[:, depth] - embeddings[:, 0]], axis=-1
        )
        features = {
            "physical_displacement_history": displacement.reshape(len(labels), -1),
            "observation_embedding": embedding_feature,
            "current_nominal_residual": residuals[:, depth - 1],
            "nominal_residual_history": residuals[:, :depth].reshape(len(labels), -1),
            "current_residual_condition": conditions[:, depth - 1],
        }
        if observed_states is not None:
            features["observed_state_delta"] = (
                observed_states[:, depth] - observed_states[:, 0]
            ).numpy()
        for name, states in memory_states.items():
            features[f"residual_memory/{name}"] = states[:, depth - 1].numpy()
        result["probes"][str(depth)] = {
            name: grouped_probe(value, labels, groups, seed=args.seed)
            for name, value in features.items()
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
