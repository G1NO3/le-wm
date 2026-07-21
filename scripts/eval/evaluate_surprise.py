#!/usr/bin/env python
"""Compare distributional VoE scores with deterministic prediction MSE."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from scipy.stats import ttest_rel

from stochastic_metrics import energy_score


VALID_STOCHASTIC = {"slip", "drop"}
PHYSICAL_VIOLATIONS = {"teleport", "interpenetration", "unsupported_motion"}


def diagnostic(scores, event_types):
    labels = np.array([event in PHYSICAL_VIOLATIONS for event in event_types], dtype=int)
    score = np.asarray(scores)
    fpr, tpr, thresholds = roc_curve(labels, score)
    index = np.where(tpr >= 0.95)[0][0]
    valid_mask = np.array([event in VALID_STOCHASTIC for event in event_types])
    valid_fpr95 = float((score[valid_mask] >= thresholds[index]).mean())
    return {
        "auroc": float(roc_auc_score(labels, score)),
        "auprc": float(average_precision_score(labels, score)),
        "valid_stochastic_fpr_at_95_tpr": valid_fpr95,
        "threshold_at_95_tpr": float(thresholds[index]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = torch.load(args.input, map_location="cpu", weights_only=True)
    event_types = payload["event_types"]
    observed = payload["observed_futures"][:, None]
    model_scores = {
        name: energy_score(samples, observed).numpy()
        for name, samples in payload["model_samples"].items()
    }
    if "deterministic_mse" in payload:
        model_scores["deterministic_mse"] = np.asarray(payload["deterministic_mse"])
    result = {name: diagnostic(score, event_types) for name, score in model_scores.items()}
    if "flow" in result and "deterministic_mse" in result:
        result["flow_vs_deterministic"] = {
            "auroc_gain": result["flow"]["auroc"] - result["deterministic_mse"]["auroc"],
            "valid_stochastic_fpr_relative_reduction": (
                result["deterministic_mse"]["valid_stochastic_fpr_at_95_tpr"]
                - result["flow"]["valid_stochastic_fpr_at_95_tpr"]
            )
            / max(result["deterministic_mse"]["valid_stochastic_fpr_at_95_tpr"], 1e-8),
        }
    if "paired_mse_ordinary" in payload and "paired_mse_violation" in payload:
        ordinary = np.asarray(payload["paired_mse_ordinary"])
        violation = np.asarray(payload["paired_mse_violation"])
        test = ttest_rel(violation, ordinary)
        result["original_prediction_mse_paired_test"] = {
            "mean_difference": float((violation - ordinary).mean()),
            "t_statistic": float(test.statistic),
            "p_value": float(test.pvalue),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
