#!/usr/bin/env python
"""Apply the preregistered persistent-residual prediction acceptance gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def evaluate(payload):
    models = payload["models"]
    recurrent = models["flow_gru_memory"]
    memoryless = models["flow"]
    horizon10 = recurrent["10"]
    horizon20 = recurrent["20"]
    paired = payload.get("paired_memory_bootstrap", {})
    checks = {
        "h10_residual_energy_skill_at_least_10pct": horizon10[
            "residual_energy_skill"
        ]
        >= 0.10,
        "h10_memory_energy_skill_at_least_5pct": horizon10["memory_energy_skill"]
        >= 0.05,
        "h20_memory_energy_skill_at_least_5pct": horizon20["memory_energy_skill"]
        >= 0.05,
        "stochastic_spread_beats_residual_mean": horizon10[
            "stochastic_energy_skill_vs_residual_mean"
        ]
        > 0,
        "coverage_90_calibrated": 0.87 <= horizon10["coverage_90"] <= 0.93,
        "lag1_error_reduced_10pct": horizon10["lag1_autocorrelation_error"]
        <= 0.90 * memoryless["10"]["lag1_autocorrelation_error"],
        "one_step_mean_error_regression_below_5pct": recurrent["1"]["mean_error"]
        <= 1.05 * memoryless["1"]["mean_error"],
        "paired_ci_beats_memoryless": paired.get("flow", {}).get(
            "ci95", [float("-inf")]
        )[0]
        > 0,
        "paired_ci_beats_reset_memory": paired.get(
            "flow_gru_memory_reset", {}
        ).get("ci95", [float("-inf")])[0]
        > 0,
        "paired_ci_beats_shuffled_memory": paired.get(
            "flow_gru_memory_shuffled", {}
        ).get("ci95", [float("-inf")])[0]
        > 0,
    }
    return {"passed": all(checks.values()), "checks": checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(json.loads(args.input.read_text()))
    result["input"] = str(args.input.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit("Persistent residual prediction gate failed")


if __name__ == "__main__":
    main()
