#!/usr/bin/env python
"""Apply the preregistered double-stack persistent-memory control gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def evaluate(payload):
    models = payload["models"]
    deterministic = models["deterministic"]
    memoryless = models["flow"]
    recurrent = models["flow_gru_memory_cvar"]
    checks = {
        "success_gain_vs_deterministic_5_points": recurrent["success_rate"]
        >= deterministic["success_rate"] + 5.0,
        "worst_decile_distance_reduced_10pct": recurrent[
            "worst_decile_distance"
        ]
        <= 0.90 * deterministic["worst_decile_distance"],
        "success_beats_memoryless_flow": recurrent["success_rate"]
        > memoryless["success_rate"],
        "worst_decile_beats_memoryless_flow": recurrent[
            "worst_decile_distance"
        ]
        < memoryless["worst_decile_distance"],
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
        raise SystemExit("Persistent residual control gate failed")


if __name__ == "__main__":
    main()
