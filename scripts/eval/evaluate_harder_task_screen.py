#!/usr/bin/env python
"""Gate a harder stochastic task before a 1,000-episode collection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.evaluate_pilot_gate import load_verified_evidence
from stochastic_metrics import pooled_mode_separation


def _deterministic_success(path):
    payload = json.loads(path.read_text())
    metrics = payload.get("metrics", payload)
    value = float(metrics["success_rate"])
    return value / 100 if value > 1 else value


def harder_task_status(
    expert_success,
    mode_separation,
    deterministic_success=None,
    *,
    expert_range=(0.5, 0.98),
    deterministic_range=(0.15, 0.6),
    minimum_gap=0.2,
    minimum_separation=1.0,
):
    if expert_success > expert_range[1]:
        return "fail_too_easy_for_expert"
    if expert_success < expert_range[0]:
        return "fail_too_hard_for_expert"
    if mode_separation < minimum_separation:
        return "fail_insufficient_stochastic_separation"
    if deterministic_success is None:
        return "needs_deterministic_baseline"
    if deterministic_success > deterministic_range[1]:
        return "fail_too_easy_for_deterministic_lewm"
    if deterministic_success < deterministic_range[0]:
        return "fail_deterministic_floor"
    if expert_success - deterministic_success < minimum_gap:
        return "fail_insufficient_control_gap"
    return "pass"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--forks", type=Path, required=True)
    parser.add_argument("--deterministic-summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expert-min", type=float, default=0.5)
    parser.add_argument("--expert-max", type=float, default=0.98)
    parser.add_argument("--deterministic-min", type=float, default=0.15)
    parser.add_argument("--deterministic-max", type=float, default=0.6)
    parser.add_argument("--minimum-gap", type=float, default=0.2)
    parser.add_argument("--minimum-separation", type=float, default=1.0)
    args = parser.parse_args()
    validation = json.loads(args.validation.read_text())
    evidence = load_verified_evidence(args.forks)
    if validation["dataset_sha256"] != evidence["dataset_sha256"]:
        raise RuntimeError("Validation and fork evidence reference different datasets")
    expert_success = float(validation["success_rate"])
    separation = float(
        pooled_mode_separation(
            evidence["h5_physical_state"],
            evidence["mode_label"],
            evidence["context_row"],
        )
    )
    deterministic = (
        _deterministic_success(args.deterministic_summary)
        if args.deterministic_summary
        else None
    )
    stage_values = evidence.get("contact_stage")
    stages = stage_values.tolist() if stage_values is not None else []
    result = {
        "expert_success": expert_success,
        "mode_separation_sd": separation,
        "deterministic_lewm_success": deterministic,
        "expert_success_range": [args.expert_min, args.expert_max],
        "deterministic_success_range": [
            args.deterministic_min, args.deterministic_max
        ],
        "minimum_expert_deterministic_gap": args.minimum_gap,
        "minimum_h5_mode_separation_sd": args.minimum_separation,
        "status": harder_task_status(
            expert_success,
            separation,
            deterministic,
            expert_range=(args.expert_min, args.expert_max),
            deterministic_range=(args.deterministic_min, args.deterministic_max),
            minimum_gap=args.minimum_gap,
            minimum_separation=args.minimum_separation,
        ),
        "dataset_sha256": validation["dataset_sha256"],
        "fork_evidence_sha256": json.loads(args.forks.with_suffix(".json").read_text())[
            "evidence_sha256"
        ],
        "contact_stages": sorted(set(stages)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
