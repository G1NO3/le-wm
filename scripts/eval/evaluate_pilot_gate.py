#!/usr/bin/env python
"""Evaluate ordered physics difficulty from pilot success and H=5 forks."""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch

from experiment_data import sha256_file
from stochastic_metrics import pooled_mode_separation
from stochastic_physics import select_calibrated_profile


def calibration_status(result):
    """Apply the ordered gate without requesting milder data for a too-easy task."""

    result = dict(result)
    try:
        result["selected_profile"] = select_calibrated_profile(result)
        result["status"] = "pass"
    except KeyError:
        result["selected_profile"] = None
        if result["strong"]["expert_success"] > 0.8:
            result["status"] = "fail_too_easy"
            result["reason"] = (
                "The strongest preregistered profile exceeds 80% success; "
                "do not spend data on the milder fallback profiles."
            )
        else:
            next_profile = next(
                name for name in ("strong", "medium", "mild") if name not in result
            )
            result["status"] = "needs_next_profile"
            result["next_profile"] = next_profile
    except RuntimeError:
        result["selected_profile"] = None
        result["status"] = "fail"
    return result


def load_verified_evidence(path):
    manifest_path = path.with_suffix(".json")
    if not manifest_path.exists():
        raise RuntimeError(f"Missing fork-evidence manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if sha256_file(path) != manifest["evidence_sha256"]:
        raise RuntimeError(f"Fork-evidence hash mismatch: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["dataset_sha256"] != manifest["dataset_sha256"]:
        raise RuntimeError("Fork evidence and manifest name different source datasets")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strong", type=Path, required=True)
    parser.add_argument("--medium", type=Path)
    parser.add_argument("--mild", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {}
    for name in ("strong", "medium", "mild"):
        path = getattr(args, name)
        if path is None:
            continue
        payload = load_verified_evidence(path)
        result[name] = {
            "expert_success": float(torch.as_tensor(payload["episode_success"]).float().mean()),
            "mode_separation_sd": float(
                pooled_mode_separation(
                    payload["h5_physical_state"],
                    payload["mode_label"],
                    payload.get("context_row"),
                )
            ),
        }
    result = calibration_status(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
