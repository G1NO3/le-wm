#!/usr/bin/env python
"""Fit frozen ridge probes on held-out nominal latent/state pairs."""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch

from physical_probes import RidgeProbe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--regularization", type=float, default=1e-3)
    args = parser.parse_args()
    payload = torch.load(args.input, map_location="cpu", weights_only=True)
    latents = payload["nominal_latents"]
    probes = {
        name: RidgeProbe(args.regularization).fit(latents, target).state_dict()
        for name, target in payload["physical_states"].items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"probes": probes, "source": str(args.input)}, args.output)


if __name__ == "__main__":
    main()
