#!/usr/bin/env python
"""Compose and save a resolved training config without importing train.py.

Invoking ``train.py --cfg job`` imports the full training stack before Hydra
prints YAML.  Some dependencies emit startup logs to stdout, which corrupts a
redirected config file.  This helper uses Hydra's composition API directly so
the output is always parseable OmegaConf YAML.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import stable_pretraining  # noqa: F401  # registers the project's eval resolver


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def main():
    args = parse_args()
    config_dir = (REPO_ROOT / "config" / "train").resolve()
    overrides = list(args.overrides)
    # Outside Hydra's decorated application there is no runtime job id.  The
    # statistics pass does not create a training run, so a null subdir is the
    # faithful and fully resolvable value used by `train.py --cfg job`.
    if not any(value.startswith("subdir=") for value in overrides):
        overrides.append("subdir=null")
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="lewm", overrides=overrides)
    OmegaConf.resolve(cfg)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, args.output)
    # Read it back so a malformed or partial artifact fails this stage rather
    # than the subsequent GPU statistics pass.
    OmegaConf.load(args.output)
    print(f"resolved_config={args.output}")


if __name__ == "__main__":
    main()
