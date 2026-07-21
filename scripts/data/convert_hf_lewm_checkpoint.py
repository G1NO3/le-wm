#!/usr/bin/env python
"""Convert an official HF LeWM state dict into the object checkpoint API."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import stable_pretraining as spt
import torch

from experiment_data import sha256_file
from jepa import JEPA
from module import ARPredictor, Embedder, MLP


def _kwargs(section):
    return {key: value for key, value in section.items() if not key.startswith("_")}


def remap_legacy_vit_keys(state):
    """Map Transformers 4.x ViT names to the locked 5.x module layout."""

    replacements = (
        ("encoder.encoder.layer.", "encoder.layers."),
        (".attention.attention.query.", ".attention.q_proj."),
        (".attention.attention.key.", ".attention.k_proj."),
        (".attention.attention.value.", ".attention.v_proj."),
        (".attention.output.dense.", ".attention.o_proj."),
        (".intermediate.dense.", ".mlp.fc1."),
        (".output.dense.", ".mlp.fc2."),
    )
    remapped = {}
    changed = 0
    for key, value in state.items():
        new_key = key
        for old, new in replacements:
            new_key = new_key.replace(old, new)
        changed += int(new_key != key)
        if new_key in remapped:
            raise RuntimeError(f"Legacy ViT remap collision at {new_key}")
        remapped[new_key] = value
    return remapped, changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite checkpoint: {args.output}")
    cfg = json.loads((args.source / "config.json").read_text())
    encoder_cfg = _kwargs(cfg["encoder"])
    encoder = spt.backbone.utils.vit_hf(
        encoder_cfg.pop("size"), **encoder_cfg
    )
    predictor = ARPredictor(**_kwargs(cfg["predictor"]))
    action_encoder = Embedder(**_kwargs(cfg["action_encoder"]))

    def mlp(name):
        section = _kwargs(cfg[name])
        section["norm_fn"] = torch.nn.BatchNorm1d
        return MLP(**section)

    model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=mlp("projector"),
        pred_proj=mlp("pred_proj"),
    )
    weights = args.source / "weights.pt"
    state = torch.load(weights, map_location="cpu", weights_only=True)
    state, remapped_keys = remap_legacy_vit_keys(state)
    model.load_state_dict(state, strict=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, args.output)
    metadata = {
        "repo_id": args.repo_id,
        "revision": args.revision,
        "source_weights_sha256": sha256_file(weights),
        "object_checkpoint_sha256": sha256_file(args.output),
        "strict_load": True,
        "legacy_vit_keys_remapped": remapped_keys,
        "output_path": str(args.output.resolve()),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
