import torch

from scripts.data.convert_hf_lewm_checkpoint import remap_legacy_vit_keys


def test_legacy_hf_vit_keys_are_remapped_without_changing_tensors():
    value = torch.ones(2)
    state, changed = remap_legacy_vit_keys(
        {
            "encoder.encoder.layer.0.attention.attention.query.weight": value,
            "predictor.pos_embedding": torch.zeros(1),
        }
    )
    assert changed == 1
    assert state["encoder.layers.0.attention.q_proj.weight"] is value
    assert "predictor.pos_embedding" in state
