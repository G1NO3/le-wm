import torch

from scripts.eval.evaluate_fetch_push_posterior import mode_contrast, sample_spread
from scripts.eval.evaluate_fetch_push_mode_probe import grouped_probe
from scripts.eval.generate_fetch_push_fork_samples import pack_actions


def test_pack_actions_normalizes_then_concatenates_frameskip_block():
    raw = torch.tensor(
        [[[1.0, 3.0], [5.0, 7.0], [9.0, 11.0], [13.0, 15.0]]]
    )
    packed = pack_actions(
        raw,
        mean=torch.tensor([1.0, 1.0]),
        std=torch.tensor([2.0, 2.0]),
        frameskip=2,
    )
    expected = torch.tensor([[[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]]])
    assert torch.equal(packed, expected)


def test_posterior_metrics_detect_contraction_and_mode_contrast():
    dispersed = torch.tensor([[[[0.0]], [[2.0]]]])
    contracted = torch.tensor([[[[0.9]], [[1.1]]]])
    assert sample_spread(contracted) < sample_spread(dispersed)

    truth = torch.tensor([[[[0.0]]], [[[2.0]]], [[[1.0]]], [[[4.0]]]])
    samples = truth.expand(-1, 3, -1, -1)
    contrast = mode_contrast(samples, truth)
    assert contrast["error_explained"] == 1.0
    assert contrast["cosine"] == 1.0
    assert contrast["magnitude_ratio"] == 1.0


def test_grouped_mode_probe_preserves_scene_pairs():
    labels = torch.tensor([0, 1] * 10).numpy()
    groups = torch.arange(10).repeat_interleave(2).numpy()
    features = labels[:, None].astype("float32")
    result = grouped_probe(features, labels, groups, seed=3)
    assert result["accuracy"] == 1.0
    assert result["balanced_accuracy"] == 1.0
