import pytest
import torch

from scripts.eval.evaluate_memory_filter import (
    axis_thresholds,
    fit_mode_probe,
    majority_exact_accuracy,
    mode_bits,
    parse_checkpoint_grid,
    probe_accuracy,
    split_probe_episodes,
)


def test_checkpoint_grid_sorts_and_deduplicates():
    assert parse_checkpoint_grid("5, 1,2,5,100") == [1, 2, 5, 100]
    with pytest.raises(ValueError):
        parse_checkpoint_grid("0,1")
    with pytest.raises(ValueError):
        parse_checkpoint_grid(" , ")


def test_mode_bits_use_midpoint_thresholds():
    modes = torch.tensor(
        [[0.5, 0.3, 0.2], [1.5, 1.2, 1.0], [0.5, 1.2, 0.2], [1.5, 0.3, 1.0]]
    )
    thresholds = axis_thresholds(modes)
    assert thresholds == pytest.approx([1.0, 0.75, 0.6])
    bits = mode_bits(modes, thresholds)
    assert bits.tolist() == [
        [False, False, False],
        [True, True, True],
        [False, True, False],
        [True, False, True],
    ]


def test_axis_thresholds_reject_constant_axis():
    with pytest.raises(ValueError):
        axis_thresholds(torch.ones(4, 3))


def test_probe_split_is_disjoint_and_deterministic():
    episodes = list(range(11))
    fit_a, eval_a = split_probe_episodes(episodes, seed=7)
    fit_b, eval_b = split_probe_episodes(episodes, seed=7)
    assert (fit_a, eval_a) == (fit_b, eval_b)
    assert not set(fit_a) & set(eval_a)
    assert sorted(fit_a + eval_a) == episodes


def test_mode_probe_recovers_separable_modes():
    generator = torch.Generator().manual_seed(0)
    bits = torch.randint(0, 2, (96, 3), generator=generator).bool()
    states = torch.zeros(96, 16)
    states[:, :3] = bits.float() * 2.0 - 1.0
    states += 0.05 * torch.randn(96, 16, generator=generator)
    probe = fit_mode_probe(states[:64], bits[:64], regularization=1e-3)
    accuracy = probe_accuracy(probe, states[64:], bits[64:])
    assert accuracy["exact_accuracy"] == 1.0
    assert all(value == 1.0 for value in accuracy["axis_accuracy"])


def test_majority_baseline_matches_class_frequency():
    fit_bits = torch.tensor(
        [[True, False, False]] * 3 + [[False, False, False]] * 2
    )
    eval_bits = torch.tensor(
        [[True, False, False], [False, True, False], [True, False, False]]
    )
    assert majority_exact_accuracy(fit_bits, eval_bits) == pytest.approx(2 / 3)
