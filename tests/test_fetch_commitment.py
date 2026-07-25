import numpy as np
import pytest
import torch

from scripts.eval.evaluate_fetch_push_commitment import (
    deterministic_index,
    push_pulse_candidates,
    stochastic_index,
)
from scripts.eval.summarize_fetch_push_commitment import (
    combine_commitment_shards,
)
from scripts.eval.evaluate_fetch_push_flow_commitment import (
    flow_candidate_futures,
)


def test_push_pulse_candidates_use_only_goal_direction_and_grid():
    state = np.zeros(28, dtype=np.float32)
    state[3:6] = [1.0, 1.0, 0.4]
    state[-3:] = [1.0, 2.0, 0.4]
    candidates, parameters = push_pulse_candidates(
        state,
        raw_horizon=6,
        speed_min=0.5,
        speed_max=1.0,
        speed_count=2,
        duration_min=2,
        duration_stride=2,
    )
    assert candidates.shape == (6, 6, 4)
    assert parameters[0] == {"speed": 0.5, "duration": 2}
    assert np.allclose(candidates[0, :2, :2], [[0.0, 0.5], [0.0, 0.5]])
    assert np.allclose(candidates[0, 2:], 0.0)


def test_stochastic_selector_avoids_mean_only_goal_crossing():
    goal = np.zeros(3, dtype=np.float32)
    # Candidate 0 has a perfect mean but both modes miss on opposite sides.
    # Candidate 1 succeeds in one mode and therefore has higher success mass.
    modes = np.array(
        [
            [[[-0.1, 0, 0]], [[0.1, 0, 0]]],
            [[[0.0, 0, 0]], [[0.2, 0, 0]]],
        ],
        dtype=np.float32,
    )
    mean = modes.mean(axis=1)
    assert deterministic_index(mean, goal) == 0
    assert stochastic_index(modes, goal, threshold=0.05, temperature=0.01) == 1


def _commitment_shard(seed, model_deterministic, model_stochastic):
    record = {
        "seed": seed,
        "oracle_deterministic_success": [False, False],
        "oracle_stochastic_success": [True, True],
        "model_deterministic_success": model_deterministic,
        "model_stochastic_success": model_stochastic,
    }
    return {
        "task": "fetch_push_hidden_friction_commitment",
        "checkpoint": "/checkpoint.pt",
        "friction_modes": [0.2, 3.0],
        "candidate_grid": {"candidate_count": 100},
        "success_threshold": 0.05,
        "success_temperature": 0.01,
        "contexts": 1,
        "episodes": 2,
        "context_seeds": [seed],
        "records": [record],
    }


def test_commitment_shards_recompute_paired_success():
    shards = [
        _commitment_shard(10, [False, False], [True, False]),
        _commitment_shard(20, [False, True], [True, True]),
    ]
    summary = combine_commitment_shards(
        shards, sources=["a.json", "b.json"], draws=2000, seed=7
    )
    assert summary["contexts"] == 2
    assert summary["learned_decision_value"]["deterministic_success_rate"] == 0.25
    assert summary["learned_decision_value"]["stochastic_success_rate"] == 0.75


def test_commitment_shards_reject_duplicate_context_seeds():
    shard = _commitment_shard(10, [False, False], [True, False])
    with pytest.raises(ValueError, match="Duplicate context seeds"):
        combine_commitment_shards(
            [shard, shard], sources=["a.json", "b.json"], draws=100, seed=7
        )


def test_commitment_shards_support_label_free_flow_selector():
    shard = _commitment_shard(10, [False, False], [True, False])
    shard["task"] = "fetch_push_hidden_friction_flow_commitment"
    shard["flow_sampling"] = {"particles": 32, "uses_privileged_mode_labels": False}
    record = shard["records"][0]
    record["flow_deterministic_success"] = record.pop(
        "model_deterministic_success"
    )
    record["flow_stochastic_success"] = record.pop("model_stochastic_success")
    summary = combine_commitment_shards(
        [shard], sources=["flow.json"], draws=100, seed=7, selector="flow"
    )
    assert summary["selector"] == "flow"
    assert summary["learned_decision_value"]["absolute_success_improvement"] == 0.5
    assert summary["flow_sampling"]["uses_privileged_mode_labels"] is False


def test_antithetic_flow_futures_share_zero_mean_noise():
    class DummyModel:
        dynamic_dim = 25

        def transition(self, state, action, *, noise, flow_steps):
            assert flow_steps == 4
            result = state.clone()
            result[..., 3] += noise[..., 0]
            return result

    initial = np.zeros(28, dtype=np.float32)
    candidates = np.zeros((2, 4, 4), dtype=np.float32)
    mean, particles = flow_candidate_futures(
        DummyModel(),
        initial,
        candidates,
        torch.device("cpu"),
        particles=4,
        flow_steps=4,
        seed=10,
    )
    assert mean.shape == (2, 2, 3)
    assert particles.shape == (2, 4, 2, 3)
    assert np.allclose(mean, 0.0, atol=1e-6)
    assert np.allclose(particles[:, :2], -particles[:, 2:], atol=1e-6)
