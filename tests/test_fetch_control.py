import numpy as np
import pytest
import torch
import json
from collections import deque
from types import SimpleNamespace

from eval import evaluate_paired_full_task, paired_fetch_success_metrics
from scripts.train_fetch_state_dynamics import transition_rows
from scripts.eval.compare_paired_fetch_control import (
    load_summary_shards,
    scene_cluster_interval,
)
from state_residual_dynamics import FetchStateResidualDynamics
from fetch_push_expert import FetchPushExpertPolicy
from residual_policy import FetchGuidedWorldModelPolicy


def test_paired_fetch_metrics_follow_seeds_not_completion_order():
    seeds = np.array([102, 100, 103, 101])
    successes = np.array([True, True, False, False])
    modes, metrics = paired_fetch_success_metrics(successes, seeds, origin=100)
    assert modes.tolist() == [0, 0, 1, 1]
    assert metrics == {
        "success_rate_low": 100.0,
        "success_rate_high": 0.0,
        "paired_both_success_rate": 0.0,
    }


def test_paired_fetch_metrics_reject_incomplete_scene():
    with pytest.raises(ValueError, match="Incomplete paired"):
        paired_fetch_success_metrics([True, False], [100, 102], origin=100)


def test_two_step_transition_rows_do_not_cross_episode_boundaries():
    rows = transition_rows([4, 3], [0, 1])
    assert rows.tolist() == [0, 1, 4]


def make_state_model():
    model = FetchStateResidualDynamics(
        state_mean=np.zeros(28, dtype=np.float32),
        state_scale=np.ones(28, dtype=np.float32),
        action_mean=np.zeros(4, dtype=np.float32),
        action_scale=np.ones(4, dtype=np.float32),
        delta_mean=np.zeros(25, dtype=np.float32),
        delta_scale=np.ones(25, dtype=np.float32),
        residual_scale=np.full(25, 0.01, dtype=np.float32),
        hidden_dim=16,
        depth=1,
        flow_hidden_dim=16,
        flow_depth=1,
    )
    for parameter in model.nominal.parameters():
        torch.nn.init.zeros_(parameter)
    return model.eval()


def test_centered_mode_residuals_preserve_nominal_mixture_mean():
    model = make_state_model()
    model.center_mode_residuals = True
    state = torch.randn(5, model.state_dim)
    action = torch.randn(5, model.action_block * model.action_dim)

    nominal_delta = model.nominal_delta(state, action)
    residuals = model.mode_residuals(state, action, nominal_delta)
    mixture_next = model.transition_mode_mixture(state, action, mode=None)

    torch.testing.assert_close(
        residuals.mean(dim=-2),
        torch.zeros_like(residuals[:, 0]),
        atol=1e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        mixture_next[:, : model.dynamic_dim],
        state[:, : model.dynamic_dim] + nominal_delta,
        atol=1e-6,
        rtol=1e-6,
    )


def test_state_dynamics_cost_supports_nominal_and_particle_planning():
    model = make_state_model()
    state = torch.zeros(1, 2, 1, 28)
    state[..., 3,] = 1.0
    info = {"state": state}
    actions = torch.zeros(1, 2, 3, 8)
    nominal = model.get_cost(info, actions)
    assert nominal.shape == (1, 2)
    assert torch.allclose(nominal, torch.ones_like(nominal))

    model.planning.update(particles=4, objective="mean", flow_steps=2)
    stochastic = model.get_cost(info, actions)
    assert stochastic.shape == (1, 2)
    assert torch.isfinite(stochastic).all()

    model.planning.update(
        particles=1,
        objective="success_probability",
        success_threshold=0.05,
        success_temperature=0.01,
    )
    deterministic_failure = model.get_cost(info, actions)
    assert deterministic_failure.shape == (1, 2)
    assert torch.all((0 <= deterministic_failure) & (deterministic_failure <= 1))

    model.planning.update(particles=4, temporal_common_noise=True)
    temporal_failure = model.get_cost(info, actions)
    assert temporal_failure.shape == (1, 2)
    assert torch.isfinite(temporal_failure).all()

    model.planning.update(kernel="mode_mixture", particles=4, objective="mean")
    mixture_cost = model.get_cost(info, actions)
    assert mixture_cost.shape == (1, 2)
    assert torch.isfinite(mixture_cost).all()


def test_state_dynamics_minimum_distance_matches_first_entry_success():
    model = make_state_model()
    # Make the nominal model advance object x by the first action component.
    def transition_mode_mixture(state, action, mode=None):
        del mode
        next_state = state.clone()
        next_state[..., 3] += action[..., 0]
        return next_state

    model.transition_mode_mixture = transition_mode_mixture
    model.planning.update(kernel="mode_mixture", particles=1, objective="mean")
    state = torch.zeros(1, 2, 1, 28)
    state[..., -3] = 1.0
    info = {"state": state}
    # Candidate 0 passes exactly through the goal then overshoots. Candidate 1
    # stops short but has the better terminal distance.
    actions = torch.zeros(1, 2, 2, 8)
    actions[0, 0, :, 0] = torch.tensor([1.0, 1.0])
    actions[0, 1, :, 0] = torch.tensor([0.4, 0.4])

    model.planning["minimum_distance"] = False
    terminal = model.get_cost(info, actions)
    assert terminal.argmin(dim=1).item() == 1

    model.planning["minimum_distance"] = True
    first_entry = model.get_cost(info, actions)
    assert first_entry.argmin(dim=1).item() == 0
    assert first_entry[0, 0].item() == pytest.approx(0.0)


def test_scene_cluster_interval_keeps_low_high_pair_together():
    deterministic = np.array([1, 0, 0, 0], dtype=bool)
    stochastic = np.array([1, 1, 1, 0], dtype=bool)
    delta, interval = scene_cluster_interval(
        deterministic, stochastic, [10, 11, 12, 13], seed=3, resamples=1000
    )
    assert delta.tolist() == [0.5, 0.5]
    assert interval.tolist() == [0.5, 0.5]


def test_control_summary_shards_reject_duplicate_seeds(tmp_path):
    paths = []
    for index, seeds in enumerate(([10, 11], [12, 13])):
        path = tmp_path / f"shard_{index}.json"
        path.write_text(
            json.dumps(
                {
                    "metrics": {
                        "episode_successes": [True, False],
                        "seeds": seeds,
                    }
                }
            )
        )
        paths.append(path)
    _, successes, seeds = load_summary_shards(paths)
    assert successes.tolist() == [True, False, True, False]
    assert seeds.tolist() == [10, 11, 12, 13]

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(paths[0].read_text())
    with pytest.raises(ValueError, match="duplicate episode seeds"):
        load_summary_shards([paths[0], duplicate])


def test_paired_full_task_uses_exact_wait_mode_seed_blocks():
    class Policy:
        _action_buffer = [deque([1]), deque([1])]
        _next_init = torch.ones(1)

    class World:
        num_envs = 2
        policy = Policy()

        def __init__(self):
            self.calls = []

        def evaluate(self, *, episodes, seed, reset_mode):
            self.calls.append((episodes, seed, reset_mode))
            return {
                "episode_successes": np.array([seed % 4 == 0, False]),
                "seeds": np.array([seed, seed + 1]),
                "success_rate": 50.0,
            }

    world = World()
    metrics = evaluate_paired_full_task(world, episodes=4, seed=10)
    assert world.calls == [(2, 10, "wait"), (2, 12, "wait")]
    assert metrics["seeds"].tolist() == [10, 11, 12, 13]
    assert metrics["success_rate"] == 25.0
    assert all(not buffer for buffer in world.policy._action_buffer)


def test_state_expert_initializer_builds_full_action_block_trajectory():
    model = make_state_model()
    policy = SimpleNamespace(
        solver=SimpleNamespace(model=model),
        expert=FetchPushExpertPolicy(),
        cfg=SimpleNamespace(horizon=3, action_block=2),
        process={},
    )
    state = np.zeros((1, 1, 28), dtype=np.float32)
    state[..., :3] = [1.2, 0.7, 0.5]
    state[..., 3:6] = [1.3, 0.7, 0.42]
    state[..., -3:] = [1.4, 0.7, 0.42]
    first = np.array([[0.25, 0.0, -0.5, 0.0]], dtype=np.float32)
    plan = FetchGuidedWorldModelPolicy._state_expert_init(
        policy, {"state": state}, first, np.array([3], dtype=np.int8)
    )
    assert plan.shape == (1, 3, 8)
    assert torch.allclose(plan[0, 0, :4], torch.from_numpy(first[0]))
    assert torch.isfinite(plan).all()


def test_balanced_episode_assignments_are_coherent_and_do_not_collapse():
    episode_ids = torch.tensor([10, 10, 11, 11, 12, 12, 13, 13])
    # Even if every transition initially favors head 0, the known balanced
    # prior assigns exactly two complete episodes to each head.
    transition_loss = torch.tensor(
        [
            [0.1, 0.5],
            [0.1, 0.5],
            [0.2, 0.5],
            [0.2, 0.5],
            [0.3, 0.5],
            [0.3, 0.5],
            [0.4, 0.5],
            [0.4, 0.5],
        ]
    )
    unique, inverse, _, assignment = (
        FetchStateResidualDynamics.balanced_episode_assignments(
            transition_loss, episode_ids
        )
    )
    assert unique.tolist() == [10, 11, 12, 13]
    assert torch.bincount(assignment, minlength=2).tolist() == [2, 2]
    transition_assignment = assignment[inverse]
    for episode in unique:
        values = transition_assignment[episode_ids == episode]
        assert values.unique().numel() == 1


def test_episode_wta_residual_loss_backpropagates_only_through_mode_heads():
    model = make_state_model().train()
    model.requires_grad_(False)
    model.mode_residual_heads.requires_grad_(True)
    state = torch.zeros(8, 28)
    action = torch.zeros(8, 8)
    delta = torch.randn(8, 25) * 0.01
    episode_ids = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    loss = model.episode_wta_residual_loss(state, action, delta, episode_ids)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in model.mode_residual_heads.parameters())
    assert all(parameter.grad is None for parameter in model.nominal.parameters())


def test_paired_episode_assignments_choose_one_head_per_scene_outcome():
    episode_ids = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    # Pair 0 prefers the identity assignment; pair 1 prefers the swap.
    transition_loss = torch.tensor(
        [
            [0.1, 0.9],
            [0.1, 0.9],
            [0.8, 0.2],
            [0.8, 0.2],
            [0.7, 0.1],
            [0.7, 0.1],
            [0.2, 0.8],
            [0.2, 0.8],
        ]
    )
    unique, inverse, _, assignment = (
        FetchStateResidualDynamics.paired_episode_assignments(
            transition_loss, episode_ids
        )
    )
    assert unique.tolist() == [0, 1, 2, 3]
    assert assignment.tolist() == [0, 1, 1, 0]
    transition_assignment = assignment[inverse]
    assert transition_assignment.tolist() == [0, 0, 1, 1, 1, 1, 0, 0]
