import numpy as np
import torch

from scripts.data.probe_fetch_slide_friction import (
    discrete_distribution_index,
    mean_state_index,
    smooth_distribution_index,
    strike_candidates,
    terminal_distances,
)
from state_residual_dynamics import FetchStateResidualDynamics


def test_strike_candidates_follow_goal_direction_and_requested_grid():
    state = np.zeros(28, dtype=np.float32)
    state[3:6] = [1.0, 2.0, 0.4]
    state[-3:] = [2.0, 2.0, 0.4]
    candidates, parameters = strike_candidates(
        state,
        horizon=5,
        speed_min=0.5,
        speed_max=1.0,
        speed_count=2,
        duration_min=1,
        duration_max=2,
        angle_max_deg=0,
        angle_count=1,
    )

    assert candidates.shape == (4, 5, 4)
    assert parameters[0] == {"angle_deg": 0.0, "speed": 0.5, "duration": 1}
    np.testing.assert_allclose(candidates[0, 0], [0.5, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(candidates[0, 1:], 0.0)
    np.testing.assert_allclose(candidates[-1, :2, 0], 1.0)
    np.testing.assert_allclose(candidates[-1, 2:], 0.0)


def test_distribution_selectors_avoid_an_unphysical_mean_outcome():
    goal = np.zeros(3, dtype=np.float32)
    terminal = np.array(
        [
            [[-0.2, 0.0, 0.0], [0.2, 0.0, 0.0]],
            [[0.01, 0.0, 0.0], [0.5, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    distances = terminal_distances(terminal, goal)

    assert mean_state_index(terminal, goal) == 0
    assert smooth_distribution_index(distances, 0.05, 0.01) == 1
    assert discrete_distribution_index(distances, 0.05) == 1


def test_pair_assignment_keeps_a_complete_pair_without_object_motion():
    model = FetchStateResidualDynamics(
        state_mean=np.zeros(28),
        state_scale=np.ones(28),
        action_mean=np.zeros(4),
        action_scale=np.ones(4),
        delta_mean=np.zeros(25),
        delta_scale=np.ones(25),
        residual_scale=np.ones(25),
        hidden_dim=16,
        depth=1,
        flow_hidden_dim=16,
        flow_depth=1,
    )
    state = torch.zeros(8, 28)
    action = torch.zeros(8, 8)
    delta = torch.zeros(8, 25)
    delta[:2, 3] = 0.1
    delta[2:4, 3] = 0.2
    episode_ids = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])

    loss, unique_ids, _, assignments = (
        model.paired_object_motion_wta_residual_loss(
            state,
            action,
            delta,
            episode_ids,
            return_assignments=True,
        )
    )

    assert torch.isfinite(loss)
    assert torch.equal(unique_ids, torch.arange(4))
    assert assignments[0] != assignments[1]
    assert assignments[2] != assignments[3]
