import numpy as np

from scripts.data.collect_stochastic_ogbench import align_commanded_actions


def test_commanded_action_rows_follow_world_transition_alignment():
    zero = np.zeros(2, dtype=np.float32)
    first = np.array([0.25, -0.5], dtype=np.float32)
    second = np.array([0.5, -0.25], dtype=np.float32)
    episode = {
        "action": [first.copy(), second.copy(), zero.copy()],
        "commanded_action": [zero.copy(), first.copy(), second.copy()],
    }

    [aligned] = list(align_commanded_actions([episode]))

    assert all(
        np.array_equal(action, commanded)
        for action, commanded in zip(
            aligned["action"], aligned["commanded_action"], strict=True
        )
    )
