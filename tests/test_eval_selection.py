import numpy as np
import pytest

from eval import reachable_context_index


class Dataset:
    lengths = np.array([3, 6], dtype=np.int64)
    offsets = np.array([0, 3], dtype=np.int64)


def test_reachable_contexts_use_internal_episode_and_positional_step():
    episodes, starts, rows = reachable_context_index(Dataset(), goal_offset=2)

    assert episodes.tolist() == [0, 1, 1, 1, 1]
    assert starts.tolist() == [0, 0, 1, 2, 3]
    assert rows.tolist() == [0, 3, 4, 5, 6]


def test_reachable_contexts_reject_too_short_dataset():
    with pytest.raises(ValueError, match="No episode"):
        reachable_context_index(Dataset(), goal_offset=6)
