import numpy as np

from scripts.data.generate_pilot_fork_evidence import (
    _choose_context,
    _stage_balanced_contexts,
)


def test_choose_context_targets_repeated_contact_stage():
    actions = np.zeros((12, 5), dtype=np.float32)
    actions[[2, 6, 9], -1] = -0.1
    contacts = np.array([0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    assert _choose_context(
        actions, 0, 12, 2, contacts, target_contact=2
    ) == 6


def test_stage_balanced_contexts_never_invent_stage_zero():
    lengths = np.array([8, 8])
    contacts = np.array([0, 1, 1, 2, 2, 2, 2, 2, 0, 1, 1, 1, 1, 1, 1, 1])
    selected = _stage_balanced_contexts(lengths, contacts, horizon=2, count=4)
    assert selected
    assert {stage for _, _, stage in selected} == {1, 2}
