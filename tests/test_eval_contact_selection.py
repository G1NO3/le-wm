from types import SimpleNamespace

import numpy as np

from eval import balanced_contact_context_indices


class ContactDataset:
    lengths = np.array([8, 8])
    offsets = np.array([0, 8])

    def get_col_data(self, key):
        assert key == "privileged/contact_index"
        return np.array([0, 1, 1, 2, 2, 3, 3, 4] * 2)


def test_reachable_contexts_are_balanced_across_contact_onsets():
    rows = np.arange(16)
    selected = balanced_contact_context_indices(
        ContactDataset(),
        rows,
        rows,
        stages=[1, 2, 3, 4],
        contexts=8,
        generator=np.random.default_rng(3),
        key="privileged/contact_index",
        selection="onset",
    )
    values = ContactDataset().get_col_data("privileged/contact_index")
    assert sorted(values[selected].tolist()) == [1, 1, 2, 2, 3, 3, 4, 4]


def test_within_stage_selection_excludes_contact_onsets():
    rows = np.arange(16)
    selected = balanced_contact_context_indices(
        ContactDataset(),
        rows,
        rows,
        stages=[1, 2, 3],
        contexts=5,
        generator=np.random.default_rng(4),
        key="privileged/contact_index",
        selection="within_stage",
    )
    assert len(selected) == 5
    assert not set(selected) & {1, 3, 5, 9, 11, 13}
