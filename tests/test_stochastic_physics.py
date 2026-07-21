from types import SimpleNamespace

import gymnasium as gym
import numpy as np

from stochastic_physics import (
    OGBenchHiddenPhysics,
    rollout_simulator_forks,
    select_calibrated_profile,
)
from scripts.eval.evaluate_pilot_gate import calibration_status
from scripts.eval.evaluate_harder_task_screen import harder_task_status


class Item:
    def __init__(self, name):
        self.name = name


class Model:
    def __init__(self):
        self.nbody = 4
        self.ngeom = 5
        self.nq = self.nv = 3
        self.body_mass = np.ones(4)
        self.geom_friction = np.ones((5, 3))
        self.geom_bodyid = np.array([0, 1, 2, 3, 3])
        self._bodies = [Item("world"), Item("object_0"), Item("table"), Item("robot")]
        self._geoms = [
            Item("world"), Item("cube"), Item("table_surface"), Item("left_pad"), Item("right_pad")
        ]

    def body(self, index):
        return self._bodies[index]

    def geom(self, index):
        return self._geoms[index]


class Environment(gym.Env):
    def __init__(self):
        self._model = Model()
        self._data = SimpleNamespace(
            qpos=np.zeros(3), qvel=np.zeros(3), xpos=np.zeros((4, 3))
        )
        self._cube_geom_ids_list = [[1]]
        self.action_space = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        self._data.qpos += 1
        return np.zeros(1, dtype=np.float32), 0.0, False, False, {}


def test_hidden_mode_reproducible_and_not_in_observation():
    wrapper = OGBenchHiddenPhysics(Environment(), profile="strong")
    obs, info1 = wrapper.reset(seed=19)
    _, info2 = wrapper.reset(seed=19)
    assert np.array_equal(info1["privileged/physics_mode"], info2["privileged/physics_mode"])
    assert obs.shape == (1,)
    assert not isinstance(obs, dict)
    assert "commanded_action" in info1
    assert "metrics/final_goal_distance" in info1
    assert "metrics/target_object_drops" in info1


def test_commanded_action_and_fork_reset_reproducibility():
    wrapper = OGBenchHiddenPhysics(Environment(), profile="strong")
    wrapper.reset(seed=5)
    state = wrapper.get_fork_state()
    action = np.array([0.25, 1.0], dtype=np.float32)
    _, _, _, _, info = wrapper.step(action)
    assert np.array_equal(info["commanded_action"], action)
    after = wrapper.get_fork_state()["qpos"].copy()
    wrapper.set_fork_state(state)
    wrapper.step(action)
    assert np.array_equal(wrapper.get_fork_state()["qpos"], after)


def test_ogbench_uses_negative_gripper_delta_for_grasp_onset():
    wrapper = OGBenchHiddenPhysics(Environment(), profile="strong")
    wrapper.reset(seed=5)
    assert wrapper._is_closing(np.array([0.0, -0.1], dtype=np.float32))
    assert not wrapper._is_closing(np.array([0.0, 0.1], dtype=np.float32))


def test_contact_index_is_episode_local_and_forked():
    wrapper = OGBenchHiddenPhysics(Environment(), profile="strong")
    _, info = wrapper.reset(seed=5)
    assert info["privileged/contact_index"] == 0
    _, _, _, _, info = wrapper.step(np.array([0.0, -0.1], dtype=np.float32))
    assert info["privileged/contact_index"] == 1
    state = wrapper.get_fork_state()
    wrapper.step(np.array([0.0, 0.1], dtype=np.float32))
    wrapper.step(np.array([0.0, -0.1], dtype=np.float32))
    assert wrapper.get_fork_state()["contact_index"] == 2
    wrapper.set_fork_state(state)
    assert wrapper.get_fork_state()["contact_index"] == 1
    _, info = wrapper.reset(seed=6)
    assert info["privileged/contact_index"] == 0


def test_calibration_uses_ordered_fallback():
    results = {
        "strong": {"expert_success": 0.2, "mode_separation_sd": 2.0},
        "medium": {"expert_success": 0.6, "mode_separation_sd": 1.1},
        "mild": {"expert_success": 0.7, "mode_separation_sd": 1.2},
    }
    assert select_calibrated_profile(results) == "medium"


def test_calibration_does_not_collect_milder_profile_when_strong_is_too_easy():
    result = calibration_status(
        {"strong": {"expert_success": 1.0, "mode_separation_sd": 2.0}}
    )
    assert result["status"] == "fail_too_easy"
    assert "next_profile" not in result


def test_harder_task_screen_requires_nontrivial_deterministic_gap():
    assert harder_task_status(0.7, 1.5, None) == "needs_deterministic_baseline"
    assert harder_task_status(0.7, 1.5, 0.4) == "pass"
    assert harder_task_status(0.99, 1.5, 0.4) == "fail_too_easy_for_expert"
    assert harder_task_status(0.7, 1.5, 0.65) == "fail_too_easy_for_deterministic_lewm"


def test_forked_realizations_have_reproducible_but_diverse_hidden_modes():
    wrapper = OGBenchHiddenPhysics(Environment(), profile="strong")
    wrapper.reset(seed=23)
    context = wrapper.get_fork_state()
    _, privileged = rollout_simulator_forks(
        wrapper,
        context,
        [np.array([0.0, 0.0], dtype=np.float32)],
        realizations=16,
        observe=lambda raw: raw._data.qpos.copy(),
    )
    modes = {
        tuple(sorted(item["physics_mode"].items())) for item in privileged
    }
    assert len(modes) > 1
