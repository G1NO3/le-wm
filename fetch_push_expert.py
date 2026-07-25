"""Closed-loop demonstration policy for friction-varying FetchPush."""

from __future__ import annotations

import numpy as np

from stable_worldmodel.policy import BasePolicy


class FetchPushExpertPolicy(BasePolicy):
    """Move behind the object, descend, then push it through the goal.

    The policy uses only the flattened Fetch observation (gripper position,
    object position, and desired goal).  Hidden friction is never read, so the
    demonstrations retain the partial-observability needed by the experiment.
    """

    def __init__(
        self,
        *,
        seed=3072,
        action_noise=0.0,
        behind_offset=0.065,
        clearance=0.10,
        push_through=0.02,
    ):
        super().__init__()
        self.seed = int(seed)
        self.action_noise = float(action_noise)
        self.behind_offset = float(behind_offset)
        self.clearance = float(clearance)
        self.push_through = float(push_through)
        self.rng = np.random.default_rng(seed)

    def set_seed(self, seed):
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)

    def set_env(self, env):
        self.env = env
        self._phase = np.zeros(env.num_envs, dtype=np.int8)

    @staticmethod
    def _current(value, index):
        value = np.asarray(value[index])
        return value[0] if value.ndim > 1 and value.shape[0] == 1 else value

    def _action(self, state, phase):
        gripper = np.asarray(state[:3], dtype=np.float64)
        object_position = np.asarray(state[3:6], dtype=np.float64)
        goal = np.asarray(state[-3:], dtype=np.float64)
        direction = goal[:2] - object_position[:2]
        distance = np.linalg.norm(direction)
        direction = direction / distance if distance > 1e-8 else np.array([1.0, 0.0])
        behind = object_position[:2] - self.behind_offset * direction

        if distance <= 0.05:
            target = gripper
        elif phase == 0:
            target = np.array([gripper[0], gripper[1], object_position[2] + self.clearance])
            if abs(gripper[2] - target[2]) < 0.015:
                phase = 1
        elif phase == 1:
            target = np.array([behind[0], behind[1], object_position[2] + self.clearance])
            if np.linalg.norm(gripper[:2] - behind) < 0.015:
                phase = 2
        elif phase == 2:
            target = np.array([behind[0], behind[1], object_position[2] + 0.005])
            if gripper[2] < object_position[2] + 0.015:
                phase = 3
        else:
            target = np.array(
                [
                    goal[0] + self.push_through * direction[0],
                    goal[1] + self.push_through * direction[1],
                    object_position[2] + 0.005,
                ]
            )

        xyz = np.clip((target - gripper) / 0.05, -1.0, 1.0)
        return np.array([*xyz, 0.0], dtype=np.float32), phase

    def get_action(self, info_dict, **kwargs):
        del kwargs
        actions = np.zeros(self.env.action_space.shape, dtype=np.float32)
        observation_key = (
            "observation" if "observation" in info_dict else "state"
        )
        for index in range(self.env.num_envs):
            step = int(np.asarray(self._current(info_dict["step_idx"], index)))
            if step == 0:
                self._phase[index] = 0
            state = self._current(info_dict[observation_key], index)
            action, phase = self._action(state, int(self._phase[index]))
            self._phase[index] = phase
            if self.action_noise:
                action += self.rng.normal(0.0, self.action_noise, action.shape)
            actions[index] = np.clip(action, -1.0, 1.0)
        return actions
