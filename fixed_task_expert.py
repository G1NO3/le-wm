"""Scripted OGBench oracle policy for fixed multi-cube benchmark tasks."""

from __future__ import annotations

import numpy as np

from ogbench.manipspace.oracles.plan.cube_plan import CubePlanOracle
from stable_worldmodel.policy import BasePolicy


class FixedTaskCubeExpertPolicy(BasePolicy):
    """Sequence OGBench's plan oracle through every cube in a fixed task."""

    def __init__(self, *, seed=3072, action_noise=0.0):
        super().__init__()
        self.seed = seed
        self.action_noise = action_noise
        self.rng = np.random.default_rng(seed)

    def set_seed(self, seed):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def set_env(self, env):
        self.env = env
        self._oracles = [
            CubePlanOracle(raw.unwrapped, noise=0.0, noise_smoothing=0.0)
            for raw in env.envs
        ]
        self._stage = np.zeros(env.num_envs, dtype=np.int64)
        self._orders = [None] * env.num_envs
        self._initialized = np.zeros(env.num_envs, dtype=bool)

    @staticmethod
    def _single_info(info_dict, index):
        return {
            key: value[index][0]
            for key, value in info_dict.items()
            if not key.startswith("_")
        }

    @staticmethod
    def _task_goals(raw):
        """Return the post-permutation target assigned to each physical cube."""

        return np.asarray(raw._data.mocap_pos[raw._cube_target_mocap_ids]).copy()

    @classmethod
    def _task_order(cls, raw):
        goals = cls._task_goals(raw)
        # Supporting cubes must be placed before cubes above them.
        return np.lexsort((np.arange(len(goals)), goals[:, 2]))

    def _reset_oracle(self, index, info):
        raw = self.env.envs[index].unwrapped
        block = int(self._orders[index][self._stage[index]])
        goal = self._task_goals(raw)[block]
        oracle_info = dict(info)
        oracle_info["privileged/target_block"] = block
        oracle_info["privileged/target_block_pos"] = goal
        oracle_info["privileged/target_block_yaw"] = np.asarray(
            info[f"privileged/block_{block}_yaw"]
        ).copy()
        # CubePlanOracle samples only a post-placement arm retreat with NumPy's
        # legacy RNG. Seed that draw reproducibly for each episode/stage.
        np.random.seed(int(self.rng.integers(0, np.iinfo(np.int32).max)))
        self._oracles[index].reset(None, oracle_info)

    def get_action(self, info_dict, **kwargs):
        del kwargs
        actions = np.zeros(self.env.action_space.shape, dtype=np.float32)
        for index in range(self.env.num_envs):
            info = self._single_info(info_dict, index)
            if info["step_idx"] == 0 or not self._initialized[index]:
                raw = self.env.envs[index].unwrapped
                self._orders[index] = self._task_order(raw)
                self._stage[index] = 0
                self._initialized[index] = True
                self._reset_oracle(index, info)

            oracle = self._oracles[index]
            action = oracle.select_action(None, info)
            if self.action_noise:
                action += self.rng.normal(0, self.action_noise, action.shape)
            actions[index] = np.clip(action, -1, 1)

            if oracle.done and self._stage[index] + 1 < len(self._orders[index]):
                self._stage[index] += 1
                self._reset_oracle(index, info)
        return actions
