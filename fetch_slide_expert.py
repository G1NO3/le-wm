"""Paired open-loop strike policy for friction-varying FetchSlide."""

from __future__ import annotations

import numpy as np

from fetch_push_expert import FetchPushExpertPolicy


class FetchSlideStrikePolicy(FetchPushExpertPolicy):
    """Approach from ordinary state, then execute one pair-matched strike.

    Strike parameters are a deterministic function of scene-pair identity, so
    low/high episodes receive identical commanded actions.  The policy never
    reads the realized friction mode.
    """

    def __init__(
        self,
        *,
        seed=4072,
        speed_min=0.4,
        speed_max=1.0,
        speed_count=13,
        duration_min=1,
        duration_max=4,
        angle_max_deg=3.0,
    ):
        super().__init__(
            seed=seed,
            action_noise=0.0,
            behind_offset=0.075,
            clearance=0.08,
        )
        self.speed_min = float(speed_min)
        self.speed_max = float(speed_max)
        self.speed_count = int(speed_count)
        self.duration_min = int(duration_min)
        self.duration_max = int(duration_max)
        self.angle_max_deg = float(angle_max_deg)
        if self.speed_count <= 0:
            raise ValueError("speed_count must be positive")
        if not 1 <= self.duration_min <= self.duration_max:
            raise ValueError("durations must satisfy 1 <= min <= max")

    def set_env(self, env):
        super().set_env(env)
        self._pulse_remaining = np.zeros(env.num_envs, dtype=np.int16)
        self._pulse_action = np.zeros((*env.action_space.shape,), dtype=np.float32)
        self._pulse_initialized = np.zeros(env.num_envs, dtype=bool)
        self._pair_occurrences = {}
        self._paired_action_cache = {}
        self._active_pair = np.full(env.num_envs, -1, dtype=np.int64)
        self._replay_pair = np.zeros(env.num_envs, dtype=bool)

    def _strike(self, state, pair_id):
        rng = np.random.default_rng(self.seed + int(pair_id))
        speeds = np.linspace(
            self.speed_min, self.speed_max, self.speed_count, dtype=np.float32
        )
        speed = float(rng.choice(speeds))
        duration = int(rng.integers(self.duration_min, self.duration_max + 1))
        angle = float(
            np.deg2rad(rng.uniform(-self.angle_max_deg, self.angle_max_deg))
        )
        object_position = np.asarray(state[3:6], dtype=np.float32)
        goal = np.asarray(state[-3:], dtype=np.float32)
        direction = goal[:2] - object_position[:2]
        direction /= max(float(np.linalg.norm(direction)), 1e-8)
        cosine, sine = np.cos(angle), np.sin(angle)
        rotated = np.array(
            [
                cosine * direction[0] - sine * direction[1],
                sine * direction[0] + cosine * direction[1],
            ],
            dtype=np.float32,
        )
        action = np.zeros(4, dtype=np.float32)
        action[:2] = speed * rotated
        return action, duration

    def get_action(self, info_dict, **kwargs):
        del kwargs
        actions = np.zeros(self.env.action_space.shape, dtype=np.float32)
        observation_key = (
            "observation" if "observation" in info_dict else "state"
        )
        if "privileged/pair_id" not in info_dict:
            raise KeyError("Paired FetchSlide collection requires privileged/pair_id")
        for index in range(self.env.num_envs):
            step = int(np.asarray(self._current(info_dict["step_idx"], index)))
            pair_id = int(
                np.asarray(
                    self._current(info_dict["privileged/pair_id"], index)
                )
            )
            if step == 0:
                self._phase[index] = 0
                self._pulse_remaining[index] = 0
                self._pulse_initialized[index] = False
                occurrence = self._pair_occurrences.get(pair_id, 0)
                if occurrence >= 2:
                    raise RuntimeError(
                        f"Scene pair {pair_id} appeared more than twice"
                    )
                self._pair_occurrences[pair_id] = occurrence + 1
                self._active_pair[index] = pair_id
                self._replay_pair[index] = occurrence == 1
                if occurrence == 0:
                    self._paired_action_cache[pair_id] = []
            if self._active_pair[index] != pair_id:
                raise RuntimeError("FetchSlide pair identity changed within an episode")
            if self._replay_pair[index]:
                cached = self._paired_action_cache[pair_id]
                if step >= len(cached):
                    raise RuntimeError(
                        f"Paired strike replay exhausted at step {step}"
                    )
                actions[index] = cached[step]
                continue
            state = self._current(info_dict[observation_key], index)
            if int(self._phase[index]) < 3:
                action, phase = self._action(state, int(self._phase[index]))
                self._phase[index] = phase
                actions[index] = action
            elif not self._pulse_initialized[index]:
                action, duration = self._strike(state, pair_id)
                self._pulse_action[index] = action
                self._pulse_remaining[index] = duration
                self._pulse_initialized[index] = True
                actions[index] = action
                self._pulse_remaining[index] -= 1
            elif self._pulse_remaining[index] > 0:
                actions[index] = self._pulse_action[index]
                self._pulse_remaining[index] -= 1
            self._paired_action_cache[pair_id].append(actions[index].copy())
        return actions
