"""Hidden-physics and grasp-slip wrappers for stochastic manipulation studies."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

import gymnasium as gym
import numpy as np


@dataclass(frozen=True)
class PhysicsProfile:
    name: str
    mass: tuple[float, float]
    surface_friction: tuple[float, float]
    pad_friction: tuple[float, float]
    slip_probability: float
    slip_pad_multiplier: float = 0.1
    slip_sim_substeps: int = 10


PHYSICS_PROFILES = {
    "strong": PhysicsProfile(
        "strong", (0.5, 1.5), (0.3, 1.2), (0.2, 1.0), 0.25
    ),
    "medium": PhysicsProfile(
        "medium", (0.75, 1.25), (0.5, 1.0), (0.5, 1.0), 0.15
    ),
    "mild": PhysicsProfile(
        "mild", (0.9, 1.1), (0.7, 1.0), (0.7, 1.0), 0.10
    ),
}


def select_calibrated_profile(pilot_results):
    """Choose the first ordered level satisfying the preregistered pilot gate."""

    for name in ("strong", "medium", "mild"):
        if name not in pilot_results:
            raise KeyError(name)
        result = pilot_results[name]
        if 0.4 <= result["expert_success"] <= 0.8 and result["mode_separation_sd"] >= 1:
            return name
    raise RuntimeError("No physics level passed the 40-80% / one-SD calibration gate")


class HiddenPhysicsWrapper(gym.Wrapper):
    """Apply episode-constant MuJoCo modes without exposing them to observations.

    Mode and slip fields are emitted under ``privileged/*`` and must not appear
    in model ``keys_to_load``. The exact input action is copied to
    ``commanded_action`` before any environment-side control processing.
    """

    def __init__(
        self,
        env,
        *,
        profile="strong",
        seed=None,
        close_threshold=0.5,
        close_when_positive=True,
        target_body_names=None,
    ):
        super().__init__(env)
        self.profile = PHYSICS_PROFILES[profile] if isinstance(profile, str) else profile
        self.rng = np.random.default_rng(seed)
        self.close_threshold = close_threshold
        self.close_when_positive = close_when_positive
        self.target_body_names = target_body_names
        self._closing = False
        self._mode = None
        self._model = None
        self._data = None
        self._baseline_mass = None
        self._baseline_friction = None
        self._cube_bodies = np.array([], dtype=int)
        self._cube_geoms = np.array([], dtype=int)
        self._surface_geoms = np.array([], dtype=int)
        self._pad_geoms = np.array([], dtype=int)
        self._initial_body_positions = None
        self._ever_lifted = None
        self._drop_count = 0
        self._grasp_retries = 0
        self._contact_index = 0

    def _resolve_simulator(self):
        raw = self.env.unwrapped
        if hasattr(raw, "_model") and hasattr(raw, "_data"):
            return raw._model, raw._data
        if hasattr(raw, "sim"):
            return raw.sim.model, raw.sim.data
        raise TypeError("HiddenPhysicsWrapper requires a MuJoCo environment")

    @staticmethod
    def _names(model, kind, count):
        accessor = getattr(model, kind)
        return [str(accessor(index).name or "").lower() for index in range(count)]

    def _discover_indices(self):
        raw = self.env.unwrapped
        if hasattr(raw, "_cube_geom_ids_list"):
            self._cube_geoms = np.unique(
                np.concatenate(raw._cube_geom_ids_list)
            ).astype(int)
            self._cube_bodies = np.unique(
                self._model.geom_bodyid[self._cube_geoms]
            ).astype(int)
        else:
            body_names = self._names(self._model, "body", self._model.nbody)
            patterns = self.target_body_names or ("object", "cube", "target_obj")
            self._cube_bodies = np.array(
                [i for i, name in enumerate(body_names) if any(x in name for x in patterns)],
                dtype=int,
            )
            self._cube_geoms = np.flatnonzero(
                np.isin(self._model.geom_bodyid, self._cube_bodies)
            )

        geom_names = self._names(self._model, "geom", self._model.ngeom)
        self._surface_geoms = np.array(
            [
                i
                for i, name in enumerate(geom_names)
                if any(x in name for x in ("table", "floor", "counter"))
            ],
            dtype=int,
        )
        self._pad_geoms = np.array(
            [
                i
                for i, name in enumerate(geom_names)
                if any(x in name for x in ("pad", "finger", "gripper"))
            ],
            dtype=int,
        )
        if not self._cube_bodies.size or not self._pad_geoms.size:
            raise RuntimeError(
                "Could not identify target bodies and gripper pads; pass target_body_names"
            )

    def _sample_mode(self):
        profile = self.profile
        return {
            "mass_multiplier": float(self.rng.choice(profile.mass)),
            "surface_friction_multiplier": float(
                self.rng.choice(profile.surface_friction)
            ),
            "pad_friction_multiplier": float(self.rng.choice(profile.pad_friction)),
        }

    def _restore_baseline(self):
        if self._model is not None and self._baseline_mass is not None:
            self._model.body_mass[:] = self._baseline_mass
            self._model.geom_friction[:] = self._baseline_friction

    def _apply_mode(self):
        self._restore_baseline()
        self._model.body_mass[self._cube_bodies] *= self._mode["mass_multiplier"]
        object_and_surface_geoms = np.unique(
            np.concatenate([self._cube_geoms, self._surface_geoms])
        )
        self._model.geom_friction[object_and_surface_geoms] *= self._mode[
            "surface_friction_multiplier"
        ]
        self._model.geom_friction[self._pad_geoms] *= self._mode[
            "pad_friction_multiplier"
        ]

    def _privileged_info(self, slip=False):
        profile_id = {"mild": 0, "medium": 1, "strong": 2}.get(
            self.profile.name, -1
        )
        return {
            "privileged/physics_profile": np.array(profile_id, dtype=np.int8),
            "privileged/physics_mode": np.array(
                [
                    self._mode["mass_multiplier"],
                    self._mode["surface_friction_multiplier"],
                    self._mode["pad_friction_multiplier"],
                ],
                dtype=np.float32,
            ),
            "privileged/slip_event": np.array(slip, dtype=np.bool_),
            "privileged/slip_sim_substeps": np.array(
                self.profile.slip_sim_substeps, dtype=np.int32
            ),
            "privileged/contact_index": np.array(
                self._contact_index, dtype=np.int32
            ),
        }

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        observation, info = self.env.reset(seed=seed, options=options)
        self._model, self._data = self._resolve_simulator()
        if self._baseline_mass is None:
            self._baseline_mass = self._model.body_mass.copy()
            self._baseline_friction = self._model.geom_friction.copy()
            self._discover_indices()
        self._mode = self._sample_mode()
        self._apply_mode()
        self._closing = False
        # Dataset-driven evaluation applies its exact qpos/qvel after reset;
        # defer the risk reference until the first commanded step.
        self._initial_body_positions = None
        self._ever_lifted = np.zeros(len(self._cube_bodies), dtype=bool)
        self._drop_count = 0
        self._grasp_retries = 0
        self._contact_index = 0
        info["commanded_action"] = np.zeros(
            self.action_space.shape, dtype=self.action_space.dtype
        )
        info.update(self._privileged_info())
        info.update(self._risk_info())
        return observation, info

    def _is_closing(self, action):
        command = float(np.asarray(action)[-1])
        return command >= self.close_threshold if self.close_when_positive else command <= -self.close_threshold

    def step(self, action):
        if self._initial_body_positions is None:
            self._initial_body_positions = self._data.xpos[self._cube_bodies].copy()
        closing = self._is_closing(action)
        grasp_onset = closing and not self._closing
        if grasp_onset:
            self._grasp_retries += 1
            self._contact_index += 1
        slip = bool(grasp_onset and self.rng.random() < self.profile.slip_probability)
        self._closing = closing
        if slip:
            nominal_pad = self._model.geom_friction[self._pad_geoms].copy()
            self._model.geom_friction[self._pad_geoms] *= self.profile.slip_pad_multiplier
        previous_callback = None
        if slip:
            # MuJoCo's control callback runs once per simulator substep. Chain
            # any controller callback and restore pad friction immediately
            # after the preregistered ten-substep transient.
            try:
                import mujoco

                previous_callback = mujoco.get_mjcb_control()
                counter = {"value": 0}

                def restore_after_substeps(model, data):
                    if previous_callback is not None:
                        previous_callback(model, data)
                    if model is self._model:
                        counter["value"] += 1
                        if counter["value"] > self.profile.slip_sim_substeps:
                            self._model.geom_friction[self._pad_geoms] = nominal_pad

                mujoco.set_mjcb_control(restore_after_substeps)
            except ImportError:
                pass
        try:
            result = self.env.step(action)
        finally:
            if slip:
                self._model.geom_friction[self._pad_geoms] = nominal_pad
                try:
                    mujoco.set_mjcb_control(previous_callback)
                except (ImportError, UnboundLocalError):
                    pass
        observation, reward, terminated, truncated, info = result
        positions = self._data.xpos[self._cube_bodies]
        newly_lifted = positions[:, 2] > self._initial_body_positions[:, 2] + 0.03
        landed_low = self._ever_lifted & (
            positions[:, 2] < self._initial_body_positions[:, 2] + 0.01
        )
        raw = self.env.unwrapped
        if hasattr(raw, "_cube_target_mocap_ids"):
            targets = self._data.mocap_pos[raw._cube_target_mocap_ids]
            at_goal = np.linalg.norm(positions - targets, axis=-1) <= 0.04
        else:
            at_goal = np.zeros(len(positions), dtype=bool)
        # A deliberate low placement at its target is not a dropped object.
        dropped = landed_low & ~at_goal
        self._drop_count += int(dropped.sum())
        self._ever_lifted[landed_low] = False
        self._ever_lifted |= newly_lifted
        info["commanded_action"] = np.asarray(action).copy()
        info.update(self._privileged_info(slip))
        info.update(self._risk_info())
        return observation, reward, terminated, truncated, info

    def _risk_info(self):
        if self._initial_body_positions is None:
            # stable-worldmodel fixes the dataset schema from reset info, so
            # predeclare every risk field before a first action is available.
            return {
                "metrics/target_object_drops": np.array(0, dtype=np.int32),
                "metrics/collateral_displacement": np.array(0.0, dtype=np.float32),
                "metrics/grasp_retries": np.array(0, dtype=np.int32),
                "metrics/final_goal_distance": np.array(0.0, dtype=np.float32),
            }
        positions = self._data.xpos[self._cube_bodies]
        displacement = np.linalg.norm(
            positions[:, :2] - self._initial_body_positions[:, :2], axis=-1
        )
        # By convention the highest-index cube is the active object for the
        # double-stack task; other tasks may override this during analysis.
        collateral = displacement[:-1] if len(displacement) > 1 else np.zeros(1)
        result = {
            "metrics/target_object_drops": np.array(self._drop_count, dtype=np.int32),
            "metrics/collateral_displacement": np.array(collateral.max(), dtype=np.float32),
            "metrics/grasp_retries": np.array(max(0, self._grasp_retries - 1), dtype=np.int32),
        }
        raw = self.env.unwrapped
        if hasattr(raw, "_cube_target_mocap_ids"):
            targets = self._data.mocap_pos[raw._cube_target_mocap_ids]
            result["metrics/final_goal_distance"] = np.array(
                np.linalg.norm(positions - targets, axis=-1).mean(), dtype=np.float32
            )
        return result

    def get_fork_state(self):
        state = {
            "qpos": self._data.qpos.copy(),
            "qvel": self._data.qvel.copy(),
            "mode": copy.deepcopy(self._mode),
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
            "closing": self._closing,
            "initial_body_positions": copy.deepcopy(self._initial_body_positions),
            "ever_lifted": self._ever_lifted.copy(),
            "drop_count": self._drop_count,
            "grasp_retries": self._grasp_retries,
            "contact_index": self._contact_index,
        }
        for name in ("act", "ctrl", "mocap_pos", "mocap_quat", "qacc_warmstart"):
            if hasattr(self._data, name):
                state[name] = getattr(self._data, name).copy()
        if hasattr(self._data, "time"):
            state["time"] = float(self._data.time)
        raw = self.env.unwrapped
        state["environment"] = {
            name: copy.deepcopy(getattr(raw, name))
            for name in ("_prev_qpos", "_prev_qvel", "_success", "_reset_next_step")
            if hasattr(raw, name)
        }
        return state

    def resample_physics_mode(self):
        """Privileged evaluation hook; never call from a deployed policy."""
        self._mode = self._sample_mode()
        self._apply_mode()
        return copy.deepcopy(self._mode)

    def set_fork_state(self, state):
        self._data.qpos[:] = state["qpos"]
        self._data.qvel[:] = state["qvel"]
        self._mode = copy.deepcopy(state["mode"])
        self.rng.bit_generator.state = copy.deepcopy(state["rng_state"])
        self._closing = bool(state["closing"])
        self._initial_body_positions = copy.deepcopy(state.get("initial_body_positions"))
        self._ever_lifted = state.get(
            "ever_lifted", np.zeros(len(self._cube_bodies), dtype=bool)
        ).copy()
        self._drop_count = int(state.get("drop_count", 0))
        self._grasp_retries = int(state.get("grasp_retries", 0))
        self._contact_index = int(state.get("contact_index", 0))
        for name in ("act", "ctrl", "mocap_pos", "mocap_quat", "qacc_warmstart"):
            if name in state and hasattr(self._data, name):
                getattr(self._data, name)[:] = state[name]
        if "time" in state and hasattr(self._data, "time"):
            self._data.time = state["time"]
        raw = self.env.unwrapped
        for name, value in state.get("environment", {}).items():
            setattr(raw, name, copy.deepcopy(value))
        self._apply_mode()
        try:
            import mujoco

            mujoco.mj_forward(self._model, self._data)
        except (ImportError, TypeError):
            if hasattr(self.env.unwrapped, "sim"):
                self.env.unwrapped.sim.forward()

    def physics_spec(self):
        return asdict(self.profile)


class OGBenchHiddenPhysics(HiddenPhysicsWrapper):
    def __init__(self, env, **kwargs):
        # OGBench uses a negative delta to close the gripper. Its plan oracle
        # normally peaks around -0.3, so the generic absolute-command
        # threshold of 0.5 would never detect a grasp onset.
        kwargs.setdefault("close_when_positive", False)
        kwargs.setdefault("close_threshold", 0.05)
        super().__init__(env, **kwargs)


class RoboCasaHiddenPhysics(HiddenPhysicsWrapper):
    def __init__(self, env, **kwargs):
        kwargs.setdefault("close_when_positive", False)
        super().__init__(env, **kwargs)


def rollout_simulator_forks(wrapper, context_state, actions, realizations, observe):
    """Roll identical commanded actions from one state under sampled modes."""

    futures = []
    privileged = []
    fork_rng_state = copy.deepcopy(context_state["rng_state"])
    for _ in range(realizations):
        wrapper.set_fork_state(context_state)
        wrapper.rng.bit_generator.state = copy.deepcopy(fork_rng_state)
        mode = wrapper.resample_physics_mode()
        trajectory = []
        slips = []
        for action in actions:
            _, _, terminated, truncated, info = wrapper.step(action)
            trajectory.append(observe(wrapper.env.unwrapped))
            slips.append(bool(info["privileged/slip_event"]))
            if terminated or truncated:
                break
        futures.append(np.asarray(trajectory))
        privileged.append({"physics_mode": mode, "slip_events": slips})
        fork_rng_state = copy.deepcopy(wrapper.rng.bit_generator.state)
    return futures, privileged
