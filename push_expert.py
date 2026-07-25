"""Scripted non-prehensile planar push policy for the OGBench single-cube env.

Keeps the gripper closed and drives the end-effector behind the cube (on the
far side from the goal), then pushes through it toward the goal. Reads the
MuJoCo state directly from the unwrapped env, so it works under the
HiddenPhysicsWrapper (which only rescales mass/friction, not geometry).

This is a data/probe policy, not an optimal controller: it is good enough to
generate pushing trajectories where the cube's motion depends pervasively on
the hidden friction/mass.
"""

from __future__ import annotations

import numpy as np

# Per-step effector delta range is [-0.05, 0.05] m in x/y/z (action_range in
# ogbench manipspace). A normalized action of magnitude 1 moves 0.05 m.
_XYZ_STEP = 0.05
_CUBE_HALF = 0.02  # cube half-width; behind offset must clear it


class PlanarPushPolicy:
    """Closed-loop planar pusher for a single object-to-goal cube task."""

    def __init__(self, *, behind_offset=0.07, safe_z=0.12, push_z=0.03,
                 gripper=-1.0, object_joint="object_joint_0"):
        self.behind_offset = behind_offset
        self.safe_z = safe_z
        self.push_z = push_z
        self.gripper = gripper
        self.object_joint = object_joint

    @staticmethod
    def _raw(env):
        return env.unwrapped

    def _read(self, env):
        raw = self._raw(env)
        eff = raw._data.site_xpos[raw._pinch_site_id].copy()
        cube = raw._data.joint(self.object_joint).qpos[:3].copy()
        goal = raw._data.mocap_pos[raw._cube_target_mocap_ids[0]].copy()
        return eff, cube, goal

    def act(self, env, return_phase=False):
        eff, cube, goal = self._read(env)
        push_dir = goal[:2] - cube[:2]
        dist = np.linalg.norm(push_dir)
        push_dir = push_dir / dist if dist > 1e-6 else np.array([1.0, 0.0])

        behind = cube[:2] - push_dir * self.behind_offset
        above_behind = np.linalg.norm(eff[:2] - behind) < 0.02
        at_push_z = eff[2] < self.push_z + 0.015

        # Three-phase state machine that never descends over the cube:
        #   lift -> reposition behind (high) -> descend clear -> push through.
        if not above_behind and eff[2] < self.safe_z - 0.02:
            phase = "lift"
            target = np.array([eff[0], eff[1], self.safe_z])
        elif not above_behind:
            phase = "reposition"
            target = np.array([behind[0], behind[1], self.safe_z])
        elif not at_push_z:
            phase = "descend"
            target = np.array([behind[0], behind[1], self.push_z])
        else:
            phase = "push"  # drive toward the goal at push height, through the cube
            target = np.array([goal[0], goal[1], self.push_z])

        delta = target - eff
        a_xyz = np.clip(delta / _XYZ_STEP, -1.0, 1.0)
        action = np.array([a_xyz[0], a_xyz[1], a_xyz[2], 0.0, self.gripper], dtype=np.float32)
        return (action, phase) if return_phase else action

    # Convenience for batched swm.World collection (one env at a time).
    def get_action(self, info_dict, **kwargs):
        raise NotImplementedError(
            "PlanarPushPolicy.act(env) reads the live simulator; use it in a "
            "single-env collection loop, not the batched World policy API."
        )
