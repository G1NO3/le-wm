#!/usr/bin/env python
"""Construct the HDF5 reader plus PushT and hidden-physics OGBench runtimes."""

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
import numpy as np
import stable_worldmodel as swm

from stochastic_physics import OGBenchHiddenPhysics


assert swm.data.HDF5Dataset.__name__ == "HDF5Dataset"

pusht = gym.make("swm/PushT-v1", render_mode="rgb_array")
pusht.reset(seed=3)
pusht.close()

cube = gym.make(
    "swm/OGBCube-v0",
    env_type="double",
    reward_task_id=5,
    ob_type="states",
    terminate_at_goal=True,
)
cube = OGBenchHiddenPhysics(cube, profile="strong")
_, info = cube.reset(seed=7)
action = np.zeros(cube.action_space.shape, dtype=np.float32)
_, _, _, _, step_info = cube.step(action)
assert cube.unwrapped.task_infos[4]["task_name"] == "task5_stack"
assert "privileged/physics_mode" in info
assert np.array_equal(step_info["commanded_action"], action)
cube.close()
print("runtime=ok hdf5 pusht ogbench-double-task5 hidden-physics")
