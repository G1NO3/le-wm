"""Trace one push episode: effector, cube, goal, action, phase per step."""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("MUJOCO_GL", "egl")
import gymnasium as gym
import numpy as np
import stable_worldmodel  # noqa
from push_expert import PlanarPushPolicy
from stochastic_physics import PhysicsProfile, OGBenchHiddenPhysics

profile = PhysicsProfile("push_strong", (0.5,1.5), (0.3,1.5), (1.0,1.0), 0.0)
env = gym.make("swm/OGBCube-v0", env_type="single", reward_task_id=2,
               ob_type="states", terminate_at_goal=False, max_episode_steps=1000)
env = OGBenchHiddenPhysics(env, profile=profile, seed=3072)
raw = env.unwrapped
pol = PlanarPushPolicy()
env.reset(seed=3072)
np.set_printoptions(precision=3, suppress=True)
print("pinch_site_id", raw._pinch_site_id, "target_mocap", raw._cube_target_mocap_ids)
cube0 = raw._data.joint("object_joint_0").qpos[:3].copy()
goal = raw._data.mocap_pos[raw._cube_target_mocap_ids[0]].copy()
print("cube0", cube0, "goal", goal, "arm_bounds", raw._arm_sampling_bounds.tolist())
for t in range(60):
    eff = raw._data.site_xpos[raw._pinch_site_id].copy()
    cube = raw._data.joint("object_joint_0").qpos[:3].copy()
    a, phase = pol.act(env, return_phase=True)
    if t % 4 == 0 or t < 5:
        cd = np.linalg.norm(cube[:2] - goal[:2])
        print(f"t={t:2d} {phase:10s} eff={eff} cube={cube} cube->goal={cd:.3f}")
    env.step(a)
cubeT = raw._data.joint("object_joint_0").qpos[:3].copy()
print("cubeT", cubeT, "moved", np.linalg.norm(cubeT[:2]-cube0[:2]))
env.close()
