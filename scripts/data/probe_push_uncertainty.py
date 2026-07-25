"""Go/no-go probe for a hidden-friction push task on OGBench single-cube.

Two questions, both answered here:
  (1) SANITY: does the scripted pusher actually move the cube toward the goal?
      (If not, the fork spread is uninformative.)
  (2) UNCERTAINTY: replaying the same commanded actions under resampled hidden
      friction/mass, how much does the cube diverge, and is that divergence
      PERVASIVE over the horizon (the property quadruple stacking lacked)?

Reports cube-position spread across realizations at each future step. A good
push task shows spread that grows and PERSISTS (does not collapse), unlike
stacking where physical state settled back to near-zero spread.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import numpy as np
import stable_worldmodel  # noqa: F401 registers swm envs

from push_expert import PlanarPushPolicy
from stochastic_physics import PhysicsProfile, OGBenchHiddenPhysics, rollout_simulator_forks


def cube_pos(raw, joint="object_joint_0"):
    return raw._data.joint(joint).qpos[:3].copy().astype(np.float32)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reward-task-id", type=int, default=2)
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--steps", type=int, default=40, help="pusher steps per episode")
    p.add_argument("--contexts", type=int, default=16, help="fork context count")
    p.add_argument("--realizations", type=int, default=24)
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--context-step", type=int, default=8, help="step at which to fork")
    p.add_argument("--seed", type=int, default=3072)
    return p.parse_args()


def make_env(reward_task_id, profile, seed):
    env = gym.make(
        "swm/OGBCube-v0",
        env_type="single",
        reward_task_id=reward_task_id,
        ob_type="states",
        terminate_at_goal=False,
        max_episode_steps=1000,
    )
    return OGBenchHiddenPhysics(env, profile=profile, seed=seed)


def main():
    args = parse_args()
    # Friction + mass are the pervasive push knobs; slip (grasp transient) off.
    profile = PhysicsProfile(
        name="push_strong",
        mass=(0.5, 1.5),
        surface_friction=(0.3, 1.5),
        pad_friction=(1.0, 1.0),
        slip_probability=0.0,
    )

    # (1) SANITY: run the pusher, measure cube->goal progress.
    env = make_env(args.reward_task_id, profile, args.seed)
    raw = env.unwrapped
    policy = PlanarPushPolicy()
    progress = []
    context_windows = []  # (fork_state, action_window, mode) captured mid-push
    for ep in range(args.episodes):
        env.reset(seed=args.seed + ep)
        cube0 = cube_pos(raw)
        goal = raw._data.mocap_pos[raw._cube_target_mocap_ids[0]].copy()
        d0 = float(np.linalg.norm(cube0[:2] - goal[:2]))
        actions = []
        fork_state = None
        for t in range(args.steps):
            a = policy.act(env)
            if t == args.context_step:
                fork_state = env.get_fork_state()
            if fork_state is not None and len(actions) < args.horizon:
                actions.append(a.copy())
            env.step(a)
        cubeT = cube_pos(raw)
        dT = float(np.linalg.norm(cubeT[:2] - goal[:2]))
        moved = float(np.linalg.norm(cubeT[:2] - cube0[:2]))
        progress.append((d0, dT, moved))
        if fork_state is not None and len(actions) == args.horizon and len(context_windows) < args.contexts:
            context_windows.append((fork_state, np.stack(actions)))

    print("=== SANITY: pusher cube->goal (planar, metres) ===")
    print(f"{'ep':>3}{'start_d':>10}{'end_d':>10}{'moved':>10}")
    for i, (d0, dT, mv) in enumerate(progress):
        print(f"{i:>3}{d0:>10.4f}{dT:>10.4f}{mv:>10.4f}")
    arr = np.array(progress)
    print(f"mean start_d {arr[:,0].mean():.4f} -> end_d {arr[:,1].mean():.4f} "
          f"| mean moved {arr[:,2].mean():.4f} | reduced dist on "
          f"{int((arr[:,1] < arr[:,0]).sum())}/{len(arr)} eps")

    # (2) UNCERTAINTY: fork each captured context under resampled physics.
    print(f"\n=== UNCERTAINTY: {len(context_windows)} contexts x {args.realizations} "
          f"realizations x H={args.horizon} ===")
    all_futures = []  # [C, R, H, 3]
    for fork_state, action_window in context_windows:
        futures, _ = rollout_simulator_forks(
            env, fork_state, action_window,
            realizations=args.realizations,
            observe=lambda r: cube_pos(r),
        )
        # futures: list of R rollouts, each list of H cube positions
        arr = np.array([[step for step in roll] for roll in futures], dtype=np.float32)
        if arr.shape[1] == args.horizon:
            all_futures.append(arr)
    env.close()

    if not all_futures:
        print("No fork futures captured; check the pusher / context step.")
        return
    fut = np.stack(all_futures)  # [C, R, H, 3]
    print(f"{'H':>3}{'cube per-dim SD':>18}{'peak-dim SD':>14}{'mean pairwise L2':>18}")
    for h in range(args.horizon):
        x = fut[:, :, h, :]  # [C, R, 3]
        sd = x.std(axis=1).mean()
        peak = x.std(axis=1).max()
        diffs = x[:, :, None, :] - x[:, None, :, :]
        pw = np.sqrt((diffs ** 2).sum(-1)).mean()
        print(f"{h+1:>3}{sd:>18.5f}{peak:>14.5f}{pw:>18.5f}")


if __name__ == "__main__":
    main()
