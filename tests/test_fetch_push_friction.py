import gymnasium as gym
import numpy as np
import stable_worldmodel  # noqa: F401 - registers the swm environments

from fetch_push_expert import FetchPushExpertPolicy
from scripts.eval.generate_fetch_push_posterior_samples import collect_scene
from stochastic_physics import (
    FetchPushHiddenFriction,
    FetchPushPairedFriction,
    PhysicsProfile,
)


def make_env(multiplier=1.0):
    profile = PhysicsProfile(
        "test", (1.0, 1.0), (multiplier, multiplier), (1.0, 1.0), 0.0
    )
    return FetchPushHiddenFriction(
        gym.make(
            "swm/FetchPush-v3", max_episode_steps=100, render_mode="rgb_array"
        ),
        profile=profile,
    )


def test_fetch_wrapper_changes_only_object_and_surface_friction():
    env = make_env(multiplier=3.0)
    before = None
    _, info = env.reset(seed=7)
    before = env._data.qpos.copy()
    baseline_mass = env._baseline_mass.copy()
    baseline_friction = env._baseline_friction.copy()

    assert np.array_equal(env._model.body_mass, baseline_mass)
    changed = np.unique(np.concatenate([env._cube_geoms, env._surface_geoms]))
    unchanged = np.setdiff1d(np.arange(env._model.ngeom), changed)
    assert np.allclose(env._model.geom_friction[changed], baseline_friction[changed] * 3)
    assert np.array_equal(env._model.geom_friction[unchanged], baseline_friction[unchanged])
    assert info["privileged/friction_multiplier"] == 3.0
    assert info["qpos"].shape == env._data.qpos.shape
    assert info["goal_position"].shape == (3,)
    assert info["goal"].shape[-1] == 3
    assert np.array_equal(env._data.qpos, before)
    assert not np.array_equal(info["goal"], env.render())
    env.close()


def test_fetch_exact_fork_replays_identical_actions():
    env = make_env(multiplier=1.0)
    env.reset(seed=13)
    context = env.get_fork_state()
    actions = [np.array([0.2, -0.1, 0.05, 0.0], dtype=np.float32)] * 5
    for action in actions:
        env.step(action)
    expected = env.get_fork_state()

    env.set_fork_state(context)
    for action in actions:
        env.step(action)
    actual = env.get_fork_state()
    assert np.array_equal(actual["qpos"], expected["qpos"])
    assert np.array_equal(actual["qvel"], expected["qvel"])
    env.close()


def test_paired_friction_uses_same_scene_and_opposite_modes():
    env = FetchPushPairedFriction(
        gym.make(
            "swm/FetchPush-v3", max_episode_steps=100, render_mode="rgb_array"
        ),
        collection_seed_origin=100,
        scene_seed_origin=500,
    )
    low_state, low_info = env.reset(seed=106)
    high_state, high_info = env.reset(seed=107)
    assert np.array_equal(low_state, high_state)
    assert np.array_equal(low_info["goal_position"], high_info["goal_position"])
    assert low_info["privileged/base_seed"] == high_info["privileged/base_seed"]
    assert low_info["privileged/pair_id"] == high_info["privileged/pair_id"] == 3
    assert low_info["privileged/friction_mode"] == 0
    assert high_info["privileged/friction_mode"] == 1
    assert low_info["privileged/friction_multiplier"] == 0.2
    assert high_info["privileged/friction_multiplier"] == 3.0
    env.close()


def test_fetch_expert_solves_low_friction_seeds():
    successes = 0
    for seed in range(8):
        env = make_env(multiplier=0.2)
        state, info = env.reset(seed=seed)
        policy = FetchPushExpertPolicy(seed=seed)
        phase = 0
        for _ in range(100):
            action, phase = policy._action(state, phase)
            state, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        successes += bool(info["success"])
        env.close()
    assert successes >= 7


def test_post_contact_context_keeps_mode_fixed_and_actions_paired():
    env = FetchPushHiddenFriction(
        gym.make(
            "swm/FetchPush-v3", max_episode_steps=100, render_mode="rgb_array"
        ),
        profile=PhysicsProfile(
            "neutral", (1.0, 1.0), (1.0, 1.0), (1.0, 1.0), 0.0
        ),
        terminate_at_goal=False,
    )
    contexts = collect_scene(
        env,
        FetchPushExpertPolicy(seed=0),
        0,
        history_size=3,
        frameskip=2,
        observation_model_steps=2,
        horizon=2,
        modes=(0.2, 3.0),
    )
    env.close()

    assert contexts is not None
    low, high = contexts
    assert low["mode"] == 0.2
    assert high["mode"] == 3.0
    assert low["context_frames"].shape[0] == 3
    assert low["memory_frames"].shape[0] == 3
    assert low["future_frames"].shape[0] == 2
    assert np.array_equal(low["actions"], high["actions"])
    assert np.array_equal(low["memory_actions"], high["memory_actions"])
    assert not np.array_equal(low["future_positions"], high["future_positions"])
