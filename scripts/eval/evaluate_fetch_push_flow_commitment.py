#!/usr/bin/env python
"""Evaluate label-free residual-flow decisions on the FetchPush commitment task.

The flow kernel and nominal model are trained without friction labels. For each
candidate push pulse, antithetic Gaussian base samples are held persistent over
the rollout. The deterministic selector scores their predictive mean, while
the stochastic selector scores balanced expected smooth failure over exactly
the same sampled futures. Exact simulator forks determine actual success.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import numpy as np
import stable_worldmodel  # noqa: F401 - registers swm environments
import torch

from fetch_push_expert import FetchPushExpertPolicy
from scripts.eval.evaluate_fetch_push_commitment import (
    collect_push_context,
    deterministic_index,
    exact_candidate_futures,
    push_pulse_candidates,
    selected_success,
    stochastic_index,
    summarize_pair,
)
from scripts.eval.generate_fetch_push_fork_samples import fixed_profile
from stochastic_physics import FetchPushHiddenFriction


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contexts", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1083072)
    parser.add_argument("--flow-seed", type=int, default=24041)
    parser.add_argument("--flow-particles", type=int, default=32)
    parser.add_argument("--flow-steps", type=int, default=4)
    parser.add_argument("--raw-horizon", type=int, default=20)
    parser.add_argument("--speed-min", type=float, default=0.1)
    parser.add_argument("--speed-max", type=float, default=1.0)
    parser.add_argument("--speed-count", type=int, default=10)
    parser.add_argument("--duration-min", type=int, default=2)
    parser.add_argument("--duration-stride", type=int, default=2)
    parser.add_argument("--success-threshold", type=float, default=0.05)
    parser.add_argument("--success-temperature", type=float, default=0.01)
    parser.add_argument("--bootstrap-draws", type=int, default=20000)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


@torch.no_grad()
def flow_candidate_futures(
    model,
    initial_state,
    candidates,
    device,
    *,
    particles,
    flow_steps,
    seed,
):
    """Return common-random-number flow futures and their predictive mean."""

    if particles < 2 or particles % 2:
        raise ValueError("flow particles must be a positive even number")
    state = torch.as_tensor(initial_state, device=device).float()
    raw = torch.as_tensor(candidates, device=device).float()
    action_blocks = raw.reshape(raw.size(0), raw.size(1) // 2, 8)
    count, horizon = action_blocks.shape[:2]
    state = state[None, None].expand(count, particles, -1).clone()

    generator = torch.Generator(device=device).manual_seed(int(seed))
    base_noise = torch.randn(
        particles // 2,
        model.dynamic_dim,
        device=device,
        dtype=state.dtype,
        generator=generator,
    )
    persistent_noise = torch.cat([base_noise, -base_noise], dim=0)
    persistent_noise = persistent_noise[None].expand(count, -1, -1)

    positions = []
    for step in range(horizon):
        block = action_blocks[:, step, None].expand(-1, particles, -1)
        state = model.transition(
            state,
            block,
            noise=persistent_noise,
            flow_steps=flow_steps,
        )
        positions.append(state[..., 3:6])
    particle_futures = torch.stack(positions, dim=2).cpu().numpy()
    return particle_futures.mean(axis=1), particle_futures


def main():
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite flow commitment result: {args.output}")
    if args.duration_stride <= 0:
        raise SystemExit("--duration-stride must be positive")
    if args.flow_particles < 2 or args.flow_particles % 2:
        raise SystemExit("--flow-particles must be a positive even number")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = torch.load(
        args.checkpoint, map_location=device, weights_only=False
    ).to(device).eval()
    env = FetchPushHiddenFriction(
        gym.make(
            "swm/FetchPush-v3",
            max_episode_steps=100,
            render_mode="rgb_array",
        ),
        profile=fixed_profile(1.0),
        terminate_at_goal=False,
    )
    policy = FetchPushExpertPolicy(seed=args.seed)

    records = []
    candidate_seed = int(args.seed)
    while len(records) < args.contexts:
        context = collect_push_context(env, policy, candidate_seed)
        candidate_seed += 1
        if context is None:
            continue
        candidates, parameters = push_pulse_candidates(
            context["state"],
            raw_horizon=args.raw_horizon,
            speed_min=args.speed_min,
            speed_max=args.speed_max,
            speed_count=args.speed_count,
            duration_min=args.duration_min,
            duration_stride=args.duration_stride,
        )
        exact = exact_candidate_futures(env, context["fork_state"], candidates)
        goal = context["state"][-3:]
        oracle_det_index = deterministic_index(exact.mean(axis=1), goal)
        oracle_stoch_index = stochastic_index(
            exact, goal, args.success_threshold, args.success_temperature
        )

        flow_mean, flow_particles = flow_candidate_futures(
            model,
            context["state"],
            candidates,
            device,
            particles=args.flow_particles,
            flow_steps=args.flow_steps,
            seed=args.flow_seed + context["seed"],
        )
        flow_det_index = deterministic_index(flow_mean, goal)
        flow_stoch_index = stochastic_index(
            flow_particles,
            goal,
            args.success_threshold,
            args.success_temperature,
        )

        record = {
            "seed": context["seed"],
            "oracle_deterministic_index": oracle_det_index,
            "oracle_stochastic_index": oracle_stoch_index,
            "flow_deterministic_index": flow_det_index,
            "flow_stochastic_index": flow_stoch_index,
            "parameters": {
                "oracle_deterministic": parameters[oracle_det_index],
                "oracle_stochastic": parameters[oracle_stoch_index],
                "flow_deterministic": parameters[flow_det_index],
                "flow_stochastic": parameters[flow_stoch_index],
            },
        }
        for prefix, index in (
            ("oracle_deterministic", oracle_det_index),
            ("oracle_stochastic", oracle_stoch_index),
            ("flow_deterministic", flow_det_index),
            ("flow_stochastic", flow_stoch_index),
        ):
            success, distance = selected_success(
                exact, index, goal, args.success_threshold
            )
            record[f"{prefix}_success"] = success.tolist()
            record[f"{prefix}_minimum_distance"] = distance.tolist()
        records.append(record)
        print(f"context {len(records)}/{args.contexts} seed={context['seed']}", flush=True)
    env.close()

    def outcomes(key):
        return np.asarray([record[key] for record in records], dtype=bool)

    oracle = summarize_pair(
        "exact distribution versus exact mean",
        outcomes("oracle_deterministic_success"),
        outcomes("oracle_stochastic_success"),
        draws=args.bootstrap_draws,
        seed=args.seed,
    )
    learned = summarize_pair(
        "label-free persistent flow samples versus their predictive mean",
        outcomes("flow_deterministic_success"),
        outcomes("flow_stochastic_success"),
        draws=args.bootstrap_draws,
        seed=args.seed + 1,
    )
    result = {
        "task": "fetch_push_hidden_friction_flow_commitment",
        "checkpoint": str(args.checkpoint.resolve()),
        "contexts": args.contexts,
        "episodes": 2 * args.contexts,
        "context_seeds": [record["seed"] for record in records],
        "friction_modes": [0.2, 3.0],
        "candidate_grid": {
            "raw_horizon": args.raw_horizon,
            "speed_min": args.speed_min,
            "speed_max": args.speed_max,
            "speed_count": args.speed_count,
            "duration_min": args.duration_min,
            "duration_stride": args.duration_stride,
            "candidate_count": len(parameters),
        },
        "flow_sampling": {
            "particles": args.flow_particles,
            "flow_steps": args.flow_steps,
            "base_seed": args.flow_seed,
            "noise": "context-seeded antithetic Gaussian",
            "temporal_coupling": "same base sample held over model horizon",
            "candidate_coupling": "common random numbers",
            "uses_privileged_mode_labels": False,
        },
        "success_threshold": args.success_threshold,
        "success_temperature": args.success_temperature,
        "oracle_decision_value": oracle,
        "learned_decision_value": learned,
        "task_gate": bool(
            oracle["strict_improvement_gate"]
            and learned["absolute_success_improvement"] > 0
        ),
        "one_validation_screen_only": True,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
