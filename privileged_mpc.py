"""Analysis-only simulator-particle cost upper bound."""

from __future__ import annotations

import copy

import numpy as np
import torch

from stochastic_metrics import aggregate_particle_cost


class PrivilegedSimulatorParticleCost:
    """Score candidate actions by exact simulator forks.

    This object is intentionally kept separate from deployable latent kernels.
    Set a privileged simulator context and physical goal before giving it to a
    solver. Identical per-particle RNG states are reused across candidates.
    """

    def __init__(
        self,
        wrapper,
        observe,
        terminal_cost,
        *,
        particles=8,
        objective="mean",
        cvar_tail_fraction=0.25,
        seed=3072,
    ):
        self.wrapper = wrapper
        self.observe = observe
        self.terminal_cost = terminal_cost
        self.particles = particles
        self.objective = objective
        self.cvar_tail_fraction = cvar_tail_fraction
        self.rng = np.random.default_rng(seed)
        self.context_state = None
        self.goal_state = None

    def set_context(self, context_state, goal_state):
        self.context_state = copy.deepcopy(context_state)
        self.goal_state = copy.deepcopy(goal_state)

    def get_cost(self, info_dict, action_candidates):
        del info_dict
        if self.context_state is None:
            raise RuntimeError("Call set_context before privileged planning")
        actions = torch.as_tensor(action_candidates).detach().cpu().numpy()
        if actions.shape[0] != 1:
            raise ValueError("Privileged upper bound currently supports batch size one")
        candidates = actions[0]
        particle_seeds = self.rng.integers(0, np.iinfo(np.int32).max, self.particles)
        costs = np.empty((len(candidates), self.particles), dtype=np.float32)
        for candidate_index, sequence in enumerate(candidates):
            for particle_index, seed in enumerate(particle_seeds):
                self.wrapper.set_fork_state(self.context_state)
                self.wrapper.rng = np.random.default_rng(int(seed))
                self.wrapper.resample_physics_mode()
                for action in sequence:
                    _, _, terminated, truncated, _ = self.wrapper.step(action)
                    if terminated or truncated:
                        break
                state = self.observe(self.wrapper.env.unwrapped)
                costs[candidate_index, particle_index] = self.terminal_cost(
                    state, self.goal_state
                )
        tensor = torch.from_numpy(costs).to(action_candidates.device).unsqueeze(0)
        return aggregate_particle_cost(
            tensor,
            objective=self.objective,
            tail_fraction=self.cvar_tail_fraction,
        )
