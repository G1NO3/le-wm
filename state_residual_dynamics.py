"""Low-dimensional Fetch dynamics with an optional flow-matched residual.

The model consumes only the ordinary 28-D Fetch observation and commanded
actions at inference. A deterministic MLP supplies the nominal
two-environment-step transition; a label-free conditional flow models its
25-D observation residual while the final three goal coordinates remain
fixed. Two optional persistent residual heads can either be supervised with
privileged friction modes to establish an explicit-mode upper bound or trained
label-free with balanced complete-episode assignments; neither path receives
the realized mode at planning time.
"""

from __future__ import annotations

import torch
from torch import nn

from residual_flow import ResidualFlow
from stochastic_metrics import aggregate_particle_cost


class StateTransitionMLP(nn.Module):
    def __init__(self, input_dim=36, output_dim=25, hidden_dim=256, depth=4):
        super().__init__()
        layers = []
        for index in range(depth):
            layers.extend(
                [
                    nn.Linear(input_dim if index == 0 else hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                ]
            )
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, state, action_block=None):
        value = state if action_block is None else torch.cat([state, action_block], dim=-1)
        return self.net(value)


class FetchStateResidualDynamics(nn.Module):
    """Planning-compatible nominal and stochastic state transition model."""

    state_dim = 28
    dynamic_dim = 25
    action_dim = 4
    action_block = 2
    required_info_keys = ("state",)

    def __init__(
        self,
        *,
        state_mean,
        state_scale,
        action_mean,
        action_scale,
        delta_mean,
        delta_scale,
        residual_scale,
        hidden_dim=256,
        depth=4,
        flow_hidden_dim=256,
        flow_depth=4,
    ):
        super().__init__()
        self.register_buffer("state_mean", torch.as_tensor(state_mean).float())
        self.register_buffer("state_scale", torch.as_tensor(state_scale).float())
        self.register_buffer("action_mean", torch.as_tensor(action_mean).float())
        self.register_buffer("action_scale", torch.as_tensor(action_scale).float())
        self.register_buffer("delta_mean", torch.as_tensor(delta_mean).float())
        self.register_buffer("delta_scale", torch.as_tensor(delta_scale).float())
        self.register_buffer("residual_scale", torch.as_tensor(residual_scale).float())
        self.nominal = StateTransitionMLP(
            input_dim=self.state_dim + self.action_block * self.action_dim,
            output_dim=self.dynamic_dim,
            hidden_dim=hidden_dim,
            depth=depth,
        )
        condition_dim = self.state_dim + self.action_block * self.action_dim + self.dynamic_dim
        self.residual_flow = ResidualFlow(
            residual_dim=self.dynamic_dim,
            condition_dim=condition_dim,
            hidden_dim=flow_hidden_dim,
            depth=flow_depth,
            time_dim=64,
        )
        self.mode_residual_heads = nn.ModuleList(
            [
                StateTransitionMLP(
                    input_dim=condition_dim,
                    output_dim=self.dynamic_dim,
                    hidden_dim=hidden_dim,
                    depth=depth,
                )
                for _ in range(2)
            ]
        )
        # Optional stochastic-only ablation. Old serialized checkpoints do not
        # carry this attribute and keep their original behavior via getattr.
        self.center_mode_residuals = False
        self.planning = {
            "particles": 1,
            "objective": "mean",
            "flow_steps": 4,
            "common_random_numbers": True,
            "sampling_seed": 3072,
        }

    @staticmethod
    def _safe_scale(scale):
        return torch.clamp(scale, min=1e-6)

    def normalize_state(self, state):
        return (state - self.state_mean) / self._safe_scale(self.state_scale)

    def normalize_raw_actions(self, action):
        mean = self.action_mean.repeat(self.action_block)
        scale = self.action_scale.repeat(self.action_block)
        return (action - mean) / self._safe_scale(scale)

    def nominal_delta(self, state, raw_action_block):
        state_norm = self.normalize_state(state)
        action_norm = self.normalize_raw_actions(raw_action_block)
        normalized_delta = self.nominal(state_norm, action_norm)
        return normalized_delta * self.delta_scale + self.delta_mean

    def residual_condition(self, state, raw_action_block, nominal_delta):
        state_norm = self.normalize_state(state)
        action_norm = self.normalize_raw_actions(raw_action_block)
        delta_norm = (nominal_delta - self.delta_mean) / self._safe_scale(self.delta_scale)
        return torch.cat([state_norm, action_norm, delta_norm], dim=-1)

    def flow_matching_loss(self, state, raw_action_block, actual_delta):
        nominal_delta = self.nominal_delta(state, raw_action_block).detach()
        target = (actual_delta - nominal_delta) / self._safe_scale(self.residual_scale)
        noise = torch.randn_like(target)
        tau = torch.rand(*target.shape[:-1], 1, device=target.device, dtype=target.dtype)
        interpolated = (1 - tau) * noise + tau * target
        velocity = self.residual_flow(
            tau,
            interpolated,
            self.residual_condition(state, raw_action_block, nominal_delta).detach(),
        )
        return (velocity - (target - noise)).square().mean()

    def sample_residual(self, state, raw_action_block, nominal_delta, noise, steps):
        sample = noise
        condition = self.residual_condition(state, raw_action_block, nominal_delta)
        dt = 1.0 / int(steps)
        for index in range(int(steps)):
            tau = sample.new_full((*sample.shape[:-1], 1), index * dt)
            sample = sample + dt * self.residual_flow(tau, sample, condition)
        return sample * self.residual_scale

    def transition(self, state, raw_action_block, *, noise=None, flow_steps=4):
        nominal_delta = self.nominal_delta(state, raw_action_block)
        delta = nominal_delta
        if noise is not None:
            delta = delta + self.sample_residual(
                state, raw_action_block, nominal_delta, noise, flow_steps
            )
        next_state = state.clone()
        next_state[..., : self.dynamic_dim] = state[..., : self.dynamic_dim] + delta
        return next_state

    def _mode_head_predictions(self, condition):
        predictions = torch.stack(
            [head(condition) for head in self.mode_residual_heads], dim=-2
        )
        if getattr(self, "center_mode_residuals", False):
            predictions = predictions - predictions.mean(dim=-2, keepdim=True)
        return predictions

    def mode_residuals(self, state, raw_action_block, nominal_delta=None):
        if nominal_delta is None:
            nominal_delta = self.nominal_delta(state, raw_action_block)
        condition = self.residual_condition(state, raw_action_block, nominal_delta)
        return self._mode_head_predictions(condition) * self.residual_scale

    def transition_mode_mixture(self, state, raw_action_block, mode=None):
        nominal_delta = self.nominal_delta(state, raw_action_block)
        residuals = self.mode_residuals(state, raw_action_block, nominal_delta)
        if mode is None:
            residual = residuals.mean(dim=-2)
        else:
            index = mode.long().unsqueeze(-1).unsqueeze(-1).expand(
                *mode.shape, 1, self.dynamic_dim
            )
            residual = residuals.gather(-2, index).squeeze(-2)
        next_state = state.clone()
        next_state[..., : self.dynamic_dim] = (
            state[..., : self.dynamic_dim] + nominal_delta + residual
        )
        return next_state

    def mode_residual_loss(self, state, raw_action_block, actual_delta, mode):
        nominal_delta = self.nominal_delta(state, raw_action_block).detach()
        target = (actual_delta - nominal_delta) / self._safe_scale(self.residual_scale)
        condition = self.residual_condition(
            state, raw_action_block, nominal_delta
        ).detach()
        predictions = self._mode_head_predictions(condition)
        index = mode.long().unsqueeze(-1).unsqueeze(-1).expand(
            *mode.shape, 1, self.dynamic_dim
        )
        selected = predictions.gather(-2, index).squeeze(-2)
        return (selected - target).square().mean()

    @staticmethod
    def balanced_episode_assignments(transition_loss, episode_ids):
        """Choose one globally balanced residual head per complete episode.

        ``transition_loss`` has shape ``[transitions, 2]``. Assigning after
        averaging within each episode prevents the head identity from changing
        from one transition to the next. The balanced two-component prior is
        known from collection, but no realized friction labels are used.
        """

        if transition_loss.ndim != 2 or transition_loss.size(-1) != 2:
            raise ValueError("transition_loss must have shape [transitions, 2]")
        if episode_ids.shape != transition_loss.shape[:1]:
            raise ValueError("episode_ids must have one value per transition")
        unique_ids, inverse = torch.unique(
            episode_ids.long(), sorted=True, return_inverse=True
        )
        episode_count = len(unique_ids)
        if episode_count == 0:
            raise ValueError("At least one episode is required")
        episode_loss = transition_loss.new_zeros(episode_count, 2)
        episode_loss.index_add_(0, inverse, transition_loss)
        counts = transition_loss.new_zeros(episode_count)
        counts.index_add_(
            0,
            inverse,
            torch.ones_like(inverse, dtype=transition_loss.dtype),
        )
        episode_loss = episode_loss / counts[:, None].clamp_min(1.0)

        if episode_count == 1:
            assignment = episode_loss.argmin(dim=-1)
        else:
            # Low head-0-minus-head-1 advantage belongs to head 0. Exactly
            # half of an even dataset is assigned to each head, preventing the
            # ordinary winner-take-all collapse in a known balanced mixture.
            advantage = episode_loss[:, 0] - episode_loss[:, 1]
            head_zero_count = episode_count // 2
            assignment = torch.ones(
                episode_count,
                device=transition_loss.device,
                dtype=torch.long,
            )
            assignment[torch.argsort(advantage)[:head_zero_count]] = 0
        return unique_ids, inverse, episode_loss, assignment

    def episode_wta_residual_loss(
        self,
        state,
        raw_action_block,
        actual_delta,
        episode_ids,
        *,
        return_assignments=False,
    ):
        """Label-free residual-head loss with episode-consistent assignments."""

        nominal_delta = self.nominal_delta(state, raw_action_block).detach()
        target = (actual_delta - nominal_delta) / self._safe_scale(self.residual_scale)
        condition = self.residual_condition(
            state, raw_action_block, nominal_delta
        ).detach()
        predictions = self._mode_head_predictions(condition)
        transition_loss = (predictions - target.unsqueeze(-2)).square().mean(dim=-1)
        unique_ids, inverse, episode_loss, assignment = (
            self.balanced_episode_assignments(transition_loss.detach(), episode_ids)
        )
        selected_head = assignment[inverse]
        loss = transition_loss.gather(-1, selected_head[:, None]).mean()
        if return_assignments:
            return loss, unique_ids, episode_loss, assignment
        return loss

    @staticmethod
    def paired_episode_assignments(transition_loss, episode_ids):
        """Assign opposite heads within each known scene pair, label-free."""

        if transition_loss.ndim != 2 or transition_loss.size(-1) != 2:
            raise ValueError("transition_loss must have shape [transitions, 2]")
        unique_ids, inverse = torch.unique(
            episode_ids.long(), sorted=True, return_inverse=True
        )
        episode_loss = transition_loss.new_zeros(len(unique_ids), 2)
        episode_loss.index_add_(0, inverse, transition_loss)
        counts = transition_loss.new_zeros(len(unique_ids))
        counts.index_add_(
            0,
            inverse,
            torch.ones_like(inverse, dtype=transition_loss.dtype),
        )
        episode_loss = episode_loss / counts[:, None].clamp_min(1.0)
        if len(unique_ids) % 2:
            raise ValueError("Paired assignment requires an even episode count")
        paired_ids = unique_ids.reshape(-1, 2)
        if not torch.all(
            (paired_ids[:, 0] // 2 == paired_ids[:, 1] // 2)
            & (paired_ids[:, 1] == paired_ids[:, 0] + 1)
        ):
            raise ValueError("Episode IDs must contain complete consecutive scene pairs")

        pair_loss = episode_loss.reshape(-1, 2, 2)
        identity_cost = pair_loss[:, 0, 0] + pair_loss[:, 1, 1]
        swapped_cost = pair_loss[:, 0, 1] + pair_loss[:, 1, 0]
        identity = identity_cost <= swapped_cost
        assignment = torch.stack(
            [torch.where(identity, 0, 1), torch.where(identity, 1, 0)],
            dim=-1,
        ).long().reshape(-1)
        return unique_ids, inverse, episode_loss, assignment

    def paired_object_motion_wta_residual_loss(
        self,
        state,
        raw_action_block,
        actual_delta,
        episode_ids,
        *,
        motion_threshold=1e-4,
        return_assignments=False,
    ):
        """Pair-contrastive modes assigned from informative object motion only."""

        nominal_delta = self.nominal_delta(state, raw_action_block).detach()
        target = (actual_delta - nominal_delta) / self._safe_scale(self.residual_scale)
        condition = self.residual_condition(
            state, raw_action_block, nominal_delta
        ).detach()
        predictions = self._mode_head_predictions(condition)
        squared = (predictions - target.unsqueeze(-2)).square()
        moving = actual_delta[..., 3:6].norm(dim=-1) > float(motion_threshold)
        object_dims = torch.as_tensor(
            [3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19],
            device=squared.device,
        )
        assignment_loss = squared.index_select(-1, object_dims).mean(dim=-1)
        # Keep complete scene pairs even when a weak randomized strike never
        # moves the puck in one member. Such rows carry zero assignment
        # evidence; the informative partner determines the pair orientation.
        # Passing every episode also preserves the strict consecutive-pair
        # invariant enforced by paired_episode_assignments.
        assignment_loss = assignment_loss * moving.to(
            assignment_loss.dtype
        ).unsqueeze(-1)
        unique_ids, _, episode_loss, assignment = self.paired_episode_assignments(
            assignment_loss.detach(), episode_ids
        )
        lookup = torch.searchsorted(unique_ids, episode_ids.long())
        if not torch.all(unique_ids[lookup] == episode_ids.long()):
            raise ValueError("Paired assignment did not cover every episode")
        selected_head = assignment[lookup]
        full_transition_loss = squared.mean(dim=-1)
        loss = full_transition_loss.gather(-1, selected_head[:, None]).mean()
        if return_assignments:
            return loss, unique_ids, episode_loss, assignment
        return loss

    def _raw_action_candidates(self, action_candidates):
        """Undo eval.py's action StandardScaler and enforce env clipping."""
        shape = action_candidates.shape
        blocked = action_candidates.reshape(*shape[:-1], self.action_block, self.action_dim)
        raw = blocked * self.action_scale + self.action_mean
        return raw.clamp(-1.0, 1.0).reshape(*shape[:-1], -1)

    def _generator(self, device):
        key = (str(device), int(self.planning.get("sampling_seed", 3072)))
        if getattr(self, "_sampling_generator_key", None) != key:
            self._sampling_generator_key = key
            self._sampling_generator = torch.Generator(device=device).manual_seed(key[1])
        return self._sampling_generator

    @torch.no_grad()
    def get_cost(self, info_dict, action_candidates):
        """Return an environment-aligned object-goal cost for each candidate.

        FetchPush terminates successfully as soon as the object enters the
        goal tolerance.  When ``minimum_distance`` is enabled, planning must
        therefore score the closest predicted approach anywhere in the
        horizon, rather than only the terminal state.  Terminal-only scoring
        incorrectly rejects plans that pass through the goal and then
        overshoot in the model rollout.
        """
        device = self.state_mean.device
        candidates = self._raw_action_candidates(action_candidates.to(device))
        state = info_dict["state"].to(device)[..., -1, :]
        batch, candidate_count, horizon = candidates.shape[:3]
        particles = int(self.planning.get("particles", 1))
        stochastic = particles > 1
        kernel = self.planning.get("kernel", "flow")

        if stochastic:
            state = state[:, :, None, :].expand(
                batch, candidate_count, particles, self.state_dim
            ).clone()
            candidates = candidates[:, :, None].expand(
                batch, candidate_count, particles, horizon, candidates.size(-1)
            )

        generator = self._generator(device)
        persistent_noise = None
        persistent_mode = None
        if stochastic and kernel == "mode_mixture":
            base_modes = torch.arange(particles, device=device) % 2
            persistent_mode = base_modes.view(1, 1, particles).expand(
                batch, candidate_count, particles
            )
        elif stochastic and self.planning.get("temporal_common_noise", False):
            if self.planning.get("common_random_numbers", True):
                persistent_noise = torch.randn(
                    batch, 1, particles, self.dynamic_dim,
                    device=device, dtype=state.dtype, generator=generator,
                ).expand(batch, candidate_count, particles, self.dynamic_dim)
            else:
                persistent_noise = torch.randn(
                    *state.shape[:-1], self.dynamic_dim,
                    device=device, dtype=state.dtype, generator=generator,
                )
        trajectory_costs = []
        for step in range(horizon):
            if kernel == "mode_mixture":
                state = self.transition_mode_mixture(
                    state, candidates[..., step, :], mode=persistent_mode
                )
            else:
                noise = persistent_noise
                if stochastic and noise is None:
                    if self.planning.get("common_random_numbers", True):
                        noise = torch.randn(
                            batch, 1, particles, self.dynamic_dim,
                            device=device, dtype=state.dtype, generator=generator,
                        ).expand(batch, candidate_count, particles, self.dynamic_dim)
                    else:
                        noise = torch.randn(
                            *state.shape[:-1], self.dynamic_dim,
                            device=device, dtype=state.dtype, generator=generator,
                        )
                state = self.transition(
                    state,
                    candidates[..., step, :],
                    noise=noise,
                    flow_steps=int(self.planning.get("flow_steps", 4)),
                )
            trajectory_costs.append(
                (state[..., 3:6] - state[..., -3:]).square().sum(dim=-1)
            )

        if self.planning.get("minimum_distance", False):
            particle_cost = torch.stack(trajectory_costs, dim=-1).min(dim=-1).values
        else:
            particle_cost = trajectory_costs[-1]
        if self.planning.get("objective") == "success_probability":
            distance = particle_cost.clamp_min(1e-12).sqrt()
            threshold = float(self.planning.get("success_threshold", 0.05))
            temperature = float(self.planning.get("success_temperature", 0.01))
            failure_probability = torch.sigmoid(
                (distance - threshold) / max(temperature, 1e-6)
            )
            return (
                failure_probability.mean(dim=-1)
                if stochastic
                else failure_probability
            )
        if not stochastic:
            return particle_cost
        return aggregate_particle_cost(
            particle_cost,
            objective=self.planning.get("objective", "mean"),
            tail_fraction=float(self.planning.get("cvar_tail_fraction", 0.25)),
            risk_weight=float(self.planning.get("risk_weight", 1.0)),
        )
