"""JEPA Implementation"""

import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from torch import nn

def detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v

class JEPA(nn.Module):

    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
        residual_flow=None,
        residual_kernel=None,
        residual_memory=None,
        residual_mean=None,
        residual_scale=None,
        residual_conditioning="conditional",
    ):
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()
        self.residual_flow = residual_flow
        self.residual_kernel = residual_kernel
        self.residual_memory = residual_memory
        self.residual_conditioning = residual_conditioning
        self.nominal_frozen = False

        # Planning settings are deliberately runtime-only and do not alter
        # upstream deterministic behavior unless explicitly configured.
        self.planning = {
            "particles": 1,
            "objective": "mean",
            "cvar_tail_fraction": 0.25,
            "flow_steps": 4,
            "particle_chunk_size": 8,
            "common_random_numbers": True,
            "collateral_penalty_weight": 0.0,
            "collateral_state_indices": [],
            "sampling_seed": 3072,
        }

        if residual_scale is None:
            residual_scale = torch.ones(1)
        if residual_mean is None:
            residual_mean = torch.zeros_like(residual_scale)
        has_kernel = residual_flow is not None or residual_kernel is not None
        # Vanilla JEPA state_dicts remain byte-for-byte compatible with
        # upstream checkpoints: optional residual buffers are non-persistent.
        self.register_buffer(
            "residual_mean", residual_mean.float(), persistent=has_kernel
        )
        self.register_buffer(
            "residual_scale", residual_scale.float(), persistent=has_kernel
        )
        self.register_buffer(
            "residual_scale_initialized",
            torch.tensor(False, dtype=torch.bool),
            persistent=has_kernel,
        )
        self.register_buffer(
            "residual_scale_frozen",
            torch.tensor(False, dtype=torch.bool),
            persistent=has_kernel,
        )

    def nominal_components(self):
        return (
            self.encoder,
            self.predictor,
            self.action_encoder,
            self.projector,
            self.pred_proj,
        )

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Load pre-centering residual checkpoints with a documented zero mean."""

        has_kernel = self.residual_flow is not None or getattr(self, "residual_kernel", None) is not None
        if has_kernel and "residual_scale" in state_dict and "residual_mean" not in state_dict:
            state_dict = dict(state_dict)
            state_dict["residual_mean"] = torch.zeros_like(self.residual_mean)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def freeze_nominal(self):
        """Freeze nominal weights and their training-time state transitions."""

        self.nominal_frozen = True
        for component in self.nominal_components():
            component.requires_grad_(False)
            component.eval()
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self, "nominal_frozen", False):
            # Lightning recursively calls train() at every epoch. Keep dropout
            # disabled and BatchNorm buffers immutable in the frozen target
            # model while leaving the residual kernel in training mode.
            for component in self.nominal_components():
                component.eval()
        return self

    def encode(self, info):
        """Encode observations and actions into embeddings.
        info: dict with pixels and action keys
        """

        pixels = info['pixels'].float()
        b = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...") # flatten for encoding
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        pixels_emb = output.last_hidden_state[:, 0]  # cls token
        emb = self.projector(pixels_emb)
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)

        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])

        return info

    def predict(self, emb, act_emb):
        """Predict next state embedding
        emb: (B, T, D)
        act_emb: (B, T, A_emb)
        """
        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0))
        return preds

    def residual_condition(self, emb, act_emb, pred_emb):
        """Build the current-transition condition without recurrent memory."""
        condition = torch.cat([emb, act_emb, pred_emb], dim=-1)
        if getattr(self, "residual_conditioning", "conditional") == "none":
            condition = torch.zeros_like(condition)
        return condition

    def init_residual_memory(self, batch_shape, *, device=None, dtype=None):
        """Create an explicit zero episode state for the residual GRU."""

        memory_module = getattr(self, "residual_memory", None)
        if memory_module is None:
            return None
        reference = self.residual_scale
        return memory_module.init(
            batch_shape,
            device=device or reference.device,
            dtype=dtype or reference.dtype,
        )

    def condition_residual(self, context, memory=None):
        """Append explicit memory to a current-transition condition."""

        memory_module = getattr(self, "residual_memory", None)
        if memory_module is None:
            if memory is not None:
                raise ValueError("A memory tensor was provided to a memoryless kernel")
            return context
        if memory is None:
            memory = self.init_residual_memory(
                context.shape[:-1], device=context.device, dtype=context.dtype
            )
        return memory_module.condition(context, memory)

    def update_residual_memory(
        self, memory, context, normalized_residual, observation_delta=None
    ):
        """Update an explicit memory state after a residual is realized."""

        memory_module = getattr(self, "residual_memory", None)
        if memory_module is None:
            if memory is not None:
                raise ValueError("Cannot update memory on a memoryless kernel")
            return None
        if memory is None:
            memory = self.init_residual_memory(
                context.shape[:-1], device=context.device, dtype=context.dtype
            )
        return memory_module.update(
            memory, context, normalized_residual, observation_delta
        )

    @torch.no_grad()
    def update_residual_statistics(self, residual, decay: float = 0.99, eps: float = 1e-3):
        """Track synchronized residual mean and diagonal scale with an EMA."""
        if not hasattr(self, "residual_mean"):
            has_kernel = self.residual_flow is not None or getattr(self, "residual_kernel", None) is not None
            self.register_buffer(
                "residual_mean",
                torch.zeros_like(self.residual_scale),
                persistent=has_kernel,
            )
        if hasattr(self, "residual_scale_frozen") and bool(self.residual_scale_frozen.item()):
            return
        values = residual.detach().flatten(0, -2).float()
        count = torch.tensor(float(values.size(0)), device=values.device)
        total = values.sum(dim=0)
        square_total = values.square().sum(dim=0)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(count)
            dist.all_reduce(total)
            dist.all_reduce(square_total)
        batch_mean = total / count.clamp_min(1)
        variance = square_total / count.clamp_min(1) - batch_mean.square()
        batch_scale = variance.clamp_min(0).sqrt()
        batch_mean = batch_mean.to(self.residual_scale)
        batch_scale = batch_scale.clamp_min(eps).to(self.residual_scale)

        if not bool(self.residual_scale_initialized.item()):
            self.residual_mean.copy_(batch_mean)
            self.residual_scale.copy_(batch_scale)
            self.residual_scale_initialized.fill_(True)
        else:
            self.residual_mean.mul_(decay).add_(batch_mean, alpha=1.0 - decay)
            self.residual_scale.mul_(decay).add_(batch_scale, alpha=1.0 - decay)
            self.residual_scale.clamp_(min=eps)

    @torch.no_grad()
    def update_residual_scale(self, residual, decay: float = 0.99, eps: float = 1e-3):
        """Backward-compatible alias for centered residual statistics."""

        self.update_residual_statistics(residual, decay=decay, eps=eps)

    @torch.no_grad()
    def freeze_residual_scale(self):
        if not hasattr(self, "residual_scale_frozen"):
            has_kernel = self.residual_flow is not None or getattr(self, "residual_kernel", None) is not None
            self.register_buffer(
                "residual_scale_frozen",
                torch.tensor(False, dtype=torch.bool, device=self.residual_scale.device),
                persistent=has_kernel,
            )
        self.residual_scale_frozen.fill_(True)

    freeze_residual_statistics = freeze_residual_scale

    def normalize_residual(self, residual, eps: float = 1e-3):
        mean = getattr(self, "residual_mean", None)
        if mean is None:
            mean = torch.zeros_like(self.residual_scale)
        mean = mean.to(device=residual.device, dtype=residual.dtype)
        scale = self.residual_scale.to(device=residual.device, dtype=residual.dtype)
        return (residual - mean) / scale.clamp_min(eps)

    def denormalize_residual(self, normalized, eps: float = 1e-3):
        mean = getattr(self, "residual_mean", None)
        if mean is None:
            mean = torch.zeros_like(self.residual_scale)
        mean = mean.to(device=normalized.device, dtype=normalized.dtype)
        scale = self.residual_scale.to(device=normalized.device, dtype=normalized.dtype)
        return mean + normalized * scale.clamp_min(eps)

    def sample_residual(
        self,
        pred_emb,
        condition,
        steps: int = 8,
        noise=None,
        *,
        memory=None,
        uniform_noise=None,
    ):
        """Euler-sample a normalized residual and unwhiten it."""
        if steps < 1:
            raise ValueError("Residual sampling requires steps >= 1")

        conditioned = self.condition_residual(condition, memory)
        if getattr(self, "residual_kernel", None) is not None:
            normalized = self.residual_kernel.sample(
                conditioned,
                noise=noise,
                uniform_noise=uniform_noise,
                steps=steps,
            )
            return self.denormalize_residual(normalized)
        if self.residual_flow is None:
            raise RuntimeError("Residual kernel is not attached to this JEPA model.")

        z = torch.randn_like(pred_emb) if noise is None else noise
        dt = 1.0 / steps
        for i in range(steps):
            tau = torch.full(
                (*z.shape[:-1], 1),
                (i + 0.5) * dt,
                device=z.device,
                dtype=z.dtype,
            )
            z = z + dt * self.residual_flow(tau, z, conditioned)

        return self.denormalize_residual(z)

    def predict_stochastic(
        self, emb, act_emb, steps: int = 8, noise=None, *, memory=None, uniform_noise=None
    ):
        pred_emb = self.predict(emb, act_emb)
        condition = self.residual_condition(emb, act_emb, pred_emb)
        return pred_emb + self.sample_residual(
            pred_emb,
            condition,
            steps,
            noise,
            memory=memory,
            uniform_noise=uniform_noise,
        )

    def observe_transition(
        self,
        memory,
        previous_emb,
        action_emb,
        next_emb,
        *,
        previous_observation=None,
        next_observation=None,
    ):
        """Update memory from one encoded real transition without target leakage."""

        prediction = self.predict(previous_emb, action_emb)[..., -1:, :]
        context = self.residual_condition(
            previous_emb[..., -1:, :], action_emb[..., -1:, :], prediction
        )
        normalized = self.normalize_residual(next_emb[..., -1:, :] - prediction)
        observation_delta = None
        if int(getattr(self.residual_memory, "observation_dim", 0)):
            if previous_observation is None or next_observation is None:
                raise ValueError(
                    "State-conditioned residual memory requires observation history"
                )
            observation_delta = (
                next_observation[..., -1, :] - previous_observation[..., -1, :]
            ).detach()
        updated = self.update_residual_memory(
            memory,
            context.squeeze(-2).detach(),
            normalized.squeeze(-2).detach(),
            observation_delta,
        )
        return updated

    def update_residual_memory_from_history(
        self,
        memory,
        embeddings,
        action_embeddings,
        *,
        observations=None,
        start_target=1,
        history_size=None,
    ):
        """Replay encoded observed transitions into an explicit memory state."""

        if getattr(self, "residual_memory", None) is None:
            return None
        if embeddings.shape[:2] != action_embeddings.shape[:2]:
            raise ValueError("Embedding and action histories must have matching axes")
        if observations is not None and observations.shape[:2] != embeddings.shape[:2]:
            raise ValueError("Observation and embedding histories must have matching axes")
        if memory is None:
            memory = self.init_residual_memory(
                embeddings.shape[:1], device=embeddings.device, dtype=embeddings.dtype
            )
        if history_size is None:
            history_size = int(self.predictor.pos_embedding.size(1))
        for target_index in range(max(1, int(start_target)), embeddings.size(1)):
            context_start = max(0, target_index - int(history_size))
            memory = self.observe_transition(
                memory,
                embeddings[:, context_start:target_index],
                action_embeddings[:, context_start:target_index],
                embeddings[:, target_index : target_index + 1],
                previous_observation=(
                    None
                    if observations is None
                    else observations[:, target_index - 1 : target_index]
                ),
                next_observation=(
                    None
                    if observations is None
                    else observations[:, target_index : target_index + 1]
                ),
            )
        return memory

    ####################
    ## Inference only ##
    ####################

    def rollout(
        self,
        info,
        action_sequence,
        history_size: int = 3,
        stochastic: bool = False,
        flow_steps: int = 8,
        particle_count: int = 1,
        common_random_numbers: bool = False,
        sampling_generator=None,
    ):
        """Rollout the model given an initial info dict and action sequence.
        pixels: (B, S, T, C, H, W)
        action_sequence: (B, S, T, action_dim)
         - S is the number of action plan samples
         - T is the time horizon
        """

        assert "pixels" in info, "pixels not in info_dict"
        H = info["pixels"].size(2)
        B, S, T = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info["action"] = act_0
        n_steps = T - H

        # copy and encode initial info dict
        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        _init = self.encode(_init)
        emb = info["emb"] = _init["emb"].unsqueeze(1).expand(B, S, -1, -1)
        _init = {k: detach_clone(v) for k, v in _init.items()}

        # flatten batch and sample dimensions for rollout
        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")
        residual_memory = None
        if getattr(self, "residual_memory", None) is not None:
            supplied_memory = info.get("residual_memory")
            if supplied_memory is None:
                residual_memory = self.init_residual_memory(
                    (B * S, 1), device=emb.device, dtype=emb.dtype
                )
            else:
                supplied_memory = supplied_memory.to(device=emb.device, dtype=emb.dtype)
                if supplied_memory.ndim == 2:
                    supplied_memory = supplied_memory[:, None].expand(B, S, -1)
                if supplied_memory.shape[:2] != (B, S):
                    raise ValueError(
                        "residual_memory must have [batch, sample, hidden] axes; "
                        f"got {tuple(supplied_memory.shape)}"
                    )
                residual_memory = rearrange(
                    supplied_memory, "b s d -> (b s) 1 d"
                ).clone()

        # rollout predictor autoregressively for n_steps
        HS = history_size
        for t in range(n_steps):
            act_emb = self.action_encoder(act)
            emb_trunc = emb[:, -HS:]  # (BS, HS, D)
            act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
            has_kernel = self.residual_flow is not None or getattr(self, "residual_kernel", None) is not None
            if stochastic and has_kernel:
                noise = self._rollout_noise(
                    pred_shape=(emb_trunc.size(0), 1, emb_trunc.size(2)),
                    batch=B,
                    samples=S,
                    particles=particle_count,
                    common=common_random_numbers,
                    device=emb_trunc.device,
                    dtype=emb_trunc.dtype,
                    generator=sampling_generator,
                )
                uniform_noise = self._rollout_uniform(
                    leading_shape=noise.shape[:-1],
                    batch=B,
                    samples=S,
                    particles=particle_count,
                    common=common_random_numbers,
                    device=emb_trunc.device,
                    dtype=emb_trunc.dtype,
                    generator=sampling_generator,
                )
                nominal = self.predict(emb_trunc, act_trunc)[:, -1:]
                condition = self.residual_condition(
                    emb_trunc[:, -1:], act_trunc[:, -1:], nominal
                )
                residual = self.sample_residual(
                    nominal,
                    condition,
                    steps=flow_steps,
                    noise=noise,
                    memory=residual_memory,
                    uniform_noise=uniform_noise,
                )
                pred_emb = nominal + residual
                if residual_memory is not None:
                    observation_dim = int(
                        getattr(self.residual_memory, "observation_dim", 0)
                    )
                    residual_memory = self.update_residual_memory(
                        residual_memory,
                        condition.detach(),
                        self.normalize_residual(residual).detach(),
                        (
                            condition.new_zeros(
                                *condition.shape[:-1], observation_dim
                            )
                            if observation_dim
                            else None
                        ),
                    )
            else:
                pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
            emb = torch.cat([emb, pred_emb], dim=1)  # (BS, T+1, D)

            next_act = act_future[:, t : t + 1, :]  # (BS, 1, action_dim)
            act = torch.cat([act, next_act], dim=1)  # (BS, T+1, action_dim)

        # predict the last state
        act_emb = self.action_encoder(act)  # (BS, T, A_emb)
        emb_trunc = emb[:, -HS:]  # (BS, HS, D)
        act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
        has_kernel = self.residual_flow is not None or getattr(self, "residual_kernel", None) is not None
        if stochastic and has_kernel:
            noise = self._rollout_noise(
                pred_shape=(emb_trunc.size(0), 1, emb_trunc.size(2)),
                batch=B,
                samples=S,
                particles=particle_count,
                common=common_random_numbers,
                device=emb_trunc.device,
                dtype=emb_trunc.dtype,
                generator=sampling_generator,
            )
            uniform_noise = self._rollout_uniform(
                leading_shape=noise.shape[:-1],
                batch=B,
                samples=S,
                particles=particle_count,
                common=common_random_numbers,
                device=emb_trunc.device,
                dtype=emb_trunc.dtype,
                generator=sampling_generator,
            )
            nominal = self.predict(emb_trunc, act_trunc)[:, -1:]
            condition = self.residual_condition(
                emb_trunc[:, -1:], act_trunc[:, -1:], nominal
            )
            residual = self.sample_residual(
                nominal,
                condition,
                steps=flow_steps,
                noise=noise,
                memory=residual_memory,
                uniform_noise=uniform_noise,
            )
            pred_emb = nominal + residual
        else:
            pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
        emb = torch.cat([emb, pred_emb], dim=1)

        # unflatten batch and sample dimensions
        pred_rollout = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        info["predicted_emb"] = pred_rollout
        info["rollout_context_length"] = H

        return info

    @staticmethod
    def _rollout_noise(
        *, pred_shape, batch, samples, particles, common, device, dtype, generator=None
    ):
        if not common or particles <= 0 or samples % particles:
            return torch.randn(
                pred_shape, device=device, dtype=dtype, generator=generator
            )
        candidates = samples // particles
        base = torch.randn(
            batch,
            1,
            particles,
            *pred_shape[1:],
            device=device,
            dtype=dtype,
            generator=generator,
        )
        return base.expand(batch, candidates, particles, *pred_shape[1:]).reshape(pred_shape)

    @staticmethod
    def _rollout_uniform(
        *, leading_shape, batch, samples, particles, common, device, dtype, generator=None
    ):
        """Draw independent component uniforms with the same CRN layout."""

        if not common or particles <= 0 or samples % particles:
            return torch.rand(
                leading_shape, device=device, dtype=dtype, generator=generator
            )
        candidates = samples // particles
        base = torch.rand(
            batch,
            1,
            particles,
            *leading_shape[1:],
            device=device,
            dtype=dtype,
            generator=generator,
        )
        return base.expand(
            batch, candidates, particles, *leading_shape[1:]
        ).reshape(leading_shape)

    def criterion(self, info_dict: dict):
        """Compute the cost between predicted embeddings and goal embeddings."""
        pred_emb = info_dict["predicted_emb"]  # (B,S, T-1, dim)
        probe = getattr(self, "physical_probe", None)
        if "goal_emb" in info_dict:
            goal_emb = info_dict["goal_emb"]
            if goal_emb.ndim == pred_emb.ndim - 1:
                goal_emb = goal_emb.unsqueeze(1)
            goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb)
            cost = F.mse_loss(
                pred_emb[..., -1:, :],
                goal_emb[..., -1:, :].detach(),
                reduction="none",
            ).sum(dim=tuple(range(2, pred_emb.ndim)))
        elif probe is not None and "goal_position" in info_dict:
            predicted_position = probe(pred_emb[..., -1, :])
            goal_position = info_dict["goal_position"]
            if goal_position.ndim > predicted_position.ndim:
                goal_position = goal_position[..., -1, :]
            goal_position = goal_position.to(predicted_position)
            cost = (predicted_position - goal_position).square().sum(dim=-1)
        else:
            raise KeyError("Planning requires either a goal image or a physical goal probe")

        weight = float(getattr(self, "planning", {}).get("collateral_penalty_weight", 0))
        indices = getattr(self, "planning", {}).get("collateral_state_indices", [])
        if probe is not None and weight > 0 and indices:
            context_index = int(info_dict.get("rollout_context_length", 1)) - 1
            initial_state = probe(pred_emb[..., context_index, :]).detach()
            final_state = probe(pred_emb[..., -1, :])
            selected = torch.as_tensor(indices, device=pred_emb.device, dtype=torch.long)
            collateral = (final_state - initial_state).index_select(-1, selected).square().sum(-1)
            cost = cost + weight * collateral

        return cost

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        """ Compute the cost of action candidates given an info dict with goal and initial state."""

        has_image_goal = "goal" in info_dict
        has_physical_goal = (
            "goal_position" in info_dict
            and getattr(self, "physical_probe", None) is not None
        )
        if not has_image_goal and not has_physical_goal:
            raise KeyError("Planning requires goal pixels or goal_position plus a probe")

        device = next(self.parameters()).device
        action_candidates = action_candidates.to(device)
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)

        if has_image_goal:
            goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
            goal["pixels"] = goal["goal"]
            for k in info_dict:
                if k.startswith("goal_"):
                    goal[k[len("goal_") :]] = goal.pop(k)
            goal.pop("action")
            goal = self.encode(goal)
            info_dict["goal_emb"] = goal["emb"]
        particles = int(getattr(self, "planning", {}).get("particles", 1))
        has_kernel = self.residual_flow is not None or getattr(self, "residual_kernel", None) is not None
        if particles <= 1 or not has_kernel:
            info_dict = self.rollout(info_dict, action_candidates)
            return self.criterion(info_dict)

        from stochastic_metrics import aggregate_particle_cost

        batch, candidates = action_candidates.shape[:2]
        generator_key = (str(action_candidates.device), int(self.planning.get("sampling_seed", 3072)))
        if getattr(self, "_sampling_generator_key", None) != generator_key:
            self._sampling_generator_key = generator_key
            self._sampling_generator = torch.Generator(device=action_candidates.device)
            self._sampling_generator.manual_seed(generator_key[1])
        all_costs = []
        chunk_size = int(self.planning.get("particle_chunk_size", particles))
        for start in range(0, particles, chunk_size):
            count = min(chunk_size, particles - start)
            chunk_actions = action_candidates[:, :, None].expand(
                batch, candidates, count, *action_candidates.shape[2:]
            ).reshape(batch, candidates * count, *action_candidates.shape[2:])
            chunk_info = {}
            for key, value in info_dict.items():
                if key == "goal_emb":
                    chunk_info[key] = value
                elif torch.is_tensor(value) and value.ndim >= 2:
                    chunk_info[key] = value[:, :, None].expand(
                        batch, candidates, count, *value.shape[2:]
                    ).reshape(batch, candidates * count, *value.shape[2:])
                else:
                    chunk_info[key] = value
            chunk_info = self.rollout(
                chunk_info,
                chunk_actions,
                stochastic=True,
                flow_steps=int(self.planning.get("flow_steps", 4)),
                particle_count=count,
                common_random_numbers=bool(
                    self.planning.get("common_random_numbers", True)
                ),
                sampling_generator=self._sampling_generator,
            )
            all_costs.append(self.criterion(chunk_info).reshape(batch, candidates, count))

        particle_cost = torch.cat(all_costs, dim=-1)
        cost = aggregate_particle_cost(
            particle_cost,
            objective=self.planning.get("objective", "mean"),
            tail_fraction=float(self.planning.get("cvar_tail_fraction", 0.25)),
            risk_weight=float(self.planning.get("risk_weight", 1.0)),
        )
        
        return cost
