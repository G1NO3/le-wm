"""Stable-worldmodel policy adapter for episode-persistent residual memory."""

from __future__ import annotations

from collections import deque

import numpy as np
import torch
import stable_worldmodel as swm

from fetch_push_expert import FetchPushExpertPolicy


class FetchGuidedWorldModelPolicy(swm.policy.WorldModelPolicy):
    """CEM initialized around the ordinary-observation Fetch push controller."""

    def __init__(self, *args, expert_seed=3072, **kwargs):
        super().__init__(*args, **kwargs)
        self.expert = FetchPushExpertPolicy(seed=expert_seed)

    def set_env(self, env):
        super().set_env(env)
        self.expert.set_env(env)

    @torch.no_grad()
    def _state_expert_init(self, info_dict, first_action, phases):
        """Roll a phase-aware expert trajectory through ordinary-state dynamics."""

        model = self.solver.model
        raw_state = np.asarray(info_dict["state"])
        if raw_state.ndim >= 3:
            raw_state = raw_state[:, -1]
        device = next(model.parameters()).device
        state = torch.as_tensor(raw_state, device=device, dtype=torch.float32)
        phases = np.asarray(phases, dtype=np.int8).copy()
        first_action = np.asarray(first_action, dtype=np.float32)
        blocks = []
        for block_index in range(self.cfg.horizon):
            raw_steps = []
            for within_block in range(self.cfg.action_block):
                if block_index == 0 and within_block == 0:
                    action = first_action.copy()
                else:
                    predicted = state.detach().cpu().numpy()
                    action = np.empty_like(first_action)
                    for index in range(len(predicted)):
                        action[index], phases[index] = self.expert._action(
                            predicted[index], int(phases[index])
                        )
                raw_steps.append(action)
            raw_block = np.stack(raw_steps, axis=1)
            flat_block = raw_block.reshape(len(raw_block), -1)
            state = model.transition(
                state,
                torch.as_tensor(flat_block, device=device, dtype=state.dtype),
            )
            blocks.append(raw_block)
        raw_plan = np.stack(blocks, axis=1)
        if "action" in self.process:
            shape = raw_plan.shape
            normalized = self.process["action"].transform(
                raw_plan.reshape(-1, shape[-1])
            ).reshape(shape)
        else:
            normalized = raw_plan
        return torch.as_tensor(normalized, dtype=torch.float32).flatten(2)

    def get_action(self, info_dict, **kwargs):
        persistent_flush = info_dict.get("_needs_flush")
        raw_action = self.expert.get_action(info_dict)
        required = getattr(self.solver.model, "required_info_keys", None)
        if required is not None and "state" in info_dict:
            self._next_init = self._state_expert_init(
                info_dict, raw_action, self.expert._phase.copy()
            )
        else:
            if "action" in self.process:
                normalized = self.process["action"].transform(raw_action)
            else:
                normalized = raw_action
            normalized = torch.as_tensor(normalized, dtype=torch.float32)
            block = normalized[:, None, :].expand(
                -1, self.cfg.action_block, -1
            ).reshape(normalized.size(0), 1, -1)
            self._next_init = block.expand(-1, self.cfg.horizon, -1).clone()
        if required is not None:
            keep = set(required) | {"_needs_flush", "terminated"}
            info_dict = {key: value for key, value in info_dict.items() if key in keep}
        action = super().get_action(info_dict, **kwargs)
        # EnvPool reuses its stacked-info dictionary and World.evaluate adds
        # this private marker outside the environment. BasePolicy preprocesses
        # into a copy, so popping it there does not clear the persistent array.
        # Consume it once, then reset it in place so receding-horizon actions
        # survive beyond the first post-reset step.
        if persistent_flush is not None:
            persistent_flush[...] = False
        return action


class ResidualWorldModelPolicy(swm.policy.WorldModelPolicy):
    """Update residual memory from real observations before each MPC solve.

    Two input layouts are supported:

    * **Window** - ``info`` already carries a multi-step, model-rate history
      whose actions match the encoder's expected dimension. The whole window
      is replayed at once (used by dataset-style callers and unit tests).
    * **Online** - a live environment reports one raw frame and the single raw
      action that produced it, at env rate. A model step spans
      ``action_block`` raw steps, so raw actions are accumulated into
      ``frameskip * action_dim`` chunks and encoded frames are buffered across
      calls; memory is updated once per completed model step. Because the
      action normalizer is per-raw-dimension, concatenating the individually
      normalized raw actions reproduces the chunk the encoder was trained on.
    """

    def set_env(self, env):
        super().set_env(env)
        self._reset_memory_state(env.num_envs)

    def _reset_memory_state(self, batch):
        self._residual_memory = None
        self._residual_memory_initialized = np.zeros(batch, dtype=bool)
        self._frame_history = None
        self._action_history = None
        self._raw_action_accum = None

    @property
    def residual_model(self):
        return self.solver.model

    def _ensure_memory(self, batch, *, device, dtype):
        model = self.residual_model
        if self._residual_memory is None or self._residual_memory.size(0) != batch:
            self._residual_memory = model.init_residual_memory(
                (batch,), device=device, dtype=dtype
            )
            self._residual_memory_initialized = np.zeros(batch, dtype=bool)

    def _model_history_size(self):
        predictor = getattr(self.residual_model, "predictor", None)
        pos = getattr(predictor, "pos_embedding", None)
        return int(pos.size(1)) if pos is not None else 3

    def _ensure_online_buffers(self, batch, history_size):
        if self._frame_history is None or len(self._frame_history) != batch:
            self._frame_history = [deque(maxlen=history_size) for _ in range(batch)]
            self._action_history = [deque(maxlen=history_size) for _ in range(batch)]
            self._raw_action_accum = [[] for _ in range(batch)]

    def _expected_action_dim(self):
        patch_embed = getattr(self.residual_model.action_encoder, "patch_embed", None)
        return int(patch_embed.in_channels) if patch_embed is not None else None

    @torch.no_grad()
    def _update_observed_memory(self, raw_info):
        model = self.residual_model
        if getattr(model, "residual_memory", None) is None:
            return None

        prepared = self._prepare_info(raw_info)
        if "pixels" not in prepared or "action" not in prepared:
            raise KeyError("Persistent residual memory requires pixel and action histories")

        expected_dim = self._expected_action_dim()
        action = prepared["action"]
        frameskip = int(getattr(getattr(self, "cfg", None), "action_block", 0) or 0)
        online = (
            expected_dim is not None
            and frameskip > 0
            and action.shape[-1] != expected_dim
            and action.shape[-1] * frameskip == expected_dim
        )
        if online:
            return self._online_memory_update(prepared, raw_info, expected_dim, frameskip)
        if expected_dim is not None and action.shape[-1] != expected_dim:
            if not getattr(self, "_warned_action_layout", False):
                self._warned_action_layout = True
                print(
                    "[residual_policy] online memory disabled: action layout "
                    f"{tuple(action.shape)} is not a frameskip chunking of the "
                    f"expected {expected_dim}-d encoder input. Using reset memory."
                )
            batch = prepared["pixels"].shape[0]
            device = next(model.parameters()).device
            self._ensure_memory(batch, device=device, dtype=torch.float32)
            return self._residual_memory.detach()

        return self._window_memory_update(prepared, raw_info)

    @torch.no_grad()
    def _online_memory_update(self, prepared, raw_info, expected_dim, frameskip):
        model = self.residual_model
        device = next(model.parameters()).device
        pixels = prepared["pixels"].to(device)
        action = torch.nan_to_num(prepared["action"].to(device).float(), 0.0)
        batch = pixels.size(0)
        history_size = self._model_history_size()
        self._ensure_memory(batch, device=device, dtype=torch.float32)
        self._ensure_online_buffers(batch, history_size)

        # A single current frame per env; encode it to a model-rate embedding.
        emb = model.encode({"pixels": pixels})["emb"][:, 0]  # (batch, dim)

        flush = raw_info.get("_needs_flush")
        flush = (
            np.asarray(flush, dtype=bool).reshape(-1)
            if flush is not None
            else np.zeros(batch, dtype=bool)
        )
        terminated = raw_info.get("terminated")
        dead = (
            np.asarray(terminated, dtype=bool).reshape(-1)
            if terminated is not None
            else np.zeros(batch, dtype=bool)
        )

        for index in range(batch):
            if flush[index]:
                self._residual_memory[index].zero_()
                self._residual_memory_initialized[index] = False
                self._frame_history[index].clear()
                self._action_history[index].clear()
                self._raw_action_accum[index].clear()
                self._frame_history[index].append(emb[index])
                continue
            if dead[index]:
                continue

            # The reported action produced the current frame; accumulate it
            # until a full model step (frameskip raw actions) is available.
            self._raw_action_accum[index].append(action[index, 0])
            if len(self._raw_action_accum[index]) < frameskip:
                continue
            chunk = torch.cat(self._raw_action_accum[index][:frameskip], dim=-1)
            del self._raw_action_accum[index][:frameskip]

            if len(self._frame_history[index]) >= 1:
                self._action_history[index].append(chunk)
                span = min(len(self._frame_history[index]), len(self._action_history[index]))
                ctx_emb = torch.stack(list(self._frame_history[index])[-span:])[None]
                ctx_chunks = torch.stack(list(self._action_history[index])[-span:])[None]
                ctx_act = model.action_encoder(ctx_chunks)
                memory = self._residual_memory[index : index + 1]
                memory = model.observe_transition(
                    memory, ctx_emb, ctx_act, emb[index : index + 1][:, None]
                )
                self._residual_memory[index] = memory.squeeze(0)
                self._residual_memory_initialized[index] = True
            self._frame_history[index].append(emb[index])
        return self._residual_memory.detach()

    @torch.no_grad()
    def _window_memory_update(self, prepared, raw_info):
        model = self.residual_model
        device = next(model.parameters()).device
        batch = {
            "pixels": prepared["pixels"].to(device),
            "action": torch.nan_to_num(prepared["action"].to(device), 0.0),
        }
        encoded = model.encode(batch)
        embeddings = encoded["emb"]
        actions = encoded["act_emb"]
        self._ensure_memory(
            embeddings.size(0), device=embeddings.device, dtype=embeddings.dtype
        )

        needs_flush = raw_info.get("_needs_flush")
        if needs_flush is not None:
            flush = np.asarray(needs_flush, dtype=bool).reshape(-1)
            for index in np.flatnonzero(flush):
                self._residual_memory[index].zero_()
                self._residual_memory_initialized[index] = False

        terminated = raw_info.get("terminated")
        dead = (
            np.asarray(terminated, dtype=bool).reshape(-1)
            if terminated is not None
            else np.zeros(embeddings.size(0), dtype=bool)
        )
        for batch_index in range(embeddings.size(0)):
            if dead[batch_index] or embeddings.size(1) < 2:
                continue
            start_target = (
                1 if not self._residual_memory_initialized[batch_index]
                else embeddings.size(1) - 1
            )
            memory = self._residual_memory[batch_index : batch_index + 1]
            memory = model.update_residual_memory_from_history(
                memory,
                embeddings[batch_index : batch_index + 1],
                actions[batch_index : batch_index + 1],
                start_target=start_target,
            )
            self._residual_memory[batch_index] = memory.squeeze(0)
            self._residual_memory_initialized[batch_index] = True
        return self._residual_memory.detach()

    def get_action(self, info_dict, **kwargs):
        model = self.residual_model
        if getattr(model, "residual_memory", None) is None:
            return super().get_action(info_dict, **kwargs)
        info_with_memory = dict(info_dict)
        memory = self._update_observed_memory(info_dict)
        info_with_memory["residual_memory"] = memory
        return super().get_action(info_with_memory, **kwargs)
