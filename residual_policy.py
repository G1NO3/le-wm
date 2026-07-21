"""Stable-worldmodel policy adapter for episode-persistent residual memory."""

from __future__ import annotations

import numpy as np
import torch
import stable_worldmodel as swm


class ResidualWorldModelPolicy(swm.policy.WorldModelPolicy):
    """Update residual memory from real observations before each MPC solve."""

    def set_env(self, env):
        super().set_env(env)
        self._residual_memory = None
        self._residual_memory_initialized = np.zeros(env.num_envs, dtype=bool)

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

    @torch.no_grad()
    def _update_observed_memory(self, raw_info):
        model = self.residual_model
        if getattr(model, "residual_memory", None) is None:
            return None

        prepared = self._prepare_info(raw_info)
        if "pixels" not in prepared or "action" not in prepared:
            raise KeyError("Persistent residual memory requires pixel and action histories")

        # Online envs report only the last raw env-step action, not the
        # frameskip-chunked, model-rate action history the encoder was trained
        # on. Until the policy reconstructs chunks from its own executed
        # actions, fall back to the reset-memory behavior instead of crashing
        # the evaluation, and log the observed layout once for the fix.
        patch_embed = getattr(self.residual_model.action_encoder, "patch_embed", None)
        expected_act_dim = int(patch_embed.in_channels) if patch_embed is not None else None
        action = prepared["action"]
        if expected_act_dim is not None and (
            action.ndim < 3 or action.shape[-1] != expected_act_dim
        ):
            if not getattr(self, "_warned_action_layout", False):
                self._warned_action_layout = True
                pixels_shape = tuple(prepared["pixels"].shape)
                print(
                    "[residual_policy] online memory disabled: expected action "
                    f"history [batch, time, {expected_act_dim}], got "
                    f"{tuple(action.shape)}; pixels {pixels_shape}. "
                    "Falling back to reset memory."
                )
            batch = prepared["pixels"].shape[0]
            device = next(self.residual_model.parameters()).device
            self._ensure_memory(batch, device=device, dtype=torch.float32)
            return self._residual_memory.detach()
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
