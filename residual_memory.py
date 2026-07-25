"""Small recurrent state for temporally coherent latent residual kernels."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ResidualMemory(nn.Module):
    """A functional GRU state updated from a condition and realized residual.

    The module never stores an episode state internally.  Callers explicitly
    create, pass, and update the memory tensor so candidate and particle axes
    cannot be mixed accidentally during planning.
    """

    def __init__(
        self,
        *,
        condition_dim: int,
        residual_dim: int,
        hidden_dim: int = 128,
        observation_dim: int = 0,
    ):
        super().__init__()
        self.condition_dim = int(condition_dim)
        self.residual_dim = int(residual_dim)
        self.hidden_dim = int(hidden_dim)
        self.observation_dim = int(observation_dim)
        input_dim = self.condition_dim + self.residual_dim + self.observation_dim
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_projection = nn.Linear(input_dim, self.hidden_dim)
        self.cell = nn.GRUCell(self.hidden_dim, self.hidden_dim)

    def init(self, batch_shape, *, device, dtype):
        if isinstance(batch_shape, int):
            batch_shape = (batch_shape,)
        return torch.zeros(*tuple(batch_shape), self.hidden_dim, device=device, dtype=dtype)

    def condition(self, context: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        if context.shape[:-1] != memory.shape[:-1]:
            raise ValueError(
                "Residual context and memory must have identical leading axes; "
                f"got {tuple(context.shape)} and {tuple(memory.shape)}"
            )
        return torch.cat([context, memory], dim=-1)

    def update(
        self,
        memory: torch.Tensor,
        context: torch.Tensor,
        normalized_residual: torch.Tensor,
        observation_delta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if context.shape[:-1] != memory.shape[:-1]:
            raise ValueError("Residual memory update received mismatched context axes")
        if normalized_residual.shape[:-1] != memory.shape[:-1]:
            raise ValueError("Residual memory update received mismatched residual axes")
        observation_dim = int(getattr(self, "observation_dim", 0))
        if observation_dim:
            if observation_delta is None:
                raise ValueError(
                    "This residual memory requires an observed-state delta"
                )
            if observation_delta.shape[:-1] != memory.shape[:-1]:
                raise ValueError(
                    "Residual memory update received mismatched observation axes"
                )
            if observation_delta.size(-1) != observation_dim:
                raise ValueError(
                    f"Expected observation delta width {observation_dim}, got "
                    f"{observation_delta.size(-1)}"
                )
            values = [context, normalized_residual, observation_delta]
        else:
            values = [context, normalized_residual]
        value = torch.cat(values, dim=-1)
        value = F.silu(self.input_projection(self.input_norm(value)))
        leading = memory.shape[:-1]
        updated = self.cell(
            value.reshape(-1, value.size(-1)),
            memory.reshape(-1, memory.size(-1)),
        )
        return updated.reshape(*leading, self.hidden_dim)


def memory_ablation_states(memory, episode_ids, hidden_modes, *, generator=None):
    """Build correct, reset, and cross-episode/cross-mode memory ablations."""

    if memory.ndim != 2:
        raise ValueError("Memory ablations require [context, hidden] tensors")
    episode_ids = torch.as_tensor(episode_ids, device=memory.device).flatten()
    hidden_modes = torch.as_tensor(hidden_modes, device=memory.device).flatten()
    if episode_ids.numel() != memory.size(0) or hidden_modes.numel() != memory.size(0):
        raise ValueError("Memory labels must have one value per context")
    source_indices = []
    for index in range(memory.size(0)):
        choices = torch.nonzero(
            (episode_ids != episode_ids[index])
            & (hidden_modes != hidden_modes[index]),
            as_tuple=False,
        ).flatten()
        if choices.numel() == 0:
            raise ValueError(
                "Every context needs a different-episode, different-mode shuffle source"
            )
        selected = torch.randint(
            choices.numel(), (1,), device=memory.device, generator=generator
        )
        source_indices.append(choices[selected].squeeze(0))
    source_indices = torch.stack(source_indices)
    return {
        "correct": memory,
        "reset": torch.zeros_like(memory),
        "shuffled": memory.index_select(0, source_indices),
        "shuffled_source_index": source_indices,
    }
