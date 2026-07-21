from types import SimpleNamespace

import json

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from experiment_data import (
    build_collection_manifest,
    episode_disjoint_split,
    sha256_json,
)
from jepa import JEPA
from residual_kernels import (
    ConditionalDiagonalGaussian,
    ConditionalGaussianMixture,
    GlobalDiagonalGaussian,
)
from residual_memory import ResidualMemory, memory_ablation_states
from residual_policy import ResidualWorldModelPolicy
from stochastic_metrics import aggregate_particle_cost
from train import residual_kernel_loss


class Encoder(nn.Module):
    def __init__(self, dim=4):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, pixels, interpolate_pos_encoding=True):
        del interpolate_pos_encoding
        value = pixels.flatten(1).mean(1, keepdim=True) + self.bias
        return SimpleNamespace(last_hidden_state=value[:, None])


class Predictor(nn.Module):
    def forward(self, emb, act):
        return emb + 0.0 * act[..., : emb.size(-1)]


def model(kernel=None):
    return JEPA(
        encoder=Encoder(),
        predictor=Predictor(),
        action_encoder=nn.Identity(),
        residual_kernel=kernel,
        residual_scale=torch.ones(4) if kernel is not None else None,
    )


def test_vanilla_checkpoint_loads_strict_without_residual_keys():
    vanilla = model()
    state = vanilla.state_dict()
    assert not any(key.startswith("residual_") for key in state)
    model().load_state_dict(state, strict=True)


def test_pre_centering_residual_checkpoint_loads_with_zero_mean():
    source = model(GlobalDiagonalGaussian(4))
    state = source.state_dict()
    del state["residual_mean"]
    target = model(GlobalDiagonalGaussian(4))
    target.residual_mean.fill_(7)
    target.load_state_dict(state, strict=True)
    assert torch.equal(target.residual_mean, torch.zeros(4))


def test_validation_does_not_update_residual_statistics():
    wm = model(GlobalDiagonalGaussian(4))
    cfg = OmegaConf.create(
        {
            "loss": {
                "residual_flow": {
                    "ema_decay": 0.9,
                    "scale_eps": 1e-3,
                    "detach_residual_target": True,
                    "detach_condition": True,
                }
            }
        }
    )
    tensors = [torch.randn(3, 2, 4) for _ in range(4)]
    before = wm.residual_scale.clone()
    before_mean = wm.residual_mean.clone()
    residual_kernel_loss(wm, *tensors, cfg, update_statistics=False)
    assert torch.equal(before, wm.residual_scale)
    assert torch.equal(before_mean, wm.residual_mean)
    assert not wm.residual_scale_initialized


def test_frozen_nominal_stays_in_eval_when_residual_head_trains():
    wm = model(ConditionalDiagonalGaussian(12, 4, hidden_dim=8, depth=1))
    wm.encoder.dropout = nn.Dropout(0.5)
    wm.freeze_nominal()

    wm.train()

    assert wm.residual_kernel.training
    assert all(not component.training for component in wm.nominal_components())
    assert all(
        not parameter.requires_grad
        for component in wm.nominal_components()
        for parameter in component.parameters()
    )


@pytest.mark.parametrize(
    "kernel",
    [ConditionalDiagonalGaussian(12, 4, hidden_dim=8, depth=1)],
)
def test_seeded_kernel_sampling_accepts_common_noise(kernel):
    condition = torch.randn(2, 3, 12)
    noise = torch.randn(2, 3, 4, generator=torch.Generator().manual_seed(7))
    first = kernel.sample(condition, noise=noise)
    second = kernel.sample(condition, noise=noise)
    assert torch.equal(first, second)
    assert first.shape == noise.shape


def test_gmm_uses_independent_component_uniforms():
    kernel = ConditionalGaussianMixture(12, 4, hidden_dim=8, depth=1)
    with torch.no_grad():
        kernel.out.weight.zero_()
        raw = kernel.out.bias.reshape(2, 1 + 2 * 4)
        raw.zero_()
        raw[0, 1:5] = -3
        raw[1, 1:5] = 3
        raw[:, 5:] = -20
    condition = torch.zeros(1, 1, 12)
    gaussian = torch.zeros(1, 1, 4)
    first = kernel.sample(
        condition, noise=gaussian, uniform_noise=torch.full((1, 1), 0.1)
    )
    second = kernel.sample(
        condition, noise=gaussian, uniform_noise=torch.full((1, 1), 0.9)
    )
    assert torch.all(first < 0)
    assert torch.all(second > 0)


def test_centered_residual_normalization_round_trips():
    wm = model(GlobalDiagonalGaussian(4))
    wm.residual_mean.copy_(torch.tensor([1.0, -2.0, 0.5, 4.0]))
    wm.residual_scale.copy_(torch.tensor([2.0, 4.0, 0.5, 8.0]))
    residual = torch.tensor([[3.0, 2.0, 1.0, -4.0]])
    normalized = wm.normalize_residual(residual)
    assert torch.allclose(normalized, torch.tensor([[1.0, 1.0, 1.0, -1.0]]))
    assert torch.allclose(wm.denormalize_residual(normalized), residual)


def test_distributed_statistics_match_concatenated_workers(monkeypatch):
    wm = model(GlobalDiagonalGaussian(4))
    local = torch.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]])
    remote = torch.tensor([[5.0, 4.0, 3.0, 2.0], [6.0, 5.0, 4.0, 3.0]])
    calls = iter(
        [
            torch.tensor(float(remote.size(0))),
            remote.sum(0),
            remote.square().sum(0),
        ]
    )
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda value: value.add_(next(calls)))
    wm.update_residual_statistics(local)
    joined = torch.cat([local, remote])
    assert torch.allclose(wm.residual_mean, joined.mean(0))
    assert torch.allclose(wm.residual_scale, joined.std(0, unbiased=False))


def recurrent_model():
    memory = ResidualMemory(condition_dim=12, residual_dim=4, hidden_dim=5)
    kernel = ConditionalDiagonalGaussian(17, 4, hidden_dim=8, depth=1)
    predictor = Predictor()
    predictor.pos_embedding = nn.Parameter(torch.zeros(1, 3, 4))
    return JEPA(
        encoder=Encoder(),
        predictor=predictor,
        action_encoder=nn.Identity(),
        residual_kernel=kernel,
        residual_memory=memory,
        residual_mean=torch.zeros(4),
        residual_scale=torch.ones(4),
    )


def test_residual_memory_is_explicit_resettable_state():
    wm = recurrent_model()
    first = wm.init_residual_memory((2, 3), device=torch.device("cpu"), dtype=torch.float32)
    second = wm.init_residual_memory((2, 3), device=torch.device("cpu"), dtype=torch.float32)
    assert first.shape == (2, 3, 5)
    assert torch.equal(first, second)
    context = torch.randn(2, 3, 12)
    updated = wm.update_residual_memory(first, context, torch.randn(2, 3, 4))
    assert updated.shape == first.shape
    assert not torch.equal(updated, first)
    assert torch.equal(first, torch.zeros_like(first))


def test_final_target_never_enters_its_own_memory():
    wm = recurrent_model()
    context = torch.randn(2, 3, 12)
    residual = torch.randn(2, 3, 4)
    changed = residual.clone()
    changed[:, -1] += 100
    first = wm.init_residual_memory((2,), device=context.device, dtype=context.dtype)
    second = first.clone()
    for index in range(2):
        first = wm.update_residual_memory(first, context[:, index], residual[:, index])
        second = wm.update_residual_memory(second, context[:, index], changed[:, index])
    assert torch.equal(first, second)


def test_memory_ablation_shuffle_changes_episode_and_mode():
    memory = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    episodes = torch.tensor([10, 11, 12, 13])
    modes = torch.tensor([0, 0, 1, 1])
    states = memory_ablation_states(
        memory,
        episodes,
        modes,
        generator=torch.Generator().manual_seed(3),
    )
    sources = states["shuffled_source_index"]
    assert torch.equal(states["correct"], memory)
    assert torch.equal(states["reset"], torch.zeros_like(memory))
    assert torch.all(episodes[sources] != episodes)
    assert torch.all(modes[sources] != modes)


def test_final_token_loss_trains_residual_memory():
    wm = recurrent_model()
    nn.init.normal_(wm.residual_kernel.out.weight, std=0.1)
    cfg = OmegaConf.create(
        {
            "loss": {
                "residual_flow": {
                    "ema_decay": 0.9,
                    "scale_eps": 1e-3,
                    "detach_residual_target": True,
                    "detach_condition": True,
                }
            }
        }
    )
    tensors = [torch.randn(3, 3, 4) for _ in range(4)]
    loss = residual_kernel_loss(wm, *tensors, cfg, update_statistics=False)
    loss.backward()
    gradient = sum(
        parameter.grad.abs().sum()
        for parameter in wm.residual_memory.parameters()
        if parameter.grad is not None
    )
    assert gradient > 0


def test_online_policy_persists_and_resets_episode_memory():
    wm = recurrent_model().eval()
    policy = ResidualWorldModelPolicy.__new__(ResidualWorldModelPolicy)
    policy.solver = SimpleNamespace(model=wm)
    policy.process = {}
    policy.transform = {}
    policy._residual_memory = None
    policy._residual_memory_initialized = np.zeros(1, dtype=bool)
    info = {
        "pixels": torch.randn(1, 3, 1, 2, 2),
        "action": torch.randn(1, 3, 4),
        "_needs_flush": np.array([True]),
    }
    first = policy._update_observed_memory(info).clone()
    assert not torch.equal(first, torch.zeros_like(first))
    continued = policy._update_observed_memory(
        {**info, "_needs_flush": np.array([False])}
    ).clone()
    assert not torch.equal(first, continued)
    reset = policy._update_observed_memory(info).clone()
    assert torch.allclose(first, reset)


def test_mean_and_worst_quartile_cvar():
    cost = torch.tensor([[[1.0, 2.0, 3.0, 20.0], [2.0, 2.0, 2.0, 2.0]]])
    assert torch.equal(aggregate_particle_cost(cost, "mean"), torch.tensor([[6.5, 2.0]]))
    assert torch.equal(
        aggregate_particle_cost(cost, "cvar", 0.25), torch.tensor([[20.0, 2.0]])
    )


class ClipDataset:
    lengths = np.array([4, 5, 6, 7, 8, 9])
    frameskip = 1
    num_steps = 2
    clip_indices = [
        (episode, start)
        for episode, length in enumerate(lengths)
        for start in range(length - 1)
    ]

    def __len__(self):
        return len(self.clip_indices)

    def __getitem__(self, index):
        return index


def test_episode_manifest_is_disjoint_and_immutable(tmp_path):
    dataset = ClipDataset()
    path = tmp_path / "split.json"
    subsets, manifest = episode_disjoint_split(
        dataset, path, seed=3, fractions=(0.5, 1 / 6, 1 / 3)
    )
    episode_groups = [set(x) for x in manifest["episodes"].values()]
    assert not (episode_groups[0] & episode_groups[1])
    assert not (episode_groups[0] & episode_groups[2])
    assert sum(len(subset) for subset in subsets.values()) == len(dataset)
    _, reused = episode_disjoint_split(
        dataset, path, seed=4, fractions=(0.5, 1 / 6, 1 / 3)
    )
    assert reused == manifest


def test_precomputed_collection_manifest_is_accepted_across_model_seeds(tmp_path):
    dataset = ClipDataset()
    path = tmp_path / "split_v2.json"
    fingerprint = sha256_json(
        {
            "episode_lengths": dataset.lengths.tolist(),
            "frameskip": dataset.frameskip,
            "num_steps": dataset.num_steps,
        }
    )
    manifest = build_collection_manifest(
        dataset.lengths,
        fingerprint,
        seed=13,
        fractions=(0.5, 1 / 6, 1 / 3),
    )
    path.write_text(json.dumps(manifest))

    subsets, reused = episode_disjoint_split(
        dataset, path, seed=999, fractions=(0.5, 1 / 6, 1 / 3)
    )

    assert reused == manifest
    assert sum(len(subset) for subset in subsets.values()) == len(dataset)


def test_collection_manifest_uses_content_hash_and_disjoint_episode_ids():
    manifest = build_collection_manifest(
        [10] * 10, "a" * 64, seed=7, fractions=(0.8, 0.1, 0.1)
    )
    groups = [set(indices) for indices in manifest["episodes"].values()]
    assert manifest["dataset_sha256"] == "a" * 64
    assert [len(group) for group in groups] == [8, 1, 1]
    assert not groups[0] & groups[1]
    assert not groups[0] & groups[2]


def test_common_random_numbers_keep_candidate_and_particle_axes_separate():
    noise = JEPA._rollout_noise(
        pred_shape=(2 * 3 * 4, 1, 5),
        batch=2,
        samples=12,
        particles=4,
        common=True,
        device=torch.device("cpu"),
        dtype=torch.float32,
    ).reshape(2, 3, 4, 1, 5)
    assert torch.equal(noise[:, 0], noise[:, 1])
    assert not torch.equal(noise[:, :, 0], noise[:, :, 1])

    uniforms = JEPA._rollout_uniform(
        leading_shape=(2 * 3 * 4, 1),
        batch=2,
        samples=12,
        particles=4,
        common=True,
        device=torch.device("cpu"),
        dtype=torch.float32,
    ).reshape(2, 3, 4, 1)
    assert torch.equal(uniforms[:, 0], uniforms[:, 1])
    assert not torch.equal(uniforms[:, :, 0], uniforms[:, :, 1])


def test_particles_one_uses_deterministic_cost():
    wm = model(GlobalDiagonalGaussian(4)).eval()
    batch, candidates, history, horizon, dim = 2, 3, 2, 4, 4
    info = {
        "pixels": torch.randn(batch, candidates, history, 1, 2, 2),
        "goal": torch.randn(batch, candidates, history, 1, 2, 2),
        "action": torch.randn(batch, candidates, history, dim),
    }
    actions = torch.randn(batch, candidates, horizon, dim)
    wm.planning["particles"] = 1
    first = wm.get_cost({k: v.clone() for k, v in info.items()}, actions)
    wm.residual_kernel = None
    second = wm.get_cost({k: v.clone() for k, v in info.items()}, actions)
    assert torch.equal(first, second)


def test_particle_mpc_returns_one_cost_per_action_candidate():
    wm = model(GlobalDiagonalGaussian(4)).eval()
    batch, candidates, history, horizon, dim = 2, 5, 2, 4, 4
    info = {
        "pixels": torch.randn(batch, candidates, history, 1, 2, 2),
        "goal": torch.randn(batch, candidates, history, 1, 2, 2),
        "action": torch.randn(batch, candidates, history, dim),
    }
    wm.planning.update(
        particles=3, particle_chunk_size=2, objective="cvar", flow_steps=1
    )
    cost = wm.get_cost(info, torch.randn(batch, candidates, horizon, dim))
    assert cost.shape == (batch, candidates)
    assert torch.isfinite(cost).all()


def test_particle_mpc_sampling_seed_reproduces_fresh_runs():
    first_model = model(GlobalDiagonalGaussian(4)).eval()
    second_model = model(GlobalDiagonalGaussian(4)).eval()
    settings = {
        "particles": 3,
        "particle_chunk_size": 2,
        "objective": "mean",
        "flow_steps": 1,
        "sampling_seed": 91,
    }
    first_model.planning.update(settings)
    second_model.planning.update(settings)
    info = {
        "pixels": torch.randn(1, 2, 2, 1, 2, 2),
        "goal": torch.randn(1, 2, 2, 1, 2, 2),
        "action": torch.randn(1, 2, 2, 4),
    }
    actions = torch.randn(1, 2, 4, 4)
    first = first_model.get_cost({k: v.clone() for k, v in info.items()}, actions)
    second = second_model.get_cost({k: v.clone() for k, v in info.items()}, actions)
    assert torch.equal(first, second)
