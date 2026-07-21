#!/usr/bin/env python
"""Train a tiny recurrent residual kernel on an AR(1) correctness problem."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch

from residual_kernels import ConditionalDiagonalGaussian
from residual_memory import ResidualMemory
from stochastic_metrics import lag1_autocorrelation_error, trajectory_energy_score


def ar1(batch, steps, *, rho, innovation_scale, generator):
    stationary_scale = innovation_scale / (1 - rho**2) ** 0.5
    values = torch.zeros(batch, steps, 1)
    values[:, 0] = stationary_scale * torch.randn(batch, 1, generator=generator)
    for index in range(1, steps):
        values[:, index] = (
            rho * values[:, index - 1]
            + innovation_scale * torch.randn(batch, 1, generator=generator)
        )
    return values


def teacher_forced_loss(memory, kernel, values):
    condition = torch.zeros(values.size(0), 1)
    state = memory.init(
        (values.size(0),), device=values.device, dtype=values.dtype
    )
    state = memory.update(state, condition, values[:, 0])
    losses = []
    for index in range(1, values.size(1)):
        conditioned = memory.condition(condition, state)
        losses.append(kernel.loss(values[:, index], conditioned))
        state = memory.update(state, condition, values[:, index])
    return torch.stack(losses).mean()


def memoryless_loss(kernel, values):
    condition = torch.zeros(*values.shape[:-1], 1)
    return kernel.loss(values, condition)


@torch.no_grad()
def sample_futures(memory, recurrent, memoryless, initial, *, samples, horizon, generator):
    contexts = initial.size(0)
    condition = torch.zeros(contexts, samples, 1)
    initial = initial[:, None].expand(contexts, samples, 1)
    state = memory.init(
        (contexts, samples), device=initial.device, dtype=initial.dtype
    )
    state = memory.update(state, condition, initial)
    recurrent_values = []
    memoryless_values = []
    for _ in range(horizon):
        recurrent_noise = torch.randn(
            contexts, samples, 1, generator=generator
        )
        recurrent_value = recurrent.sample(
            memory.condition(condition, state), noise=recurrent_noise
        )
        recurrent_values.append(recurrent_value)
        state = memory.update(state, condition, recurrent_value)
        memoryless_values.append(
            memoryless.sample(
                condition,
                noise=torch.randn(contexts, samples, 1, generator=generator),
            )
        )
    return torch.stack(recurrent_values, 2), torch.stack(memoryless_values, 2)


def run(steps=300, seed=3072):
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    rho = 0.9
    innovation_scale = 0.2
    train = ar1(
        2048, 8, rho=rho, innovation_scale=innovation_scale, generator=generator
    )
    test = ar1(
        512, 8, rho=rho, innovation_scale=innovation_scale, generator=generator
    )
    memory = ResidualMemory(condition_dim=1, residual_dim=1, hidden_dim=8)
    recurrent = ConditionalDiagonalGaussian(9, 1, hidden_dim=32, depth=2)
    memoryless = ConditionalDiagonalGaussian(1, 1, hidden_dim=32, depth=2)
    optimizer = torch.optim.Adam(
        [*memory.parameters(), *recurrent.parameters(), *memoryless.parameters()],
        lr=3e-3,
    )
    for _ in range(steps):
        indices = torch.randint(train.size(0), (128,), generator=generator)
        batch = train[indices]
        loss = teacher_forced_loss(memory, recurrent, batch) + memoryless_loss(
            memoryless, batch[:, 1:]
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    recurrent_nll = float(teacher_forced_loss(memory, recurrent, test).detach())
    memoryless_nll = float(memoryless_loss(memoryless, test[:, 1:]).detach())
    contexts = 32
    futures = 128
    horizon = 6
    initial = test[:contexts, 0]
    truth = torch.empty(contexts, futures, horizon, 1)
    for context in range(contexts):
        current = initial[context].expand(futures, 1).clone()
        for index in range(horizon):
            current = rho * current + innovation_scale * torch.randn(
                futures, 1, generator=generator
            )
            truth[context, :, index] = current
    recurrent_samples, memoryless_samples = sample_futures(
        memory,
        recurrent,
        memoryless,
        initial,
        samples=futures,
        horizon=horizon,
        generator=generator,
    )
    recurrent_energy = float(trajectory_energy_score(recurrent_samples, truth).mean())
    memoryless_energy = float(trajectory_energy_score(memoryless_samples, truth).mean())
    recurrent_lag_error = float(
        lag1_autocorrelation_error(recurrent_samples, truth).mean()
    )
    memoryless_lag_error = float(
        lag1_autocorrelation_error(memoryless_samples, truth).mean()
    )
    result = {
        "seed": seed,
        "steps": steps,
        "recurrent_nll": recurrent_nll,
        "memoryless_nll": memoryless_nll,
        "recurrent_trajectory_energy": recurrent_energy,
        "memoryless_trajectory_energy": memoryless_energy,
        "recurrent_lag1_error": recurrent_lag_error,
        "memoryless_lag1_error": memoryless_lag_error,
        "passed": (
            recurrent_nll < 0.8 * memoryless_nll
            and recurrent_energy < 0.9 * memoryless_energy
            and recurrent_lag_error < memoryless_lag_error
        ),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(steps=args.steps, seed=args.seed)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    if not result["passed"]:
        raise SystemExit("Persistent-memory AR(1) smoke gate failed")


if __name__ == "__main__":
    main()
