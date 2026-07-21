import torch

from stochastic_metrics import (
    cross_time_covariance_error,
    energy_score,
    energy_skill_score,
    interval_metrics,
    mean_error_explained_fraction,
    lag1_autocorrelation_error,
    mmd_rbf,
    pooled_mode_separation,
    sliced_wasserstein,
    trajectory_energy_score,
)


def test_proper_scores_prefer_matching_distribution():
    generator = torch.Generator().manual_seed(11)
    truth = torch.randn(16, 64, 3, generator=generator)
    match = torch.randn(16, 64, 3, generator=generator)
    shifted = match + 4
    assert energy_score(match, truth).mean() < energy_score(shifted, truth).mean()
    assert mmd_rbf(match, truth).mean() < mmd_rbf(shifted, truth).mean()
    assert sliced_wasserstein(match, truth).mean() < sliced_wasserstein(shifted, truth).mean()


def test_residual_skill_separates_mean_correction_from_stochastic_gain():
    observations = torch.full((2, 3, 1), 2.0)
    nominal = torch.zeros(2, 1, 1)
    halfway = torch.ones(2, 4, 1)
    perfect = torch.full((2, 4, 1), 2.0)

    assert torch.isclose(
        energy_skill_score(halfway, observations, nominal), torch.tensor(0.5)
    )
    assert torch.isclose(
        mean_error_explained_fraction(halfway, observations, nominal),
        torch.tensor(0.75),
    )
    assert torch.isclose(
        energy_skill_score(perfect, observations, nominal), torch.tensor(1.0)
    )


def test_stochastic_spread_can_hurt_relative_to_residual_mean():
    observations = torch.ones(1, 2, 1)
    spread = torch.tensor([[[-1.0], [3.0]]])
    residual_mean = spread.mean(dim=1, keepdim=True)
    assert energy_skill_score(spread, observations, residual_mean) < 0


def test_interval_metrics_report_coverage_and_width():
    samples = torch.linspace(-1, 1, 101).view(1, 101, 1)
    observations = torch.zeros(1, 5, 1)
    coverage, width = interval_metrics(samples, observations, 0.8)
    assert coverage.item() == 1
    assert 1.5 < width.item() < 1.7


def test_pooled_mode_separation_uses_hidden_groups():
    values = torch.tensor([[0.0], [0.2], [2.0], [2.2]])
    labels = torch.tensor([0, 0, 1, 1])
    assert pooled_mode_separation(values, labels) > 10


def test_grouped_mode_separation_removes_context_offsets():
    groups = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    labels = torch.tensor([False, False, True, True] * 2)
    context_offset = groups.float().unsqueeze(1) * 1000
    mode_effect = labels.float().unsqueeze(1) * 2
    noise = torch.tensor([[-0.1], [0.1], [-0.1], [0.1]] * 2)
    values = context_offset + mode_effect + noise
    assert pooled_mode_separation(values, labels, groups) > 10


def test_trajectory_metrics_prefer_temporally_matching_samples():
    generator = torch.Generator().manual_seed(19)
    innovations = torch.randn(8, 64, 6, 1, generator=generator)
    truth = torch.zeros_like(innovations)
    for index in range(1, truth.size(2)):
        truth[:, :, index] = 0.9 * truth[:, :, index - 1] + innovations[:, :, index]
    matching = truth + 0.05 * torch.randn(
        truth.shape, generator=generator
    )
    shuffled = matching[:, :, torch.randperm(6, generator=generator)]
    assert trajectory_energy_score(matching, truth).mean() < trajectory_energy_score(
        shuffled, truth
    ).mean()
    assert lag1_autocorrelation_error(matching, truth).mean() < lag1_autocorrelation_error(
        shuffled, truth
    ).mean()
    assert cross_time_covariance_error(matching, truth).mean() < cross_time_covariance_error(
        shuffled, truth
    ).mean()
