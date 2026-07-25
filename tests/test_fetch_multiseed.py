import torch

from scripts.eval.summarize_fetch_push_multiseed import crossed_bootstrap


def test_crossed_bootstrap_detects_sampling_beyond_mean():
    model = torch.full((3, 12), 1.0)
    deterministic = torch.full((3, 12), 3.0)
    residual_mean = torch.full((3, 12), 2.0)

    result = crossed_bootstrap(
        model,
        deterministic,
        residual_mean,
        draws=100,
        seed=7,
    )

    assert result["versus_deterministic"]["mean_improvement"] == 2.0
    assert result["versus_deterministic"]["ci95"][0] > 0
    assert result["versus_residual_mean"]["mean_improvement"] == 1.0
    assert result["versus_residual_mean"]["ci95"][0] > 0
