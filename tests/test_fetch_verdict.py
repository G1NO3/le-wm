from scripts.eval.summarize_fetch_push_results import model_verdict


def test_strict_verdict_requires_stochastic_skill_beyond_residual_mean():
    heldout = {
        "flow": {
            "energy_score": 0.8,
            "residual_energy_skill": 0.2,
            "stochastic_energy_skill_vs_residual_mean": -0.01,
        }
    }
    fork = {
        "models": {
            "flow": {
                "10": {
                    "energy_score": 0.7,
                    "residual_energy_skill": 0.3,
                    "stochastic_energy_skill_vs_residual_mean": 0.1,
                }
            }
        },
        "paired_model_bootstrap": {
            "flow": {
                "versus_deterministic": {"mean_improvement": 0.3, "ci95": [0.1, 0.5]},
                "versus_residual_mean": {"mean_improvement": 0.1, "ci95": [0.02, 0.2]},
            }
        },
    }
    result = model_verdict("flow", heldout, fork, 10)
    assert not result["strict_stochastic_help"]
    assert not result["gates"]["heldout_stochastic_skill_positive"]
