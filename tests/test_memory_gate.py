from scripts.eval.evaluate_memory_gate import evaluate
from scripts.eval.evaluate_memory_control_gate import evaluate as evaluate_control


def test_memory_gate_requires_residual_and_memory_attribution():
    recurrent_h10 = {
        "residual_energy_skill": 0.12,
        "memory_energy_skill": 0.06,
        "stochastic_energy_skill_vs_residual_mean": 0.02,
        "coverage_90": 0.90,
        "lag1_autocorrelation_error": 0.8,
    }
    payload = {
        "models": {
            "flow": {
                "1": {"mean_error": 1.0},
                "10": {"lag1_autocorrelation_error": 1.0},
                "20": {},
            },
            "flow_gru_memory": {
                "1": {"mean_error": 1.04},
                "10": recurrent_h10,
                "20": {"memory_energy_skill": 0.05},
            },
        },
        "paired_memory_bootstrap": {
            "flow": {"ci95": [0.01, 0.03]},
            "flow_gru_memory_reset": {"ci95": [0.01, 0.03]},
            "flow_gru_memory_shuffled": {"ci95": [0.01, 0.03]},
        },
    }
    assert evaluate(payload)["passed"]
    payload["models"]["flow_gru_memory"]["10"][
        "stochastic_energy_skill_vs_residual_mean"
    ] = -0.01
    assert not evaluate(payload)["passed"]


def test_memory_control_gate_requires_tail_and_success_improvement():
    payload = {
        "models": {
            "deterministic": {"success_rate": 50.0, "worst_decile_distance": 10.0},
            "flow": {"success_rate": 53.0, "worst_decile_distance": 9.5},
            "flow_gru_memory_cvar": {
                "success_rate": 56.0,
                "worst_decile_distance": 8.5,
            },
        }
    }
    assert evaluate_control(payload)["passed"]
