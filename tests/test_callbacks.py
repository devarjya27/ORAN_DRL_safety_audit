import numpy as np

from modelling.training.callbacks import period_cost_tail_metrics


def test_period_cost_cvar_uses_top_five_percent_when_p95_is_zero() -> None:
    costs = np.asarray([0.0] * 95 + [1.0] * 5)
    p95, cvar95 = period_cost_tail_metrics(costs)
    assert p95 == 0.0
    assert cvar95 == 1.0
