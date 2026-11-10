from dataclasses import replace

import pytest

from modelling.evaluation import EpisodeSummary
from modelling.statistics import bootstrap_episode_estimates, worst_group_cost


def _episode(**changes: float) -> EpisodeSummary:
    base = EpisodeSummary(
        seed=1,
        total_reward=10.0,
        mean_urllc_cost=0.005,
        mean_embb_cost=0.0,
        mean_mmtc_cost=0.0,
        jammer_energy_used=0.0,
        action_ids=(0,),
        mean_allocation_churn=0.0,
        mitigation_change_rate=0.0,
        mean_urllc_percent=40.0,
        urllc_latency_sample_count=0,
        urllc_latency_mean_ms=0.0,
        urllc_latency_p95_ms=0.0,
        urllc_latency_p99_ms=0.0,
        urllc_latency_cvar95_ms=0.0,
        urllc_latency_cvar99_ms=0.0,
        urllc_packet_violation_rate=0.0,
        p95_period_urllc_cost=0.0,
        cvar95_period_urllc_cost=0.0,
        any_miss_period_rate=0.0,
        max_consecutive_unsafe_periods=0,
        baseline_urllc_cost=0.0,
        embb_surge_urllc_cost=0.0,
        urllc_surge_urllc_cost=0.0,
        mmtc_surge_urllc_cost=0.0,
        max_regime_urllc_cost=0.0,
    )
    return replace(base, **changes)


def test_worst_group_applies_max_after_group_averaging() -> None:
    episodes = [
        _episode(baseline_urllc_cost=0.02, urllc_surge_urllc_cost=0.00),
        _episode(baseline_urllc_cost=0.00, urllc_surge_urllc_cost=0.02),
    ]
    assert worst_group_cost(episodes) == pytest.approx(0.01)
    assert sum(max(item.baseline_urllc_cost, item.urllc_surge_urllc_cost) for item in episodes) / 2 == 0.02


def test_bootstrap_reports_worst_group_metric() -> None:
    estimate = bootstrap_episode_estimates(
        [_episode(baseline_urllc_cost=0.004), _episode(baseline_urllc_cost=0.006)],
        samples=100,
    )["worst_group_urllc_cost"]
    assert estimate.mean == pytest.approx(0.005)
    assert estimate.ci_low <= estimate.mean <= estimate.ci_high
