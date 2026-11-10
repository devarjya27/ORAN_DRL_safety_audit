"""Paired, seed-controlled policy evaluation for the fast simulator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from modelling.contracts import Action, EnvironmentConfig
from modelling.envs.slicing_env import SafeSliceEnv
from modelling.traces import ActionDependentTraceError, ExogenousTrace


class Policy(Protocol):
    def act(self, observation: object) -> Action: ...


@dataclass(frozen=True, slots=True)
class EpisodeSummary:
    seed: int
    total_reward: float
    mean_urllc_cost: float
    mean_embb_cost: float
    mean_mmtc_cost: float
    jammer_energy_used: float
    action_ids: tuple[int, ...]
    mean_allocation_churn: float
    mitigation_change_rate: float
    mean_urllc_percent: float
    urllc_latency_sample_count: int
    urllc_latency_mean_ms: float
    urllc_latency_p95_ms: float
    urllc_latency_p99_ms: float
    urllc_latency_cvar95_ms: float
    urllc_latency_cvar99_ms: float
    urllc_packet_violation_rate: float
    p95_period_urllc_cost: float
    cvar95_period_urllc_cost: float
    any_miss_period_rate: float
    max_consecutive_unsafe_periods: int
    baseline_urllc_cost: float
    embb_surge_urllc_cost: float
    urllc_surge_urllc_cost: float
    mmtc_surge_urllc_cost: float
    max_regime_urllc_cost: float


def evaluate_episode(
    config: EnvironmentConfig,
    policy: Policy,
    *,
    seed: int,
    trace: ExogenousTrace | None = None,
) -> EpisodeSummary:
    """Evaluate one policy with an explicit seed and record every action."""

    env = SafeSliceEnv(config, trace=trace)
    observation, _ = env.reset(seed=seed)
    total_reward = 0.0
    costs = [0.0, 0.0, 0.0]
    action_ids: list[int] = []
    action_history: list[Action] = []
    energy_used = 0.0
    urllc_latencies_ms: list[float] = []
    total_urllc_arrivals = 0
    total_expired_urllc = 0
    period_urllc_costs: list[float] = []
    regime_urllc_costs: dict[str, list[float]] = {}
    unsafe_streak = 0
    max_unsafe_streak = 0
    while True:
        action = policy.act(observation)
        action_ids.append(action.action_id)
        action_history.append(action)
        result = env.step(action)
        total_reward += result.reward
        for index, cost in enumerate(result.costs):
            costs[index] += cost
        period_urllc_costs.append(result.costs[0])
        regime = str(result.info.get("traffic_regime", "unknown"))
        regime_urllc_costs.setdefault(regime, []).append(result.costs[0])
        unsafe_streak = unsafe_streak + 1 if result.costs[0] > 0.0 else 0
        max_unsafe_streak = max(max_unsafe_streak, unsafe_streak)
        energy_used = result.info["jammer"].energy_used  # type: ignore[union-attr]
        urllc_latencies_ms.extend(result.info["urllc_latencies_ms"])  # type: ignore[arg-type]
        total_urllc_arrivals += int(result.info["arrivals"]["urllc"])  # type: ignore[index]
        total_expired_urllc += int(result.info["expired_urllc"])
        observation = result.observation
        if result.terminated or result.truncated:
            break
    periods = len(action_ids)
    allocation_churn = [
        sum(abs(current - previous) for current, previous in zip(action.shares, prior.shares)) / 200.0
        for prior, action in zip(action_history, action_history[1:])
    ]
    mitigation_changes = [
        float(action.mitigation != prior.mitigation)
        for prior, action in zip(action_history, action_history[1:])
    ]
    sorted_latencies = sorted(urllc_latencies_ms)
    if sorted_latencies:
        p95_index = min(len(sorted_latencies) - 1, int(0.95 * (len(sorted_latencies) - 1)))
        p99_index = min(len(sorted_latencies) - 1, int(0.99 * (len(sorted_latencies) - 1)))
        cvar_start = min(len(sorted_latencies) - 1, int(0.95 * len(sorted_latencies)))
        cvar99_start = min(len(sorted_latencies) - 1, int(0.99 * len(sorted_latencies)))
        latency_mean = sum(sorted_latencies) / len(sorted_latencies)
        latency_p95 = sorted_latencies[p95_index]
        latency_p99 = sorted_latencies[p99_index]
        latency_cvar95 = sum(sorted_latencies[cvar_start:]) / len(sorted_latencies[cvar_start:])
        latency_cvar99 = sum(sorted_latencies[cvar99_start:]) / len(sorted_latencies[cvar99_start:])
    else:
        latency_mean = latency_p95 = latency_p99 = latency_cvar95 = latency_cvar99 = 0.0
    sorted_period_costs = sorted(period_urllc_costs)
    period_p95_index = min(len(sorted_period_costs) - 1, int(0.95 * (len(sorted_period_costs) - 1)))
    period_cvar_start = min(len(sorted_period_costs) - 1, int(0.95 * len(sorted_period_costs)))
    period_p95 = sorted_period_costs[period_p95_index]
    period_cvar95 = sum(sorted_period_costs[period_cvar_start:]) / len(sorted_period_costs[period_cvar_start:])
    regime_means = {name: sum(values) / len(values) for name, values in regime_urllc_costs.items()}
    return EpisodeSummary(
        seed=seed,
        total_reward=total_reward,
        mean_urllc_cost=costs[0] / periods,
        mean_embb_cost=costs[1] / periods,
        mean_mmtc_cost=costs[2] / periods,
        jammer_energy_used=energy_used,
        action_ids=tuple(action_ids),
        mean_allocation_churn=sum(allocation_churn) / max(1, len(allocation_churn)),
        mitigation_change_rate=sum(mitigation_changes) / max(1, len(mitigation_changes)),
        mean_urllc_percent=sum(action.urllc_percent for action in action_history) / periods,
        urllc_latency_sample_count=len(sorted_latencies),
        urllc_latency_mean_ms=latency_mean,
        urllc_latency_p95_ms=latency_p95,
        urllc_latency_p99_ms=latency_p99,
        urllc_latency_cvar95_ms=latency_cvar95,
        urllc_latency_cvar99_ms=latency_cvar99,
        urllc_packet_violation_rate=total_expired_urllc / max(1, total_urllc_arrivals),
        p95_period_urllc_cost=period_p95,
        cvar95_period_urllc_cost=period_cvar95,
        any_miss_period_rate=sum(cost > 0.0 for cost in period_urllc_costs) / periods,
        max_consecutive_unsafe_periods=max_unsafe_streak,
        baseline_urllc_cost=regime_means.get("baseline", 0.0),
        embb_surge_urllc_cost=regime_means.get("embb_surge", 0.0),
        urllc_surge_urllc_cost=regime_means.get("urllc_surge", 0.0),
        mmtc_surge_urllc_cost=regime_means.get("mmtc_surge", 0.0),
        max_regime_urllc_cost=max(regime_means.values()),
    )


def evaluate_paired(config: EnvironmentConfig, policies: dict[str, Policy], seeds: list[int]) -> dict[str, list[EpisodeSummary]]:
    """Evaluate policies with matched seeds and fixed traces when valid.

    Adaptive (reactive) jammers cannot be frozen without changing their attack
    semantics. In that case each policy receives the same split random seeds,
    while its legitimate action-dependent interaction with the attacker remains
    intact.
    """

    traces: dict[int, ExogenousTrace | None] = {}
    for seed in seeds:
        try:
            traces[seed] = ExogenousTrace.generate(config, seed=seed)
        except ActionDependentTraceError:
            traces[seed] = None

    return {
        name: [
            evaluate_episode(
                config,
                policy,
                seed=seed,
                trace=traces[seed],
            )
            for seed in seeds
        ]
        for name, policy in policies.items()
    }
