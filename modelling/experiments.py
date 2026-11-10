"""Paired experiment execution, uncertainty estimates, and portable reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
from typing import Callable

import numpy as np

from modelling.config import scenario_fingerprint
from modelling.contracts import EnvironmentConfig
from modelling.evaluation import EpisodeSummary, Policy, evaluate_paired


@dataclass(frozen=True, slots=True)
class MetricEstimate:
    """Mean and deterministic nonparametric bootstrap confidence interval."""

    mean: float
    ci_low: float
    ci_high: float
    sample_count: int


@dataclass(frozen=True, slots=True)
class PolicyReport:
    name: str
    episodes: tuple[EpisodeSummary, ...]
    metrics: dict[str, MetricEstimate]


@dataclass(frozen=True, slots=True)
class ExperimentReport:
    """Immutable report sufficient to reproduce a paired fast-simulator run."""

    schema_version: int
    scenario_fingerprint: str
    seeds: tuple[int, ...]
    bootstrap_samples: int
    policies: tuple[PolicyReport, ...]

    def write(self, output_directory: str | Path) -> None:
        """Write JSON summary and one row per policy/seed to CSV."""

        directory = Path(output_directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "report.json").write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        rows = [
            {"policy": policy.name, **asdict(episode)}
            for policy in self.policies
            for episode in policy.episodes
        ]
        if rows:
            with (directory / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)


def run_paired_experiment(
    config: EnvironmentConfig,
    policies: dict[str, Policy],
    seeds: list[int],
    *,
    bootstrap_samples: int = 2_000,
) -> ExperimentReport:
    """Evaluate named policies on matched conditions and estimate uncertainty."""

    if not policies:
        raise ValueError("at least one policy is required")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be non-empty and unique")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    results = evaluate_paired(config, policies, seeds)
    reports = tuple(
        PolicyReport(name=name, episodes=tuple(episodes), metrics=_summarize(episodes, bootstrap_samples))
        for name, episodes in results.items()
    )
    return ExperimentReport(
        schema_version=2,
        scenario_fingerprint=scenario_fingerprint(config),
        seeds=tuple(seeds),
        bootstrap_samples=bootstrap_samples,
        policies=reports,
    )


def paired_difference(
    report: ExperimentReport,
    candidate: str,
    baseline: str,
    metric: str,
    *,
    bootstrap_samples: int | None = None,
) -> MetricEstimate:
    """Estimate candidate-minus-baseline on identical seeds, preserving pairing."""

    reports = {policy.name: policy for policy in report.policies}
    try:
        candidate_episodes = reports[candidate].episodes
        baseline_episodes = reports[baseline].episodes
    except KeyError as error:
        raise ValueError(f"unknown policy: {error.args[0]}") from error
    if len(candidate_episodes) != len(baseline_episodes):
        raise ValueError("paired reports have unequal episode counts")
    accessor = _metric_accessor(metric)
    differences = np.asarray(
        [accessor(candidate_episode) - accessor(baseline_episode) for candidate_episode, baseline_episode in zip(candidate_episodes, baseline_episodes)],
        dtype=np.float64,
    )
    return _bootstrap(differences, bootstrap_samples or report.bootstrap_samples)


def _summarize(episodes: tuple[EpisodeSummary, ...] | list[EpisodeSummary], bootstrap_samples: int) -> dict[str, MetricEstimate]:
    return {
        name: _bootstrap(np.asarray([accessor(episode) for episode in episodes], dtype=np.float64), bootstrap_samples)
        for name, accessor in _METRICS.items()
    }


_METRICS: dict[str, Callable[[EpisodeSummary], float]] = {
    "total_reward": lambda episode: episode.total_reward,
    "mean_urllc_cost": lambda episode: episode.mean_urllc_cost,
    "max_regime_urllc_cost": lambda episode: episode.max_regime_urllc_cost,
    "p95_period_urllc_cost": lambda episode: episode.p95_period_urllc_cost,
    "cvar95_period_urllc_cost": lambda episode: episode.cvar95_period_urllc_cost,
    "any_miss_period_rate": lambda episode: episode.any_miss_period_rate,
    "max_consecutive_unsafe_periods": lambda episode: float(episode.max_consecutive_unsafe_periods),
    "baseline_urllc_cost": lambda episode: episode.baseline_urllc_cost,
    "embb_surge_urllc_cost": lambda episode: episode.embb_surge_urllc_cost,
    "urllc_surge_urllc_cost": lambda episode: episode.urllc_surge_urllc_cost,
    "mmtc_surge_urllc_cost": lambda episode: episode.mmtc_surge_urllc_cost,
    "mean_embb_cost": lambda episode: episode.mean_embb_cost,
    "mean_mmtc_cost": lambda episode: episode.mean_mmtc_cost,
    "mean_urllc_percent": lambda episode: episode.mean_urllc_percent,
    "mean_allocation_churn": lambda episode: episode.mean_allocation_churn,
    "mitigation_change_rate": lambda episode: episode.mitigation_change_rate,
    "jammer_energy_used": lambda episode: episode.jammer_energy_used,
    "urllc_latency_mean_ms": lambda episode: episode.urllc_latency_mean_ms,
    "urllc_latency_p95_ms": lambda episode: episode.urllc_latency_p95_ms,
    "urllc_latency_p99_ms": lambda episode: episode.urllc_latency_p99_ms,
    "urllc_latency_cvar95_ms": lambda episode: episode.urllc_latency_cvar95_ms,
    "urllc_latency_cvar99_ms": lambda episode: episode.urllc_latency_cvar99_ms,
    "urllc_packet_violation_rate": lambda episode: episode.urllc_packet_violation_rate,
}


def _metric_accessor(metric: str) -> Callable[[EpisodeSummary], float]:
    try:
        return _METRICS[metric]
    except KeyError as error:
        raise ValueError(f"unknown metric: {metric}; expected one of {sorted(_METRICS)}") from error


def _bootstrap(samples: np.ndarray, bootstrap_samples: int) -> MetricEstimate:
    if len(samples) == 0:
        raise ValueError("cannot summarize an empty episode set")
    # The fixed generator makes report regeneration byte-for-byte stable.
    generator = np.random.default_rng(0)
    resamples = generator.choice(samples, size=(bootstrap_samples, len(samples)), replace=True).mean(axis=1)
    low, high = np.quantile(resamples, (0.025, 0.975))
    return MetricEstimate(float(samples.mean()), float(low), float(high), len(samples))
