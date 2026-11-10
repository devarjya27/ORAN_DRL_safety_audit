"""Deterministic episode-level domain randomization shared by env and traces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from modelling.contracts import EnvironmentConfig, JammerFamily, TrafficProfile


@dataclass(frozen=True, slots=True)
class EpisodeJammerDomain:
    """Jammer family and resource envelope selected for one episode."""

    family: JammerFamily
    power: float
    energy_budget: float


def sample_episode_jammer_domain(
    config: EnvironmentConfig,
    rng: np.random.Generator,
) -> EpisodeJammerDomain:
    """Sample a reproducible training domain without exposing it to policy observations."""

    if config.jammer_population:
        family = config.jammer_population[int(rng.integers(len(config.jammer_population)))]
    else:
        family = config.jammer_family
    power = _sample_range(rng, config.jammer_power_range, config.jammer_power)
    budget = _sample_range(rng, config.jammer_energy_budget_range, config.jammer_energy_budget)
    return EpisodeJammerDomain(family=family, power=power, energy_budget=budget)


def build_traffic_phase_schedule(config: EnvironmentConfig, rng: np.random.Generator) -> tuple[int, ...]:
    """Build a deterministic per-episode phase schedule without policy-visible time."""

    if config.traffic_profile is TrafficProfile.STATIONARY:
        return tuple(0 for _ in range(config.episode_steps))
    schedule: list[int] = []
    while len(schedule) < config.episode_steps:
        phases = np.arange(4)
        if config.traffic_randomize_regime_order:
            phases = rng.permutation(phases)
        for phase_value in phases:
            jitter = config.traffic_regime_jitter_steps
            duration = config.traffic_regime_steps
            if jitter:
                duration += int(rng.integers(-jitter, jitter + 1))
            schedule.extend([int(phase_value)] * max(1, duration))
            if len(schedule) >= config.episode_steps:
                break
    return tuple(schedule[: config.episode_steps])


def _sample_range(
    rng: np.random.Generator,
    bounds: tuple[float, float] | None,
    default: float,
) -> float:
    if bounds is None:
        return default
    low, high = bounds
    return low if low == high else float(rng.uniform(low, high))
