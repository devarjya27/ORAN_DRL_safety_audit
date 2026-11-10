"""Statistical estimators for regime-conditional policy evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from modelling.evaluation import EpisodeSummary


REGIME_COST_FIELDS = (
    "baseline_urllc_cost",
    "embb_surge_urllc_cost",
    "urllc_surge_urllc_cost",
    "mmtc_surge_urllc_cost",
)


@dataclass(frozen=True, slots=True)
class BootstrapEstimate:
    mean: float
    ci_low: float
    ci_high: float


def worst_group_cost(episodes: list[EpisodeSummary] | tuple[EpisodeSummary, ...]) -> float:
    """Maximum of episode-averaged regime costs.

    The maximum is applied after estimating each group mean.  Averaging an
    episode-wise maximum would estimate a different, upward-biased quantity.
    """

    if not episodes:
        raise ValueError("worst-group cost requires at least one episode")
    return max(float(np.mean([getattr(episode, field) for episode in episodes])) for field in REGIME_COST_FIELDS)


def bootstrap_episode_estimates(
    episodes: list[EpisodeSummary] | tuple[EpisodeSummary, ...],
    *,
    samples: int = 5_000,
    seed: int = 0,
) -> dict[str, BootstrapEstimate]:
    """Bootstrap reward, mean cost, and worst-group cost over episodes."""

    if not episodes or samples <= 0:
        raise ValueError("episodes and samples must be non-empty/positive")
    rewards = np.asarray([episode.total_reward for episode in episodes], dtype=np.float64)
    mean_costs = np.asarray([episode.mean_urllc_cost for episode in episodes], dtype=np.float64)
    group_costs = np.asarray(
        [[getattr(episode, field) for field in REGIME_COST_FIELDS] for episode in episodes],
        dtype=np.float64,
    )
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(episodes), size=(samples, len(episodes)))
    reward_draws = rewards[indices].mean(axis=1)
    mean_cost_draws = mean_costs[indices].mean(axis=1)
    worst_draws = group_costs[indices].mean(axis=1).max(axis=1)
    return {
        "total_reward": _estimate(float(rewards.mean()), reward_draws),
        "mean_urllc_cost": _estimate(float(mean_costs.mean()), mean_cost_draws),
        "worst_group_urllc_cost": _estimate(float(group_costs.mean(axis=0).max()), worst_draws),
    }


def hierarchical_episode_estimates(
    runs: list[list[EpisodeSummary] | tuple[EpisodeSummary, ...]],
    *,
    samples: int = 5_000,
    seed: int = 0,
) -> dict[str, BootstrapEstimate]:
    """Resample training runs and episodes, recomputing each run's group max."""

    if not runs or any(not run for run in runs) or samples <= 0:
        raise ValueError("runs and samples must be non-empty/positive")
    reward_arrays = [np.asarray([episode.total_reward for episode in run]) for run in runs]
    mean_arrays = [np.asarray([episode.mean_urllc_cost for episode in run]) for run in runs]
    group_arrays = [
        np.asarray([[getattr(episode, field) for field in REGIME_COST_FIELDS] for episode in run])
        for run in runs
    ]
    reward_point = float(np.mean([values.mean() for values in reward_arrays]))
    mean_point = float(np.mean([values.mean() for values in mean_arrays]))
    worst_point = float(np.mean([values.mean(axis=0).max() for values in group_arrays]))
    generator = np.random.default_rng(seed)
    reward_draws = np.empty(samples)
    mean_draws = np.empty(samples)
    worst_draws = np.empty(samples)
    for draw in range(samples):
        selected_runs = generator.integers(0, len(runs), size=len(runs))
        reward_values: list[float] = []
        mean_values: list[float] = []
        worst_values: list[float] = []
        for run_index in selected_runs:
            episode_count = len(runs[int(run_index)])
            indices = generator.integers(0, episode_count, size=episode_count)
            reward_values.append(float(reward_arrays[int(run_index)][indices].mean()))
            mean_values.append(float(mean_arrays[int(run_index)][indices].mean()))
            worst_values.append(float(group_arrays[int(run_index)][indices].mean(axis=0).max()))
        reward_draws[draw] = np.mean(reward_values)
        mean_draws[draw] = np.mean(mean_values)
        worst_draws[draw] = np.mean(worst_values)
    return {
        "total_reward": _estimate(reward_point, reward_draws),
        "mean_urllc_cost": _estimate(mean_point, mean_draws),
        "worst_group_urllc_cost": _estimate(worst_point, worst_draws),
    }


def _estimate(point: float, draws: np.ndarray) -> BootstrapEstimate:
    low, high = np.quantile(draws, (0.025, 0.975))
    return BootstrapEstimate(point, float(low), float(high))
