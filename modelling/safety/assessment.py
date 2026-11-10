"""Evaluate safety-shield decisions against realized simulator outcomes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Protocol

from modelling.contracts import Action, EnvironmentConfig, Observation
from modelling.envs.slicing_env import SafeSliceEnv
from modelling.safety.shield import SafetyDecision, ShieldResult


class Policy(Protocol):
    def act(self, observation: Observation) -> Action: ...


class Shield(Protocol):
    def apply(self, observation: Observation, actor_action: Action) -> ShieldResult: ...


@dataclass(frozen=True, slots=True)
class ShieldEpisodeMetrics:
    seed: int
    periods: int
    interventions: int
    stale_fallbacks: int
    no_safe_candidates: int
    realized_unsafe_periods: int
    actor_false_negatives: int
    mean_predicted_risk: float


@dataclass(frozen=True, slots=True)
class ShieldAssessment:
    """Aggregate shield metrics across independent, explicitly seeded episodes."""

    episodes: tuple[ShieldEpisodeMetrics, ...]

    @property
    def periods(self) -> int:
        return sum(episode.periods for episode in self.episodes)

    def rates(self) -> dict[str, float]:
        periods = max(1, self.periods)
        return {
            "intervention_rate": sum(episode.interventions for episode in self.episodes) / periods,
            "stale_fallback_rate": sum(episode.stale_fallbacks for episode in self.episodes) / periods,
            "no_safe_candidate_rate": sum(episode.no_safe_candidates for episode in self.episodes) / periods,
            "realized_unsafe_rate": sum(episode.realized_unsafe_periods for episode in self.episodes) / periods,
            "actor_false_negative_rate": sum(episode.actor_false_negatives for episode in self.episodes) / periods,
        }

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"episodes": [asdict(episode) for episode in self.episodes], "rates": self.rates()}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def assess_shield(
    config: EnvironmentConfig,
    policy: Policy,
    shield: Shield,
    *,
    seeds: list[int],
    unsafe_cost_threshold: float = 0.0,
) -> ShieldAssessment:
    """Assess only observable decisions and realized cost; jammer state stays hidden."""

    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be non-empty and unique")
    if unsafe_cost_threshold < 0.0:
        raise ValueError("unsafe_cost_threshold must be non-negative")
    episode_metrics: list[ShieldEpisodeMetrics] = []
    for seed in seeds:
        environment = SafeSliceEnv(config)
        observation, _ = environment.reset(seed=seed)
        interventions = stale = no_safe = unsafe = false_negatives = 0
        risks: list[float] = []
        periods = 0
        while True:
            actor_action = policy.act(observation)
            decision = shield.apply(observation, actor_action)
            result = environment.step(decision.action)
            realized_unsafe = result.costs[0] > unsafe_cost_threshold
            interventions += int(decision.decision is not SafetyDecision.ACTOR)
            stale += int(decision.decision is SafetyDecision.STALE_OBSERVATION)
            no_safe += int(decision.decision is SafetyDecision.NO_SAFE_CANDIDATE)
            unsafe += int(realized_unsafe)
            false_negatives += int(decision.decision is SafetyDecision.ACTOR and realized_unsafe)
            risks.append(decision.predicted_risk)
            periods += 1
            observation = result.observation
            if result.terminated or result.truncated:
                break
        episode_metrics.append(
            ShieldEpisodeMetrics(
                seed,
                periods,
                interventions,
                stale,
                no_safe,
                unsafe,
                false_negatives,
                sum(risks) / periods,
            )
        )
    return ShieldAssessment(tuple(episode_metrics))
