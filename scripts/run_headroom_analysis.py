#!/usr/bin/env python3
"""Measure adaptive headroom with privileged and observable regime policies.

The privileged policy receives the simulator's true traffic-regime label.  It
is not a deployable baseline; it tests whether the benchmark rewards adaptation
at all.  A transparent heuristic uses only delayed/lost offered-load KPMs.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from modelling.action_catalog import ActionCatalog
from modelling.config import load_environment_config
from modelling.contracts import Action, EnvironmentConfig, Observation
from modelling.domain_randomization import build_traffic_phase_schedule
from modelling.evaluation import EpisodeSummary, evaluate_episode
from modelling.statistics import bootstrap_episode_estimates, worst_group_cost


REGIMES = ("baseline", "embb_surge", "urllc_surge", "mmtc_surge")
DEFAULT_CANDIDATES = (45, 46, 50, 51, 60, 61, 65, 66)


def _traffic_schedule(config: EnvironmentConfig, seed: int) -> tuple[str, ...]:
    seed_sequence = np.random.SeedSequence(seed)
    *_, traffic_seed = seed_sequence.spawn(7)
    phases = build_traffic_phase_schedule(config, np.random.default_rng(traffic_seed))
    return tuple("stationary" if config.traffic_profile.value == "stationary" else REGIMES[phase] for phase in phases)


class PrivilegedRegimePolicy:
    """Catalogue mapping indexed by the hidden true traffic regime."""

    def __init__(self, catalog: ActionCatalog, schedule: tuple[str, ...], mapping: dict[str, int]) -> None:
        self.catalog = catalog
        self.schedule = schedule
        self.mapping = mapping

    def act(self, observation: Observation) -> Action:
        regime = self.schedule[min(observation.step, len(self.schedule) - 1)]
        action_id = self.mapping.get(regime, self.mapping.get("baseline", 65))
        return self.catalog.get(action_id)


class OfferedLoadHeuristicPolicy:
    """Infer the active surge from delayed offered-load KPMs."""

    def __init__(self, catalog: ActionCatalog, config: EnvironmentConfig, mapping: dict[str, int]) -> None:
        self.catalog = catalog
        self.base_rates = np.asarray(config.traffic_arrival_rates, dtype=np.float64)
        self.mapping = mapping

    def act(self, observation: Observation) -> Action:
        if not observation.feature_valid or not any(observation.offered_loads):
            return self.catalog.get(observation.previous_action_id)
        ratios = np.asarray(observation.offered_loads, dtype=np.float64) / np.maximum(self.base_rates, 1e-9)
        index = int(np.argmax(ratios))
        regime = REGIMES[index + 1] if ratios[index] >= 1.5 else "baseline"
        return self.catalog.get(self.mapping[regime])


def _evaluate_mapping(
    config: EnvironmentConfig,
    mapping: dict[str, int],
    seeds: Iterable[int],
) -> list[EpisodeSummary]:
    catalog = ActionCatalog.build(config)
    return [
        evaluate_episode(
            config,
            PrivilegedRegimePolicy(catalog, _traffic_schedule(config, seed), mapping),
            seed=seed,
        )
        for seed in seeds
    ]


def _objective(episodes: list[EpisodeSummary], limit: float) -> tuple[int, float, float]:
    reward = float(np.mean([episode.total_reward for episode in episodes]))
    cost = worst_group_cost(episodes)
    return (int(cost <= limit), -max(0.0, cost - limit), reward)


def tune_mapping(
    config: EnvironmentConfig,
    seeds: tuple[int, ...],
    candidates: tuple[int, ...],
    limit: float,
) -> dict[str, int]:
    """Coordinate-search a regime mapping from static-frontier candidates."""

    best_mapping: dict[str, int] | None = None
    best_score: tuple[int, float, float] | None = None
    for initial in (60, 61, 65, 66):
        mapping = {regime: initial for regime in REGIMES}
        for _ in range(3):
            changed = False
            for regime in REGIMES:
                local_mapping = mapping
                local_score = _objective(_evaluate_mapping(config, mapping, seeds), limit)
                for candidate in candidates:
                    trial = dict(mapping)
                    trial[regime] = candidate
                    score = _objective(_evaluate_mapping(config, trial, seeds), limit)
                    if score > local_score:
                        local_mapping, local_score = trial, score
                changed |= local_mapping != mapping
                mapping = local_mapping
            if not changed:
                break
        score = _objective(_evaluate_mapping(config, mapping, seeds), limit)
        if best_score is None or score > best_score:
            best_mapping, best_score = mapping, score
    assert best_mapping is not None
    return best_mapping


def _summary(episodes: list[EpisodeSummary]) -> dict[str, dict[str, float]]:
    estimates = bootstrap_episode_estimates(episodes)
    return {
        name: {"mean": estimate.mean, "ci_low": estimate.ci_low, "ci_high": estimate.ci_high}
        for name, estimate in estimates.items()
    }


def evaluate_scenario(config: EnvironmentConfig, mapping: dict[str, int], seeds: tuple[int, ...]) -> dict[str, object]:
    catalog = ActionCatalog.build(config)
    oracle = _evaluate_mapping(config, mapping, seeds)
    heuristic = [
        evaluate_episode(config, OfferedLoadHeuristicPolicy(catalog, config, mapping), seed=seed)
        for seed in seeds
    ]
    return {
        "privileged_regime_oracle": {"metrics": _summary(oracle), "episodes": [asdict(item) for item in oracle]},
        "offered_load_heuristic": {"metrics": _summary(heuristic), "episodes": [asdict(item) for item in heuristic]},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("study/v7-paper-final/results/headroom/report.json"))
    parser.add_argument("--limit", type=float, default=0.01)
    parser.add_argument(
        "--tuning-limit",
        type=float,
        default=0.005,
        help="validation safety margin used to select the privileged mapping",
    )
    parser.add_argument("--validation-seeds", type=int, nargs="*", default=list(range(1001, 1021)))
    parser.add_argument("--evaluation-episodes", type=int, default=100)
    arguments = parser.parse_args()

    training_config = load_environment_config(Path("configs/research_dynamic.yaml"))
    mapping = tune_mapping(
        training_config,
        tuple(arguments.validation_seeds),
        DEFAULT_CANDIDATES,
        arguments.tuning_limit,
    )
    scenarios = {
        "training_matched": (Path("configs/research_dynamic.yaml"), 10_000),
        "constant_stress": (Path("configs/research_validation_constant.yaml"), 20_000),
        "unseen_sweep": (Path("configs/research_heldout_sweep.yaml"), 30_000),
    }
    payload = {
        "schema_version": 1,
        "safety_limit": arguments.limit,
        "tuning_limit": arguments.tuning_limit,
        "validation_seeds": arguments.validation_seeds,
        "candidate_action_ids": DEFAULT_CANDIDATES,
        "selected_mapping": mapping,
        "scenarios": {
            name: evaluate_scenario(
                load_environment_config(config_path),
                mapping,
                tuple(range(seed_base, seed_base + arguments.evaluation_episodes)),
            )
            for name, (config_path, seed_base) in scenarios.items()
        },
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Selected mapping: {mapping}")
    print(f"Wrote headroom analysis to {arguments.output}")


if __name__ == "__main__":
    main()
