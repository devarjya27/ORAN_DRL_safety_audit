#!/usr/bin/env python3
"""Diagnose whether a small labeled sweep sample can recalibrate the shield."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path

from modelling.action_catalog import ActionCatalog
from modelling.config import load_environment_config
from modelling.contracts import Action, Observation
from modelling.evaluation import evaluate_episode
from modelling.safety.assessment import assess_shield
from modelling.safety.learned_risk import (
    LearnedRiskShield,
    calibrate_upper_bound_threshold,
    collect_risk_samples,
    load_risk_ensemble,
)
from modelling.safety.shield import ShieldConfig
from modelling.statistics import bootstrap_episode_estimates
from modelling.training.train_recurrent_ppo import load_recurrent_policy


class _ShieldedPolicy:
    def __init__(self, actor: object, shield: LearnedRiskShield) -> None:
        self.actor = actor
        self.shield = shield

    def act(self, observation: Observation) -> Action:
        return self.shield.apply(observation, self.actor.act(observation)).action  # type: ignore[union-attr]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path("study/v7-paper-final"))
    parser.add_argument("--config", type=Path, default=Path("configs/research_heldout_sweep.yaml"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("study/v7-paper-final/results/sweep_recalibration.csv"),
    )
    parser.add_argument("--calibration-episodes", type=int, default=20)
    parser.add_argument("--evaluation-episodes", type=int, default=100)
    parser.add_argument("--target-unsafe-rate", type=float, default=0.01)
    parser.add_argument("--minimum-accepted-fraction", type=float, default=0.25)
    arguments = parser.parse_args()
    config = load_environment_config(arguments.config)
    catalog = ActionCatalog.build(config)
    rows: list[dict[str, object]] = []
    calibration_seeds = list(range(40_000, 40_000 + arguments.calibration_episodes))
    evaluation_seeds = list(range(30_000, 30_000 + arguments.evaluation_episodes))
    for seed in range(11, 16):
        artifact_dir = arguments.study / "artifacts" / f"seed-{seed}"
        actor = load_recurrent_policy(artifact_dir / "recurrent.pt", config, allow_scenario_mismatch=True)
        ensemble = load_risk_ensemble(
            artifact_dir / "risk/risk_ensemble.pt",
            catalog,
            config,
            allow_scenario_mismatch=True,
        )
        samples = collect_risk_samples(
            config,
            actor,
            seeds=calibration_seeds,
            exploration_probability=0.25,
            exploration_seed=50_000 + seed,
        )
        calibration = calibrate_upper_bound_threshold(
            ensemble,
            samples,
            target_unsafe_rate=arguments.target_unsafe_rate,
            unsafe_cost_threshold=0.01,
            minimum_accepted_fraction=arguments.minimum_accepted_fraction,
        )
        shield = LearnedRiskShield(
            catalog,
            config,
            ensemble,
            replace(ShieldConfig(), maximum_predicted_urllc_risk=calibration.threshold),
        )
        episodes = [
            evaluate_episode(config, _ShieldedPolicy(actor, shield), seed=episode_seed)
            for episode_seed in evaluation_seeds
        ]
        estimates = bootstrap_episode_estimates(episodes)
        cost = estimates["worst_group_urllc_cost"]
        reward = estimates["total_reward"]
        status = "supported" if cost.ci_high <= 0.01 else "violating" if cost.ci_low > 0.01 else "uncertain"
        assessment = assess_shield(config, actor, shield, seeds=evaluation_seeds)
        periods = sum(item.periods for item in assessment.episodes)
        interventions = sum(item.interventions for item in assessment.episodes)
        false_negatives = sum(item.actor_false_negatives for item in assessment.episodes)
        accepted = periods - interventions
        rows.append(
            {
                "training_seed": seed,
                "calibration_episodes": arguments.calibration_episodes,
                "calibration_threshold": calibration.threshold,
                "calibration_accepted_fraction": calibration.accepted_fraction,
                "calibration_unsafe_rate": calibration.observed_unsafe_rate,
                "calibration_target_met": calibration.target_met,
                "fresh_mean_reward": reward.mean,
                "fresh_worst_group_cost": cost.mean,
                "fresh_cost_ci_low": cost.ci_low,
                "fresh_cost_ci_high": cost.ci_high,
                "fresh_status": status,
                "fresh_accepted_action_unsafe_rate": false_negatives / accepted if accepted else float("nan"),
                "fresh_realized_unsafe_rate": assessment.rates()["realized_unsafe_rate"],
                "fresh_intervention_rate": assessment.rates()["intervention_rate"],
            }
        )
        print(f"evaluated recalibration seed {seed}", flush=True)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote sweep-recalibration diagnostic to {arguments.output}")


if __name__ == "__main__":
    main()
