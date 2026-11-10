#!/usr/bin/env python3
"""Evaluate every fixed catalogue action on untouched scenario traces."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from modelling.action_catalog import ActionCatalog
from modelling.config import load_environment_config
from modelling.evaluation import evaluate_episode
from modelling.statistics import bootstrap_episode_estimates


class _FixedPolicy:
    def __init__(self, action: object) -> None:
        self.action = action

    def act(self, observation: object) -> object:  # noqa: ARG002
        return self.action


def _status(low: float, high: float, limit: float) -> str:
    if high <= limit:
        return "supported"
    if low > limit:
        return "violating"
    return "uncertain"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/research_dynamic.yaml"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("study/v7-paper-final/results/static_catalogue_fresh.csv"),
    )
    parser.add_argument("--seed-start", type=int, default=10_000)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--bootstrap-samples", type=int, default=5_000)
    parser.add_argument("--limit", type=float, default=0.01)
    arguments = parser.parse_args()
    config = load_environment_config(arguments.config)
    catalog = ActionCatalog.build(config)
    seeds = range(arguments.seed_start, arguments.seed_start + arguments.episodes)
    rows: list[dict[str, object]] = []
    for action in catalog.actions:
        episodes = [evaluate_episode(config, _FixedPolicy(action), seed=seed) for seed in seeds]
        estimates = bootstrap_episode_estimates(episodes, samples=arguments.bootstrap_samples)
        reward = estimates["total_reward"]
        mean_cost = estimates["mean_urllc_cost"]
        worst_cost = estimates["worst_group_urllc_cost"]
        rows.append(
            {
                "action_id": action.action_id,
                "embb_percent": action.embb_percent,
                "urllc_percent": action.urllc_percent,
                "mmtc_percent": action.mmtc_percent,
                "mitigation": action.mitigation.value,
                "episodes": len(episodes),
                "mean_reward": reward.mean,
                "reward_ci_low": reward.ci_low,
                "reward_ci_high": reward.ci_high,
                "mean_urllc_cost": mean_cost.mean,
                "mean_cost_ci_low": mean_cost.ci_low,
                "mean_cost_ci_high": mean_cost.ci_high,
                "mean_status": _status(mean_cost.ci_low, mean_cost.ci_high, arguments.limit),
                "max_regime_urllc_cost": worst_cost.mean,
                "cost_ci_low": worst_cost.ci_low,
                "cost_ci_high": worst_cost.ci_high,
                "worst_group_status": _status(worst_cost.ci_low, worst_cost.ci_high, arguments.limit),
                "point_category": (
                    "worst_group_feasible"
                    if worst_cost.mean <= arguments.limit
                    else "mean_only_feasible"
                    if mean_cost.mean <= arguments.limit
                    else "mean_infeasible"
                ),
            }
        )
        print(f"evaluated action {action.action_id + 1}/{len(catalog.actions)}", flush=True)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} fresh-trace static estimates to {arguments.output}")


if __name__ == "__main__":
    main()
