#!/usr/bin/env python3
"""Sensitivity sweep over traffic bursts, KPM impairment, and safety limit."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path

from modelling.action_catalog import ActionCatalog
from modelling.baselines import ConservativeURLLCPolicy, JointValidatedStaticPolicy
from modelling.config import load_environment_config
from modelling.evaluation import evaluate_episode
from modelling.statistics import bootstrap_episode_estimates
from scripts.run_headroom_analysis import OfferedLoadHeuristicPolicy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--headroom",
        type=Path,
        default=Path("study/v7-paper-final/results/headroom_tuned005/report.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("study/v7-paper-final/results/sensitivity.csv"))
    parser.add_argument("--episodes", type=int, default=100)
    arguments = parser.parse_args()
    base = load_environment_config(Path("configs/research_dynamic.yaml"))
    mapping = json.loads(arguments.headroom.read_text(encoding="utf-8"))["selected_mapping"]
    seeds = tuple(range(50_000, 50_000 + arguments.episodes))
    impairment_levels = (
        (0, 0.0, "none"),
        (1, 0.02, "nominal"),
        (2, 0.08, "severe"),
        (5, 0.25, "harsh"),
    )
    rows: list[dict[str, object]] = []
    for surge in (1.8, 2.2, 2.6):
        for delay, loss, impairment in impairment_levels:
            config = replace(
                base,
                traffic_surge_multiplier=surge,
                kpm_delay_steps=delay,
                kpm_loss_probability=loss,
            )
            catalog = ActionCatalog.build(config)
            policies = {
                "joint_validated_static": JointValidatedStaticPolicy(catalog),
                "conservative_reserve": ConservativeURLLCPolicy(catalog),
                "offered_load_heuristic": OfferedLoadHeuristicPolicy(catalog, config, mapping),
            }
            for name, policy in policies.items():
                episodes = [evaluate_episode(config, policy, seed=seed) for seed in seeds]
                estimates = bootstrap_episode_estimates(episodes, samples=2_000)
                reward = estimates["total_reward"]
                cost = estimates["worst_group_urllc_cost"]
                for limit in (0.005, 0.01, 0.02):
                    status = "supported" if cost.ci_high <= limit else "violating" if cost.ci_low > limit else "uncertain"
                    rows.append(
                        {
                            "surge_multiplier": surge,
                            "impairment": impairment,
                            "kpm_delay_steps": delay,
                            "kpm_loss_probability": loss,
                            "limit": limit,
                            "policy": name,
                            "mean_reward": reward.mean,
                            "reward_ci_low": reward.ci_low,
                            "reward_ci_high": reward.ci_high,
                            "max_regime_cost": cost.mean,
                            "cost_ci_low": cost.ci_low,
                            "cost_ci_high": cost.ci_high,
                            "feasibility_status": status,
                        }
                    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} sensitivity rows to {arguments.output}")


if __name__ == "__main__":
    main()
