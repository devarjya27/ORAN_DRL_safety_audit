#!/usr/bin/env python3
"""Reevaluate retained and final checkpoints with one metric and uncertainty.

Historical training CSVs used different aggregation orders for feed-forward
and recurrent validation.  This script reloads every artifact and computes the
mean of per-episode worst-regime costs uniformly, with bootstrap intervals.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from modelling.action_catalog import ActionCatalog
from modelling.config import load_environment_config
from modelling.evaluation import evaluate_episode
from modelling.features import FactorizedActionCodec, FeatureEncoder
from modelling.statistics import bootstrap_episode_estimates
from modelling.training.train_dqn import DQNPolicy
from modelling.training.train_ppo import PPOPolicy
from modelling.training.train_recurrent_ppo import load_recurrent_policy


def _load_policy(algorithm: str, path: Path, config: object, metadata: dict[str, object]) -> object:
    catalog = ActionCatalog.build(config)  # type: ignore[arg-type]
    encoder = FeatureEncoder(config, catalog)  # type: ignore[arg-type]
    if algorithm == "ppo":
        from stable_baselines3 import PPO

        codec = FactorizedActionCodec(catalog) if metadata.get("factorized_actions") else None
        return PPOPolicy(PPO.load(path, device="cpu"), catalog, encoder, codec)
    if algorithm == "dqn":
        from stable_baselines3 import DQN

        return DQNPolicy(DQN.load(path, device="cpu"), catalog, encoder)
    return load_recurrent_policy(path, config)  # type: ignore[arg-type]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path("study/v7-paper-final"))
    parser.add_argument("--config", type=Path, default=Path("configs/research_dynamic.yaml"))
    parser.add_argument("--output", type=Path, default=Path("study/v7-paper-final/checkpoint_audit.csv"))
    parser.add_argument("--seeds", type=int, nargs="*", default=list(range(1001, 1021)))
    parser.add_argument("--bootstrap-samples", type=int, default=5_000)
    parser.add_argument("--limit", type=float, default=0.01)
    arguments = parser.parse_args()
    config = load_environment_config(arguments.config)
    rows: list[dict[str, object]] = []
    for algorithm in ("ppo", "recurrent", "dqn"):
        for training_seed in range(11, 16):
            directory = arguments.study / "artifacts" / f"seed-{training_seed}"
            metadata = json.loads((directory / f"{algorithm}.metadata.json").read_text(encoding="utf-8"))
            suffix = ".pt" if algorithm == "recurrent" else ".zip"
            for state in ("retained", "final"):
                stem = algorithm if state == "retained" else f"{algorithm}.final"
                path = directory / f"{stem}{suffix}"
                policy = _load_policy(algorithm, path, config, metadata)
                episodes = [evaluate_episode(config, policy, seed=seed) for seed in arguments.seeds]
                estimates = bootstrap_episode_estimates(episodes, samples=arguments.bootstrap_samples)
                cost = estimates["worst_group_urllc_cost"]
                reward = estimates["total_reward"]
                mean_cost = estimates["mean_urllc_cost"]
                action_ids = sorted({action for episode in episodes for action in episode.action_ids})
                status = (
                    "supported"
                    if cost.ci_high <= arguments.limit
                    else "violating"
                    if cost.ci_low > arguments.limit
                    else "uncertain"
                )
                rows.append(
                    {
                        "algorithm": algorithm,
                        "training_seed": training_seed,
                        "state": state,
                        "episodes": len(episodes),
                        "mean_reward": reward.mean,
                        "reward_ci_low": reward.ci_low,
                        "reward_ci_high": reward.ci_high,
                        "mean_urllc_cost": mean_cost.mean,
                        "max_regime_urllc_cost": cost.mean,
                        "cost_ci_low": cost.ci_low,
                        "cost_ci_high": cost.ci_high,
                        "feasibility_status": status,
                        "unique_action_count": len(action_ids),
                        "action_ids": ";".join(str(value) for value in action_ids),
                    }
                )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} common-metric checkpoint estimates to {arguments.output}")


if __name__ == "__main__":
    main()
