#!/usr/bin/env python3
"""Fit and calibrate a learned URLLC-risk shield from a YAML scenario."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from modelling.action_catalog import ActionCatalog
from modelling.config import load_environment_config
from modelling.policies import baseline_policy
from modelling.safety.learned_risk import (
    EnsembleConfig,
    RiskEnsemble,
    calibrate_upper_bound_threshold,
    collect_risk_samples,
    save_risk_ensemble,
)
from modelling.training.train_ppo import PPOPolicy
from modelling.training.train_recurrent_ppo import load_recurrent_policy
from modelling.training.train_dqn import DQNPolicy


def _load_policy(kind: str, artifact: Path | None, catalog: ActionCatalog, config):
    if kind == "baseline":
        return baseline_policy("demand_proportional", catalog)
    if artifact is None:
        raise ValueError(f"--artifact is required for --policy-kind {kind}")
    if kind == "ppo":
        return PPOPolicy.load(artifact, config)
    if kind == "recurrent":
        return load_recurrent_policy(artifact, config)
    if kind == "dqn":
        return DQNPolicy.load(artifact, config)
    raise ValueError(f"unsupported policy kind: {kind}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--validation-config", type=Path, help="optional action-compatible validation scenario")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy-kind", choices=("baseline", "ppo", "recurrent", "dqn"), default="baseline")
    parser.add_argument("--artifact", type=Path, help="PPO/recurrent actor checkpoint when --policy-kind is not baseline")
    parser.add_argument("--train-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--validation-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--target-unsafe-rate", type=float, default=0.01)
    parser.add_argument(
        "--unsafe-cost-threshold",
        type=float,
        default=0.01,
        help="per-period URLLC cost above which a calibration sample is unsafe",
    )
    parser.add_argument("--minimum-accepted-fraction", type=float, default=0.25)
    parser.add_argument("--members", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--exploration-probability",
        type=float,
        default=0.25,
        help="fraction of training samples drawn from random feasible catalogue actions",
    )
    parser.add_argument(
        "--validation-exploration-probability",
        type=float,
        default=0.25,
        help="catalogue exploration used to calibrate actions beyond the actor's dominant mode",
    )
    arguments = parser.parse_args()

    config = load_environment_config(arguments.config)
    validation_config = config if arguments.validation_config is None else load_environment_config(arguments.validation_config)
    catalog = ActionCatalog.build(config)
    policy = _load_policy(arguments.policy_kind, arguments.artifact, catalog, config)
    validation_catalog = ActionCatalog.build(validation_config)
    if validation_catalog.actions != catalog.actions:
        raise ValueError("validation scenario must retain the training action catalogue")
    train_samples = collect_risk_samples(
        config,
        policy,
        seeds=arguments.train_seeds,
        exploration_probability=arguments.exploration_probability,
        exploration_seed=arguments.seed,
    )
    validation_samples = collect_risk_samples(
        validation_config,
        policy,
        seeds=arguments.validation_seeds,
        exploration_probability=arguments.validation_exploration_probability,
        exploration_seed=arguments.seed + 1_000_003,
    )
    ensemble = RiskEnsemble(
        catalog,
        EnsembleConfig(
            members=arguments.members,
            hidden_size=arguments.hidden_size,
            epochs=arguments.epochs,
            seed=arguments.seed,
        ),
    )
    losses = ensemble.fit(train_samples)
    calibration = calibrate_upper_bound_threshold(
        ensemble,
        validation_samples,
        target_unsafe_rate=arguments.target_unsafe_rate,
        unsafe_cost_threshold=arguments.unsafe_cost_threshold,
        minimum_accepted_fraction=arguments.minimum_accepted_fraction,
    )
    if not calibration.target_met:
        raise RuntimeError(
            "risk calibration could not meet the unsafe-rate target at the required coverage; "
            "collect more diverse samples or improve the predictor before evaluating the shield"
        )
    arguments.output.mkdir(parents=True, exist_ok=True)
    artifact = save_risk_ensemble(ensemble, arguments.output / "risk_ensemble", config)
    (arguments.output / "calibration.json").write_text(
        json.dumps(
            {
                "policy_kind": arguments.policy_kind,
                "policy_artifact": None if arguments.artifact is None else str(arguments.artifact),
                "seed": arguments.seed,
                "exploration_probability": arguments.exploration_probability,
                "validation_exploration_probability": arguments.validation_exploration_probability,
                "unsafe_cost_threshold": arguments.unsafe_cost_threshold,
                "validation_config": None if arguments.validation_config is None else str(arguments.validation_config),
                "train_samples": len(train_samples),
                "validation_samples": len(validation_samples),
                "member_final_losses": losses,
                "calibration": asdict(calibration),
                "artifact": asdict(artifact),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote learned-risk artifact and calibration to {arguments.output}")


if __name__ == "__main__":
    main()
