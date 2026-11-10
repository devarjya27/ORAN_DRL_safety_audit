#!/usr/bin/env python3
"""Train a reproducible feed-forward PPO baseline from a YAML scenario."""

from __future__ import annotations

import argparse
from pathlib import Path

from modelling.config import load_environment_config
from modelling.training import PPOTrainingConfig, train_baseline_ppo
from modelling.training.lagrangian import LagrangianConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--entropy-coefficient", type=float, default=0.001)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--device", default="cpu", help="PyTorch device; CPU is recommended for SB3 MLP PPO")
    parser.add_argument("--no-progress", action="store_true", help="disable the terminal training progress bar")
    parser.add_argument("--flat-actions", action="store_true", help="use the legacy 105-way categorical head instead of factorized actions")
    parser.add_argument("--n-envs", type=int, default=8, help="parallel training environments")
    parser.add_argument("--training-metrics", type=Path, help="CSV path for completed-episode reward history")
    parser.add_argument("--validation-metrics", type=Path, help="CSV path for deterministic fixed-seed evaluations")
    parser.add_argument("--eval-freq", type=int, default=10000)
    parser.add_argument("--validation-seeds", type=int, nargs="+", default=[101, 102, 103, 104, 105])
    parser.add_argument("--shielded", action="store_true")
    parser.add_argument("--urllc-cost-limit", type=float)
    parser.add_argument("--lagrangian-learning-rate", type=float, default=0.1)
    parser.add_argument("--initial-lagrange-multiplier", type=float, default=0.0)
    parser.add_argument(
        "--max-regime-urllc-cost-limit",
        type=float,
        help="optional checkpoint constraint on the worst observed traffic-regime mean cost",
    )
    arguments = parser.parse_args()
    constraint = None
    if arguments.urllc_cost_limit is not None:
        constraint = LagrangianConfig(
            urllc_cost_limit=arguments.urllc_cost_limit,
            multiplier_learning_rate=arguments.lagrangian_learning_rate,
            initial_multiplier=arguments.initial_lagrange_multiplier,
        )
    metadata = train_baseline_ppo(
        training=PPOTrainingConfig(
            total_timesteps=arguments.timesteps,
            seed=arguments.seed,
            n_steps=arguments.n_steps,
            batch_size=arguments.batch_size,
            n_epochs=arguments.epochs,
            learning_rate=arguments.learning_rate,
            clip_range=arguments.clip_range,
            entropy_coefficient=arguments.entropy_coefficient,
            target_kl=arguments.target_kl,
            device=arguments.device,
            progress_bar=not arguments.no_progress,
            factorized_actions=not arguments.flat_actions,
            n_envs=arguments.n_envs,
            eval_freq=arguments.eval_freq,
            validation_seeds=tuple(arguments.validation_seeds),
            max_regime_urllc_cost_limit=arguments.max_regime_urllc_cost_limit,
            use_safety_shield=arguments.shielded,
            constraint=constraint,
        ),
        output_path=arguments.output,
        config=load_environment_config(arguments.config),
        episode_metrics_path=arguments.training_metrics,
        validation_metrics_path=arguments.validation_metrics,
    )
    print(f"Wrote PPO model to {arguments.output.with_suffix('.zip')}; final multiplier={metadata.final_lagrange_multiplier}")


if __name__ == "__main__":
    main()
