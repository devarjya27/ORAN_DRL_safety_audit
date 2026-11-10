#!/usr/bin/env python3
"""Train the custom recurrent PPO baseline from a YAML scenario."""

from __future__ import annotations

import argparse
from pathlib import Path

from modelling.config import load_environment_config
from modelling.training.lagrangian import LagrangianConfig
from modelling.training.recurrent_ppo import RecurrentPPOConfig
from modelling.training.train_recurrent_ppo import RecurrentTrainingConfig, train_recurrent_ppo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--n-envs", type=int, default=8, help="parallel recurrent rollout environments")
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--sequences-per-batch", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--clip-range", type=float, default=0.15)
    parser.add_argument("--entropy-coefficient", type=float, default=0.005)
    parser.add_argument("--target-kl", type=float, default=0.015)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shielded", action="store_true")
    parser.add_argument("--urllc-cost-limit", type=float)
    parser.add_argument("--max-regime-urllc-cost-limit", type=float)
    parser.add_argument("--no-progress", action="store_true", help="disable the terminal training progress bar")
    parser.add_argument("--eval-every-updates", type=int, default=10)
    parser.add_argument("--validation-seeds", type=int, nargs="+", default=[101, 102, 103, 104, 105])
    arguments = parser.parse_args()
    constraint = None if arguments.urllc_cost_limit is None else LagrangianConfig(urllc_cost_limit=arguments.urllc_cost_limit)
    metadata = train_recurrent_ppo(
        training=RecurrentTrainingConfig(
            updates=arguments.updates,
            horizon=arguments.horizon,
            hidden_size=arguments.hidden_size,
            n_envs=arguments.n_envs,
            seed=arguments.seed,
            use_safety_shield=arguments.shielded,
            progress_bar=not arguments.no_progress,
            ppo=RecurrentPPOConfig(
                learning_rate=arguments.learning_rate,
                clip_range=arguments.clip_range,
                epochs=arguments.epochs,
                sequence_length=arguments.sequence_length,
                sequences_per_batch=arguments.sequences_per_batch,
                entropy_coefficient=arguments.entropy_coefficient,
                target_kl=arguments.target_kl,
            ),
            constraint=constraint,
            eval_every_updates=arguments.eval_every_updates,
            validation_seeds=tuple(arguments.validation_seeds),
            max_regime_urllc_cost_limit=arguments.max_regime_urllc_cost_limit,
        ),
        output_path=arguments.output,
        config=load_environment_config(arguments.config),
    )
    print(f"Wrote recurrent PPO model to {arguments.output.with_suffix('.pt')}; final multiplier={metadata.final_lagrange_multiplier}")


if __name__ == "__main__":
    main()
