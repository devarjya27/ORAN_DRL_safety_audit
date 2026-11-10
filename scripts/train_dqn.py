#!/usr/bin/env python3
"""Train the normalized discrete DQN comparison baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

from modelling.config import load_environment_config
from modelling.training.train_dqn import DQNTrainingConfig, train_dqn


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--learning-starts", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--target-update-interval", type=int, default=20_000)
    parser.add_argument("--exploration-fraction", type=float, default=0.5)
    parser.add_argument("--exploration-final-epsilon", type=float, default=0.1)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--urllc-cost-limit", type=float, default=0.01)
    parser.add_argument("--max-regime-urllc-cost-limit", type=float)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--validation-seeds", type=int, nargs="+", default=[101, 102, 103, 104, 105])
    parser.add_argument("--no-progress", action="store_true")
    arguments = parser.parse_args()
    train_dqn(
        training=DQNTrainingConfig(
            total_timesteps=arguments.timesteps,
            seed=arguments.seed,
            learning_rate=arguments.learning_rate,
            buffer_size=arguments.buffer_size,
            learning_starts=arguments.learning_starts,
            batch_size=arguments.batch_size,
            target_update_interval=arguments.target_update_interval,
            exploration_fraction=arguments.exploration_fraction,
            exploration_final_epsilon=arguments.exploration_final_epsilon,
            hidden_sizes=tuple(arguments.hidden_sizes),
            checkpoint_urllc_cost_limit=arguments.urllc_cost_limit,
            checkpoint_max_regime_urllc_cost_limit=arguments.max_regime_urllc_cost_limit,
            eval_freq=arguments.eval_freq,
            validation_seeds=tuple(arguments.validation_seeds),
            progress_bar=not arguments.no_progress,
        ),
        output_path=arguments.output,
        config=load_environment_config(arguments.config),
    )
    print(f"Wrote DQN model to {arguments.output.with_suffix('.zip')}")


if __name__ == "__main__":
    main()
