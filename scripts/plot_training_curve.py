#!/usr/bin/env python3
"""Plot reward versus completed training episode from PPO CSV artifacts."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True, help="*.training.csv emitted by train_ppo.py")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=int, default=10, help="moving-average window in episodes")
    parser.add_argument("--validation-metrics", type=Path, help="deterministic *.validation.csv from train_ppo.py")
    parser.add_argument("--episode-steps", type=int, default=200, help="used to align validation timesteps to episode count")
    parser.add_argument("--title", default="PPO training reward")
    arguments = parser.parse_args()
    if arguments.window <= 0:
        raise ValueError("--window must be positive")
    with arguments.metrics.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("training metrics has no completed episodes; increase --timesteps")
    episodes = np.asarray([int(row["episode"]) for row in rows])
    rewards = np.asarray([float(row["total_reward"]) for row in rows])
    window = min(arguments.window, len(rewards))
    smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(episodes, rewards, color="#93c5fd", linewidth=1.0, label="episode reward")
    axis.plot(episodes[window - 1 :], smoothed, color="#1d4ed8", linewidth=2.2, label=f"{window}-episode moving average")
    if arguments.validation_metrics is not None:
        with arguments.validation_metrics.open(encoding="utf-8", newline="") as handle:
            validation = list(csv.DictReader(handle))
        if validation:
            validation_episodes = np.asarray([float(row["timesteps"]) / arguments.episode_steps for row in validation])
            means = np.asarray([float(row["mean_reward"]) for row in validation])
            deviations = np.asarray([float(row["std_reward"]) for row in validation])
            axis.plot(validation_episodes, means, color="#dc2626", marker="o", linewidth=2.0, label="deterministic validation")
            axis.fill_between(validation_episodes, means - deviations, means + deviations, color="#fecaca", alpha=0.35)
    axis.set_xlabel("Completed training episode")
    axis.set_ylabel("Total reward")
    axis.set_title(arguments.title)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, dpi=200)
    print(f"Wrote {arguments.output}")


if __name__ == "__main__":
    main()
