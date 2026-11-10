#!/usr/bin/env python3
"""Plot recurrent PPO optimization, validation, safety cost, and dual state."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"metrics file has no rows: {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--validation-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--urllc-cost-limit", type=float, default=0.01)
    parser.add_argument("--title", default="Recurrent constrained PPO diagnostics")
    arguments = parser.parse_args()
    training = _rows(arguments.metrics)
    validation = _rows(arguments.validation_metrics)
    steps = np.asarray([int(row["timesteps"]) for row in training])
    validation_steps = np.asarray([int(row["timesteps"]) for row in validation])

    figure, axes = plt.subplots(3, 1, figsize=(8.5, 9.0), sharex=True)
    axes[0].plot(steps, [float(row["mean_step_reward"]) for row in training], color="#1d4ed8")
    axes[0].set_ylabel("Mean step reward")
    axes[0].grid(alpha=0.25)

    validation_mean = np.asarray([float(row["mean_reward"]) for row in validation])
    validation_std = np.asarray([float(row["std_reward"]) for row in validation])
    axes[1].plot(validation_steps, validation_mean, color="#15803d", marker="o", label="fixed-seed validation")
    axes[1].fill_between(
        validation_steps,
        validation_mean - validation_std,
        validation_mean + validation_std,
        color="#bbf7d0",
        alpha=0.45,
    )
    axes[1].set_ylabel("Episode reward")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")

    costs = np.asarray([float(row["mean_urllc_cost"]) for row in training])
    axes[2].plot(steps, costs, color="#dc2626", label="rollout URLLC cost")
    axes[2].axhline(arguments.urllc_cost_limit, color="#7f1d1d", linestyle="--", label="cost limit")
    axes[2].set_ylabel("URLLC cost")
    axes[2].set_xlabel("Environment timesteps")
    axes[2].grid(alpha=0.25)
    multiplier_axis = axes[2].twinx()
    multiplier_axis.plot(
        steps,
        [float(row["lagrange_multiplier"]) for row in training],
        color="#7c3aed",
        alpha=0.75,
        label="Lagrange multiplier",
    )
    multiplier_axis.set_ylabel("Multiplier")
    lines, labels = axes[2].get_legend_handles_labels()
    dual_lines, dual_labels = multiplier_axis.get_legend_handles_labels()
    axes[2].legend(lines + dual_lines, labels + dual_labels, loc="best")

    figure.suptitle(arguments.title)
    figure.tight_layout()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, dpi=200)
    print(f"Wrote {arguments.output}")


if __name__ == "__main__":
    main()
