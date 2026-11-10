#!/usr/bin/env python3
"""Plot fixed-seed validation convergence across independent training seeds."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load(path: Path, metric: str) -> dict[int, float]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {int(row["timesteps"]): float(row[metric]) for row in csv.DictReader(handle)}


def _run_label(path: Path) -> str:
    for part in reversed(path.parts):
        if part.startswith("seed-"):
            return part
    return path.stem


def _curve(paths: list[Path], metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    runs = [_load(path, metric) for path in paths]
    common = sorted(set.intersection(*(set(run) for run in runs)))
    if not common:
        raise ValueError("validation files do not share evaluation timesteps")
    values = np.asarray([[run[step] for step in common] for run in runs], dtype=np.float64)
    generator = np.random.default_rng(0)
    bootstrap = np.asarray(
        [values[generator.integers(0, len(values), size=len(values))].mean(axis=0) for _ in range(5_000)]
    )
    return (
        np.asarray(common),
        values.mean(axis=0),
        np.quantile(bootstrap, 0.025, axis=0),
        np.quantile(bootstrap, 0.975, axis=0),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppo", type=Path, nargs="+", required=True)
    parser.add_argument("--dqn", type=Path, nargs="+", required=True)
    parser.add_argument("--recurrent", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Fixed-seed validation convergence")
    parser.add_argument("--reference-label", default=None)
    parser.add_argument("--reference-reward", type=float, default=None)
    parser.add_argument("--reference-urllc-cost", type=float, default=None)
    parser.add_argument(
        "--show-individual",
        action="store_true",
        help="draw each training seed behind the cross-seed mean",
    )
    arguments = parser.parse_args()
    reference_values = (arguments.reference_reward, arguments.reference_urllc_cost)
    if arguments.reference_label is not None and any(value is None for value in reference_values):
        parser.error("--reference-label requires --reference-reward and --reference-urllc-cost")
    if arguments.reference_label is None and any(value is not None for value in reference_values):
        parser.error("reference values require --reference-label")
    figure, axes = plt.subplots(3, 1, figsize=(9.0, 10.5), sharex=True)
    algorithms = (
        ("PPO", arguments.ppo, "#2563eb"),
        ("DQN", arguments.dqn, "#ea580c"),
        ("Recurrent PPO-L", arguments.recurrent, "#16a34a"),
    )
    for label, paths, color in algorithms:
        for axis, metric in zip(axes, ("mean_reward", "mean_urllc_cost", "max_regime_urllc_cost")):
            if arguments.show_individual:
                for index, path in enumerate(paths):
                    run = _load(path, metric)
                    run_steps = np.asarray(sorted(run))
                    axis.plot(
                        run_steps,
                        [run[step] for step in run_steps],
                        color=color,
                        linewidth=1.0,
                        alpha=0.38,
                        linestyle=("--", ":", "-.")[index % 3],
                        label=f"{label} {_run_label(path)}",
                    )
            steps, mean, low, high = _curve(paths, metric)
            axis.plot(steps, mean, linewidth=2.4, color=color, label=f"{label} mean (n={len(paths)})")
            axis.fill_between(steps, low, high, color=color, alpha=0.18)
    axes[0].set_ylabel("Mean episode reward")
    axes[0].set_title(arguments.title)
    axes[1].set_ylabel("Mean URLLC cost")
    axes[2].set_ylabel("Worst-regime URLLC cost")
    axes[2].set_xlabel("Environment timesteps")
    axes[1].axhline(0.01, color="#991b1b", linestyle="--", linewidth=1.5, label="constraint (0.01)")
    axes[2].axhline(0.01, color="#991b1b", linestyle="--", linewidth=1.5, label="constraint (0.01)")
    if arguments.reference_label is not None:
        axes[0].axhline(
            arguments.reference_reward,
            color="#4b5563",
            linestyle=":",
            linewidth=1.8,
            label=arguments.reference_label,
        )
        axes[1].axhline(
            arguments.reference_urllc_cost,
            color="#4b5563",
            linestyle=":",
            linewidth=1.8,
            label=arguments.reference_label,
        )
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, dpi=220, bbox_inches="tight")
    print(f"Wrote {arguments.output}")


if __name__ == "__main__":
    main()
