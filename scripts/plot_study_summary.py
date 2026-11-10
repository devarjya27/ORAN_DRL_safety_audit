#!/usr/bin/env python3
"""Aggregate multi-seed study reports into paper-ready hierarchical plots."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRICS = {
    "total_reward": ("total_reward", "Cumulative reward", True),
    "mean_urllc_cost": ("mean_urllc_cost", "URLLC deadline cost", False),
    "max_regime_urllc_cost": ("max_regime_urllc_cost", "Worst-regime URLLC cost", False),
    "urllc_latency_cvar99_ms": ("urllc_latency_cvar99_ms", "URLLC CVaR99 (ms)", False),
    "max_consecutive_unsafe_periods": ("max_consecutive_unsafe_periods", "Maximum unsafe-period run", False),
    "mean_embb_cost": ("mean_embb_cost", "eMBB service cost", False),
    "mean_mmtc_cost": ("mean_mmtc_cost", "mMTC service cost", False),
    "mean_allocation_churn": ("mean_allocation_churn", "Allocation churn", False),
}

DISPLAY = {
    "equal_static": "Equal static",
    "validated_static": "Validated static",
    "conservative_urllc": "Conservative static",
    "demand_proportional": "Demand proportional",
    "ppo": "PPO",
    "dqn": "DQN",
    "recurrent_ppo": "Recurrent PPO-L",
    "heuristic_shield": "Recurrent + rules",
    "learned_shield": "Risk-ensemble shield",
}

ORDER = tuple(DISPLAY)


def _load_runs(paths: list[Path], selected: set[str] | None = None) -> dict[str, list[list[dict[str, object]]]]:
    collected: dict[str, list[list[dict[str, object]]]] = {}
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        for policy in report["policies"]:
            name = str(policy["name"])
            if selected is not None and name not in selected:
                continue
            collected.setdefault(name, []).append(policy["episodes"])
    return collected


def _hierarchical_estimate(
    runs: list[list[dict[str, object]]],
    field: str,
    *,
    bootstrap_samples: int = 5_000,
) -> dict[str, float | int]:
    arrays = [np.asarray([float(episode[field]) for episode in run], dtype=np.float64) for run in runs]
    if not arrays or any(len(values) == 0 for values in arrays):
        raise ValueError("every policy run must contain evaluation episodes")
    run_means = np.asarray([values.mean() for values in arrays])
    generator = np.random.default_rng(0)
    bootstrap = np.empty(bootstrap_samples, dtype=np.float64)
    for index in range(bootstrap_samples):
        selected_runs = generator.integers(0, len(arrays), size=len(arrays))
        means = []
        for run_index in selected_runs:
            values = arrays[int(run_index)]
            means.append(generator.choice(values, size=len(values), replace=True).mean())
        bootstrap[index] = float(np.mean(means))
    low, high = np.quantile(bootstrap, (0.025, 0.975))
    return {
        "mean": float(run_means.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "training_runs": len(arrays),
        "episodes_per_run": int(len(arrays[0])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm-reports", type=Path, nargs="+", required=True)
    parser.add_argument("--shield-reports", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Multi-seed conditional-safety evaluation")
    arguments = parser.parse_args()

    algorithm = _load_runs(arguments.algorithm_reports)
    shield = _load_runs(arguments.shield_reports, {"heuristic_shield", "learned_shield"})
    runs = dict(algorithm)
    runs.update(shield)
    # Static policies have no training seed. Repeating the identical controller
    # across learned-model directories would falsely inflate its run count.
    for name in ("equal_static", "validated_static", "conservative_urllc", "demand_proportional"):
        if name in runs:
            runs[name] = runs[name][:1]
    names = [name for name in ORDER if name in runs]
    summary = {
        name: {
            metric: _hierarchical_estimate(runs[name], field)
            for metric, (field, _, _) in METRICS.items()
        }
        for name in names
    }
    arguments.output.mkdir(parents=True, exist_ok=True)
    (arguments.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (arguments.output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("policy", "metric", "mean", "ci_low", "ci_high", "training_runs", "episodes_per_run"),
        )
        writer.writeheader()
        for name in names:
            for metric, estimate in summary[name].items():
                writer.writerow({"policy": name, "metric": metric, **estimate})

    colors = plt.cm.tab10(np.linspace(0.0, 0.9, len(names)))
    labels = [DISPLAY[name] for name in names]
    figure, axes = plt.subplots(2, 4, figsize=(20, 9.5))
    for axis, (metric, (_, label, higher_is_better)) in zip(axes.flat, METRICS.items()):
        means = np.asarray([summary[name][metric]["mean"] for name in names], dtype=float)
        lower = means - np.asarray([summary[name][metric]["ci_low"] for name in names], dtype=float)
        upper = np.asarray([summary[name][metric]["ci_high"] for name in names], dtype=float) - means
        bars = axis.barh(labels, means, xerr=np.vstack((lower, upper)), color=colors, capsize=3)
        axis.set_title(label)
        axis.grid(axis="x", alpha=0.25)
        axis.tick_params(axis="y", labelsize=8)
        axis.invert_yaxis()
        axis.margins(x=0.16)
        axis.set_xlim(left=0.0)
        axis.set_xlabel("higher is better" if higher_is_better else "lower is better", fontsize=8)
        for bar, value in zip(bars, means):
            axis.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f" {value:.3g}", ha="left", va="center", fontsize=7)
    figure.suptitle(arguments.title)
    figure.tight_layout()
    figure.savefig(arguments.output / "dashboard.png", dpi=220, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.5, 6.0))
    for color, name in zip(colors, names):
        reward = summary[name]["total_reward"]
        safety = summary[name]["max_regime_urllc_cost"]
        axis.errorbar(
            safety["mean"],
            reward["mean"],
            xerr=[[safety["mean"] - safety["ci_low"]], [safety["ci_high"] - safety["mean"]]],
            yerr=[[reward["mean"] - reward["ci_low"]], [reward["ci_high"] - reward["mean"]]],
            fmt="o",
            capsize=3,
            color=color,
            label=DISPLAY[name],
        )
    axis.set_xlabel("Worst-regime URLLC deadline cost (lower is safer)")
    axis.set_ylabel("Cumulative reward (higher is better)")
    axis.set_title(f"{arguments.title}: conditional-safety–utility frontier")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(arguments.output / "safety_utility_frontier.png", dpi=220, bbox_inches="tight")
    print(f"Wrote aggregate plots and summaries to {arguments.output}")


if __name__ == "__main__":
    main()
