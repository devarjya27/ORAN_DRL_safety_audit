#!/usr/bin/env python3
"""Generate the compact, publication-ready figures used by the final paper."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LIMIT = 0.01
COLORS = {
    "unsafe": "#b91c1c",
    "mean_only": "#d97706",
    "joint": "#15803d",
    "ppo": "#2563eb",
    "dqn": "#c2410c",
    "recurrent_ppo": "#7c3aed",
    "heuristic_shield": "#64748b",
    "learned_shield": "#0891b2",
    "mean_validated_static": "#d97706",
    "joint_validated_static": "#15803d",
    "conservative_urllc": "#334155",
    "privileged_regime_oracle": "#111827",
    "offered_load_heuristic": "#db2777",
}
LABELS = {
    "ppo": "PPO-L",
    "dqn": "DQN",
    "recurrent_ppo": "Recurrent PPO-L",
    "recurrent": "Recurrent PPO-L",
    "heuristic_shield": "Rule shield",
    "learned_shield": "Risk-ensemble shield",
    "mean_validated_static": "Mean-validated static",
    "joint_validated_static": "Group-validated static",
    "conservative_urllc": "50% URLLC reserve",
    "privileged_regime_oracle": "Privileged traffic-group oracle",
    "offered_load_heuristic": "Observed-load heuristic",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9.5,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "grid.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save(figure: plt.Figure, output: Path, name: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / f"{name}.pdf", bbox_inches="tight")
    figure.savefig(output / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def static_frontier(study: Path, output: Path) -> None:
    rows = _read_csv(study / "results/static_catalogue_fresh.csv")
    points = []
    for row in rows:
        mean_cost = float(row["mean_urllc_cost"])
        worst_cost = float(row["max_regime_urllc_cost"])
        category = "joint" if mean_cost <= LIMIT and worst_cost <= LIMIT else "mean_only" if mean_cost <= LIMIT else "unsafe"
        points.append((int(row["action_id"]), float(row["mean_reward"]), mean_cost, worst_cost, category))
    category_counts = {category: sum(point[4] == category for point in points) for category in ("joint", "mean_only", "unsafe")}
    if len(points) != 105 or category_counts != {"joint": 46, "mean_only": 11, "unsafe": 48}:
        raise ValueError(f"unexpected publication frontier: {len(points)=}, {category_counts=}")

    figure, axes = plt.subplots(1, 2, figsize=(7.15, 3.0), gridspec_kw={"width_ratios": (1.1, 1.0)})
    for axis in axes:
        for category, label in (("unsafe", "Mean-infeasible"), ("mean_only", "Mean-only feasible"), ("joint", "Worst-group feasible")):
            selected = [point for point in points if point[4] == category]
            axis.scatter(
                [point[3] for point in selected],
                [point[1] for point in selected],
                s=18,
                alpha=0.72,
                color=COLORS[category],
                edgecolor="white",
                linewidth=0.35,
                label=label,
            )
        axis.axvspan(0, LIMIT, color="#dcfce7", alpha=0.45, zorder=-2)
        axis.axvline(LIMIT, color="#991b1b", linestyle="--", linewidth=1.0)
        axis.set_xlabel("Worst-group URLLC cost")
        axis.set_ylabel("Mean episode return")
    axes[0].set_title("All 105 catalogue actions")
    axes[0].set_xlim(left=-0.005)
    axes[1].set_title("Safety-boundary detail")
    axes[1].set_xlim(-0.001, 0.05)
    axes[1].set_ylim(174.5, 184.5)
    for action_id, reward, _, worst, _ in points:
        if action_id not in {60, 61, 65}:
            continue
        axes[1].scatter([worst], [reward], s=44, facecolor="none", edgecolor="#111827", linewidth=1.1, zorder=5)
        offsets = {60: (-38, 8), 61: (6, -13), 65: (7, 5)}
        axes[1].annotate(
            f"a{action_id}",
            (worst, reward),
            xytext=offsets[action_id],
            textcoords="offset points",
            arrowprops={"arrowstyle": "-", "color": "#475569", "lw": 0.7},
        )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.05))
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    _save(figure, output, "static_conditional_frontier")


def checkpoint_stability(study: Path, output: Path) -> None:
    seeds = (11, 12, 13, 14, 15)
    algorithms = (("ppo", "PPO-L"), ("recurrent", "Recurrent PPO-L"), ("dqn", "DQN"))
    audit = _read_csv(study / "checkpoint_audit_fresh.csv")
    figure, axes = plt.subplots(1, 3, figsize=(7.15, 2.7), sharey=True)
    for axis, (algorithm, title) in zip(axes, algorithms):
        records = {
            (int(row["training_seed"]), row["state"]): row
            for row in audit
            if row["algorithm"] == algorithm
        }
        retained = [records[(seed, "retained")] for seed in seeds]
        final = [records[(seed, "final")] for seed in seeds]
        selected_values = [float(row["max_regime_urllc_cost"]) for row in retained]
        final_values = [float(row["max_regime_urllc_cost"]) for row in final]
        for x, selected, final_value in zip(seeds, selected_values, final_values):
            axis.plot([x, x], [selected, final_value], color="#94a3b8", linewidth=1.0, zorder=1)
        for rows, values, color, marker, label in (
            (retained, selected_values, COLORS["ppo"], "o", "Safety-ranked retained"),
            (final, final_values, COLORS["unsafe"], "X", "Final optimizer state"),
        ):
            lower = [value - float(row["cost_ci_low"]) for value, row in zip(values, rows)]
            upper = [float(row["cost_ci_high"]) - value for value, row in zip(values, rows)]
            axis.errorbar(
                seeds,
                values,
                yerr=np.asarray((lower, upper)),
                fmt=marker,
                color=color,
                markersize=5,
                capsize=2,
                linewidth=0.9,
                label=label,
                zorder=3,
            )
        axis.axhspan(1e-5, LIMIT, color="#dcfce7", alpha=0.45, zorder=-2)
        axis.axhline(LIMIT, color="#991b1b", linestyle="--", linewidth=1.0)
        axis.set_yscale("log")
        axis.set_xticks(seeds)
        axis.set_xlabel("Training seed")
        axis.set_title(title)
    axes[0].set_ylabel("Worst-group URLLC cost")
    axes[0].set_ylim(1e-4, 0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.08))
    figure.tight_layout(rect=(0, 0.09, 1, 1))
    _save(figure, output, "checkpoint_safety_stability")


def shield_per_seed(study: Path, output: Path) -> None:
    seeds = (11, 12, 13, 14, 15)
    scenarios = (("id_population", "Training-matched"), ("constant_stress", "Constant stress"))
    policies = ("recurrent", "heuristic_shield", "learned_shield")
    markers = {"recurrent": "o", "heuristic_shield": "s", "learned_shield": "D"}
    paper_summary = json.loads((study / "paper_summary.json").read_text(encoding="utf-8"))["scenarios"]
    figure, axes = plt.subplots(2, 1, figsize=(3.45, 4.1), sharex=True)
    for axis, (scenario, title) in zip(axes, scenarios):
        collected = {}
        for policy in policies:
            summary_key = "recurrent_ppo" if policy == "recurrent" else policy
            runs = paper_summary[scenario][summary_key]["runs"]
            by_seed = {
                int(run["training_seed"]): float(run["metrics"]["worst_group_urllc_cost"]["mean"])
                for run in runs
            }
            collected[policy] = [by_seed[seed] for seed in seeds]
        for policy in policies:
            color = COLORS["recurrent_ppo"] if policy == "recurrent" else COLORS[policy]
            axis.plot(
                seeds,
                collected[policy],
                color=color,
                marker=markers[policy],
                linewidth=1.7,
                markersize=4.5,
                label=LABELS[policy],
            )
        axis.axhspan(0, LIMIT, color="#dcfce7", alpha=0.45, zorder=-2)
        axis.axhline(LIMIT, color="#991b1b", linestyle="--", linewidth=1.0)
        axis.set_title(title)
        axis.set_ylabel("Worst-group cost")
        axis.set_xticks(seeds)
    axes[0].set_ylim(0, 0.025)
    axes[1].set_ylim(0, 0.03)
    axes[1].set_xlabel("Training seed")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=1, frameon=False, bbox_to_anchor=(0.5, -0.01))
    figure.tight_layout(rect=(0, 0.16, 1, 1))
    _save(figure, output, "shield_per_seed")


def _aggregate_metric(summary: dict[str, object], policy: str, metric: str) -> tuple[float, float, float]:
    values = summary[policy][metric]  # type: ignore[index]
    return float(values["mean"]), float(values["ci_low"]), float(values["ci_high"])  # type: ignore[index]


def scenario_frontiers(study: Path, output: Path) -> None:
    scenarios = (("id_population", "Training-matched"), ("constant_stress", "Constant stress"), ("heldout_sweep", "Unseen sweep jammer"))
    policies = (
        "mean_validated_static",
        "joint_validated_static",
        "conservative_urllc",
        "ppo",
        "dqn",
        "recurrent_ppo",
        "learned_shield",
        "privileged_regime_oracle",
        "offered_load_heuristic",
    )
    markers = ("v", "P", "s", "o", "X", "^", "D", "*", "h")
    paper_summary = json.loads((study / "paper_summary.json").read_text(encoding="utf-8"))["scenarios"]
    figure, axes = plt.subplots(1, 3, figsize=(7.15, 3.05))
    for axis, (scenario, title) in zip(axes, scenarios):
        summary = paper_summary[scenario]
        for policy, marker in zip(policies, markers):
            cost_values = summary[policy]["metrics"]["worst_group_urllc_cost"]
            reward_values = summary[policy]["metrics"]["total_reward"]
            cost = float(cost_values["mean"])
            cost_low = float(cost_values["ci_low"])
            cost_high = float(cost_values["ci_high"])
            reward = float(reward_values["mean"])
            reward_low = float(reward_values["ci_low"])
            reward_high = float(reward_values["ci_high"])
            plot_cost = max(cost, 1e-5)
            plot_cost_low = max(cost_low, 1e-5)
            plot_cost_high = max(cost_high, plot_cost)
            axis.errorbar(
                plot_cost,
                reward,
                xerr=[[max(0.0, plot_cost - plot_cost_low)], [max(0.0, plot_cost_high - plot_cost)]],
                yerr=[[max(0.0, reward - reward_low)], [max(0.0, reward_high - reward)]],
                fmt=marker,
                markersize=5.2,
                color=COLORS[policy],
                capsize=2,
                linewidth=0.9,
                label=LABELS[policy],
            )
        axis.axvspan(1e-5, LIMIT, color="#dcfce7", alpha=0.45, zorder=-2)
        axis.axvline(LIMIT, color="#991b1b", linestyle="--", linewidth=1.0)
        axis.set_xscale("log")
        axis.set_xlim(8e-6, 0.35)
        axis.set_title(title)
        axis.set_xlabel("Worst-group URLLC cost")
    axes[0].set_ylabel("Mean episode return")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.13))
    figure.tight_layout(rect=(0, 0.24, 1, 1))
    _save(figure, output, "scenario_safety_utility")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path("study/v7-paper-final"))
    parser.add_argument("--output", type=Path, default=Path("paper/figures/final"))
    arguments = parser.parse_args()
    _style()
    static_frontier(arguments.study, arguments.output)
    checkpoint_stability(arguments.study, arguments.output)
    shield_per_seed(arguments.study, arguments.output)
    scenario_frontiers(arguments.study, arguments.output)
    print(f"Wrote paper figures to {arguments.output}")


if __name__ == "__main__":
    main()
