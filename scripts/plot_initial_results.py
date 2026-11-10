#!/usr/bin/env python3
"""Create a compact paper-ready dashboard from a paired experiment report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PANELS = (
    ("total_reward", "Cumulative reward", True),
    ("mean_urllc_cost", "Mean URLLC deadline cost", False),
    ("urllc_latency_p99_ms", "URLLC p99 latency (ms)", False),
    ("urllc_latency_cvar99_ms", "URLLC CVaR99 latency (ms)", False),
    ("mean_urllc_percent", "Mean URLLC reservation (%)", False),
    ("mean_allocation_churn", "Mean allocation churn", False),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="O-RAN DRL Audit initial baseline comparison")
    arguments = parser.parse_args()
    report = json.loads(arguments.report.read_text(encoding="utf-8"))
    policies = report["policies"]
    labels = [policy["name"].replace("_", " ") for policy in policies]
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#4f46e5"]
    figure, axes = plt.subplots(2, 3, figsize=(13, 7))
    for axis, (metric, label, higher_is_better) in zip(axes.flat, PANELS):
        estimates = [policy["metrics"][metric] for policy in policies]
        means = [estimate["mean"] for estimate in estimates]
        errors = [
            [estimate["mean"] - estimate["ci_low"] for estimate in estimates],
            [estimate["ci_high"] - estimate["mean"] for estimate in estimates],
        ]
        bars = axis.bar(labels, means, yerr=errors, capsize=4, color=colors[: len(labels)])
        axis.set_title(label, fontsize=10)
        axis.tick_params(axis="x", rotation=18, labelsize=8)
        axis.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, means):
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3g}", ha="center", va="bottom", fontsize=8)
        axis.set_xlabel("higher is better" if higher_is_better else "lower is better", fontsize=8)
    figure.suptitle(arguments.title, fontsize=14)
    figure.tight_layout()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, dpi=200, bbox_inches="tight")
    print(f"Wrote {arguments.output}")


if __name__ == "__main__":
    main()
