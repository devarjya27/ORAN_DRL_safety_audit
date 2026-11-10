#!/usr/bin/env python3
"""Render a metric comparison with bootstrap confidence intervals from report.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True, help="paired/report.json from run_model_experiment.py")
    parser.add_argument("--metric", default="urllc_latency_cvar99_ms")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title")
    arguments = parser.parse_args()
    payload = json.loads(arguments.report.read_text(encoding="utf-8"))
    labels: list[str] = []
    means: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    for policy in payload["policies"]:
        try:
            estimate = policy["metrics"][arguments.metric]
        except KeyError as error:
            raise ValueError(f"metric {arguments.metric!r} is absent from the report") from error
        labels.append(policy["name"])
        means.append(estimate["mean"])
        lower.append(estimate["mean"] - estimate["ci_low"])
        upper.append(estimate["ci_high"] - estimate["mean"])
    figure, axis = plt.subplots(figsize=(max(6, len(labels) * 1.5), 4))
    axis.bar(labels, means, yerr=[lower, upper], capsize=5, color="#3b82b6")
    axis.set_ylabel(arguments.metric.replace("_", " "))
    axis.set_title(arguments.title or arguments.metric.replace("_", " "))
    axis.tick_params(axis="x", rotation=20)
    figure.tight_layout()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, dpi=180)
    print(f"Wrote {arguments.output}")


if __name__ == "__main__":
    main()
