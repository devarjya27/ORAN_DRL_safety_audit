#!/usr/bin/env python3
"""Summarize learned-shield calibration transfer from saved assessments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


SCENARIOS = ("id_population", "constant_stress", "heldout_sweep")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path("study/v7-paper-final"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("study/v7-paper-final/results/shield_transfer.csv"),
    )
    arguments = parser.parse_args()
    rows: list[dict[str, object]] = []
    for scenario in SCENARIOS:
        seed_rows: list[tuple[float, float, float]] = []
        for seed in range(11, 16):
            path = (
                arguments.study
                / "results"
                / scenario
                / f"seed-{seed}"
                / "shielded"
                / "learned_assessment.json"
            )
            report = json.loads(path.read_text(encoding="utf-8"))
            periods = sum(item["periods"] for item in report["episodes"])
            interventions = sum(item["interventions"] for item in report["episodes"])
            false_negatives = sum(item["actor_false_negatives"] for item in report["episodes"])
            realized_unsafe = sum(item["realized_unsafe_periods"] for item in report["episodes"])
            accepted = periods - interventions
            seed_rows.append(
                (
                    false_negatives / accepted,
                    realized_unsafe / periods,
                    interventions / periods,
                )
            )
        rows.append(
            {
                "scenario": scenario,
                "training_seeds": len(seed_rows),
                "episodes_per_seed": len(report["episodes"]),
                "accepted_action_unsafe_rate": sum(row[0] for row in seed_rows) / len(seed_rows),
                "realized_unsafe_rate": sum(row[1] for row in seed_rows) / len(seed_rows),
                "intervention_rate": sum(row[2] for row in seed_rows) / len(seed_rows),
            }
        )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} shield-transfer rows to {arguments.output}")


if __name__ == "__main__":
    main()
