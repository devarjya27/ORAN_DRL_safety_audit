#!/usr/bin/env python3
"""Build the paper's common-estimator scenario summary from raw episodes."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from modelling.action_catalog import ActionCatalog
from modelling.config import load_environment_config
from modelling.evaluation import EpisodeSummary, evaluate_episode
from modelling.statistics import bootstrap_episode_estimates, hierarchical_episode_estimates


LEARNED = ("ppo", "dqn", "recurrent_ppo")
SHIELDED = ("heuristic_shield", "learned_shield")
STATIC = ("mean_validated_static", "joint_validated_static", "conservative_urllc")
SCENARIOS = ("id_population", "constant_stress", "heldout_sweep")


class _FixedActionPolicy:
    def __init__(self, action: object) -> None:
        self.action = action

    def act(self, observation: object) -> object:  # noqa: ARG002
        return self.action


def _episodes(payload: list[dict[str, object]]) -> list[EpisodeSummary]:
    return [EpisodeSummary(**{**item, "action_ids": tuple(item["action_ids"])}) for item in payload]  # type: ignore[arg-type]


def _policy_episodes(path: Path, name: str) -> list[EpisodeSummary]:
    report = json.loads(path.read_text(encoding="utf-8"))
    policy = next(item for item in report["policies"] if item["name"] == name)
    return _episodes(policy["episodes"])


def _estimate_policy(runs: list[list[EpisodeSummary]], seeds: list[int | None], limit: float) -> dict[str, object]:
    aggregate = hierarchical_episode_estimates(runs)
    run_rows = []
    counts = {"supported": 0, "uncertain": 0, "violating": 0}
    for seed, episodes in zip(seeds, runs):
        estimates = bootstrap_episode_estimates(episodes)
        cost = estimates["worst_group_urllc_cost"]
        status = "supported" if cost.ci_high <= limit else "violating" if cost.ci_low > limit else "uncertain"
        counts[status] += 1
        run_rows.append(
            {
                "training_seed": seed,
                "status": status,
                "metrics": {name: asdict(estimate) for name, estimate in estimates.items()},
            }
        )
    return {
        "metrics": {name: asdict(estimate) for name, estimate in aggregate.items()},
        "feasibility_counts": counts,
        "runs": run_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path("study/v7-paper-final"))
    parser.add_argument(
        "--headroom",
        type=Path,
        default=Path("study/v7-paper-final/results/headroom_tuned005/report.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("study/v7-paper-final/paper_summary.json"))
    parser.add_argument("--limit", type=float, default=0.01)
    arguments = parser.parse_args()
    headroom = json.loads(arguments.headroom.read_text(encoding="utf-8"))
    headroom_names = {
        "id_population": "training_matched",
        "constant_stress": "constant_stress",
        "heldout_sweep": "unseen_sweep",
    }
    validation_config = load_environment_config(Path("configs/research_dynamic.yaml"))
    validation_catalog = ActionCatalog.build(validation_config)
    static_validation = {}
    for action_id in (60, 61, 65):
        episodes = [
            evaluate_episode(validation_config, _FixedActionPolicy(validation_catalog.get(action_id)), seed=seed)  # type: ignore[arg-type]
            for seed in range(1001, 1021)
        ]
        static_validation[str(action_id)] = {
            name: asdict(estimate) for name, estimate in bootstrap_episode_estimates(episodes).items()
        }
    output: dict[str, object] = {
        "schema_version": 1,
        "safety_limit": arguments.limit,
        "static_validation": static_validation,
        "scenarios": {},
    }
    for scenario in SCENARIOS:
        policies: dict[str, object] = {}
        for name in STATIC:
            run = _policy_episodes(arguments.study / "results/paper_static" / scenario / "report.json", name)
            policies[name] = _estimate_policy([run], [None], arguments.limit)
        for name in LEARNED:
            runs = [
                _policy_episodes(arguments.study / "results" / scenario / f"seed-{seed}/algorithms/report.json", name)
                for seed in range(11, 16)
            ]
            policies[name] = _estimate_policy(runs, list(range(11, 16)), arguments.limit)
        for name in SHIELDED:
            runs = [
                _policy_episodes(arguments.study / "results" / scenario / f"seed-{seed}/shielded/paired/report.json", name)
                for seed in range(11, 16)
            ]
            policies[name] = _estimate_policy(runs, list(range(11, 16)), arguments.limit)
        source = headroom["scenarios"][headroom_names[scenario]]
        for name in ("privileged_regime_oracle", "offered_load_heuristic"):
            run = _episodes(source[name]["episodes"])
            policies[name] = _estimate_policy([run], [None], arguments.limit)
        output["scenarios"][scenario] = policies  # type: ignore[index]
    arguments.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote common-estimator paper summary to {arguments.output}")


if __name__ == "__main__":
    main()
