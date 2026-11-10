#!/usr/bin/env python3
"""Compare static, on-policy, off-policy, and recurrent policies on matched traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from modelling.action_catalog import ActionCatalog
from modelling.config import load_environment_config, scenario_fingerprint
from modelling.experiments import run_paired_experiment
from modelling.policies import baseline_policy
from modelling.training.train_dqn import DQNPolicy
from modelling.training.train_ppo import PPOPolicy
from modelling.training.train_recurrent_ppo import load_recurrent_policy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/research_dynamic.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--ppo", type=Path)
    parser.add_argument("--dqn", type=Path)
    parser.add_argument("--recurrent", type=Path)
    parser.add_argument("--allow-heldout-scenario", action="store_true")
    arguments = parser.parse_args()
    config = load_environment_config(arguments.config)
    catalog = ActionCatalog.build(config)
    policies = {
        "equal_static": baseline_policy("equal_static", catalog),
        "validated_static": baseline_policy("validated_static", catalog),
        "conservative_urllc": baseline_policy("conservative_urllc", catalog),
        "demand_proportional": baseline_policy("demand_proportional", catalog),
    }
    if arguments.ppo is not None:
        policies["ppo"] = PPOPolicy.load(arguments.ppo, config, allow_scenario_mismatch=arguments.allow_heldout_scenario)
    if arguments.dqn is not None:
        policies["dqn"] = DQNPolicy.load(arguments.dqn, config, allow_scenario_mismatch=arguments.allow_heldout_scenario)
    if arguments.recurrent is not None:
        policies["recurrent_ppo"] = load_recurrent_policy(
            arguments.recurrent,
            config,
            allow_scenario_mismatch=arguments.allow_heldout_scenario,
        )
    arguments.output.mkdir(parents=True, exist_ok=True)
    run_paired_experiment(config, policies, arguments.seeds).write(arguments.output)
    (arguments.output / "comparison_manifest.json").write_text(
        json.dumps(
            {
                "scenario_fingerprint": scenario_fingerprint(config),
                "seeds": arguments.seeds,
                "artifacts": {
                    "ppo": None if arguments.ppo is None else str(arguments.ppo),
                    "dqn": None if arguments.dqn is None else str(arguments.dqn),
                    "recurrent": None if arguments.recurrent is None else str(arguments.recurrent),
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote matched algorithm comparison to {arguments.output}")


if __name__ == "__main__":
    main()
