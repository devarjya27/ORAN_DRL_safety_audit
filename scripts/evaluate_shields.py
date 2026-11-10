#!/usr/bin/env python3
"""Compare an actor policy with heuristic and calibrated learned shields."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from modelling.action_catalog import ActionCatalog
from modelling.config import load_environment_config
from modelling.experiments import run_paired_experiment
from modelling.policies import baseline_policy
from modelling.safety.assessment import assess_shield
from modelling.safety.learned_risk import LearnedRiskShield, load_risk_ensemble
from modelling.safety.shield import ShieldConfig, SafetyShield


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--risk-artifact", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", default="demand_proportional")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument(
        "--allow-heldout-scenario",
        action="store_true",
        help="permit a risk model trained on another scenario only when the action catalogue still matches",
    )
    arguments = parser.parse_args()

    config = load_environment_config(arguments.config)
    catalog = ActionCatalog.build(config)
    policy = baseline_policy(arguments.policy, catalog)
    calibration = json.loads(arguments.calibration.read_text(encoding="utf-8"))["calibration"]
    shield_config = replace(ShieldConfig(), maximum_predicted_urllc_risk=float(calibration["threshold"]))
    learned = LearnedRiskShield(
        catalog,
        config,
        load_risk_ensemble(
            arguments.risk_artifact,
            catalog,
            config,
            allow_scenario_mismatch=arguments.allow_heldout_scenario,
        ),
        shield_config,
    )
    heuristic = SafetyShield(catalog, config, shield_config)
    arguments.output.mkdir(parents=True, exist_ok=True)
    run_paired_experiment(config, {arguments.policy: policy}, arguments.seeds).write(arguments.output / "unshielded")
    assess_shield(config, policy, heuristic, seeds=arguments.seeds).write_json(arguments.output / "heuristic_shield.json")
    assess_shield(config, policy, learned, seeds=arguments.seeds).write_json(arguments.output / "learned_shield.json")
    print(f"Wrote paired unshielded and shield assessments to {arguments.output}")


if __name__ == "__main__":
    main()
