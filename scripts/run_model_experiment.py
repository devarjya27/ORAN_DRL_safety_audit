#!/usr/bin/env python3
"""Run model/baseline comparisons and write reproducible paired result artifacts."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Protocol

from modelling.action_catalog import ActionCatalog
from modelling.config import load_environment_config, scenario_fingerprint
from modelling.contracts import Action, Observation
from modelling.evaluation import Policy
from modelling.experiments import run_paired_experiment
from modelling.policies import baseline_policy
from modelling.safety.assessment import assess_shield
from modelling.safety.learned_risk import LearnedRiskShield, load_risk_ensemble
from modelling.safety.shield import SafetyShield, ShieldConfig
from modelling.training.train_ppo import PPOPolicy
from modelling.training.train_recurrent_ppo import load_recurrent_policy
from modelling.training.train_dqn import DQNPolicy


class _ShieldedPolicy:
    def __init__(self, policy: Policy, shield: object) -> None:
        self.policy = policy
        self.shield = shield

    def act(self, observation: Observation) -> Action:
        return self.shield.apply(observation, self.policy.act(observation)).action  # type: ignore[union-attr]


def _load_policy(kind: str, artifact: Path | None, catalog: ActionCatalog, config, allow_scenario_mismatch: bool) -> Policy:
    if kind == "baseline":
        return baseline_policy("demand_proportional", catalog)
    if artifact is None:
        raise ValueError(f"--artifact is required for --policy-kind {kind}")
    if kind == "ppo":
        return PPOPolicy.load(artifact, config, allow_scenario_mismatch=allow_scenario_mismatch)
    if kind == "recurrent":
        return load_recurrent_policy(artifact, config, allow_scenario_mismatch=allow_scenario_mismatch)
    if kind == "dqn":
        return DQNPolicy.load(artifact, config, allow_scenario_mismatch=allow_scenario_mismatch)
    raise ValueError(f"unsupported policy kind: {kind}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--policy-kind", choices=("baseline", "ppo", "recurrent", "dqn"), default="baseline")
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--baseline-name", default="demand_proportional")
    parser.add_argument("--risk-artifact", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--allow-heldout-scenario", action="store_true")
    parser.add_argument("--unsafe-cost-threshold", type=float, default=0.01)
    arguments = parser.parse_args()
    config = load_environment_config(arguments.config)
    catalog = ActionCatalog.build(config)
    actor: Policy = (
        baseline_policy(arguments.baseline_name, catalog)
        if arguments.policy_kind == "baseline"
        else _load_policy(arguments.policy_kind, arguments.artifact, catalog, config, arguments.allow_heldout_scenario)
    )
    policies: dict[str, Policy] = {arguments.policy_kind: actor}
    heuristic = SafetyShield(catalog, config)
    policies["heuristic_shield"] = _ShieldedPolicy(actor, heuristic)
    learned = None
    if arguments.risk_artifact is not None or arguments.calibration is not None:
        if arguments.risk_artifact is None or arguments.calibration is None:
            raise ValueError("--risk-artifact and --calibration must be supplied together")
        calibration_document = json.loads(arguments.calibration.read_text(encoding="utf-8"))
        calibration = calibration_document["calibration"]
        unsafe_cost_threshold = float(
            calibration_document.get("unsafe_cost_threshold", arguments.unsafe_cost_threshold)
        )
        shield_config = replace(ShieldConfig(), maximum_predicted_urllc_risk=float(calibration["threshold"]))
        learned = LearnedRiskShield(
            catalog,
            config,
            load_risk_ensemble(arguments.risk_artifact, catalog, config, allow_scenario_mismatch=arguments.allow_heldout_scenario),
            shield_config,
        )
        policies["learned_shield"] = _ShieldedPolicy(actor, learned)
    else:
        unsafe_cost_threshold = arguments.unsafe_cost_threshold
    arguments.output.mkdir(parents=True, exist_ok=True)
    report = run_paired_experiment(config, policies, arguments.seeds)
    report.write(arguments.output / "paired")
    assess_shield(
        config, actor, heuristic, seeds=arguments.seeds, unsafe_cost_threshold=unsafe_cost_threshold
    ).write_json(arguments.output / "heuristic_assessment.json")
    if learned is not None:
        assess_shield(
            config, actor, learned, seeds=arguments.seeds, unsafe_cost_threshold=unsafe_cost_threshold
        ).write_json(arguments.output / "learned_assessment.json")
    (arguments.output / "manifest.json").write_text(
        json.dumps(
            {
                "scenario_fingerprint": scenario_fingerprint(config),
                "policy_kind": arguments.policy_kind,
                "policy_artifact": None if arguments.artifact is None else str(arguments.artifact),
                "risk_artifact": None if arguments.risk_artifact is None else str(arguments.risk_artifact),
                "seeds": arguments.seeds,
                "unsafe_cost_threshold": unsafe_cost_threshold,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote paired report and shield assessments to {arguments.output}")


if __name__ == "__main__":
    main()
