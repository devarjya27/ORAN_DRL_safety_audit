#!/usr/bin/env python3
"""Run reproducible paired static-baseline experiments from a YAML scenario."""

from __future__ import annotations

import argparse
from pathlib import Path

from modelling.baselines import (
    ConservativeURLLCPolicy,
    DemandProportionalPolicy,
    EqualStaticPolicy,
    JointValidatedStaticPolicy,
    ValidatedStaticPolicy,
)
from modelling.config import load_environment_config
from modelling.envs.slicing_env import SafeSliceEnv
from modelling.experiments import run_paired_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    arguments = parser.parse_args()

    config = load_environment_config(arguments.config)
    catalog = SafeSliceEnv(config).catalog
    policies = {
        "equal_static": EqualStaticPolicy(catalog),
        "mean_validated_static": ValidatedStaticPolicy(catalog),
        "joint_validated_static": JointValidatedStaticPolicy(catalog),
        "conservative_urllc": ConservativeURLLCPolicy(catalog),
        "demand_proportional": DemandProportionalPolicy(catalog),
    }
    report = run_paired_experiment(
        config,
        policies,
        arguments.seeds,
        bootstrap_samples=arguments.bootstrap_samples,
    )
    report.write(arguments.output)
    print(f"Wrote {arguments.output / 'report.json'} and {arguments.output / 'episodes.csv'}")


if __name__ == "__main__":
    main()
