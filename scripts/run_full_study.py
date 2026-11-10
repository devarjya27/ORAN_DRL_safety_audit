#!/usr/bin/env python3
"""Run the leakage-safe multi-model, multi-seed O-RAN DRL Audit experiment matrix."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PIPELINE_SCHEMA_VERSION = 7
PPO_HYPERPARAMETERS = {
    "n_steps": 256,
    "batch_size": 256,
    "epochs": 10,
    "learning_rate": 3e-4,
    "clip_range": 0.2,
    "entropy_coefficient": 0.001,
    "target_kl": 0.03,
}


@dataclass(frozen=True)
class Scenario:
    name: str
    config: str
    seed_base: int
    training_match: bool = False


SCENARIOS = (
    Scenario("id_population", "configs/research_dynamic.yaml", 10_000, True),
    Scenario("constant_stress", "configs/research_validation_constant.yaml", 20_000),
    Scenario("heldout_sweep", "configs/research_heldout_sweep.yaml", 30_000),
    Scenario("stationary_diagnostic", "configs/default.yaml", 40_000),
)


def _run(command: list[str], *, expected: Path | None, force: bool, environment: dict[str, str]) -> None:
    if expected is not None and expected.is_file() and not force:
        print(f"[resume] {expected}", flush=True)
        return
    print(f"[run] {shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def _seed_range(start: int, count: int) -> list[str]:
    return [str(value) for value in range(start, start + count)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("study/v7"))
    parser.add_argument("--training-seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--evaluation-episodes", type=int, default=100)
    parser.add_argument("--validation-episodes", type=int, default=20)
    parser.add_argument("--risk-episodes", type=int, default=20)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--eval-freq", type=int, default=20_000)
    parser.add_argument("--recurrent-eval-updates", type=int, default=25)
    parser.add_argument("--urllc-cost-limit", type=float, default=0.01)
    parser.add_argument("--ppo-initial-lagrange-multiplier", type=float, default=1.0)
    parser.add_argument("--max-regime-urllc-cost-limit", type=float, default=0.01)
    parser.add_argument("--stages", nargs="+", choices=("train", "shield", "evaluate", "plot"), default=["train", "shield", "evaluate", "plot"])
    parser.add_argument("--no-shield", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--force", action="store_true", help="rerun completed artifacts instead of resuming")
    arguments = parser.parse_args()
    if arguments.timesteps <= 0 or arguments.evaluation_episodes <= 0 or arguments.validation_episodes <= 0:
        raise ValueError("timesteps and episode counts must be positive")
    if (
        not 0.0 <= arguments.urllc_cost_limit <= 1.0
        or arguments.ppo_initial_lagrange_multiplier < 0.0
        or not 0.0 <= arguments.max_regime_urllc_cost_limit <= 1.0
    ):
        raise ValueError("invalid PPO safety limits or initial Lagrange multiplier")
    if len(set(arguments.training_seeds)) != len(arguments.training_seeds):
        raise ValueError("training seeds must be unique")

    output = arguments.output if arguments.output.is_absolute() else ROOT / arguments.output
    artifacts = output / "artifacts"
    results = output / "results"
    plots = output / "plots"
    output.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(ROOT) if not existing_pythonpath else f"{ROOT}:{existing_pythonpath}"
    environment.setdefault("MPLCONFIGDIR", "/tmp/modelling-mpl")
    python = sys.executable
    validation_seeds = _seed_range(1_001, arguments.validation_episodes)
    risk_train_seeds = _seed_range(2_001, arguments.risk_episodes)
    risk_validation_seeds = _seed_range(3_001, arguments.risk_episodes)
    recurrent_updates = math.ceil(arguments.timesteps / (arguments.horizon * arguments.n_envs))

    manifest = {
        "pipeline_schema_version": PIPELINE_SCHEMA_VERSION,
        "training_seeds": arguments.training_seeds,
        "timesteps": arguments.timesteps,
        "evaluation_episodes": arguments.evaluation_episodes,
        "validation_episodes": arguments.validation_episodes,
        "risk_episodes": arguments.risk_episodes,
        "n_envs": arguments.n_envs,
        "horizon": arguments.horizon,
        "recurrent_updates": recurrent_updates,
        "urllc_cost_limit": arguments.urllc_cost_limit,
        "ppo_initial_lagrange_multiplier": arguments.ppo_initial_lagrange_multiplier,
        "max_regime_urllc_cost_limit": arguments.max_regime_urllc_cost_limit,
        "ppo_hyperparameters": PPO_HYPERPARAMETERS,
        "scenarios": [asdict(scenario) for scenario in SCENARIOS],
    }
    manifest_path = output / "study_manifest.json"
    if manifest_path.is_file() and not arguments.force:
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest != manifest:
            raise RuntimeError(
                f"{manifest_path} belongs to a different pipeline or experiment matrix; "
                "use a new output directory, or pass --force to replace its artifacts"
            )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if not arguments.skip_tests and "train" in arguments.stages:
        _run([python, "-m", "pytest", "-q"], expected=None, force=True, environment=environment)

    if "train" in arguments.stages:
        for seed in arguments.training_seeds:
            seed_dir = artifacts / f"seed-{seed}"
            _run(
                [
                    python,
                    "scripts/train_ppo.py",
                    "--config",
                    "configs/research_dynamic.yaml",
                    "--output",
                    str(seed_dir / "ppo"),
                    "--timesteps",
                    str(arguments.timesteps),
                    "--seed",
                    str(seed),
                    "--n-envs",
                    str(arguments.n_envs),
                    "--device",
                    "cpu",
                    "--n-steps",
                    str(PPO_HYPERPARAMETERS["n_steps"]),
                    "--batch-size",
                    str(PPO_HYPERPARAMETERS["batch_size"]),
                    "--epochs",
                    str(PPO_HYPERPARAMETERS["epochs"]),
                    "--learning-rate",
                    str(PPO_HYPERPARAMETERS["learning_rate"]),
                    "--clip-range",
                    str(PPO_HYPERPARAMETERS["clip_range"]),
                    "--entropy-coefficient",
                    str(PPO_HYPERPARAMETERS["entropy_coefficient"]),
                    "--target-kl",
                    str(PPO_HYPERPARAMETERS["target_kl"]),
                    "--urllc-cost-limit",
                    str(arguments.urllc_cost_limit),
                    "--initial-lagrange-multiplier",
                    str(arguments.ppo_initial_lagrange_multiplier),
                    "--max-regime-urllc-cost-limit",
                    str(arguments.max_regime_urllc_cost_limit),
                    "--eval-freq",
                    str(arguments.eval_freq),
                    "--validation-seeds",
                    *validation_seeds,
                ],
                expected=(seed_dir / "ppo.zip"),
                force=arguments.force,
                environment=environment,
            )
            _run(
                [
                    python,
                    "scripts/train_dqn.py",
                    "--config",
                    "configs/research_dynamic.yaml",
                    "--output",
                    str(seed_dir / "dqn"),
                    "--timesteps",
                    str(arguments.timesteps),
                    "--seed",
                    str(seed),
                    "--learning-starts",
                    "50000",
                    "--eval-freq",
                    str(arguments.eval_freq),
                    "--max-regime-urllc-cost-limit",
                    str(arguments.max_regime_urllc_cost_limit),
                    "--validation-seeds",
                    *validation_seeds,
                ],
                expected=(seed_dir / "dqn.zip"),
                force=arguments.force,
                environment=environment,
            )
            _run(
                [
                    python,
                    "scripts/train_recurrent_ppo.py",
                    "--config",
                    "configs/research_dynamic.yaml",
                    "--output",
                    str(seed_dir / "recurrent"),
                    "--updates",
                    str(recurrent_updates),
                    "--horizon",
                    str(arguments.horizon),
                    "--seed",
                    str(seed),
                    "--hidden-size",
                    "128",
                    "--n-envs",
                    str(arguments.n_envs),
                    "--sequence-length",
                    "16",
                    "--sequences-per-batch",
                    "8",
                    "--epochs",
                    "4",
                    "--urllc-cost-limit",
                    str(arguments.urllc_cost_limit),
                    "--max-regime-urllc-cost-limit",
                    str(arguments.max_regime_urllc_cost_limit),
                    "--eval-every-updates",
                    str(arguments.recurrent_eval_updates),
                    "--validation-seeds",
                    *validation_seeds,
                ],
                expected=(seed_dir / "recurrent.pt"),
                force=arguments.force,
                environment=environment,
            )

    if "shield" in arguments.stages and not arguments.no_shield:
        for seed in arguments.training_seeds:
            seed_dir = artifacts / f"seed-{seed}"
            risk_dir = seed_dir / "risk"
            _run(
                [
                    python,
                    "scripts/train_risk_shield.py",
                    "--config",
                    "configs/research_dynamic.yaml",
                    "--validation-config",
                    "configs/research_validation_constant.yaml",
                    "--policy-kind",
                    "recurrent",
                    "--artifact",
                    str(seed_dir / "recurrent.pt"),
                    "--output",
                    str(risk_dir),
                    "--train-seeds",
                    *risk_train_seeds,
                    "--validation-seeds",
                    *risk_validation_seeds,
                    "--target-unsafe-rate",
                    "0.01",
                    "--exploration-probability",
                    "0.25",
                    "--seed",
                    str(seed),
                ],
                expected=(risk_dir / "calibration.json"),
                force=arguments.force,
                environment=environment,
            )

    if "evaluate" in arguments.stages:
        for scenario in SCENARIOS:
            evaluation_seeds = _seed_range(scenario.seed_base, arguments.evaluation_episodes)
            for seed in arguments.training_seeds:
                seed_dir = artifacts / f"seed-{seed}"
                algorithm_dir = results / scenario.name / f"seed-{seed}" / "algorithms"
                comparison = [
                    python,
                    "scripts/run_algorithm_comparison.py",
                    "--config",
                    scenario.config,
                    "--output",
                    str(algorithm_dir),
                    "--seeds",
                    *evaluation_seeds,
                    "--ppo",
                    str(seed_dir / "ppo.zip"),
                    "--dqn",
                    str(seed_dir / "dqn.zip"),
                    "--recurrent",
                    str(seed_dir / "recurrent.pt"),
                ]
                if not scenario.training_match:
                    comparison.append("--allow-heldout-scenario")
                _run(
                    comparison,
                expected=(algorithm_dir / "comparison_manifest.json"),
                    force=arguments.force,
                    environment=environment,
                )
                if not arguments.no_shield:
                    shield_dir = results / scenario.name / f"seed-{seed}" / "shielded"
                    shield_command = [
                        python,
                        "scripts/run_model_experiment.py",
                        "--config",
                        scenario.config,
                        "--output",
                        str(shield_dir),
                        "--seeds",
                        *evaluation_seeds,
                        "--policy-kind",
                        "recurrent",
                        "--artifact",
                        str(seed_dir / "recurrent.pt"),
                        "--risk-artifact",
                        str(seed_dir / "risk/risk_ensemble.pt"),
                        "--calibration",
                        str(seed_dir / "risk/calibration.json"),
                    ]
                    if not scenario.training_match:
                        shield_command.append("--allow-heldout-scenario")
                    _run(
                        shield_command,
                        expected=(shield_dir / "manifest.json"),
                        force=arguments.force,
                        environment=environment,
                    )

    if "plot" in arguments.stages:
        ppo_validation = [artifacts / f"seed-{seed}" / "ppo.validation.csv" for seed in arguments.training_seeds]
        dqn_validation = [artifacts / f"seed-{seed}" / "dqn.validation.csv" for seed in arguments.training_seeds]
        recurrent_validation = [artifacts / f"seed-{seed}" / "recurrent.validation.csv" for seed in arguments.training_seeds]
        _run(
            [
                python,
                "scripts/plot_multi_seed_validation.py",
                "--ppo",
                *map(str, ppo_validation),
                "--dqn",
                *map(str, dqn_validation),
                "--recurrent",
                *map(str, recurrent_validation),
                "--output",
                str(plots / "multi_seed_validation.png"),
            ],
            expected=None,
            force=True,
            environment=environment,
        )
        for scenario in SCENARIOS:
            algorithm_reports = [
                results / scenario.name / f"seed-{seed}" / "algorithms/report.json"
                for seed in arguments.training_seeds
            ]
            command = [
                python,
                "scripts/plot_study_summary.py",
                "--algorithm-reports",
                *map(str, algorithm_reports),
            ]
            if not arguments.no_shield:
                shield_reports = [
                    results / scenario.name / f"seed-{seed}" / "shielded/paired/report.json"
                    for seed in arguments.training_seeds
                ]
                command.extend(["--shield-reports", *map(str, shield_reports)])
            command.extend(
                [
                    "--output",
                    str(plots / scenario.name),
                    "--title",
                    f"O-RAN DRL Audit: {scenario.name.replace('_', ' ')}",
                ]
            )
            _run(command, expected=None, force=True, environment=environment)
    print(f"Full study artifacts are under {output}")


if __name__ == "__main__":
    main()
