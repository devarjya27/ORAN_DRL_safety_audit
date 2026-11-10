#!/usr/bin/env python3
"""Run a resumable PPO-only action-representation diagnostic."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from modelling.config import load_environment_config, scenario_fingerprint


ROOT = Path(__file__).resolve().parents[1]
PPO_HYPERPARAMETERS = {
    "n_steps": 256,
    "batch_size": 256,
    "epochs": 10,
    "learning_rate": 3e-4,
    "clip_range": 0.2,
    "entropy_coefficient": 0.001,
    "target_kl": 0.03,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("study/ppo-burst-diagnostic"))
    parser.add_argument("--config", type=Path, default=Path("configs/research_dynamic.yaml"))
    parser.add_argument("--training-seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--validation-episodes", type=int, default=20)
    parser.add_argument("--eval-freq", type=int, default=20_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--action-mode", choices=("flat", "factorized"), default="factorized")
    parser.add_argument("--urllc-cost-limit", type=float, default=0.01)
    parser.add_argument("--lagrangian-learning-rate", type=float, default=0.1)
    parser.add_argument("--initial-lagrange-multiplier", type=float, default=1.0)
    parser.add_argument("--max-regime-urllc-cost-limit", type=float, default=0.01)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    if arguments.timesteps <= 0 or arguments.validation_episodes <= 0 or arguments.eval_freq <= 0:
        raise ValueError("timesteps, validation episodes, and evaluation frequency must be positive")
    if (
        not 0.0 <= arguments.urllc_cost_limit <= 1.0
        or arguments.lagrangian_learning_rate <= 0.0
        or arguments.initial_lagrange_multiplier < 0.0
        or not 0.0 <= arguments.max_regime_urllc_cost_limit <= 1.0
    ):
        raise ValueError("invalid URLLC constraint or Lagrangian multiplier parameters")
    if len(set(arguments.training_seeds)) != len(arguments.training_seeds):
        raise ValueError("training seeds must be unique")

    output = arguments.output if arguments.output.is_absolute() else ROOT / arguments.output
    config_path = arguments.config if arguments.config.is_absolute() else ROOT / arguments.config
    config = load_environment_config(config_path)
    manifest = {
        "diagnostic_schema_version": 4,
        "action_mode": arguments.action_mode,
        "scenario_config": str(config_path.relative_to(ROOT)),
        "scenario_fingerprint": scenario_fingerprint(config),
        "training_seeds": arguments.training_seeds,
        "timesteps": arguments.timesteps,
        "validation_episodes": arguments.validation_episodes,
        "eval_freq": arguments.eval_freq,
        "n_envs": arguments.n_envs,
        "urllc_cost_limit": arguments.urllc_cost_limit,
        "lagrangian_learning_rate": arguments.lagrangian_learning_rate,
        "initial_lagrange_multiplier": arguments.initial_lagrange_multiplier,
        "max_regime_urllc_cost_limit": arguments.max_regime_urllc_cost_limit,
        "ppo_hyperparameters": PPO_HYPERPARAMETERS,
    }
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "study_manifest.json"
    if manifest_path.is_file() and not arguments.force:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise RuntimeError(f"{manifest_path} belongs to a different diagnostic; choose a new output directory")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(ROOT) if not existing_pythonpath else f"{ROOT}:{existing_pythonpath}"
    validation_seeds = [str(seed) for seed in range(1_001, 1_001 + arguments.validation_episodes)]
    for seed in arguments.training_seeds:
        artifact = output / "artifacts" / f"seed-{seed}" / "ppo"
        expected = artifact.with_suffix(".metadata.json")
        if expected.is_file() and artifact.with_suffix(".zip").is_file() and not arguments.force:
            print(f"[resume] {artifact.with_suffix('.zip')}", flush=True)
            continue
        command = [
            sys.executable,
            "scripts/train_ppo.py",
            "--config",
            str(config_path),
            "--output",
            str(artifact),
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
            "--lagrangian-learning-rate",
            str(arguments.lagrangian_learning_rate),
            "--initial-lagrange-multiplier",
            str(arguments.initial_lagrange_multiplier),
            "--max-regime-urllc-cost-limit",
            str(arguments.max_regime_urllc_cost_limit),
            "--eval-freq",
            str(arguments.eval_freq),
            "--validation-seeds",
            *validation_seeds,
        ]
        if arguments.action_mode == "flat":
            command.append("--flat-actions")
        if arguments.no_progress:
            command.append("--no-progress")
        print(f"[run] {shlex.join(command)}", flush=True)
        subprocess.run(command, cwd=ROOT, env=environment, check=True)
    print(f"PPO diagnostic artifacts are under {output}")


if __name__ == "__main__":
    main()
