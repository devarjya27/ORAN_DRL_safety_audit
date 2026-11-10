"""Reproducible discrete off-policy DQN baseline and inference artifact."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

from modelling.action_catalog import ActionCatalog
from modelling.config import scenario_fingerprint
from modelling.contracts import Action, EnvironmentConfig, Observation
from modelling.envs.gym_adapter import SafeSliceGymEnv
from modelling.features import FEATURE_SCHEMA_VERSION, FeatureEncoder
from modelling.training.callbacks import VALIDATION_FIELDS, build_episode_reward_callback, build_validation_callback


@dataclass(frozen=True, slots=True)
class DQNTrainingConfig:
    total_timesteps: int
    seed: int = 0
    learning_rate: float = 5e-5
    buffer_size: int = 1_000_000
    learning_starts: int = 50_000
    batch_size: int = 256
    train_frequency: int = 4
    gradient_steps: int = 1
    gamma: float = 0.99
    tau: float = 1.0
    target_update_interval: int = 20_000
    exploration_fraction: float = 0.50
    exploration_final_epsilon: float = 0.10
    hidden_sizes: tuple[int, ...] = (128, 128)
    max_gradient_norm: float = 10.0
    checkpoint_urllc_cost_limit: float | None = 0.01
    checkpoint_max_regime_urllc_cost_limit: float | None = None
    eval_freq: int = 10_000
    validation_seeds: tuple[int, ...] = (101, 102, 103, 104, 105)
    device: str = "cpu"
    progress_bar: bool = True

    def __post_init__(self) -> None:
        if min(self.total_timesteps, self.buffer_size, self.batch_size, self.train_frequency, self.gradient_steps) <= 0:
            raise ValueError("DQN timestep, replay, batch, and update settings must be positive")
        if self.learning_starts < 0 or self.target_update_interval <= 0 or self.eval_freq <= 0:
            raise ValueError("invalid DQN warm-up or evaluation cadence")
        if not 0.0 < self.learning_rate or not 0.0 < self.gamma <= 1.0 or not 0.0 < self.tau <= 1.0:
            raise ValueError("invalid DQN learning rate, gamma, or tau")
        if not 0.0 <= self.exploration_final_epsilon <= 1.0 or not 0.0 < self.exploration_fraction <= 1.0:
            raise ValueError("invalid DQN exploration schedule")
        if not self.hidden_sizes or min(self.hidden_sizes) <= 0 or not self.validation_seeds:
            raise ValueError("DQN hidden sizes and validation seeds must be configured")
        if self.checkpoint_urllc_cost_limit is not None and not 0.0 <= self.checkpoint_urllc_cost_limit <= 1.0:
            raise ValueError("DQN checkpoint URLLC limit must be in [0, 1]")
        if (
            self.checkpoint_max_regime_urllc_cost_limit is not None
            and not 0.0 <= self.checkpoint_max_regime_urllc_cost_limit <= 1.0
        ):
            raise ValueError("DQN checkpoint worst-regime URLLC limit must be in [0, 1]")


def train_dqn(
    *,
    training: DQNTrainingConfig,
    output_path: Path,
    config: EnvironmentConfig | None = None,
) -> None:
    try:
        from stable_baselines3 import DQN
    except ImportError as exc:
        raise RuntimeError("Stable-Baselines3 is required for DQN training") from exc
    environment_config = config or EnvironmentConfig()
    catalog = ActionCatalog.build(environment_config)
    encoder = FeatureEncoder(environment_config, catalog)

    def make_environment() -> SafeSliceGymEnv:
        return SafeSliceGymEnv(environment_config, feature_encoder=encoder, factorized_actions=False)

    environment = make_environment()
    model_path = output_path if output_path.suffix == ".zip" else output_path.with_suffix(".zip")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    best_model_path = model_path.with_name(f"{model_path.stem}.best.zip")
    final_model_path = model_path.with_name(f"{model_path.stem}.final.zip")
    model = DQN(
        "MlpPolicy",
        environment,
        seed=training.seed,
        learning_rate=training.learning_rate,
        buffer_size=training.buffer_size,
        learning_starts=training.learning_starts,
        batch_size=training.batch_size,
        train_freq=training.train_frequency,
        gradient_steps=training.gradient_steps,
        gamma=training.gamma,
        tau=training.tau,
        target_update_interval=training.target_update_interval,
        exploration_fraction=training.exploration_fraction,
        exploration_final_eps=training.exploration_final_epsilon,
        max_grad_norm=training.max_gradient_norm,
        policy_kwargs={"net_arch": list(training.hidden_sizes)},
        device=training.device,
        verbose=0,
    )
    episode_callback = build_episode_reward_callback()
    validation_callback = build_validation_callback(
        make_environment,
        training.validation_seeds,
        training.eval_freq,
        best_model_path,
        maximum_mean_urllc_cost=training.checkpoint_urllc_cost_limit,
        maximum_regime_urllc_cost=training.checkpoint_max_regime_urllc_cost_limit,
    )
    model.learn(total_timesteps=training.total_timesteps, callback=[episode_callback, validation_callback], progress_bar=training.progress_bar)
    model.save(final_model_path)
    if best_model_path.is_file():
        DQN.load(best_model_path, device=training.device).save(model_path)
    else:
        model.save(model_path)
    metadata = {
        "schema_version": 2,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "scenario_fingerprint": scenario_fingerprint(environment_config),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "observation_size": encoder.size,
        "action_count": len(catalog),
        "best_validation_reward": validation_callback.best_mean_reward if validation_callback.records else 0.0,
        "best_validation_urllc_cost": validation_callback.best_mean_urllc_cost if validation_callback.records else None,
        "best_validation_max_regime_urllc_cost": (
            validation_callback.best_max_regime_urllc_cost if validation_callback.records else None
        ),
        "best_validation_feasible": validation_callback.best_feasible if validation_callback.records else False,
        "best_validation_timestep": validation_callback.best_timestep,
        "training": asdict(training),
    }
    model_path.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(
        model_path.with_suffix(".training.csv"),
        ("episode", "total_reward"),
        ({"episode": index, "total_reward": reward} for index, reward in enumerate(episode_callback.episode_rewards, 1)),
    )
    _write_csv(
        model_path.with_suffix(".validation.csv"),
        VALIDATION_FIELDS,
        validation_callback.records,
    )
    environment.close()


class DQNPolicy:
    def __init__(self, model: Any, catalog: ActionCatalog, encoder: FeatureEncoder) -> None:
        self.model = model
        self.catalog = catalog
        self.encoder = encoder

    @classmethod
    def load(cls, path: Path, config: EnvironmentConfig, *, allow_scenario_mismatch: bool = False) -> "DQNPolicy":
        from stable_baselines3 import DQN

        model_path = path if path.suffix == ".zip" else path.with_suffix(".zip")
        metadata = json.loads(model_path.with_suffix(".metadata.json").read_text(encoding="utf-8"))
        if metadata.get("schema_version") != 2:
            raise ValueError("unsupported DQN artifact schema")
        catalog = ActionCatalog.build(config)
        encoder = FeatureEncoder(config, catalog)
        if metadata["scenario_fingerprint"] != scenario_fingerprint(config) and not allow_scenario_mismatch:
            raise ValueError("DQN artifact scenario fingerprint does not match the requested environment")
        if metadata["feature_schema_version"] != FEATURE_SCHEMA_VERSION or metadata["observation_size"] != encoder.size:
            raise ValueError("DQN artifact feature schema mismatch")
        return cls(DQN.load(model_path, device="cpu"), catalog, encoder)

    def act(self, observation: Observation) -> Action:
        action, _ = self.model.predict(self.encoder.encode(observation), deterministic=True)
        return self.catalog.get(int(action))


def _write_csv(path: Path, fields: tuple[str, ...], rows: Any) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
