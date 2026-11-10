"""Reproducible feed-forward PPO baseline training and inference artifacts.

This module is intentionally the *unconstrained* baseline. It is the
reference point for the later recurrent PPO-Lagrangian controller, not the
paper's final safe policy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import csv
from typing import Any

import numpy as np

from modelling.action_catalog import ActionCatalog
from modelling.config import scenario_fingerprint
from modelling.contracts import Action, EnvironmentConfig, Observation
from modelling.envs.gym_adapter import SafeSliceGymEnv
from modelling.safety.shield import SafetyShield
from modelling.training.lagrangian import (
    LagrangianConfig,
    LagrangianController,
    URLLCCostPenaltyWrapper,
    build_lagrangian_callback,
)
from modelling.training.callbacks import VALIDATION_FIELDS, build_episode_reward_callback, build_validation_callback
from modelling.features import FEATURE_SCHEMA_VERSION, FactorizedActionCodec, FeatureEncoder


ARTIFACT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class PPOTrainingConfig:
    """Small, explicit PPO hyperparameter surface for the baseline."""

    total_timesteps: int
    seed: int = 0
    learning_rate: float = 3e-4
    clip_range: float = 0.2
    entropy_coefficient: float = 0.001
    target_kl: float | None = 0.03
    n_steps: int = 256
    batch_size: int = 256
    n_envs: int = 1
    n_epochs: int = 10
    # SB3's MLP PPO is generally faster on CPU; GPU remains an explicit option.
    device: str = "cpu"
    use_safety_shield: bool = False
    constraint: LagrangianConfig | None = None
    progress_bar: bool = True
    factorized_actions: bool = True
    eval_freq: int = 10_000
    validation_seeds: tuple[int, ...] = (101, 102, 103, 104, 105)
    max_regime_urllc_cost_limit: float | None = None

    def __post_init__(self) -> None:
        if self.total_timesteps <= 0 or self.n_steps <= 0 or self.batch_size <= 0 or self.n_epochs <= 0 or self.n_envs <= 0 or self.eval_freq <= 0:
            raise ValueError("PPO timestep, rollout, batch, environment, and epoch counts must be positive")
        if self.learning_rate <= 0.0 or not 0.0 < self.clip_range < 1.0 or self.entropy_coefficient < 0.0:
            raise ValueError("invalid PPO learning rate, clip range, or entropy coefficient")
        if not self.validation_seeds:
            raise ValueError("validation_seeds cannot be empty")
        if self.max_regime_urllc_cost_limit is not None and not 0.0 <= self.max_regime_urllc_cost_limit <= 1.0:
            raise ValueError("max_regime_urllc_cost_limit must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class PPOArtifactMetadata:
    """Compatibility data needed before an artifact is trusted for inference."""

    schema_version: int
    created_at_utc: str
    scenario_fingerprint: str
    action_count: int
    observation_size: int
    training: PPOTrainingConfig
    stable_baselines3_version: str
    torch_version: str
    final_lagrange_multiplier: float | None
    feature_schema_version: int
    factorized_actions: bool
    best_validation_reward: float
    best_validation_timestep: int
    best_validation_urllc_cost: float = float("nan")
    best_validation_max_regime_urllc_cost: float = float("nan")
    best_validation_feasible: bool = False


def train_baseline_ppo(
    *,
    training: PPOTrainingConfig,
    output_path: Path,
    config: EnvironmentConfig | None = None,
    episode_metrics_path: Path | None = None,
    validation_metrics_path: Path | None = None,
) -> PPOArtifactMetadata:
    """Train PPO and persist model-adjacent compatibility metadata."""

    PPO, sb3_version, torch_version = _load_training_dependencies()
    from stable_baselines3.common.env_util import make_vec_env

    environment_config = config or EnvironmentConfig()
    controller = LagrangianController(training.constraint) if training.constraint is not None else None

    def make_base_env() -> SafeSliceGymEnv:
        catalog = ActionCatalog.build(environment_config)
        shield = SafetyShield(catalog, environment_config) if training.use_safety_shield else None
        return SafeSliceGymEnv(
            environment_config,
            shield=shield,
            feature_encoder=FeatureEncoder(environment_config, catalog),
            factorized_actions=training.factorized_actions,
        )

    def make_env() -> SafeSliceGymEnv:
        env = make_base_env()
        if controller is not None:
            env = URLLCCostPenaltyWrapper(env, controller)
        return env

    environment = make_vec_env(make_env, n_envs=training.n_envs, seed=training.seed)

    if training.batch_size > (training.n_steps * training.n_envs):
        raise ValueError("batch_size cannot exceed n_steps * n_envs for the baseline PPO runner")
    model = PPO(
        "MlpPolicy",
        environment,
        seed=training.seed,
        learning_rate=training.learning_rate,
        clip_range=training.clip_range,
        ent_coef=training.entropy_coefficient,
        target_kl=training.target_kl,
        n_steps=training.n_steps,
        batch_size=training.batch_size,
        n_epochs=training.n_epochs,
        verbose=0,
        device=training.device,
    )
    model_path = _model_path(output_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    best_model_path = model_path.with_name(f"{model_path.stem}.best.zip")
    final_model_path = model_path.with_name(f"{model_path.stem}.final.zip")
    episode_callback = build_episode_reward_callback()
    callbacks = [episode_callback]
    validation_callback = build_validation_callback(
        make_base_env,
        training.validation_seeds,
        training.eval_freq,
        best_model_path,
        maximum_mean_urllc_cost=None if training.constraint is None else training.constraint.urllc_cost_limit,
        maximum_regime_urllc_cost=training.max_regime_urllc_cost_limit,
        lagrange_multiplier_getter=None if controller is None else lambda: controller.multiplier,
    )
    callbacks.append(validation_callback)
    if controller is not None:
        callbacks.append(build_lagrangian_callback(controller))
    callback: object = callbacks[0] if len(callbacks) == 1 else callbacks
    model.learn(total_timesteps=training.total_timesteps, progress_bar=training.progress_bar, callback=callback)
    # SB3 runs step callbacks while collecting a rollout, before optimizing on
    # that rollout. Evaluate once more after learn() so the last CSV row and
    # checkpoint selection include the parameters persisted as *.final.zip.
    validation_callback.evaluate_current_policy(post_training=True)
    model.save(final_model_path)
    if best_model_path.is_file():
        PPO.load(best_model_path, device=training.device).save(model_path)
    else:
        model.save(model_path)
    metadata = PPOArtifactMetadata(
        schema_version=ARTIFACT_SCHEMA_VERSION,
        created_at_utc=datetime.now(UTC).isoformat(),
        scenario_fingerprint=scenario_fingerprint(environment_config),
        action_count=len(ActionCatalog.build(environment_config)),
        observation_size=int(environment.observation_space.shape[0]),
        training=training,
        stable_baselines3_version=sb3_version,
        torch_version=torch_version,
        final_lagrange_multiplier=None if controller is None else controller.multiplier,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        factorized_actions=training.factorized_actions,
        best_validation_reward=(validation_callback.best_mean_reward if validation_callback.records else 0.0),
        best_validation_timestep=validation_callback.best_timestep,
        best_validation_urllc_cost=(
            validation_callback.best_mean_urllc_cost if validation_callback.records else float("nan")
        ),
        best_validation_max_regime_urllc_cost=(
            validation_callback.best_max_regime_urllc_cost if validation_callback.records else float("nan")
        ),
        best_validation_feasible=validation_callback.best_feasible,
    )
    _metadata_path(model_path).write_text(json.dumps(_metadata_json(metadata), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metrics_path = episode_metrics_path or model_path.with_suffix(".training.csv")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("episode", "total_reward"))
        writer.writeheader()
        writer.writerows(
            {"episode": episode, "total_reward": reward}
            for episode, reward in enumerate(episode_callback.episode_rewards, start=1)
        )
    validation_path = validation_metrics_path or model_path.with_suffix(".validation.csv")
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    with validation_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=VALIDATION_FIELDS)
        writer.writeheader()
        writer.writerows(validation_callback.records)
    environment.close()
    return metadata


class PPOPolicy:
    """Catalogue policy wrapper that refuses incompatible PPO artifacts."""

    def __init__(
        self,
        model: Any,
        catalog: ActionCatalog,
        feature_encoder: FeatureEncoder | None,
        action_codec: FactorizedActionCodec | None,
    ) -> None:
        self._model = model
        self._catalog = catalog
        self._feature_encoder = feature_encoder
        self._action_codec = action_codec

    @classmethod
    def load(cls, model_path: Path, config: EnvironmentConfig, *, allow_scenario_mismatch: bool = False) -> "PPOPolicy":
        PPO, _, _ = _load_training_dependencies()
        resolved_model_path = _model_path(model_path)
        metadata = _read_metadata(_metadata_path(resolved_model_path))
        catalog = ActionCatalog.build(config)
        feature_encoder = FeatureEncoder(config, catalog) if metadata.feature_schema_version == FEATURE_SCHEMA_VERSION else None
        expected_observation_size = feature_encoder.size if feature_encoder is not None else 17
        if metadata.scenario_fingerprint != scenario_fingerprint(config) and not allow_scenario_mismatch:
            raise ValueError("PPO artifact scenario fingerprint does not match the requested environment")
        if metadata.action_count != len(catalog) or metadata.observation_size != expected_observation_size:
            raise ValueError("PPO artifact action or observation contract does not match the requested environment")
        codec = FactorizedActionCodec(catalog) if metadata.factorized_actions else None
        return cls(PPO.load(resolved_model_path, device="auto"), catalog, feature_encoder, codec)

    def act(self, observation: Observation) -> Action:
        vector = self._feature_encoder.encode(observation) if self._feature_encoder is not None else np.asarray(observation.vector(), dtype=np.float32)
        action, _ = self._model.predict(vector, deterministic=True)
        action_id = self._action_codec.decode(action) if self._action_codec is not None else int(action)
        return self._catalog.get(action_id)


def _load_training_dependencies() -> tuple[Any, str, str]:
    try:
        import stable_baselines3
        import torch
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise RuntimeError(
            "Stable-Baselines3 is not installed in the selected Python 3.13 environment. "
            "Install it without dependencies to reuse the existing PyTorch build."
        ) from exc
    return PPO, stable_baselines3.__version__, torch.__version__


def _model_path(path: Path) -> Path:
    return path if path.suffix == ".zip" else path.with_suffix(".zip")


def _metadata_path(model_path: Path) -> Path:
    return model_path.with_suffix(".metadata.json")


def _metadata_json(metadata: PPOArtifactMetadata) -> dict[str, object]:
    payload = asdict(metadata)
    payload["training"] = asdict(metadata.training)
    if metadata.training.constraint is not None:
        payload["training"]["constraint"] = asdict(metadata.training.constraint)
    return payload


def _read_metadata(path: Path) -> PPOArtifactMetadata:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["schema_version"] != ARTIFACT_SCHEMA_VERSION:
            raise ValueError("unsupported PPO artifact schema")
        return PPOArtifactMetadata(
            schema_version=int(payload["schema_version"]),
            created_at_utc=str(payload["created_at_utc"]),
            scenario_fingerprint=str(payload["scenario_fingerprint"]),
            action_count=int(payload["action_count"]),
            observation_size=int(payload["observation_size"]),
            training=PPOTrainingConfig(**_read_training_config(payload["training"])),
            stable_baselines3_version=str(payload["stable_baselines3_version"]),
            torch_version=str(payload["torch_version"]),
            final_lagrange_multiplier=(
                None if payload.get("final_lagrange_multiplier") is None else float(payload["final_lagrange_multiplier"])
            ),
            feature_schema_version=int(payload.get("feature_schema_version", 1)),
            factorized_actions=bool(payload.get("factorized_actions", False)),
            best_validation_reward=float(payload.get("best_validation_reward", float("nan"))),
            best_validation_timestep=int(payload.get("best_validation_timestep", 0)),
            best_validation_urllc_cost=float(payload.get("best_validation_urllc_cost", float("nan"))),
            best_validation_max_regime_urllc_cost=float(
                payload.get("best_validation_max_regime_urllc_cost", float("nan"))
            ),
            best_validation_feasible=bool(payload.get("best_validation_feasible", False)),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid PPO artifact metadata: {path}") from error


def _read_training_config(payload: dict[str, object]) -> dict[str, object]:
    values = dict(payload)
    constraint = values.get("constraint")
    if constraint is not None:
        if not isinstance(constraint, dict):
            raise ValueError("invalid PPO constraint configuration")
        values["constraint"] = LagrangianConfig(**constraint)
    if "validation_seeds" in values:
        values["validation_seeds"] = tuple(int(seed) for seed in values["validation_seeds"])  # type: ignore[arg-type]
    return values
