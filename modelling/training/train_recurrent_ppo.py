"""Runnable fast-simulator training loop for the recurrent PPO prototype."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import csv
from copy import deepcopy

import numpy as np
import torch

from modelling.action_catalog import ActionCatalog
from modelling.config import scenario_fingerprint
from modelling.contracts import EnvironmentConfig
from modelling.contracts import Action, Observation
from modelling.envs.gym_adapter import SafeSliceGymEnv
from modelling.models.recurrent_actor_critic import RecurrentActorCritic
from modelling.safety.shield import SafetyShield
from modelling.training.lagrangian import LagrangianConfig, LagrangianController
from modelling.training.recurrent_ppo import RecurrentPPOConfig, RecurrentPPOUpdate, RecurrentPPOUpdater
from modelling.training.recurrent_runner import collect_recurrent_rollout, reset_recurrent_runner
from modelling.features import FEATURE_SCHEMA_VERSION, FactorizedActionCodec, FeatureEncoder


@dataclass(frozen=True, slots=True)
class RecurrentTrainingConfig:
    updates: int
    horizon: int = 128
    hidden_size: int = 128
    n_envs: int = 1
    seed: int = 0
    use_safety_shield: bool = False
    progress_bar: bool = True
    ppo: RecurrentPPOConfig = RecurrentPPOConfig()
    constraint: LagrangianConfig | None = None
    eval_every_updates: int = 10
    validation_seeds: tuple[int, ...] = (101, 102, 103, 104, 105)
    constraint_cost_ema_alpha: float = 0.05
    max_regime_urllc_cost_limit: float | None = None

    def __post_init__(self) -> None:
        if self.updates <= 0 or self.horizon <= 0 or self.hidden_size <= 0 or self.n_envs <= 0:
            raise ValueError("updates, horizon, hidden_size, and n_envs must be positive")
        if self.horizon % self.ppo.sequence_length:
            raise ValueError("horizon must divide evenly into recurrent PPO sequences")
        if self.eval_every_updates <= 0 or not self.validation_seeds:
            raise ValueError("recurrent validation cadence and seeds must be configured")
        if not 0.0 < self.constraint_cost_ema_alpha <= 1.0:
            raise ValueError("constraint_cost_ema_alpha must be in (0, 1]")
        if self.max_regime_urllc_cost_limit is not None and not 0.0 <= self.max_regime_urllc_cost_limit <= 1.0:
            raise ValueError("max_regime_urllc_cost_limit must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class RecurrentTrainingMetadata:
    schema_version: int
    created_at_utc: str
    scenario_fingerprint: str
    action_count: int
    observation_size: int
    training: RecurrentTrainingConfig
    final_lagrange_multiplier: float | None
    last_update: RecurrentPPOUpdate
    torch_version: str
    best_validation_reward: float
    best_validation_urllc_cost: float
    best_validation_max_regime_urllc_cost: float
    best_validation_feasible: bool
    best_update: int


class RecurrentPolicy:
    """Stateful catalogue-policy adapter; resets GRU belief at episode start."""

    def __init__(self, model: RecurrentActorCritic, catalog: ActionCatalog, feature_encoder: FeatureEncoder) -> None:
        self.model = model
        self.catalog = catalog
        self.feature_encoder = feature_encoder
        self._hidden_state: torch.Tensor | None = None

    def act(self, observation: Observation) -> Action:
        device = next(self.model.parameters()).device
        starts = observation.step == 0 or self._hidden_state is None
        if starts:
            self._hidden_state = self.model.initial_state(1, device=device)
        assert self._hidden_state is not None
        with torch.no_grad():
            result = self.model.act(
                torch.as_tensor(self.feature_encoder.encode(observation), dtype=torch.float32, device=device).unsqueeze(0),
                hidden_state=self._hidden_state,
                episode_starts=torch.tensor([starts], dtype=torch.bool, device=device),
                deterministic=True,
            )
        self._hidden_state = result.hidden_state
        return self.catalog.get(int(result.action.item()))


def train_recurrent_ppo(
    *,
    training: RecurrentTrainingConfig,
    output_path: Path,
    config: EnvironmentConfig | None = None,
) -> RecurrentTrainingMetadata:
    """Train, checkpoint, and fingerprint a GRU PPO policy on the fast simulator."""

    environment_config = config or EnvironmentConfig()
    torch.manual_seed(training.seed)
    np.random.seed(training.seed)
    catalog = ActionCatalog.build(environment_config)
    feature_encoder = FeatureEncoder(environment_config, catalog)
    action_codec = FactorizedActionCodec(catalog)
    for allocation_index in range(action_codec.shape[0]):
        for mitigation_index in range(action_codec.shape[1]):
            expected = allocation_index * action_codec.shape[1] + mitigation_index
            if action_codec.decode((allocation_index, mitigation_index)) != expected:
                raise RuntimeError("recurrent factorization requires allocation-major catalogue ordering")
    environments = [
        SafeSliceGymEnv(
            environment_config,
            shield=SafetyShield(catalog, environment_config) if training.use_safety_shield else None,
            feature_encoder=feature_encoder,
        )
        for _ in range(training.n_envs)
    ]
    model = RecurrentActorCritic(
        feature_encoder.size,
        len(catalog),
        hidden_size=training.hidden_size,
        action_factors=action_codec.shape,
    )
    updater = RecurrentPPOUpdater(model, training.ppo)
    controller = LagrangianController(training.constraint) if training.constraint is not None else None
    state = reset_recurrent_runner(environments, model, seed=training.seed)
    last_update: RecurrentPPOUpdate | None = None
    best_reward = float("-inf")
    best_urllc_cost = float("inf")
    best_max_regime_cost = float("inf")
    best_safety_violation = float("inf")
    best_feasible = False
    best_update = 0
    best_state: dict[str, torch.Tensor] | None = None
    training_records: list[dict[str, float | int]] = []
    validation_records: list[dict[str, float | int]] = []
    smoothed_urllc_cost: float | None = None
    try:
        from tqdm.auto import trange
    except ImportError:  # pragma: no cover - tqdm is part of the training extra
        trange = range  # type: ignore[assignment]
    for update_index in trange(training.updates, desc="Recurrent PPO updates", disable=not training.progress_bar):
        rollout = collect_recurrent_rollout(environments, model, state, horizon=training.horizon)
        state = rollout.state
        last_update = updater.update(
            rollout.buffer,
            urllc_lagrange_multiplier=0.0 if controller is None else controller.multiplier,
        )
        mean_costs = rollout.buffer.costs.mean(axis=(0, 1))
        smoothed_urllc_cost = (
            float(mean_costs[0])
            if smoothed_urllc_cost is None
            else (1.0 - training.constraint_cost_ema_alpha) * smoothed_urllc_cost
            + training.constraint_cost_ema_alpha * float(mean_costs[0])
        )
        if controller is not None:
            controller.update(smoothed_urllc_cost)
        completed_update = update_index + 1
        training_records.append(
            {
                "update": completed_update,
                "timesteps": completed_update * training.horizon * training.n_envs,
                "mean_step_reward": float(rollout.buffer.rewards.mean()),
                "mean_urllc_cost": float(mean_costs[0]),
                "smoothed_urllc_cost": smoothed_urllc_cost,
                "mean_embb_cost": float(mean_costs[1]),
                "mean_mmtc_cost": float(mean_costs[2]),
                "lagrange_multiplier": 0.0 if controller is None else controller.multiplier,
                "policy_loss": last_update.policy_loss,
                "reward_value_loss": last_update.reward_value_loss,
                "cost_value_loss": last_update.cost_value_loss,
                "entropy": last_update.entropy,
                "approximate_kl": last_update.approximate_kl,
            }
        )
        if completed_update % training.eval_every_updates == 0 or completed_update == training.updates:
            mean_reward, std_reward, validation_costs, unique_actions, dominant_fraction, burst_metrics = _evaluate_recurrent(
                model, environment_config, training.validation_seeds
            )
            cost_limit = None if training.constraint is None else training.constraint.urllc_cost_limit
            max_regime_limit = training.max_regime_urllc_cost_limit
            mean_excess = 0.0 if cost_limit is None else max(0.0, validation_costs[0] - cost_limit)
            regime_excess = (
                0.0
                if max_regime_limit is None
                else max(0.0, burst_metrics["max_regime_urllc_cost"] - max_regime_limit)
            )
            safety_violation = max(mean_excess, regime_excess)
            constraint_applied = cost_limit is not None or max_regime_limit is not None
            feasible = not constraint_applied or safety_violation == 0.0
            validation_records.append(
                {
                    "update": completed_update,
                    "timesteps": completed_update * training.horizon * training.n_envs,
                    "mean_reward": mean_reward,
                    "std_reward": std_reward,
                    "mean_urllc_cost": validation_costs[0],
                    "mean_embb_cost": validation_costs[1],
                    "mean_mmtc_cost": validation_costs[2],
                    **burst_metrics,
                    "constraint_applied": int(constraint_applied),
                    "constraint_feasible": int(feasible),
                    "unique_actions": unique_actions,
                    "dominant_action_fraction": dominant_fraction,
                    "episodes": len(training.validation_seeds),
                }
            )
            better = (
                mean_reward > best_reward
                if not constraint_applied
                else (
                    feasible and not best_feasible
                    or feasible == best_feasible
                    and (
                        mean_reward > best_reward
                        if feasible
                        else (safety_violation, -mean_reward) < (best_safety_violation, -best_reward)
                    )
                )
            )
            if better:
                best_reward = mean_reward
                best_urllc_cost = validation_costs[0]
                best_max_regime_cost = burst_metrics["max_regime_urllc_cost"]
                best_safety_violation = safety_violation
                best_feasible = feasible
                best_update = completed_update
                best_state = deepcopy({name: tensor.detach().cpu() for name, tensor in model.state_dict().items()})
    assert last_update is not None
    checkpoint_path = _checkpoint_path(output_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_payload = {
            "schema_version": 2,
            "scenario_fingerprint": scenario_fingerprint(environment_config),
            "observation_size": feature_encoder.size,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "action_count": len(catalog),
            "action_factors": action_codec.shape,
            "hidden_size": training.hidden_size,
            "cost_count": 3,
            "model_state_dict": model.state_dict(),
        }
    torch.save(checkpoint_payload, checkpoint_path.with_name(f"{checkpoint_path.stem}.final.pt"))
    checkpoint_payload["model_state_dict"] = best_state if best_state is not None else model.state_dict()
    torch.save(checkpoint_payload, checkpoint_path)
    metadata = RecurrentTrainingMetadata(
        schema_version=2,
        created_at_utc=datetime.now(UTC).isoformat(),
        scenario_fingerprint=scenario_fingerprint(environment_config),
        action_count=len(catalog),
        observation_size=feature_encoder.size,
        training=training,
        final_lagrange_multiplier=None if controller is None else controller.multiplier,
        last_update=last_update,
        torch_version=torch.__version__,
        best_validation_reward=best_reward,
        best_validation_urllc_cost=best_urllc_cost,
        best_validation_max_regime_urllc_cost=best_max_regime_cost,
        best_validation_feasible=best_feasible,
        best_update=best_update,
    )
    _metadata_path(checkpoint_path).write_text(json.dumps(_jsonable(asdict(metadata)), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with checkpoint_path.with_suffix(".training.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(training_records[0]))
        writer.writeheader()
        writer.writerows(training_records)
    with checkpoint_path.with_suffix(".validation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(validation_records[0]))
        writer.writeheader()
        writer.writerows(validation_records)
    for environment in environments:
        environment.close()
    return metadata


def _evaluate_recurrent(
    model: RecurrentActorCritic,
    config: EnvironmentConfig,
    seeds: tuple[int, ...],
) -> tuple[float, float, tuple[float, float, float], int, float, dict[str, float | int]]:
    from modelling.evaluation import evaluate_episode

    catalog = ActionCatalog.build(config)
    summaries = [
        evaluate_episode(config, RecurrentPolicy(model, catalog, FeatureEncoder(config, catalog)), seed=seed)
        for seed in seeds
    ]
    rewards = [summary.total_reward for summary in summaries]
    costs = tuple(
        float(np.mean([getattr(summary, field) for summary in summaries]))
        for field in ("mean_urllc_cost", "mean_embb_cost", "mean_mmtc_cost")
    )
    action_ids = np.asarray([action for summary in summaries for action in summary.action_ids])
    _, counts = np.unique(action_ids, return_counts=True)
    burst_fields = (
        "p95_period_urllc_cost",
        "cvar95_period_urllc_cost",
        "any_miss_period_rate",
        "baseline_urllc_cost",
        "embb_surge_urllc_cost",
        "urllc_surge_urllc_cost",
        "mmtc_surge_urllc_cost",
        "max_regime_urllc_cost",
    )
    burst_metrics: dict[str, float | int] = {
        field: float(np.nanmean([getattr(summary, field) for summary in summaries])) for field in burst_fields
    }
    burst_metrics["max_consecutive_unsafe_periods"] = max(
        summary.max_consecutive_unsafe_periods for summary in summaries
    )
    return (
        float(np.mean(rewards)),
        float(np.std(rewards)),
        costs,
        int(len(counts)),
        float(counts.max() / counts.sum()),
        burst_metrics,
    )


def load_recurrent_model(
    checkpoint_path: Path,
    config: EnvironmentConfig,
    *,
    device: str = "cpu",
    allow_scenario_mismatch: bool = False,
) -> RecurrentActorCritic:
    """Load a checkpoint only when its action/observation scenario contract matches."""

    payload = torch.load(_checkpoint_path(checkpoint_path), map_location=device, weights_only=True)
    catalog = ActionCatalog.build(config)
    feature_encoder = FeatureEncoder(config, catalog)
    if payload.get("schema_version") != 2 or (
        payload.get("scenario_fingerprint") != scenario_fingerprint(config) and not allow_scenario_mismatch
    ):
        raise ValueError("recurrent checkpoint scenario fingerprint does not match the requested environment")
    if payload.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError("recurrent checkpoint feature schema is unsupported")
    if payload.get("observation_size") != feature_encoder.size or payload.get("action_count") != len(catalog):
        raise ValueError("recurrent checkpoint action or observation contract mismatch")
    model = RecurrentActorCritic(
        observation_size=feature_encoder.size,
        action_count=len(catalog),
        hidden_size=int(payload["hidden_size"]),
        cost_count=int(payload["cost_count"]),
        action_factors=tuple(int(value) for value in payload["action_factors"]),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def load_recurrent_policy(
    checkpoint_path: Path,
    config: EnvironmentConfig,
    *,
    device: str = "cpu",
    allow_scenario_mismatch: bool = False,
) -> RecurrentPolicy:
    """Load a checkpoint as an evaluation-policy adapter."""

    catalog = ActionCatalog.build(config)
    return RecurrentPolicy(
        load_recurrent_model(checkpoint_path, config, device=device, allow_scenario_mismatch=allow_scenario_mismatch),
        catalog,
        FeatureEncoder(config, catalog),
    )


def _checkpoint_path(path: Path) -> Path:
    return path if path.suffix == ".pt" else path.with_suffix(".pt")


def _metadata_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_suffix(".metadata.json")


def _jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value
