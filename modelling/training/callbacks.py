"""Optional SB3 callbacks used for reproducible training artifacts."""

from __future__ import annotations

from typing import Any
import numpy as np


VALIDATION_FIELDS = (
    "timesteps", "mean_reward", "std_reward", "mean_urllc_cost", "mean_embb_cost", "mean_mmtc_cost",
    "p95_period_urllc_cost", "cvar95_period_urllc_cost", "any_miss_period_rate",
    "max_consecutive_unsafe_periods", "baseline_urllc_cost", "embb_surge_urllc_cost",
    "urllc_surge_urllc_cost", "mmtc_surge_urllc_cost", "max_regime_urllc_cost",
    "constraint_applied", "constraint_feasible", "unique_actions", "dominant_action_fraction", "exploration_rate",
    "mean_max_q", "mean_absolute_q", "training_loss", "policy_gradient_loss", "value_loss", "entropy_loss",
    "approximate_kl", "clip_fraction", "explained_variance", "lagrange_multiplier", "episodes", "post_training",
)


def period_cost_tail_metrics(costs: np.ndarray) -> tuple[float, float]:
    """Return empirical p95 and top-5% CVaR without zero-quantile dilution."""

    values = np.sort(np.asarray(costs, dtype=np.float64))
    if values.size == 0:
        raise ValueError("period cost metrics require at least one observation")
    p95_index = min(values.size - 1, int(0.95 * (values.size - 1)))
    cvar_start = min(values.size - 1, int(0.95 * values.size))
    return float(values[p95_index]), float(np.mean(values[cvar_start:]))


def build_episode_reward_callback() -> Any:
    """Collect completed-episode rewards without changing the environment API."""

    try:
        from stable_baselines3.common.callbacks import BaseCallback
    except ImportError as exc:  # pragma: no cover - guarded by the PPO entry point
        raise RuntimeError("Stable-Baselines3 is required for episode reward logging") from exc

    class _EpisodeRewardCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__(verbose=0)
            self._running_rewards: list[float] = []
            self.episode_rewards: list[float] = []

        def _on_training_start(self) -> None:
            self._running_rewards = [0.0] * self.training_env.num_envs

        def _on_step(self) -> bool:
            for index, (reward, done) in enumerate(zip(self.locals["rewards"], self.locals["dones"])):
                self._running_rewards[index] += float(reward)
                if done:
                    self.episode_rewards.append(self._running_rewards[index])
                    self._running_rewards[index] = 0.0
            return True

    return _EpisodeRewardCallback()


def build_validation_callback(
    environment_factory: Any,
    seeds: tuple[int, ...],
    eval_freq: int,
    best_model_path: Any | None = None,
    maximum_mean_urllc_cost: float | None = None,
    maximum_regime_urllc_cost: float | None = None,
    lagrange_multiplier_getter: Any | None = None,
) -> Any:
    """Evaluate on fixed seeds and select a constraint-aware checkpoint.

    Feasible candidates are ranked by reward.  Before any feasible candidate
    exists, the least-violating candidate is retained (reward breaks ties).
    This avoids silently calling a high-reward, unsafe snapshot the best model.
    """

    if not seeds or eval_freq <= 0:
        raise ValueError("validation seeds and eval_freq must be non-empty/positive")
    try:
        from stable_baselines3.common.callbacks import BaseCallback
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Stable-Baselines3 is required for validation evaluation") from exc

    class _ValidationCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__(verbose=0)
            self.last_evaluation = 0
            self.records: list[dict[str, float | int]] = []
            self.best_mean_reward = float("-inf")
            self.best_mean_urllc_cost = float("inf")
            self.best_feasible = False
            self.best_safety_violation = float("inf")
            self.best_max_regime_urllc_cost = float("inf")
            self.best_timestep = 0

        def _is_better(self, mean_reward: float, mean_urllc_cost: float, max_regime_cost: float) -> bool:
            if maximum_mean_urllc_cost is None and maximum_regime_urllc_cost is None:
                return mean_reward > self.best_mean_reward
            mean_excess = (
                0.0 if maximum_mean_urllc_cost is None else max(0.0, mean_urllc_cost - maximum_mean_urllc_cost)
            )
            regime_excess = (
                0.0
                if maximum_regime_urllc_cost is None
                else max(0.0, max_regime_cost - maximum_regime_urllc_cost)
            )
            safety_violation = max(mean_excess, regime_excess)
            feasible = safety_violation == 0.0
            if feasible != self.best_feasible:
                return feasible
            if feasible:
                return mean_reward > self.best_mean_reward
            return (safety_violation, -mean_reward) < (self.best_safety_violation, -self.best_mean_reward)

        def evaluate_current_policy(self, *, post_training: bool = False) -> None:
            """Evaluate the current parameters, including after the last PPO update."""

            timestep = int(self.model.num_timesteps)
            rewards: list[float] = []
            costs: list[tuple[float, float, float]] = []
            action_ids: list[int] = []
            validation_observations: list[np.ndarray] = []
            regime_costs: dict[str, list[float]] = {}
            maximum_unsafe_streaks: list[int] = []
            for seed in seeds:
                environment = environment_factory()
                observation, _ = environment.reset(seed=seed)
                total_reward = 0.0
                unsafe_streak = 0
                maximum_unsafe_streak = 0
                while True:
                    validation_observations.append(np.asarray(observation, dtype=np.float32))
                    action, _ = self.model.predict(observation, deterministic=True)
                    observation, reward, terminated, truncated, info = environment.step(action)
                    total_reward += float(reward)
                    step_costs = info.get("costs")
                    if step_costs is None or len(step_costs) != 3:
                        raise RuntimeError("validation environment must expose three costs in info['costs']")
                    numeric_costs = tuple(float(value) for value in step_costs)
                    costs.append(numeric_costs)
                    regime = str(info.get("traffic_regime", "unknown"))
                    regime_costs.setdefault(regime, []).append(numeric_costs[0])
                    unsafe_streak = unsafe_streak + 1 if numeric_costs[0] > 0.0 else 0
                    maximum_unsafe_streak = max(maximum_unsafe_streak, unsafe_streak)
                    action_ids.append(int(info.get("applied_action_id", np.asarray(action).reshape(-1)[0])))
                    if terminated or truncated:
                        break
                environment.close()
                rewards.append(total_reward)
                maximum_unsafe_streaks.append(maximum_unsafe_streak)
            mean_costs = np.mean(np.asarray(costs, dtype=np.float64), axis=0)
            urllc_costs = np.asarray([cost[0] for cost in costs], dtype=np.float64)
            p95_urllc_cost, cvar95_urllc_cost = period_cost_tail_metrics(urllc_costs)
            named_regime_costs = {
                regime: (float(np.mean(values)) if values else float("nan"))
                for regime, values in regime_costs.items()
            }
            max_regime_urllc_cost = max(named_regime_costs.values())
            _, action_counts = np.unique(np.asarray(action_ids), return_counts=True)
            mean_reward = float(np.mean(rewards))
            mean_urllc_cost = float(mean_costs[0])
            constraint_applied = maximum_mean_urllc_cost is not None or maximum_regime_urllc_cost is not None
            feasible = (
                (maximum_mean_urllc_cost is None or mean_urllc_cost <= maximum_mean_urllc_cost)
                and (maximum_regime_urllc_cost is None or max_regime_urllc_cost <= maximum_regime_urllc_cost)
            )
            mean_max_q = float("nan")
            mean_absolute_q = float("nan")
            q_network = getattr(getattr(self.model, "policy", None), "q_net", None)
            if q_network is not None:
                import torch

                with torch.no_grad():
                    q_values = q_network(
                        torch.as_tensor(np.stack(validation_observations), device=self.model.device)
                    )
                mean_max_q = float(q_values.max(dim=1).values.mean().cpu())
                mean_absolute_q = float(q_values.abs().mean().cpu())
            self.records.append(
                {
                    "timesteps": timestep,
                    "mean_reward": mean_reward,
                    "std_reward": float(np.std(rewards)),
                    "mean_urllc_cost": mean_urllc_cost,
                    "mean_embb_cost": float(mean_costs[1]),
                    "mean_mmtc_cost": float(mean_costs[2]),
                    "p95_period_urllc_cost": p95_urllc_cost,
                    "cvar95_period_urllc_cost": cvar95_urllc_cost,
                    "any_miss_period_rate": float(np.mean(urllc_costs > 0.0)),
                    "max_consecutive_unsafe_periods": int(max(maximum_unsafe_streaks)),
                    "baseline_urllc_cost": named_regime_costs.get("baseline", float("nan")),
                    "embb_surge_urllc_cost": named_regime_costs.get("embb_surge", float("nan")),
                    "urllc_surge_urllc_cost": named_regime_costs.get("urllc_surge", float("nan")),
                    "mmtc_surge_urllc_cost": named_regime_costs.get("mmtc_surge", float("nan")),
                    "max_regime_urllc_cost": max_regime_urllc_cost,
                    "constraint_applied": int(constraint_applied),
                    "constraint_feasible": int(feasible) if constraint_applied else -1,
                    "unique_actions": int(len(action_counts)),
                    "dominant_action_fraction": float(action_counts.max() / action_counts.sum()),
                    "exploration_rate": float(getattr(self.model, "exploration_rate", float("nan"))),
                    "mean_max_q": mean_max_q,
                    "mean_absolute_q": mean_absolute_q,
                    "training_loss": float(self.model.logger.name_to_value.get("train/loss", float("nan"))),
                    "policy_gradient_loss": float(
                        self.model.logger.name_to_value.get("train/policy_gradient_loss", float("nan"))
                    ),
                    "value_loss": float(self.model.logger.name_to_value.get("train/value_loss", float("nan"))),
                    "entropy_loss": float(self.model.logger.name_to_value.get("train/entropy_loss", float("nan"))),
                    "approximate_kl": float(self.model.logger.name_to_value.get("train/approx_kl", float("nan"))),
                    "clip_fraction": float(self.model.logger.name_to_value.get("train/clip_fraction", float("nan"))),
                    "explained_variance": float(
                        self.model.logger.name_to_value.get("train/explained_variance", float("nan"))
                    ),
                    "lagrange_multiplier": (
                        float("nan") if lagrange_multiplier_getter is None else float(lagrange_multiplier_getter())
                    ),
                    "episodes": len(rewards),
                    "post_training": int(post_training),
                }
            )
            if self._is_better(mean_reward, mean_urllc_cost, max_regime_urllc_cost):
                self.best_mean_reward = mean_reward
                self.best_mean_urllc_cost = mean_urllc_cost
                self.best_max_regime_urllc_cost = max_regime_urllc_cost
                self.best_safety_violation = max(
                    0.0 if maximum_mean_urllc_cost is None else mean_urllc_cost - maximum_mean_urllc_cost,
                    0.0 if maximum_regime_urllc_cost is None else max_regime_urllc_cost - maximum_regime_urllc_cost,
                )
                self.best_feasible = constraint_applied and feasible
                self.best_timestep = timestep
                if best_model_path is not None:
                    self.model.save(best_model_path)
            self.logger.record("validation/mean_reward", self.records[-1]["mean_reward"])
            self.logger.record("validation/mean_urllc_cost", mean_urllc_cost)
            self.logger.record("validation/max_regime_urllc_cost", max_regime_urllc_cost)
            self.logger.record("validation/dominant_action_fraction", self.records[-1]["dominant_action_fraction"])
            self.last_evaluation = timestep

        def _on_step(self) -> bool:
            if self.num_timesteps - self.last_evaluation < eval_freq:
                return True
            self.evaluate_current_policy()
            return True

    return _ValidationCallback()
