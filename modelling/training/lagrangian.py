"""First primal-dual constraint layer for the feed-forward PPO baseline.

The controller adjusts the multiplier from observed per-period URLLC cost.
It is deliberately separate from the environment: the raw reward and cost
remain available in ``info`` for reporting, while PPO sees the augmented
reward. A cost critic and recurrent constrained policy are later increments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym


@dataclass(frozen=True, slots=True)
class LagrangianConfig:
    """Constraint target and stable projected-dual update parameters."""

    urllc_cost_limit: float = 0.01
    multiplier_learning_rate: float = 0.1
    initial_multiplier: float = 0.0
    max_multiplier: float = 100.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.urllc_cost_limit <= 1.0:
            raise ValueError("urllc_cost_limit must be in [0, 1]")
        if self.multiplier_learning_rate <= 0.0 or self.initial_multiplier < 0.0 or self.max_multiplier <= 0.0:
            raise ValueError("invalid Lagrangian multiplier configuration")
        if self.initial_multiplier > self.max_multiplier:
            raise ValueError("initial_multiplier cannot exceed max_multiplier")


class LagrangianController:
    """Mutable projected multiplier shared by the reward wrapper and callback."""

    def __init__(self, config: LagrangianConfig) -> None:
        self.config = config
        self.multiplier = config.initial_multiplier

    def penalize(self, reward: float, urllc_cost: float) -> float:
        return reward - self.multiplier * urllc_cost

    def update(self, mean_urllc_cost: float) -> float:
        delta = self.config.multiplier_learning_rate * (mean_urllc_cost - self.config.urllc_cost_limit)
        self.multiplier = min(self.config.max_multiplier, max(0.0, self.multiplier + delta))
        return self.multiplier


class URLLCCostPenaltyWrapper(gym.Wrapper[Any, Any, Any, Any]):
    """Expose constrained reward while retaining raw outcomes in ``info``."""

    def __init__(self, env: gym.Env[Any, Any], controller: LagrangianController) -> None:
        super().__init__(env)
        self.controller = controller

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        augmented_info = dict(info)
        try:
            urllc_cost = float(augmented_info["costs"][0])
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError("constrained wrapper requires info['costs'][0] for URLLC cost") from error
        augmented_info["raw_reward"] = reward
        augmented_info["urllc_cost"] = urllc_cost
        augmented_info["lagrange_multiplier"] = self.controller.multiplier
        return observation, self.controller.penalize(reward, urllc_cost), terminated, truncated, augmented_info


def build_lagrangian_callback(controller: LagrangianController) -> Any:
    """Create an SB3 callback lazily, preserving simulator-only imports."""

    try:
        from stable_baselines3.common.callbacks import BaseCallback
    except ImportError as exc:  # pragma: no cover - exercised by training dependency guard
        raise RuntimeError("Stable-Baselines3 is required for Lagrangian PPO training") from exc

    class _LagrangianCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__(verbose=0)
            self.costs: list[float] = []

        def _on_step(self) -> bool:
            self.costs.extend(float(info["urllc_cost"]) for info in self.locals["infos"])
            return True

        def _on_rollout_end(self) -> None:
            if self.costs:
                mean_cost = sum(self.costs) / len(self.costs)
                multiplier = controller.update(mean_cost)
                self.logger.record("constraint/mean_urllc_cost", mean_cost)
                self.logger.record("constraint/lagrange_multiplier", multiplier)
                self.costs.clear()

    return _LagrangianCallback()
