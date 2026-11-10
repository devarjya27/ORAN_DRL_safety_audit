"""Gymnasium adapter for the deterministic O-RAN DRL Audit environment."""

from __future__ import annotations

import numpy as np

from modelling.contracts import EnvironmentConfig, Observation
from modelling.envs.slicing_env import SafeSliceEnv
from modelling.safety.shield import SafetyShield
from modelling.features import FactorizedActionCodec, FeatureEncoder


try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover - selected environment supplies Gymnasium
    raise ImportError("Gymnasium is required for SafeSliceGymEnv") from exc


class SafeSliceGymEnv(gym.Env[np.ndarray, int]):
    """SB3-compatible adapter with discrete catalogue actions and float32 observations."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: EnvironmentConfig | None = None,
        *,
        shield: SafetyShield | None = None,
        feature_encoder: FeatureEncoder | None = None,
        factorized_actions: bool = False,
    ) -> None:
        super().__init__()
        self.core = SafeSliceEnv(config)
        if shield is not None and shield.catalog.actions != self.core.catalog.actions:
            raise ValueError("shield catalogue must exactly match the environment catalogue")
        self.shield = shield
        self.feature_encoder = feature_encoder
        self.action_codec = FactorizedActionCodec(self.core.catalog) if factorized_actions else None
        self._last_observation: Observation | None = None
        self.action_space = (
            spaces.MultiDiscrete(np.asarray(self.action_codec.shape, dtype=np.int64))
            if self.action_codec is not None
            else spaces.Discrete(self.core.action_count)
        )
        observation_size = self.feature_encoder.size if self.feature_encoder is not None else 17
        high = 1.0 if self.feature_encoder is not None else np.inf
        self.observation_space = spaces.Box(low=0.0, high=high, shape=(observation_size,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict[str, object] | None = None) -> tuple[np.ndarray, dict[str, object]]:
        super().reset(seed=seed)
        observation, info = self.core.reset(seed=seed)
        self._last_observation = observation
        return self._as_array(observation), info

    def step(self, action: int | np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        action_id = self.action_codec.decode(action) if self.action_codec is not None else int(action)
        proposed_action = self.core.catalog.get(action_id)
        shield_result = None
        selected_action = proposed_action
        if self.shield is not None:
            if self._last_observation is None:
                raise RuntimeError("call reset() before step()")
            shield_result = self.shield.apply(self._last_observation, proposed_action)
            selected_action = shield_result.action
        result = self.core.step(selected_action)
        info = dict(result.info)
        info["costs"] = result.costs
        info["actor_action_id"] = proposed_action.action_id
        info["applied_action_id"] = selected_action.action_id
        if shield_result is not None:
            info["shield_decision"] = shield_result.decision.value
            info["shield_predicted_risk"] = shield_result.predicted_risk
            info["shield_actor_risk"] = shield_result.actor_risk
        self._last_observation = result.observation
        return (
            self._as_array(result.observation),
            result.reward,
            result.terminated,
            result.truncated,
            info,
        )

    def _as_array(self, observation: Observation) -> np.ndarray:
        if self.feature_encoder is not None:
            return self.feature_encoder.encode(observation)
        return np.asarray(observation.vector(), dtype=np.float32)
