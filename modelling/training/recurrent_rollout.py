"""Sequence-preserving rollout storage and GAE for recurrent PPO."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class RecurrentBatch:
    observations: Tensor
    actions: Tensor
    old_log_probabilities: Tensor
    advantages: Tensor
    returns: Tensor
    cost_advantages: Tensor
    cost_returns: Tensor
    episode_starts: Tensor
    initial_hidden_states: Tensor


class RecurrentRolloutBuffer:
    """Fixed-horizon buffer that keeps each environment trajectory contiguous."""

    def __init__(
        self,
        horizon: int,
        environments: int,
        observation_size: int,
        hidden_size: int,
        *,
        cost_count: int = 3,
        device: torch.device | str = "cpu",
    ) -> None:
        if min(horizon, environments, observation_size, hidden_size, cost_count) <= 0:
            raise ValueError("rollout dimensions must be positive")
        self.horizon = horizon
        self.environments = environments
        self.observation_size = observation_size
        self.hidden_size = hidden_size
        self.cost_count = cost_count
        self.device = torch.device(device)
        self.position = 0
        self.observations = np.zeros((horizon, environments, observation_size), dtype=np.float32)
        self.actions = np.zeros((horizon, environments), dtype=np.int64)
        self.log_probabilities = np.zeros((horizon, environments), dtype=np.float32)
        self.rewards = np.zeros((horizon, environments), dtype=np.float32)
        self.costs = np.zeros((horizon, environments, cost_count), dtype=np.float32)
        self.dones = np.zeros((horizon, environments), dtype=np.bool_)
        self.episode_starts = np.zeros((horizon, environments), dtype=np.bool_)
        self.reward_values = np.zeros((horizon, environments), dtype=np.float32)
        self.cost_values = np.zeros((horizon, environments, cost_count), dtype=np.float32)
        self.hidden_states = np.zeros((horizon, environments, hidden_size), dtype=np.float32)
        self.advantages = np.zeros((horizon, environments), dtype=np.float32)
        self.cost_advantages = np.zeros((horizon, environments, cost_count), dtype=np.float32)

    @property
    def full(self) -> bool:
        return self.position == self.horizon

    def add(
        self,
        *,
        observations: np.ndarray,
        actions: np.ndarray,
        log_probabilities: np.ndarray,
        rewards: np.ndarray,
        costs: np.ndarray,
        dones: np.ndarray,
        episode_starts: np.ndarray,
        reward_values: np.ndarray,
        cost_values: np.ndarray,
        hidden_states: np.ndarray,
    ) -> None:
        """Add one vectorized environment transition, validating all contracts."""

        if self.full:
            raise RuntimeError("rollout buffer is full; reset it before adding another transition")
        expected = {
            "observations": (self.environments, self.observation_size),
            "actions": (self.environments,),
            "log_probabilities": (self.environments,),
            "rewards": (self.environments,),
            "costs": (self.environments, self.cost_count),
            "dones": (self.environments,),
            "episode_starts": (self.environments,),
            "reward_values": (self.environments,),
            "cost_values": (self.environments, self.cost_count),
            "hidden_states": (self.environments, self.hidden_size),
        }
        supplied = locals()
        for name, shape in expected.items():
            if supplied[name].shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {supplied[name].shape}")
        index = self.position
        self.observations[index] = observations
        self.actions[index] = actions
        self.log_probabilities[index] = log_probabilities
        self.rewards[index] = rewards
        self.costs[index] = costs
        self.dones[index] = dones
        self.episode_starts[index] = episode_starts
        self.reward_values[index] = reward_values
        self.cost_values[index] = cost_values
        self.hidden_states[index] = hidden_states
        self.position += 1

    def compute_advantages(
        self,
        last_reward_values: np.ndarray,
        last_cost_values: np.ndarray,
        *,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        """Compute reward and per-cost generalized advantages in place."""

        if not self.full:
            raise RuntimeError("fill the rollout before computing advantages")
        if last_reward_values.shape != (self.environments,) or last_cost_values.shape != (self.environments, self.cost_count):
            raise ValueError("last value estimates have incompatible shapes")
        if not 0.0 <= gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
            raise ValueError("gamma and gae_lambda must be in [0, 1]")
        next_reward_advantage = np.zeros(self.environments, dtype=np.float32)
        next_cost_advantage = np.zeros((self.environments, self.cost_count), dtype=np.float32)
        for step in range(self.horizon - 1, -1, -1):
            if step == self.horizon - 1:
                next_reward_value, next_cost_value = last_reward_values, last_cost_values
            else:
                next_reward_value, next_cost_value = self.reward_values[step + 1], self.cost_values[step + 1]
            nonterminal = (~self.dones[step]).astype(np.float32)
            reward_delta = self.rewards[step] + gamma * next_reward_value * nonterminal - self.reward_values[step]
            cost_delta = self.costs[step] + gamma * next_cost_value * nonterminal[:, None] - self.cost_values[step]
            next_reward_advantage = reward_delta + gamma * gae_lambda * nonterminal * next_reward_advantage
            next_cost_advantage = cost_delta + gamma * gae_lambda * nonterminal[:, None] * next_cost_advantage
            self.advantages[step] = next_reward_advantage
            self.cost_advantages[step] = next_cost_advantage

    def iterate_sequences(self, sequence_length: int, sequences_per_batch: int) -> list[RecurrentBatch]:
        """Return shuffled contiguous sequence mini-batches without flattening time."""

        if not self.full:
            raise RuntimeError("fill the rollout before creating mini-batches")
        if sequence_length <= 0 or self.horizon % sequence_length:
            raise ValueError("sequence_length must divide the rollout horizon")
        if sequences_per_batch <= 0:
            raise ValueError("sequences_per_batch must be positive")
        starts = [(environment, step) for environment in range(self.environments) for step in range(0, self.horizon, sequence_length)]
        generator = np.random.default_rng(0)
        generator.shuffle(starts)
        batches: list[RecurrentBatch] = []
        for offset in range(0, len(starts), sequences_per_batch):
            selected = starts[offset : offset + sequences_per_batch]
            batches.append(self._batch(selected, sequence_length))
        return batches

    def reset(self) -> None:
        self.position = 0

    def _batch(self, starts: list[tuple[int, int]], sequence_length: int) -> RecurrentBatch:
        def sequence(array: np.ndarray) -> np.ndarray:
            return np.stack([array[start : start + sequence_length, environment] for environment, start in starts])

        return RecurrentBatch(
            observations=torch.as_tensor(sequence(self.observations), device=self.device),
            actions=torch.as_tensor(sequence(self.actions), device=self.device),
            old_log_probabilities=torch.as_tensor(sequence(self.log_probabilities), device=self.device),
            advantages=torch.as_tensor(sequence(self.advantages), device=self.device),
            returns=torch.as_tensor(sequence(self.advantages + self.reward_values), device=self.device),
            cost_advantages=torch.as_tensor(sequence(self.cost_advantages), device=self.device),
            cost_returns=torch.as_tensor(sequence(self.cost_advantages + self.cost_values), device=self.device),
            episode_starts=torch.as_tensor(sequence(self.episode_starts), device=self.device),
            initial_hidden_states=torch.as_tensor(
                np.stack([self.hidden_states[start, environment] for environment, start in starts]), device=self.device
            ),
        )
