"""Masked recurrent PPO update kernel with separate reward and cost critics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.distributions import Categorical

from modelling.models.recurrent_actor_critic import RecurrentActorCritic
from modelling.training.recurrent_rollout import RecurrentRolloutBuffer


@dataclass(frozen=True, slots=True)
class RecurrentPPOConfig:
    learning_rate: float = 1e-4
    clip_range: float = 0.15
    epochs: int = 4
    sequence_length: int = 16
    sequences_per_batch: int = 8
    entropy_coefficient: float = 0.005
    reward_value_coefficient: float = 0.5
    cost_value_coefficient: float = 0.5
    max_gradient_norm: float = 0.5
    target_kl: float | None = 0.015

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0 or not 0.0 < self.clip_range < 1.0 or self.epochs <= 0:
            raise ValueError("invalid recurrent PPO optimizer settings")
        if self.sequence_length <= 0 or self.sequences_per_batch <= 0:
            raise ValueError("sequence_length and sequences_per_batch must be positive")
        if min(self.entropy_coefficient, self.reward_value_coefficient, self.cost_value_coefficient, self.max_gradient_norm) < 0.0:
            raise ValueError("PPO coefficients must be non-negative")
        if self.target_kl is not None and self.target_kl <= 0.0:
            raise ValueError("target_kl must be positive when configured")


@dataclass(frozen=True, slots=True)
class RecurrentPPOUpdate:
    loss: float
    policy_loss: float
    reward_value_loss: float
    cost_value_loss: float
    entropy: float
    approximate_kl: float
    updates: int


class RecurrentPPOUpdater:
    """Apply PPO minibatch updates without flattening recurrent trajectories."""

    def __init__(self, model: RecurrentActorCritic, config: RecurrentPPOConfig | None = None) -> None:
        self.model = model
        self.config = config or RecurrentPPOConfig()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)

    def update(self, buffer: RecurrentRolloutBuffer, *, urllc_lagrange_multiplier: float = 0.0) -> RecurrentPPOUpdate:
        if urllc_lagrange_multiplier < 0.0:
            raise ValueError("urllc_lagrange_multiplier must be non-negative")
        batches = buffer.iterate_sequences(self.config.sequence_length, self.config.sequences_per_batch)
        totals = {"loss": 0.0, "policy": 0.0, "reward_value": 0.0, "cost_value": 0.0, "entropy": 0.0, "kl": 0.0}
        updates = 0
        self.model.train()
        for _ in range(self.config.epochs):
            for batch in batches:
                output = self.model.forward_sequence(
                    batch.observations,
                    hidden_state=batch.initial_hidden_states,
                    episode_starts=batch.episode_starts,
                )
                distribution = Categorical(logits=output.logits)
                log_probabilities = distribution.log_prob(batch.actions)
                ratio = torch.exp(log_probabilities - batch.old_log_probabilities)
                advantage = batch.advantages - urllc_lagrange_multiplier * batch.cost_advantages[:, :, 0]
                advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
                unclipped = ratio * advantage
                clipped = torch.clamp(ratio, 1.0 - self.config.clip_range, 1.0 + self.config.clip_range) * advantage
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                # Huber loss prevents transient critic errors from dominating
                # the shared recurrent representation and actor gradients.
                reward_value_loss = F.smooth_l1_loss(output.reward_values, batch.returns)
                cost_value_loss = F.smooth_l1_loss(output.cost_values, batch.cost_returns)
                entropy = distribution.entropy().mean()
                loss = (
                    policy_loss
                    + self.config.reward_value_coefficient * reward_value_loss
                    + self.config.cost_value_coefficient * cost_value_loss
                    - self.config.entropy_coefficient * entropy
                )
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_gradient_norm)
                self.optimizer.step()
                totals["loss"] += float(loss.detach())
                totals["policy"] += float(policy_loss.detach())
                totals["reward_value"] += float(reward_value_loss.detach())
                totals["cost_value"] += float(cost_value_loss.detach())
                totals["entropy"] += float(entropy.detach())
                log_ratio = log_probabilities - batch.old_log_probabilities
                approximate_kl = ((torch.exp(log_ratio) - 1.0) - log_ratio).mean()
                totals["kl"] += float(approximate_kl.detach())
                updates += 1
                if self.config.target_kl is not None and totals["kl"] / updates > self.config.target_kl:
                    break
            if self.config.target_kl is not None and totals["kl"] / updates > self.config.target_kl:
                break
        return RecurrentPPOUpdate(
            loss=totals["loss"] / updates,
            policy_loss=totals["policy"] / updates,
            reward_value_loss=totals["reward_value"] / updates,
            cost_value_loss=totals["cost_value"] / updates,
            entropy=totals["entropy"] / updates,
            approximate_kl=totals["kl"] / updates,
            updates=updates,
        )
