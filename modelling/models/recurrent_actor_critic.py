"""Recurrent actor and separate reward/cost critics for partial observability.

The GRU hidden state is the initial jammer/telemetry belief representation. It
is reset explicitly at episode boundaries so delayed KPM history from one
scenario can never leak into the next one.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.distributions import Categorical


@dataclass(frozen=True, slots=True)
class RecurrentPolicyOutput:
    """Outputs for a complete batch of observation sequences."""

    logits: Tensor
    reward_values: Tensor
    cost_values: Tensor
    hidden_state: Tensor


@dataclass(frozen=True, slots=True)
class RecurrentAction:
    """One action sample plus values required by a PPO rollout buffer."""

    action: Tensor
    log_probability: Tensor
    reward_value: Tensor
    cost_values: Tensor
    hidden_state: Tensor


class RecurrentActorCritic(nn.Module):
    """GRU belief encoder, categorical actor, and independent value heads."""

    def __init__(
        self,
        observation_size: int,
        action_count: int,
        *,
        hidden_size: int = 128,
        cost_count: int = 3,
        action_factors: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        if observation_size <= 0 or action_count <= 1 or hidden_size <= 0 or cost_count <= 0:
            raise ValueError("invalid recurrent actor-critic dimensions")
        self.observation_size = observation_size
        self.action_count = action_count
        self.hidden_size = hidden_size
        self.cost_count = cost_count
        if action_factors is not None and action_factors[0] * action_factors[1] != action_count:
            raise ValueError("action factors must multiply to action_count")
        self.action_factors = action_factors
        self.encoder = nn.Sequential(nn.Linear(observation_size, hidden_size), nn.Tanh())
        # GRUCell lets us apply a reset mask at any individual sequence step.
        self.belief_cell = nn.GRUCell(hidden_size, hidden_size)
        if action_factors is None:
            self.actor = nn.Linear(hidden_size, action_count)
            self.allocation_actor = None
            self.mitigation_actor = None
        else:
            self.actor = None
            self.allocation_actor = nn.Linear(hidden_size, action_factors[0])
            self.mitigation_actor = nn.Linear(hidden_size, action_factors[1])
        self.reward_critic = nn.Linear(hidden_size, 1)
        self.cost_critic = nn.Linear(hidden_size, cost_count)

    def initial_state(self, batch_size: int, *, device: torch.device | str | None = None) -> Tensor:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        return torch.zeros(batch_size, self.hidden_size, device=device)

    def forward_sequence(
        self,
        observations: Tensor,
        *,
        hidden_state: Tensor | None = None,
        episode_starts: Tensor | None = None,
    ) -> RecurrentPolicyOutput:
        """Evaluate ``[batch, time, features]`` observations with reset masking."""

        if observations.ndim != 3 or observations.shape[-1] != self.observation_size:
            raise ValueError("observations must have shape [batch, time, observation_size]")
        batch_size, steps, _ = observations.shape
        if hidden_state is None:
            hidden_state = self.initial_state(batch_size, device=observations.device)
        if hidden_state.shape != (batch_size, self.hidden_size):
            raise ValueError("hidden_state has an incompatible shape")
        if episode_starts is None:
            episode_starts = torch.zeros(batch_size, steps, dtype=torch.bool, device=observations.device)
        if episode_starts.shape != (batch_size, steps):
            raise ValueError("episode_starts must have shape [batch, time]")
        states: list[Tensor] = []
        state = hidden_state
        for index in range(steps):
            state = state * (~episode_starts[:, index]).to(dtype=state.dtype).unsqueeze(-1)
            state = self.belief_cell(self.encoder(observations[:, index]), state)
            states.append(state)
        beliefs = torch.stack(states, dim=1)
        if self.action_factors is None:
            assert self.actor is not None
            logits = self.actor(beliefs)
        else:
            assert self.allocation_actor is not None and self.mitigation_actor is not None
            # Catalogue ordering is allocation-major, mitigation-minor.  The
            # additive logits define the product of two categorical policies
            # while preserving the existing flat action-id rollout contract.
            logits = (
                self.allocation_actor(beliefs).unsqueeze(-1)
                + self.mitigation_actor(beliefs).unsqueeze(-2)
            ).flatten(start_dim=-2)
        return RecurrentPolicyOutput(
            logits=logits,
            reward_values=self.reward_critic(beliefs).squeeze(-1),
            cost_values=self.cost_critic(beliefs),
            hidden_state=state,
        )

    @torch.no_grad()
    def act(
        self,
        observations: Tensor,
        *,
        hidden_state: Tensor | None = None,
        episode_starts: Tensor | None = None,
        deterministic: bool = False,
    ) -> RecurrentAction:
        """Choose one catalogue action from ``[batch, observation_size]`` input."""

        if observations.ndim != 2:
            raise ValueError("act expects shape [batch, observation_size]")
        starts = None if episode_starts is None else episode_starts.unsqueeze(1)
        output = self.forward_sequence(observations.unsqueeze(1), hidden_state=hidden_state, episode_starts=starts)
        distribution = Categorical(logits=output.logits[:, 0])
        action = output.logits[:, 0].argmax(dim=-1) if deterministic else distribution.sample()
        return RecurrentAction(
            action=action,
            log_probability=distribution.log_prob(action),
            reward_value=output.reward_values[:, 0],
            cost_values=output.cost_values[:, 0],
            hidden_state=output.hidden_state,
        )
