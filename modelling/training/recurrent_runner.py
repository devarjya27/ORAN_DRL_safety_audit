"""Single-environment recurrent rollout collection for the fast simulator."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor

from modelling.envs.gym_adapter import SafeSliceGymEnv
from modelling.models.recurrent_actor_critic import RecurrentActorCritic
from modelling.training.recurrent_rollout import RecurrentRolloutBuffer


@dataclass(frozen=True, slots=True)
class RecurrentRunnerState:
    observation: np.ndarray
    hidden_state: Tensor
    episode_start: np.ndarray


@dataclass(frozen=True, slots=True)
class CollectedRollout:
    buffer: RecurrentRolloutBuffer
    state: RecurrentRunnerState


def reset_recurrent_runner(
    environment: SafeSliceGymEnv | Sequence[SafeSliceGymEnv],
    model: RecurrentActorCritic,
    *,
    seed: int | None = None,
) -> RecurrentRunnerState:
    """Reset one or more environments and create zero belief states."""

    environments = _as_environments(environment)
    observations = [
        item.reset(seed=None if seed is None else seed + index)[0]
        for index, item in enumerate(environments)
    ]
    device = next(model.parameters()).device
    return RecurrentRunnerState(
        np.stack(observations),
        model.initial_state(len(environments), device=device),
        np.ones(len(environments), dtype=np.bool_),
    )


def collect_recurrent_rollout(
    environment: SafeSliceGymEnv | Sequence[SafeSliceGymEnv],
    model: RecurrentActorCritic,
    state: RecurrentRunnerState,
    *,
    horizon: int,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> CollectedRollout:
    """Collect one horizon while respecting terminal hidden-state boundaries."""

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    device = next(model.parameters()).device
    environments = _as_environments(environment)
    environment_count = len(environments)
    if state.observation.shape != (environment_count, model.observation_size):
        raise ValueError("runner state does not match the recurrent environment batch")
    buffer = RecurrentRolloutBuffer(
        horizon,
        environment_count,
        model.observation_size,
        model.hidden_size,
        cost_count=model.cost_count,
        device=device,
    )
    model.eval()
    current = state
    for _ in range(horizon):
        observation_tensor = torch.as_tensor(current.observation, dtype=torch.float32, device=device)
        start_tensor = torch.as_tensor(current.episode_start, dtype=torch.bool, device=device)
        with torch.no_grad():
            action = model.act(
                observation_tensor,
                hidden_state=current.hidden_state,
                episode_starts=start_tensor,
                deterministic=False,
            )
        transitions = [item.step(int(selected)) for item, selected in zip(environments, action.action.tolist())]
        next_observations: list[np.ndarray] = []
        rewards: list[float] = []
        costs: list[tuple[float, float, float]] = []
        dones: list[bool] = []
        for item, (next_observation, reward, terminated, truncated, info) in zip(environments, transitions):
            done = terminated or truncated
            if done:
                next_observation, _ = item.reset()
            next_observations.append(next_observation)
            rewards.append(float(reward))
            costs.append(tuple(float(value) for value in info["costs"]))
            dones.append(done)
        buffer.add(
            observations=current.observation,
            actions=action.action.detach().cpu().numpy().astype(np.int64),
            log_probabilities=action.log_probability.detach().cpu().numpy().astype(np.float32),
            rewards=np.asarray(rewards, dtype=np.float32),
            costs=np.asarray(costs, dtype=np.float32),
            dones=np.asarray(dones, dtype=np.bool_),
            episode_starts=current.episode_start,
            reward_values=action.reward_value.detach().cpu().numpy().astype(np.float32),
            cost_values=action.cost_values.detach().cpu().numpy().astype(np.float32),
            hidden_states=current.hidden_state.detach().cpu().numpy().astype(np.float32),
        )
        current = RecurrentRunnerState(
            np.stack(next_observations),
            action.hidden_state,
            np.asarray(dones, dtype=np.bool_),
        )
    with torch.no_grad():
        bootstrap = model.forward_sequence(
            torch.as_tensor(current.observation, dtype=torch.float32, device=device).unsqueeze(1),
            hidden_state=current.hidden_state,
            episode_starts=torch.as_tensor(current.episode_start, dtype=torch.bool, device=device).unsqueeze(1),
        )
    buffer.compute_advantages(
        bootstrap.reward_values[:, 0].detach().cpu().numpy(),
        bootstrap.cost_values[:, 0].detach().cpu().numpy(),
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    return CollectedRollout(buffer, current)


def _as_environments(environment: SafeSliceGymEnv | Sequence[SafeSliceGymEnv]) -> tuple[SafeSliceGymEnv, ...]:
    environments = (environment,) if isinstance(environment, SafeSliceGymEnv) else tuple(environment)
    if not environments:
        raise ValueError("at least one recurrent rollout environment is required")
    return environments
