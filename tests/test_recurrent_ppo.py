import numpy as np
import torch

from modelling.models.recurrent_actor_critic import RecurrentActorCritic
from modelling.training.recurrent_ppo import RecurrentPPOConfig, RecurrentPPOUpdater
from modelling.training.recurrent_rollout import RecurrentRolloutBuffer


def test_recurrent_ppo_updates_actor_and_both_critics() -> None:
    generator = np.random.default_rng(7)
    torch.manual_seed(7)
    model = RecurrentActorCritic(3, 4, hidden_size=6, cost_count=2)
    buffer = RecurrentRolloutBuffer(4, 1, 3, 6, cost_count=2)
    for step in range(4):
        observations = generator.normal(size=(1, 3)).astype(np.float32)
        with torch.no_grad():
            action = model.act(torch.as_tensor(observations), deterministic=False)
        buffer.add(
            observations=observations,
            actions=action.action.numpy(),
            log_probabilities=action.log_probability.numpy(),
            rewards=np.asarray([1.0], dtype=np.float32),
            costs=np.asarray([[0.2, 0.1]], dtype=np.float32),
            dones=np.asarray([step == 3]),
            episode_starts=np.asarray([step == 0]),
            reward_values=action.reward_value.numpy(),
            cost_values=action.cost_values.numpy(),
            hidden_states=np.zeros((1, 6), dtype=np.float32),
        )
    buffer.compute_advantages(np.zeros(1, dtype=np.float32), np.zeros((1, 2), dtype=np.float32))
    original_actor = model.actor.weight.detach().clone()
    result = RecurrentPPOUpdater(
        model,
        RecurrentPPOConfig(epochs=1, sequence_length=2, sequences_per_batch=2),
    ).update(buffer, urllc_lagrange_multiplier=0.5)
    assert result.updates == 1
    assert np.isfinite(result.loss)
    assert not torch.equal(original_actor, model.actor.weight)
