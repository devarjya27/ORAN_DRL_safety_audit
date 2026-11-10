import torch

from modelling.models.recurrent_actor_critic import RecurrentActorCritic


def test_recurrent_actor_critic_sequence_shapes_and_action_range() -> None:
    torch.manual_seed(3)
    model = RecurrentActorCritic(observation_size=17, action_count=12, hidden_size=8)
    observations = torch.randn(2, 4, 17)
    output = model.forward_sequence(observations)
    assert output.logits.shape == (2, 4, 12)
    assert output.reward_values.shape == (2, 4)
    assert output.cost_values.shape == (2, 4, 3)
    action = model.act(observations[:, 0], deterministic=True)
    assert torch.all((0 <= action.action) & (action.action < 12))


def test_episode_start_mask_removes_previous_episode_belief() -> None:
    torch.manual_seed(9)
    model = RecurrentActorCritic(observation_size=17, action_count=4, hidden_size=8)
    model.eval()
    first = torch.randn(1, 17)
    second = torch.randn(1, 17)
    sequence = torch.stack((first, second), dim=1)
    reset_output = model.forward_sequence(sequence, episode_starts=torch.tensor([[False, True]]))
    fresh_output = model.forward_sequence(second.unsqueeze(1), episode_starts=torch.tensor([[True]]))
    assert torch.allclose(reset_output.logits[:, 1], fresh_output.logits[:, 0])
    assert torch.allclose(reset_output.cost_values[:, 1], fresh_output.cost_values[:, 0])


def test_factorized_recurrent_actor_preserves_flat_catalogue_shape() -> None:
    model = RecurrentActorCritic(6, 15, hidden_size=8, action_factors=(3, 5))
    output = model.forward_sequence(torch.randn(2, 4, 6))
    assert output.logits.shape == (2, 4, 15)
    assert model.actor is None
    assert model.allocation_actor is not None
    assert model.mitigation_actor is not None
