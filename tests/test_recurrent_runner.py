from modelling.contracts import EnvironmentConfig
from modelling.envs.gym_adapter import SafeSliceGymEnv
from modelling.models.recurrent_actor_critic import RecurrentActorCritic
from modelling.training.recurrent_runner import collect_recurrent_rollout, reset_recurrent_runner


def test_runner_collects_rollout_and_marks_episode_boundaries() -> None:
    environment = SafeSliceGymEnv(EnvironmentConfig(episode_steps=2))
    model = RecurrentActorCritic(17, environment.action_space.n, hidden_size=8)
    state = reset_recurrent_runner(environment, model, seed=4)
    collected = collect_recurrent_rollout(environment, model, state, horizon=4)
    assert collected.buffer.full
    assert collected.buffer.episode_starts[:, 0].tolist() == [True, False, True, False]
    assert collected.buffer.advantages.shape == (4, 1)


def test_runner_collects_parallel_recurrent_environments() -> None:
    environments = [SafeSliceGymEnv(EnvironmentConfig(episode_steps=3)) for _ in range(3)]
    model = RecurrentActorCritic(17, environments[0].action_space.n, hidden_size=8)
    state = reset_recurrent_runner(environments, model, seed=7)
    collected = collect_recurrent_rollout(environments, model, state, horizon=4)
    assert collected.buffer.observations.shape == (4, 3, 17)
    assert collected.buffer.episode_starts[:, 0].tolist() == [True, False, False, True]
    assert collected.buffer.advantages.shape == (4, 3)
