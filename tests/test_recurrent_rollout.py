import numpy as np
import pytest

from modelling.training.recurrent_rollout import RecurrentRolloutBuffer


def _add(buffer: RecurrentRolloutBuffer, reward: float, *, done: bool = False) -> None:
    buffer.add(
        observations=np.zeros((1, 2), dtype=np.float32),
        actions=np.zeros(1, dtype=np.int64),
        log_probabilities=np.zeros(1, dtype=np.float32),
        rewards=np.asarray([reward], dtype=np.float32),
        costs=np.zeros((1, 1), dtype=np.float32),
        dones=np.asarray([done]),
        episode_starts=np.asarray([False]),
        reward_values=np.zeros(1, dtype=np.float32),
        cost_values=np.zeros((1, 1), dtype=np.float32),
        hidden_states=np.zeros((1, 3), dtype=np.float32),
    )


def test_gae_respects_terminal_boundaries_and_sequence_shapes() -> None:
    buffer = RecurrentRolloutBuffer(4, 1, 2, 3, cost_count=1)
    _add(buffer, 1.0)
    _add(buffer, 1.0, done=True)
    _add(buffer, 2.0)
    _add(buffer, 2.0)
    buffer.compute_advantages(np.zeros(1, dtype=np.float32), np.zeros((1, 1), dtype=np.float32), gamma=1.0, gae_lambda=1.0)
    assert buffer.advantages[:, 0].tolist() == pytest.approx([2.0, 1.0, 4.0, 2.0])
    batch = buffer.iterate_sequences(sequence_length=2, sequences_per_batch=2)[0]
    assert batch.observations.shape == (2, 2, 2)
    assert batch.initial_hidden_states.shape == (2, 3)


def test_rollout_rejects_partial_or_nondivisible_sequences() -> None:
    buffer = RecurrentRolloutBuffer(3, 1, 2, 3, cost_count=1)
    with pytest.raises(RuntimeError, match="fill"):
        buffer.iterate_sequences(1, 1)
    _add(buffer, 0.0)
    _add(buffer, 0.0)
    _add(buffer, 0.0)
    with pytest.raises(ValueError, match="divide"):
        buffer.iterate_sequences(2, 1)
