import json

from modelling.contracts import EnvironmentConfig
from modelling.envs.slicing_env import SafeSliceEnv
from modelling.training.train_dqn import DQNPolicy, DQNTrainingConfig, train_dqn


def test_dqn_trains_logs_validation_and_loads(tmp_path) -> None:
    config = EnvironmentConfig(episode_steps=4, jammer_energy_budget=0.0)
    artifact = tmp_path / "dqn"
    train_dqn(
        training=DQNTrainingConfig(
            total_timesteps=64,
            learning_starts=8,
            batch_size=8,
            buffer_size=128,
            eval_freq=32,
            validation_seeds=(10, 11),
            checkpoint_max_regime_urllc_cost_limit=1.0,
            progress_bar=False,
        ),
        output_path=artifact,
        config=config,
    )
    assert (tmp_path / "dqn.validation.csv").read_text(encoding="utf-8").count("\n") >= 2
    metadata = json.loads((tmp_path / "dqn.metadata.json").read_text(encoding="utf-8"))
    assert 0.0 <= metadata["best_validation_max_regime_urllc_cost"] <= 1.0
    observation, _ = SafeSliceEnv(config).reset(seed=1)
    assert DQNPolicy.load(artifact, config).act(observation) in SafeSliceEnv(config).catalog.actions
