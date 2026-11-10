import csv
import json

import pytest

pytest.importorskip("stable_baselines3")

from modelling.contracts import EnvironmentConfig
from modelling.envs.slicing_env import SafeSliceEnv
from modelling.training import PPOPolicy, PPOTrainingConfig, train_baseline_ppo
from modelling.training.lagrangian import LagrangianConfig


def test_ppo_artifact_trains_loads_and_selects_catalogue_actions(tmp_path) -> None:
    config = EnvironmentConfig(episode_steps=4, jammer_energy_budget=0.0)
    artifact = tmp_path / "baseline"
    metadata = train_baseline_ppo(
        training=PPOTrainingConfig(
            total_timesteps=64,
            n_steps=32,
            batch_size=32,
            n_epochs=1,
            use_safety_shield=True,
            constraint=LagrangianConfig(urllc_cost_limit=0.0),
            max_regime_urllc_cost_limit=1.0,
            progress_bar=False,
            eval_freq=32,
            validation_seeds=(20, 21),
        ),
        output_path=artifact,
        config=config,
    )
    assert (tmp_path / "baseline.zip").is_file()
    assert (tmp_path / "baseline.best.zip").is_file()
    assert (tmp_path / "baseline.final.zip").is_file()
    metadata_path = tmp_path / "baseline.metadata.json"
    assert json.loads(metadata_path.read_text(encoding="utf-8"))["scenario_fingerprint"] == metadata.scenario_fingerprint
    assert metadata.final_lagrange_multiplier is not None
    assert metadata.best_validation_timestep > 0
    with (tmp_path / "baseline.validation.csv").open(encoding="utf-8", newline="") as handle:
        validation = list(csv.DictReader(handle))
    assert validation[-1]["post_training"] == "1"
    assert "cvar95_period_urllc_cost" in validation[-1]
    assert "max_regime_urllc_cost" in validation[-1]
    assert int(validation[-1]["timesteps"]) >= 64
    assert 0.0 <= metadata.best_validation_max_regime_urllc_cost <= 1.0
    policy = PPOPolicy.load(artifact, config)
    observation, _ = SafeSliceEnv(config).reset(seed=3)
    assert policy.act(observation).action_id in range(metadata.action_count)


def test_ppo_policy_rejects_artifact_for_a_different_scenario(tmp_path) -> None:
    config = EnvironmentConfig(episode_steps=4)
    artifact = tmp_path / "baseline"
    train_baseline_ppo(
        training=PPOTrainingConfig(total_timesteps=32, n_steps=32, batch_size=32, n_epochs=1, progress_bar=False),
        output_path=artifact,
        config=config,
    )
    with pytest.raises(ValueError, match="fingerprint"):
        PPOPolicy.load(artifact, EnvironmentConfig(episode_steps=4, jammer_power=0.2))
