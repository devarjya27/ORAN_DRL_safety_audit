import csv
import json

from modelling.contracts import EnvironmentConfig
from modelling.envs.slicing_env import SafeSliceEnv
from modelling.training.lagrangian import LagrangianConfig
from modelling.training.recurrent_ppo import RecurrentPPOConfig
from modelling.training.train_recurrent_ppo import (
    RecurrentTrainingConfig,
    load_recurrent_model,
    load_recurrent_policy,
    train_recurrent_ppo,
)


def test_recurrent_training_saves_loads_and_fingerprints(tmp_path) -> None:
    config = EnvironmentConfig(episode_steps=2, jammer_energy_budget=0.0)
    training = RecurrentTrainingConfig(
        updates=2,
        horizon=4,
        hidden_size=8,
        ppo=RecurrentPPOConfig(epochs=1, sequence_length=2, sequences_per_batch=2),
        constraint=LagrangianConfig(urllc_cost_limit=0.0),
        eval_every_updates=1,
        validation_seeds=(31, 32),
        max_regime_urllc_cost_limit=1.0,
    )
    artifact = tmp_path / "recurrent"
    metadata = train_recurrent_ppo(training=training, output_path=artifact, config=config)
    assert (tmp_path / "recurrent.pt").is_file()
    assert metadata.final_lagrange_multiplier is not None
    assert metadata.best_update > 0
    assert (tmp_path / "recurrent.training.csv").is_file()
    assert (tmp_path / "recurrent.validation.csv").is_file()
    with (tmp_path / "recurrent.validation.csv").open(encoding="utf-8", newline="") as handle:
        validation = list(csv.DictReader(handle))
    assert "max_regime_urllc_cost" in validation[-1]
    assert "cvar95_period_urllc_cost" in validation[-1]
    assert 0.0 <= metadata.best_validation_max_regime_urllc_cost <= 1.0
    payload = json.loads((tmp_path / "recurrent.metadata.json").read_text(encoding="utf-8"))
    assert payload["last_update"]["updates"] == 1
    model = load_recurrent_model(artifact, config)
    assert model.hidden_size == 8
    observation, _ = SafeSliceEnv(config).reset(seed=1)
    assert load_recurrent_policy(artifact, config).act(observation).action_id >= 0
