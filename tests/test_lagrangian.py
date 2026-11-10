import pytest

from modelling.contracts import EnvironmentConfig
from modelling.envs.gym_adapter import SafeSliceGymEnv
from modelling.training.lagrangian import LagrangianConfig, LagrangianController, URLLCCostPenaltyWrapper


def test_projected_dual_update_and_augmented_reward() -> None:
    controller = LagrangianController(LagrangianConfig(urllc_cost_limit=0.1, multiplier_learning_rate=2.0))
    assert controller.update(0.4) == pytest.approx(0.6)
    assert controller.penalize(1.0, 0.5) == pytest.approx(0.7)
    assert controller.update(0.0) == pytest.approx(0.4)


def test_cost_wrapper_exposes_raw_reward_and_cost() -> None:
    environment = URLLCCostPenaltyWrapper(
        SafeSliceGymEnv(EnvironmentConfig(episode_steps=2)),
        LagrangianController(LagrangianConfig(initial_multiplier=2.0)),
    )
    environment.reset(seed=4)
    _, reward, _, _, info = environment.step(0)
    assert info["urllc_cost"] == info["costs"][0]
    assert reward == pytest.approx(info["raw_reward"] - 2.0 * info["urllc_cost"])
