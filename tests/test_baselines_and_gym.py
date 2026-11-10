import numpy as np

from modelling.action_catalog import ActionCatalog
from modelling.baselines import ConservativeURLLCPolicy, DemandProportionalPolicy, EqualStaticPolicy, ValidatedStaticPolicy
from modelling.contracts import EnvironmentConfig, JammerFamily
from modelling.evaluation import evaluate_paired
from modelling.envs.gym_adapter import SafeSliceGymEnv
from modelling.envs.slicing_env import SafeSliceEnv
from modelling.safety import SafetyShield


def test_baselines_only_choose_catalogue_actions() -> None:
    config = EnvironmentConfig()
    env = SafeSliceEnv(config)
    observation, _ = env.reset(seed=7)
    catalog = ActionCatalog.build(config)
    for policy in (
        EqualStaticPolicy(catalog), ValidatedStaticPolicy(catalog),
        ConservativeURLLCPolicy(catalog), DemandProportionalPolicy(catalog),
    ):
        assert policy.act(observation) in catalog.actions


def test_gym_adapter_has_stable_shape_and_costs() -> None:
    env = SafeSliceGymEnv(EnvironmentConfig(episode_steps=2))
    observation, info = env.reset(seed=11)
    assert observation.shape == (17,)
    assert observation.dtype == np.float32
    assert info["action_count"] == env.action_space.n
    _, _, _, _, step_info = env.step(0)
    assert len(step_info["costs"]) == 3


def test_gym_adapter_reports_shielded_action() -> None:
    config = EnvironmentConfig(kpm_delay_steps=1)
    unshielded = SafeSliceGymEnv(config)
    shielded = SafeSliceGymEnv(config, shield=SafetyShield(unshielded.core.catalog, config))
    shielded.reset(seed=4)
    _, _, _, _, info = shielded.step(0)
    assert "shield_decision" in info
    _, _, _, _, stale_info = shielded.step(0)
    assert stale_info["shield_decision"] == "stale_observation"
    assert stale_info["applied_action_id"] != stale_info["actor_action_id"]


def test_paired_evaluation_records_comparable_runs() -> None:
    config = EnvironmentConfig(episode_steps=4, jammer_family=JammerFamily.RANDOM)
    catalog = ActionCatalog.build(config)
    policies = {"equal": EqualStaticPolicy(catalog), "demand": DemandProportionalPolicy(catalog)}
    results = evaluate_paired(config, policies, seeds=[3, 5])
    assert set(results) == set(policies)
    assert [summary.seed for summary in results["equal"]] == [3, 5]
    assert all(len(summary.action_ids) == config.episode_steps for summaries in results.values() for summary in summaries)
