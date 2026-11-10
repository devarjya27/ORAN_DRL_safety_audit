from dataclasses import dataclass

from modelling.action_catalog import ActionCatalog
from modelling.baselines import EqualStaticPolicy
from modelling.contracts import Action, EnvironmentConfig, Observation
from modelling.safety.learned_risk import (
    EnsembleConfig,
    LearnedRiskShield,
    RiskEnsemble,
    RiskEstimate,
    calibrate_upper_bound_threshold,
    collect_risk_samples,
    load_risk_ensemble,
    save_risk_ensemble,
)
from modelling.safety.shield import SafetyDecision


@dataclass(frozen=True)
class _ShareRiskPredictor:
    def estimate(self, observation: Observation, action: Action) -> RiskEstimate:
        risk = max(0.0, (50.0 - action.urllc_percent) / 100.0)
        return RiskEstimate(risk, 0.01, min(1.0, risk + 0.02))


def test_learned_shield_projects_using_upper_confidence() -> None:
    config = EnvironmentConfig()
    catalog = ActionCatalog.build(config)
    shield = LearnedRiskShield(catalog, config, _ShareRiskPredictor())
    observation = Observation(0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 0.0, 0.0, 0, True, 0)
    actor = min(catalog.actions, key=lambda action: action.urllc_percent)
    result = shield.apply(observation, actor)
    assert result.decision is SafetyDecision.PROJECTED
    assert result.action.urllc_percent >= 40


def test_risk_ensemble_trains_on_nonprivileged_simulator_samples() -> None:
    config = EnvironmentConfig(episode_steps=4)
    catalog = ActionCatalog.build(config)
    samples = collect_risk_samples(config, EqualStaticPolicy(catalog), seeds=[1, 2, 3])
    ensemble = RiskEnsemble(catalog, EnsembleConfig(members=2, hidden_size=8, epochs=2, batch_size=4))
    losses = ensemble.fit(samples, seed=4)
    estimate = ensemble.estimate(samples[0].observation, samples[0].action)
    assert len(losses) == 2
    assert 0.0 <= estimate.upper_confidence <= 1.0
    batched = ensemble.estimate_actions(samples[0].observation, catalog.actions[:5])
    assert len(batched) == 5
    assert batched[0] == ensemble.estimate(samples[0].observation, catalog.actions[0])


def test_risk_sample_exploration_is_seeded_and_covers_catalogue_actions() -> None:
    config = EnvironmentConfig(episode_steps=30)
    catalog = ActionCatalog.build(config)
    first = collect_risk_samples(
        config,
        EqualStaticPolicy(catalog),
        seeds=[1],
        exploration_probability=1.0,
        exploration_seed=12,
    )
    second = collect_risk_samples(
        config,
        EqualStaticPolicy(catalog),
        seeds=[1],
        exploration_probability=1.0,
        exploration_seed=12,
    )
    assert [sample.action.action_id for sample in first] == [sample.action.action_id for sample in second]
    assert len({sample.action.action_id for sample in first}) > 10


def test_risk_ensemble_seed_reproduces_initialization_and_training() -> None:
    config = EnvironmentConfig(episode_steps=3)
    catalog = ActionCatalog.build(config)
    samples = collect_risk_samples(config, EqualStaticPolicy(catalog), seeds=[1, 2])
    ensemble_config = EnsembleConfig(members=2, hidden_size=8, epochs=1, batch_size=4, seed=9)
    first = RiskEnsemble(catalog, ensemble_config)
    second = RiskEnsemble(catalog, ensemble_config)
    assert first.fit(samples) == second.fit(samples)
    assert all(
        left.equal(right)
        for left, right in zip(first.state_dict().values(), second.state_dict().values())
    )


def test_calibration_and_artifact_round_trip(tmp_path) -> None:
    config = EnvironmentConfig(episode_steps=3)
    catalog = ActionCatalog.build(config)
    samples = collect_risk_samples(config, EqualStaticPolicy(catalog), seeds=[4, 5])
    ensemble = RiskEnsemble(catalog, EnsembleConfig(members=2, hidden_size=8, epochs=1, batch_size=4))
    ensemble.fit(samples)
    calibration = calibrate_upper_bound_threshold(ensemble, samples, target_unsafe_rate=1.0)
    assert calibration.accepted_count == len(samples)
    save_risk_ensemble(ensemble, tmp_path / "risk", config)
    restored = load_risk_ensemble(tmp_path / "risk", catalog, config)
    assert restored.estimate(samples[0].observation, samples[0].action).upper_confidence >= 0.0
    heldout = EnvironmentConfig(episode_steps=3, jammer_family="sweep")
    heldout_catalog = ActionCatalog.build(heldout)
    restored_heldout = load_risk_ensemble(
        tmp_path / "risk",
        heldout_catalog,
        heldout,
        allow_scenario_mismatch=True,
    )
    assert restored_heldout.input_size == restored.input_size
