from modelling.baselines import EqualStaticPolicy
from modelling.contracts import EnvironmentConfig, JammerFamily
from modelling.envs.slicing_env import SafeSliceEnv
from modelling.evaluation import evaluate_paired
from modelling.traces import ExogenousTrace


def test_trace_replays_seeded_external_events() -> None:
    config = EnvironmentConfig(episode_steps=5)
    trace = ExogenousTrace.generate(config, seed=17)
    direct = SafeSliceEnv(config)
    replayed = SafeSliceEnv(config, trace=trace)
    direct_observation, _ = direct.reset(seed=17)
    replayed_observation, _ = replayed.reset(seed=999)
    for _ in range(config.episode_steps):
        action = direct.catalog.get(0)
        direct_result = direct.step(action)
        replayed_result = replayed.step(action)
        assert direct_result.info["arrivals"] == replayed_result.info["arrivals"]
        assert direct_result.info["jammer"] == replayed_result.info["jammer"]
        direct_observation = direct_result.observation
        replayed_observation = replayed_result.observation
    assert direct_observation == replayed_observation


def test_trace_json_round_trip(tmp_path) -> None:
    config = EnvironmentConfig(episode_steps=3)
    original = ExogenousTrace.generate(config, seed=8)
    trace_path = tmp_path / "trace.json"
    original.write_json(trace_path)
    assert ExogenousTrace.read_json(trace_path) == original


def test_reactive_jammer_cannot_be_misrepresented_as_fixed_trace() -> None:
    config = EnvironmentConfig(jammer_family=JammerFamily.REACTIVE)
    try:
        ExogenousTrace.generate(config, seed=0)
    except ValueError as error:
        assert "action-dependent" in str(error)
    else:
        raise AssertionError("expected action-dependent jammer trace to be rejected")


def test_paired_evaluation_is_repeatable() -> None:
    config = EnvironmentConfig(episode_steps=5)
    policy = EqualStaticPolicy(SafeSliceEnv(config).catalog)
    first = evaluate_paired(config, {"equal": policy}, [3, 4])
    second = evaluate_paired(config, {"equal": policy}, [3, 4])
    assert first == second


def test_paired_evaluation_preserves_reactive_jammer_semantics() -> None:
    config = EnvironmentConfig(episode_steps=3, jammer_family=JammerFamily.REACTIVE)
    policy = EqualStaticPolicy(SafeSliceEnv(config).catalog)
    result = evaluate_paired(config, {"equal": policy}, [6])
    assert len(result["equal"]) == 1


def test_population_trace_records_the_sampled_episode_domain() -> None:
    config = EnvironmentConfig(
        episode_steps=3,
        jammer_population=(JammerFamily.NONE, JammerFamily.CONSTANT, JammerFamily.RANDOM),
        jammer_power_range=(0.2, 0.8),
        jammer_energy_budget_range=(10.0, 50.0),
    )
    trace = ExogenousTrace.generate(config, seed=12)
    trace.validate_for(config)
    assert trace.jammer_family in config.jammer_population
    assert 0.2 <= trace.jammer_power <= 0.8
    assert 10.0 <= trace.jammer_energy_budget <= 50.0
