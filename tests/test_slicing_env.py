from dataclasses import replace

from modelling.contracts import EnvironmentConfig, JammerFamily, TrafficProfile
from modelling.envs.slicing_env import SafeSliceEnv


def _trace(config: EnvironmentConfig, seed: int, steps: int = 8) -> list[tuple[float, tuple[float, float, float], tuple[float, float, float], bool]]:
    env = SafeSliceEnv(config)
    observation, _ = env.reset(seed=seed)
    trace = [(0.0, observation.queues, observation.throughputs, observation.feature_valid)]
    for step in range(steps):
        result = env.step(step % env.action_count)
        trace.append((result.reward, result.observation.queues, result.observation.throughputs, result.observation.feature_valid))
    return trace


def test_identical_seed_and_action_trace_is_deterministic() -> None:
    config = EnvironmentConfig(jammer_family=JammerFamily.RANDOM, kpm_delay_steps=1)
    assert _trace(config, seed=17) == _trace(config, seed=17)


def test_unseeded_episode_resets_continue_deterministically_after_initial_seed() -> None:
    config = EnvironmentConfig(episode_steps=2, jammer_population=(JammerFamily.CONSTANT, JammerFamily.RANDOM))
    first = SafeSliceEnv(config)
    second = SafeSliceEnv(config)
    first.reset(seed=91)
    second.reset(seed=91)
    for _ in range(2):
        assert first.step(0) == second.step(0)
    first_observation, first_info = first.reset()
    second_observation, second_info = second.reset()
    assert first_info == second_info
    assert first_observation == second_observation
    for _ in range(2):
        assert first.step(0) == second.step(0)


def test_episode_domain_randomization_is_seeded_and_spans_training_population() -> None:
    config = EnvironmentConfig(
        episode_steps=1,
        jammer_population=(JammerFamily.NONE, JammerFamily.CONSTANT, JammerFamily.RANDOM, JammerFamily.REACTIVE),
        jammer_power_range=(0.2, 0.8),
        jammer_energy_budget_range=(10.0, 50.0),
    )
    first_pass = [SafeSliceEnv(config).reset(seed=seed)[1] for seed in range(20)]
    second_pass = [SafeSliceEnv(config).reset(seed=seed)[1] for seed in range(20)]
    assert first_pass == second_pass
    assert len({episode["jammer_family"] for episode in first_pass}) == 4
    assert all(0.2 <= float(episode["jammer_power"]) <= 0.8 for episode in first_pass)
    assert all(10.0 <= float(episode["jammer_energy_budget"]) <= 50.0 for episode in first_pass)


def test_zero_power_jammer_matches_no_jammer() -> None:
    no_jammer = EnvironmentConfig(jammer_family=JammerFamily.NONE)
    zero_power = replace(no_jammer, jammer_family=JammerFamily.CONSTANT, jammer_power=0.0)
    assert _trace(no_jammer, seed=23) == _trace(zero_power, seed=23)


def test_jammer_never_exceeds_energy_budget() -> None:
    config = EnvironmentConfig(
        jammer_family=JammerFamily.CONSTANT,
        jammer_power=0.6,
        jammer_energy_budget=1.0,
        episode_steps=10,
    )
    env = SafeSliceEnv(config)
    env.reset(seed=5)
    for _ in range(10):
        status = env.step(0).info["jammer"]
        assert status.energy_used <= config.jammer_energy_budget
        assert status.energy_remaining >= 0.0


def test_delayed_observation_is_marked_invalid_during_warmup() -> None:
    env = SafeSliceEnv(EnvironmentConfig(kpm_delay_steps=2))
    env.reset(seed=1)
    first = env.step(0).observation
    second = env.step(0).observation
    third = env.step(0).observation
    assert not first.feature_valid
    assert not second.feature_valid
    assert third.feature_valid


def test_urllc_latency_samples_are_exported_in_diagnostics() -> None:
    env = SafeSliceEnv(EnvironmentConfig(episode_steps=2, jammer_energy_budget=0.0))
    env.reset(seed=1)
    result = env.step(0)
    assert all(latency >= 0.0 for latency in result.info["urllc_latencies_ms"])


def test_region_avoidance_reduces_targeted_jammer_exposure() -> None:
    config = EnvironmentConfig(
        jammer_family=JammerFamily.CONSTANT,
        jammer_power=1.0,
        jammer_energy_budget=10.0,
        total_service_packets=30,
        traffic_arrival_rates=(0.0, 30.0, 0.0),
        channel_fading_std=0.0,
    )
    normal = SafeSliceEnv(config)
    avoiding = SafeSliceEnv(config)
    normal.reset(seed=8)
    avoiding.reset(seed=8)
    normal_action = next(action for action in normal.catalog.actions if action.mitigation.value == "normal" and action.urllc_percent == 20)
    avoid_action = next(action for action in avoiding.catalog.actions if action.shares == normal_action.shares and action.mitigation.value == "avoid_region_low")
    normal_result = normal.step(normal_action)
    avoid_result = avoiding.step(avoid_action)
    assert avoid_result.info["true_metrics"].throughputs[1] > normal_result.info["true_metrics"].throughputs[1]


def test_dynamic_traffic_exposes_distinct_slice_surge_regimes() -> None:
    config = EnvironmentConfig(
        episode_steps=8,
        traffic_profile=TrafficProfile.CYCLIC_SURGES,
        traffic_regime_steps=2,
        traffic_surge_multiplier=2.0,
    )
    env = SafeSliceEnv(config)
    env.reset(seed=3)
    regimes = []
    rates = []
    for _ in range(8):
        result = env.step(0)
        regimes.append(result.info["traffic_regime"])
        rates.append(result.info["offered_arrival_rates"])
    assert regimes == ["baseline", "baseline", "embb_surge", "embb_surge", "urllc_surge", "urllc_surge", "mmtc_surge", "mmtc_surge"]
    assert rates[2][0] == 2.0 * config.traffic_arrival_rates[0]
    assert rates[4][1] == 2.0 * config.traffic_arrival_rates[1]


def test_randomized_traffic_schedule_is_seeded_and_not_fixed_order() -> None:
    config = EnvironmentConfig(
        episode_steps=40,
        traffic_profile=TrafficProfile.CYCLIC_SURGES,
        traffic_regime_steps=5,
        traffic_randomize_regime_order=True,
        traffic_regime_jitter_steps=2,
    )

    def regimes(seed: int) -> list[str]:
        env = SafeSliceEnv(config)
        env.reset(seed=seed)
        return [str(env.step(0).info["traffic_regime"]) for _ in range(config.episode_steps)]

    assert regimes(7) == regimes(7)
    assert regimes(7) != regimes(8)


def test_observation_reports_the_action_just_applied() -> None:
    env = SafeSliceEnv(EnvironmentConfig(kpm_delay_steps=0))
    env.reset(seed=1)
    action = env.catalog.get(17)
    result = env.step(action)
    assert result.observation.previous_action_id == action.action_id
