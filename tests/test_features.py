import numpy as np

from modelling.action_catalog import ActionCatalog
from modelling.contracts import EnvironmentConfig, TrafficProfile
from modelling.envs.slicing_env import SafeSliceEnv
from modelling.features import FactorizedActionCodec, FeatureEncoder


def test_feature_encoder_is_bounded_and_semantic() -> None:
    config = EnvironmentConfig()
    env = SafeSliceEnv(config)
    observation, _ = env.reset(seed=2)
    encoded = FeatureEncoder(config, env.catalog).encode(observation)
    assert encoded.shape == (29,)
    assert np.all((0.0 <= encoded) & (encoded <= 1.0))


def test_feature_encoder_exposes_delayed_offered_load_kpms() -> None:
    config = EnvironmentConfig(
        episode_steps=8,
        traffic_profile=TrafficProfile.CYCLIC_SURGES,
        traffic_regime_steps=2,
        traffic_surge_multiplier=2.0,
        kpm_delay_steps=0,
    )
    env = SafeSliceEnv(config)
    env.reset(seed=3)
    baseline = FeatureEncoder(config, env.catalog).encode(env.step(0).observation)
    env.step(0)
    surge = FeatureEncoder(config, env.catalog).encode(env.step(0).observation)
    # Offered-load features follow region interference and precede the
    # previous-action features. The eMBB KPM doubles in the eMBB surge.
    offered_start = 14 + config.jammer_regions
    assert baseline[offered_start] == 0.5
    assert surge[offered_start] == 1.0


def test_factorized_action_codec_round_trips_every_catalogue_action() -> None:
    catalog = ActionCatalog.build(EnvironmentConfig())
    codec = FactorizedActionCodec(catalog)
    assert codec.shape == (21, 5)
    for action in catalog.actions:
        assert codec.decode(codec.encode(action.action_id)) == action.action_id
