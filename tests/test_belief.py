from modelling.belief import JammerBeliefEstimator
from modelling.contracts import Observation


def _observation(*, interference: float, valid: bool = True, age: int = 0, urllc_delivery: float = 1.0) -> Observation:
    return Observation(
        step=1,
        queues=(0.0, 0.0, 0.0),
        head_of_line_delay_ms=(0.0, 0.0, 0.0),
        throughputs=(0.0, 0.0, 0.0),
        delivery_rates=(1.0, urllc_delivery, 1.0),
        deadline_miss_fraction=0.0,
        interference=interference,
        previous_action_id=0,
        feature_valid=valid,
        feature_age_steps=age,
    )


def test_belief_reacts_to_visible_interference_without_diagnostics() -> None:
    estimator = JammerBeliefEstimator()
    quiet = estimator.update(_observation(interference=0.0))
    active = estimator.update(_observation(interference=1.0, urllc_delivery=0.4))
    assert active.interference_estimate > quiet.interference_estimate
    assert active.activity_probability > quiet.activity_probability
    assert active.urllc_delivery_anomaly == 0.6


def test_stale_observation_reduces_belief_confidence() -> None:
    estimator = JammerBeliefEstimator()
    estimator.update(_observation(interference=0.8))
    stale = estimator.update(_observation(interference=0.0, valid=False, age=3))
    assert stale.telemetry_confidence == 0.0
    assert stale.interference_estimate > 0.0
