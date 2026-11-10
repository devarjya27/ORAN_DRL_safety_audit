"""Interpretable jammer-belief features derived only from policy-visible KPMs."""

from __future__ import annotations

from dataclasses import dataclass

from modelling.contracts import Observation, Slice


@dataclass(frozen=True, slots=True)
class JammerBeliefConfig:
    """Smoothing and freshness settings for the non-privileged estimator."""

    smoothing: float = 0.30
    activity_threshold: float = 0.20
    stale_confidence_decay: float = 0.25

    def __post_init__(self) -> None:
        if not 0.0 < self.smoothing <= 1.0 or not 0.0 <= self.activity_threshold < 1.0:
            raise ValueError("invalid jammer-belief smoothing or threshold")
        if not 0.0 <= self.stale_confidence_decay <= 1.0:
            raise ValueError("stale_confidence_decay must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class JammerBelief:
    """Belief summary suitable for an auxiliary recurrent-policy feature block."""

    interference_estimate: float
    activity_probability: float
    telemetry_confidence: float
    urllc_delivery_anomaly: float

    def vector(self) -> tuple[float, float, float, float]:
        return (
            self.interference_estimate,
            self.activity_probability,
            self.telemetry_confidence,
            self.urllc_delivery_anomaly,
        )


class JammerBeliefEstimator:
    """Stateful EWMA estimator that never reads simulator jammer diagnostics."""

    def __init__(self, config: JammerBeliefConfig | None = None) -> None:
        self.config = config or JammerBeliefConfig()
        self.reset()

    def reset(self) -> None:
        self._interference_estimate = 0.0
        self._confidence = 1.0

    def update(self, observation: Observation) -> JammerBelief:
        """Update from an observation; stale/lost KPMs reduce confidence only."""

        freshness = 1.0 if observation.feature_valid else max(
            0.0,
            1.0 - self.config.stale_confidence_decay * (observation.feature_age_steps + 1),
        )
        if observation.feature_valid:
            self._interference_estimate = (
                self.config.smoothing * min(1.0, max(0.0, observation.interference))
                + (1.0 - self.config.smoothing) * self._interference_estimate
            )
        self._confidence = freshness
        active_evidence = max(
            0.0,
            (self._interference_estimate - self.config.activity_threshold) / (1.0 - self.config.activity_threshold),
        )
        # A delivery collapse contributes evidence but is discounted under stale telemetry.
        urllc_delivery = min(1.0, max(0.0, observation.delivery_rates[Slice.URLLC.value_index]))
        anomaly = 1.0 - urllc_delivery
        probability = min(1.0, self._confidence * (0.75 * active_evidence + 0.25 * anomaly))
        return JammerBelief(self._interference_estimate, probability, self._confidence, anomaly)
