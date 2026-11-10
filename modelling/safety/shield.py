"""Deterministic first safety shield for O-RAN DRL Audit.

This is deliberately an interpretable pre-learning shield. It validates data
freshness, estimates near-term URLLC pressure from observable queue and HOL
signals, projects unsafe actor choices to a feasible catalogue action, and
uses a fixed fallback when the observation itself is not trustworthy. A later
cost-model ensemble will replace only the risk estimator, not the action and
fallback contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import exp
from typing import Protocol

from modelling.action_catalog import ActionCatalog
from modelling.contracts import Action, EnvironmentConfig, MitigationMode, Observation, Slice


class SafetyDecision(StrEnum):
    ACTOR = "actor"
    PROJECTED = "projected"
    STALE_OBSERVATION = "stale_observation"
    NO_SAFE_CANDIDATE = "no_safe_candidate"


@dataclass(frozen=True, slots=True)
class ShieldConfig:
    """Conservative thresholds for the deterministic shield."""

    max_observation_age_steps: int = 2
    maximum_predicted_urllc_risk: float = 0.20
    fallback_urllc_percent: int = 50
    high_interference_threshold: float = 0.35


@dataclass(frozen=True, slots=True)
class ShieldResult:
    action: Action
    decision: SafetyDecision
    predicted_risk: float
    actor_risk: float


class Policy(Protocol):
    def act(self, observation: Observation) -> Action: ...


class SafetyShield:
    """Project a proposed action onto a conservative feasible action set."""

    def __init__(
        self,
        catalog: ActionCatalog,
        environment: EnvironmentConfig,
        config: ShieldConfig | None = None,
    ) -> None:
        self.catalog = catalog
        self.environment = environment
        self.config = config or ShieldConfig()
        self.fallback_action = self._select_fallback()

    def apply(self, observation: Observation, actor_action: Action) -> ShieldResult:
        if actor_action not in self.catalog.actions:
            raise ValueError("actor_action must belong to the shield catalogue")
        actor_risk = self.estimate_urllc_risk(observation, actor_action)
        if not observation.feature_valid or observation.feature_age_steps > self.config.max_observation_age_steps:
            return ShieldResult(
                action=self.fallback_action,
                decision=SafetyDecision.STALE_OBSERVATION,
                predicted_risk=self.estimate_urllc_risk(observation, self.fallback_action),
                actor_risk=actor_risk,
            )
        if actor_risk <= self.config.maximum_predicted_urllc_risk:
            return ShieldResult(actor_action, SafetyDecision.ACTOR, actor_risk, actor_risk)
        safe_actions = [
            action
            for action in self.catalog.actions
            if self.estimate_urllc_risk(observation, action) <= self.config.maximum_predicted_urllc_risk
        ]
        if not safe_actions:
            return ShieldResult(
                action=self.fallback_action,
                decision=SafetyDecision.NO_SAFE_CANDIDATE,
                predicted_risk=self.estimate_urllc_risk(observation, self.fallback_action),
                actor_risk=actor_risk,
            )
        action = min(safe_actions, key=lambda candidate: self._projection_score(candidate, actor_action))
        return ShieldResult(
            action=action,
            decision=SafetyDecision.PROJECTED,
            predicted_risk=self.estimate_urllc_risk(observation, action),
            actor_risk=actor_risk,
        )

    def estimate_urllc_risk(self, observation: Observation, action: Action) -> float:
        """Estimate next-period deadline risk using only policy-visible data.

        The estimate intentionally trades sophistication for inspectability. It
        is replaced by an ensemble upper-confidence bound in the learned-shield
        phase, while keeping this public method for calibration comparisons.
        """

        queue_pressure = min(1.0, observation.queues[Slice.URLLC.value_index] / 20.0)
        deadline = max(1.0, float(self.environment.urllc_deadline_ms))
        hol_pressure = min(1.0, observation.head_of_line_delay_ms[Slice.URLLC.value_index] / deadline)
        interference_pressure = min(1.0, observation.interference)
        required_share = 20.0 + 25.0 * queue_pressure + 25.0 * hol_pressure + 15.0 * interference_pressure
        mitigation_credit = 0.0
        if action.mitigation is MitigationMode.URLLC_RESERVE:
            mitigation_credit = 7.0
        elif action.mitigation is MitigationMode.ROBUST_SCHEDULER:
            mitigation_credit = 5.0
        elif action.mitigation is MitigationMode.AVOID_REGION_LOW and observation.interference >= self.config.high_interference_threshold:
            mitigation_credit = 8.0
        elif action.mitigation is MitigationMode.AVOID_REGION_HIGH and observation.interference >= self.config.high_interference_threshold:
            mitigation_credit = 8.0
        margin = action.urllc_percent + mitigation_credit - required_share
        return 1.0 / (1.0 + exp(margin / 5.0))

    def _select_fallback(self) -> Action:
        eligible = [
            action
            for action in self.catalog.actions
            if action.urllc_percent >= self.config.fallback_urllc_percent
        ]
        if not eligible:
            raise ValueError("catalogue has no fallback action with the configured URLLC reserve")
        return min(
            eligible,
            key=lambda action: (
                action.urllc_percent,
                int(action.mitigation is not MitigationMode.URLLC_RESERVE),
                action.action_id,
            ),
        )

    @staticmethod
    def _projection_score(candidate: Action, actor_action: Action) -> tuple[int, int, int]:
        distance = sum(abs(candidate_share - actor_share) for candidate_share, actor_share in zip(candidate.shares, actor_action.shares))
        mitigation_change = int(candidate.mitigation != actor_action.mitigation)
        return (distance, mitigation_change, candidate.action_id)


@dataclass(frozen=True, slots=True)
class ShieldedPolicy:
    """Wrap any catalogue policy so evaluations expose shield interventions."""

    policy: Policy
    shield: SafetyShield

    def act(self, observation: Observation) -> Action:
        return self.shield.apply(observation, self.policy.act(observation)).action
