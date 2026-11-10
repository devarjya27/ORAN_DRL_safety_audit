"""Transparent, non-learning policies required for meaningful comparison."""

from __future__ import annotations

from dataclasses import dataclass

from modelling.action_catalog import ActionCatalog
from modelling.contracts import Action, MitigationMode, Observation


def _nearest_action(
    catalog: ActionCatalog,
    requested_shares: tuple[float, float, float],
    preferred_mitigation: MitigationMode = MitigationMode.NORMAL,
) -> Action:
    """Return a deterministic feasible catalogue action nearest to a request."""

    def score(action: Action) -> tuple[float, int, int]:
        distance = sum(abs(actual - requested) for actual, requested in zip(action.shares, requested_shares))
        mitigation_penalty = int(action.mitigation != preferred_mitigation)
        return (distance, mitigation_penalty, action.action_id)

    return min(catalog.actions, key=score)


@dataclass(frozen=True, slots=True)
class EqualStaticPolicy:
    """Static equal-like allocation projected to the feasible catalogue."""

    catalog: ActionCatalog

    def act(self, observation: Observation) -> Action:  # noqa: ARG002 - common policy protocol
        return _nearest_action(self.catalog, (100 / 3, 100 / 3, 100 / 3))


@dataclass(frozen=True, slots=True)
class ValidatedStaticPolicy:
    """Highest-return fixed allocation satisfying only the mean-cost limit."""

    catalog: ActionCatalog

    def act(self, observation: Observation) -> Action:  # noqa: ARG002 - common policy protocol
        return _nearest_action(self.catalog, (40.0, 30.0, 30.0), MitigationMode.URLLC_RESERVE)


@dataclass(frozen=True, slots=True)
class JointValidatedStaticPolicy:
    """Highest-return fixed allocation satisfying mean and conditional limits."""

    catalog: ActionCatalog

    def act(self, observation: Observation) -> Action:  # noqa: ARG002 - common policy protocol
        return _nearest_action(self.catalog, (40.0, 40.0, 20.0), MitigationMode.NORMAL)


@dataclass(frozen=True, slots=True)
class ConservativeURLLCPolicy:
    """Static allocation with a fixed URLLC reserve for safety comparison."""

    catalog: ActionCatalog
    urllc_percent: int = 50

    def act(self, observation: Observation) -> Action:  # noqa: ARG002 - common policy protocol
        remaining = 100 - self.urllc_percent
        return _nearest_action(
            self.catalog,
            (remaining * 0.60, float(self.urllc_percent), remaining * 0.40),
            MitigationMode.URLLC_RESERVE,
        )


@dataclass(frozen=True, slots=True)
class DemandProportionalPolicy:
    """Allocates shares in proportion to observed queues with an URLLC floor."""

    catalog: ActionCatalog
    urllc_floor_percent: float = 25.0

    def act(self, observation: Observation) -> Action:
        queues = observation.queues
        total = sum(queues)
        if total <= 0.0 or not observation.feature_valid:
            return _nearest_action(self.catalog, (35.0, self.urllc_floor_percent, 100.0 - 35.0 - self.urllc_floor_percent))
        shares = [100.0 * queue / total for queue in queues]
        if shares[1] < self.urllc_floor_percent:
            deficit = self.urllc_floor_percent - shares[1]
            donors = shares[0] + shares[2]
            if donors > 0.0:
                shares[0] -= deficit * shares[0] / donors
                shares[2] -= deficit * shares[2] / donors
            shares[1] = self.urllc_floor_percent
        mitigation = MitigationMode.ROBUST_SCHEDULER if observation.interference >= 0.35 else MitigationMode.NORMAL
        return _nearest_action(self.catalog, tuple(shares), mitigation)
