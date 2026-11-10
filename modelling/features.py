"""Versioned, bounded observation features shared by training and inference."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from modelling.action_catalog import ActionCatalog
from modelling.contracts import EnvironmentConfig, Observation


FEATURE_SCHEMA_VERSION = 5


@dataclass(frozen=True, slots=True)
class FeatureEncoder:
    """Encode physical KPMs and semantic previous-action features into [0, 1]."""

    config: EnvironmentConfig
    catalog: ActionCatalog

    @property
    def size(self) -> int:
        # queues(3), HOL(3), throughput(3), delivery(3), miss/interference(2),
        # per-region interference, offered loads(3), previous shares(3), previous mitigation
        # one-hot, valid/age(2). Episode progress is deliberately excluded:
        # policies must react to KPMs instead of memorizing a regime timetable.
        return 22 + self.config.jammer_regions + len(self.config.mitigation_modes)

    def encode(self, observation: Observation) -> np.ndarray:
        previous = self.catalog.get(observation.previous_action_id)
        queue_anchors = [max(1.0, rate * 20.0) for rate in self.config.traffic_arrival_rates]
        hol_anchor = max(float(self.config.control_period_ms), float(self.config.urllc_deadline_ms))
        throughputs = [
            _bounded(value / max(1.0, rate))
            for value, rate in zip(observation.throughputs, self.config.traffic_arrival_rates)
        ]
        mitigation = [float(previous.mitigation is mode) for mode in self.config.mitigation_modes]
        offered_loads = [
            _bounded(value / max(1.0, baseline * self.config.traffic_surge_multiplier))
            for value, baseline in zip(observation.offered_loads, self.config.traffic_arrival_rates)
        ]
        age_anchor = max(1.0, float(self.config.kpm_delay_steps + 5))
        features = (
            *(_bounded(value / anchor) for value, anchor in zip(observation.queues, queue_anchors)),
            *(_bounded(value / hol_anchor) for value in observation.head_of_line_delay_ms),
            *throughputs,
            *(_bounded(value) for value in observation.delivery_rates),
            _bounded(observation.deadline_miss_fraction),
            _bounded(observation.interference),
            *(
                _bounded(value)
                for value in (
                    observation.region_interference
                    if len(observation.region_interference) == self.config.jammer_regions
                    else tuple(observation.interference / self.config.jammer_regions for _ in range(self.config.jammer_regions))
                )
            ),
            *offered_loads,
            *(share / 100.0 for share in previous.shares),
            *mitigation,
            float(observation.feature_valid),
            _bounded(observation.feature_age_steps / age_anchor),
        )
        encoded = np.asarray(features, dtype=np.float32)
        if encoded.shape != (self.size,):
            raise RuntimeError(f"feature schema produced {encoded.shape}, expected {(self.size,)}")
        return encoded


def _bounded(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


@dataclass(frozen=True, slots=True)
class FactorizedActionCodec:
    """Losslessly factor catalogue actions into allocation and mitigation indices."""

    catalog: ActionCatalog
    _allocations: tuple[tuple[int, int, int], ...] = field(init=False, repr=False)
    _mitigations: tuple[object, ...] = field(init=False, repr=False)
    _lookup: dict[tuple[tuple[int, int, int], object], int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        allocations = tuple(dict.fromkeys(action.shares for action in self.catalog.actions))
        mitigations = tuple(self.catalog.config.mitigation_modes)
        lookup = {(action.shares, action.mitigation): action.action_id for action in self.catalog.actions}
        if len(lookup) != len(allocations) * len(mitigations):
            raise ValueError("factorized action codec requires a complete allocation/mitigation product catalogue")
        object.__setattr__(self, "_allocations", allocations)
        object.__setattr__(self, "_mitigations", mitigations)
        object.__setattr__(self, "_lookup", lookup)

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self._allocations), len(self._mitigations))

    def decode(self, indices: np.ndarray | tuple[int, int] | list[int]) -> int:
        allocation_index, mitigation_index = (int(value) for value in indices)
        if not 0 <= allocation_index < len(self._allocations) or not 0 <= mitigation_index < len(self._mitigations):
            raise ValueError("factorized action index is outside the catalogue")
        return self._lookup[(self._allocations[allocation_index], self._mitigations[mitigation_index])]

    def encode(self, action_id: int) -> tuple[int, int]:
        action = self.catalog.get(action_id)
        return (self._allocations.index(action.shares), self._mitigations.index(action.mitigation))
