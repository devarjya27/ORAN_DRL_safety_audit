"""Stable contracts shared by the simulator, future policy, and xApp layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final


class Slice(StrEnum):
    """Canonical slice order used by every vector in the prototype."""

    EMBB = "embb"
    URLLC = "urllc"
    MMTC = "mmtc"

    @property
    def value_index(self) -> int:
        return SLICE_ORDER.index(self)


SLICE_ORDER: Final[tuple[Slice, Slice, Slice]] = (Slice.EMBB, Slice.URLLC, Slice.MMTC)


class MitigationMode(StrEnum):
    """Coarse mitigation modes suitable for a near-real-time control loop."""

    NORMAL = "normal"
    URLLC_RESERVE = "urllc_reserve"
    AVOID_REGION_LOW = "avoid_region_low"
    AVOID_REGION_HIGH = "avoid_region_high"
    ROBUST_SCHEDULER = "robust_scheduler"


class JammerFamily(StrEnum):
    """Jammer families available in the first simulator increment."""

    NONE = "none"
    CONSTANT = "constant"
    RANDOM = "random"
    SWEEP = "sweep"
    REACTIVE = "reactive"


class TrafficProfile(StrEnum):
    """Workload evolution used to distinguish static from adaptive policies."""

    STATIONARY = "stationary"
    CYCLIC_SURGES = "cyclic_surges"


@dataclass(frozen=True, slots=True)
class Action:
    """A validated catalogue action expressed in integer PRB percentages."""

    action_id: int
    embb_percent: int
    urllc_percent: int
    mmtc_percent: int
    mitigation: MitigationMode

    def __post_init__(self) -> None:
        shares = (self.embb_percent, self.urllc_percent, self.mmtc_percent)
        if any(share < 0 or share > 100 for share in shares):
            raise ValueError(f"PRB shares must be in [0, 100], got {shares}")
        if sum(shares) != 100:
            raise ValueError(f"PRB shares must sum to 100, got {shares}")

    @property
    def shares(self) -> tuple[int, int, int]:
        return (self.embb_percent, self.urllc_percent, self.mmtc_percent)

    def share(self, slice_id: Slice) -> int:
        return self.shares[SLICE_ORDER.index(slice_id)]


@dataclass(frozen=True, slots=True)
class EnvironmentConfig:
    """Configuration for the fast, deterministic prototype environment."""

    control_period_ms: int = 100
    episode_steps: int = 200
    total_service_packets: int = 90
    action_increment_percent: int = 10
    min_shares: tuple[int, int, int] = (20, 20, 10)
    max_shares: tuple[int, int, int] = (70, 70, 60)
    traffic_arrival_rates: tuple[float, float, float] = (26.0, 8.0, 14.0)
    traffic_profile: TrafficProfile = TrafficProfile.STATIONARY
    traffic_regime_steps: int = 50
    traffic_surge_multiplier: float = 1.8
    traffic_randomize_regime_order: bool = False
    traffic_regime_jitter_steps: int = 0
    urllc_deadline_ms: int = 5
    jammer_family: JammerFamily = JammerFamily.RANDOM
    jammer_population: tuple[JammerFamily, ...] = ()
    jammer_power: float = 0.55
    jammer_power_range: tuple[float, float] | None = None
    jammer_energy_budget: float = 40.0
    jammer_energy_budget_range: tuple[float, float] | None = None
    jammer_regions: int = 2
    channel_fading_std: float = 0.08
    channel_mean_reversion: float = 0.15
    kpm_delay_steps: int = 1
    kpm_loss_probability: float = 0.0
    kpm_noise_std: float = 0.0
    mitigation_modes: tuple[MitigationMode, ...] = field(
        default=(
            MitigationMode.NORMAL,
            MitigationMode.URLLC_RESERVE,
            MitigationMode.AVOID_REGION_LOW,
            MitigationMode.AVOID_REGION_HIGH,
            MitigationMode.ROBUST_SCHEDULER,
        )
    )

    def __post_init__(self) -> None:
        if self.control_period_ms <= 0 or self.episode_steps <= 0:
            raise ValueError("control_period_ms and episode_steps must be positive")
        if self.total_service_packets < 0:
            raise ValueError("total_service_packets must be non-negative")
        if self.action_increment_percent <= 0 or 100 % self.action_increment_percent:
            raise ValueError("action_increment_percent must divide 100")
        if len(self.min_shares) != 3 or len(self.max_shares) != 3:
            raise ValueError("share bounds must contain one value per canonical slice")
        if any(low < 0 or high > 100 or low > high for low, high in zip(self.min_shares, self.max_shares)):
            raise ValueError("invalid min_shares or max_shares")
        if sum(self.min_shares) > 100 or sum(self.max_shares) < 100:
            raise ValueError("share bounds cannot form a 100-percent allocation")
        if len(self.traffic_arrival_rates) != 3 or any(rate < 0 for rate in self.traffic_arrival_rates):
            raise ValueError("traffic_arrival_rates must contain three non-negative rates")
        if self.traffic_regime_steps <= 0 or self.traffic_surge_multiplier < 1.0:
            raise ValueError("invalid traffic regime configuration")
        if self.traffic_regime_jitter_steps < 0:
            raise ValueError("traffic_regime_jitter_steps must be non-negative")
        if self.urllc_deadline_ms <= 0:
            raise ValueError("urllc_deadline_ms must be positive")
        if not 0.0 <= self.jammer_power <= 1.0 or self.jammer_energy_budget < 0:
            raise ValueError("invalid jammer power or energy budget")
        if len(set(self.jammer_population)) != len(self.jammer_population):
            raise ValueError("jammer_population must not contain duplicates")
        if self.jammer_power_range is not None:
            low, high = self.jammer_power_range
            if not 0.0 <= low <= high <= 1.0:
                raise ValueError("jammer_power_range must be ordered within [0, 1]")
        if self.jammer_energy_budget_range is not None:
            low, high = self.jammer_energy_budget_range
            if not 0.0 <= low <= high:
                raise ValueError("jammer_energy_budget_range must be non-negative and ordered")
        if self.jammer_regions < 1:
            raise ValueError("jammer_regions must be positive")
        if self.channel_fading_std < 0.0 or not 0.0 < self.channel_mean_reversion <= 1.0:
            raise ValueError("invalid channel fading configuration")
        if self.kpm_delay_steps < 0 or not 0.0 <= self.kpm_loss_probability <= 1.0:
            raise ValueError("invalid KPM impairment configuration")
        if self.kpm_noise_std < 0:
            raise ValueError("kpm_noise_std must be non-negative")

    @property
    def urllc_deadline_steps(self) -> int:
        """Deadline expressed in control periods, rounded up conservatively."""

        return max(1, -(-self.urllc_deadline_ms // self.control_period_ms))


@dataclass(frozen=True, slots=True)
class Observation:
    """What a policy receives; true simulator state is intentionally excluded."""

    step: int
    queues: tuple[float, float, float]
    head_of_line_delay_ms: tuple[float, float, float]
    throughputs: tuple[float, float, float]
    delivery_rates: tuple[float, float, float]
    deadline_miss_fraction: float
    interference: float
    previous_action_id: int
    feature_valid: bool
    feature_age_steps: int
    region_interference: tuple[float, ...] = ()
    # Delayed traffic-volume KPMs. These are measurements available to the
    # controller, not the simulator's hidden traffic-regime label.
    offered_loads: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def vector(self) -> tuple[float, ...]:
        """Stable feature ordering for the future Gymnasium/SB3 adapter."""

        return (
            *self.queues,
            *self.head_of_line_delay_ms,
            *self.throughputs,
            *self.delivery_rates,
            self.deadline_miss_fraction,
            self.interference,
            float(self.previous_action_id),
            float(self.feature_valid),
            float(self.feature_age_steps),
        )


@dataclass(frozen=True, slots=True)
class JammerStatus:
    """Diagnostic-only jammer state; it is never exposed in Observation."""

    active: bool
    region: int | None
    power: float
    energy_used: float
    energy_remaining: float


@dataclass(frozen=True, slots=True)
class StepResult:
    """Gymnasium-compatible transition payload without a runtime dependency."""

    observation: Observation
    reward: float
    costs: tuple[float, float, float]
    terminated: bool
    truncated: bool
    info: dict[str, object]
