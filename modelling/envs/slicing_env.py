"""Deterministic fast environment for the first O-RAN DRL Audit research stage."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque

import numpy as np

from modelling.action_catalog import ActionCatalog
from modelling.contracts import (
    Action,
    EnvironmentConfig,
    JammerFamily,
    JammerStatus,
    MitigationMode,
    Observation,
    Slice,
    StepResult,
    TrafficProfile,
)
from modelling.domain_randomization import EpisodeJammerDomain, build_traffic_phase_schedule, sample_episode_jammer_domain
from modelling.traces import ExogenousTrace


@dataclass(slots=True)
class _Metrics:
    queues: tuple[float, float, float]
    hol_delay_ms: tuple[float, float, float]
    throughputs: tuple[float, float, float]
    delivery_rates: tuple[float, float, float]
    deadline_miss_fraction: float
    interference: float
    region_interference: tuple[float, ...]
    offered_loads: tuple[float, float, float]


class SafeSliceEnv:
    """A queue-level simulator with packet-level URLLC deadline accounting.

    The public ``reset``/``step`` surface intentionally mirrors Gymnasium.
    A Gymnasium wrapper will be added once the core invariants and baselines
    are stable. Jammer status and true metrics are diagnostics only; policy
    observations pass through the configured delay/loss/noise channel.
    """

    def __init__(self, config: EnvironmentConfig | None = None, *, trace: ExogenousTrace | None = None) -> None:
        self.config = config or EnvironmentConfig()
        if trace is not None:
            trace.validate_for(self.config)
        self.trace = trace
        self.catalog = ActionCatalog.build(self.config)
        self._arrival_rng = np.random.default_rng()
        self._channel_rng = np.random.default_rng()
        self._jammer_rng = np.random.default_rng()
        self._telemetry_rng = np.random.default_rng()
        self._noise_rng = np.random.default_rng()
        self._domain_rng = np.random.default_rng()
        self._traffic_rng = np.random.default_rng()
        self._episode_seed_rng = np.random.default_rng()
        self._step = 0
        self._embb_queue = 0.0
        self._mmtc_queue = 0.0
        self._urllc_arrivals: Deque[float] = deque()
        self._last_action = self.catalog.actions[0]
        self._jammer_energy_used = 0.0
        self._region_quality = np.ones(self.config.jammer_regions, dtype=np.float64)
        self._telemetry: list[_Metrics] = []
        self._last_observation: Observation | None = None
        self._offered_rates = self.config.traffic_arrival_rates
        self._traffic_schedule = tuple(0 for _ in range(self.config.episode_steps))
        self._episode_jammer = EpisodeJammerDomain(
            self.config.jammer_family,
            self.config.jammer_power,
            self.config.jammer_energy_budget,
        )

    @property
    def action_count(self) -> int:
        return len(self.catalog)

    def reset(self, *, seed: int | None = None) -> tuple[Observation, dict[str, object]]:
        if seed is not None:
            self._episode_seed_rng = np.random.default_rng(seed)
            episode_seed = seed
        else:
            episode_seed = int(self._episode_seed_rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
        self._reset_random_streams(episode_seed)
        self._traffic_schedule = build_traffic_phase_schedule(self.config, self._traffic_rng)
        self._episode_jammer = (
            EpisodeJammerDomain(
                self.trace.jammer_family,
                self.trace.jammer_power,
                self.trace.jammer_energy_budget,
            )
            if self.trace is not None
            else sample_episode_jammer_domain(self.config, self._domain_rng)
        )
        self._step = 0
        self._embb_queue = 0.0
        self._mmtc_queue = 0.0
        self._urllc_arrivals.clear()
        self._last_action = self.catalog.actions[0]
        self._jammer_energy_used = 0.0
        self._region_quality = np.ones(self.config.jammer_regions, dtype=np.float64)
        self._telemetry.clear()
        self._offered_rates = self.config.traffic_arrival_rates
        initial = _Metrics(
            queues=(0.0, 0.0, 0.0),
            hol_delay_ms=(0.0, 0.0, 0.0),
            throughputs=(0.0, 0.0, 0.0),
            delivery_rates=(1.0, 1.0, 1.0),
            deadline_miss_fraction=0.0,
            interference=0.0,
            region_interference=tuple(0.0 for _ in range(self.config.jammer_regions)),
            offered_loads=self._offered_rates,
        )
        self._last_observation = self._to_observation(initial, valid=True, age_steps=0)
        return self._last_observation, {
            "action_count": self.action_count,
            "jammer_family": self._episode_jammer.family.value,
            "jammer_power": self._episode_jammer.power,
            "jammer_energy_budget": self._episode_jammer.energy_budget,
            "episode_seed": episode_seed,
        }

    def _reset_random_streams(self, seed: int | None) -> None:
        """Split exogenous processes so policies can be evaluated on paired traces."""

        seed_sequence = np.random.SeedSequence(seed)
        arrival_seed, channel_seed, jammer_seed, telemetry_seed, noise_seed, domain_seed, traffic_seed = seed_sequence.spawn(7)
        self._arrival_rng = np.random.default_rng(arrival_seed)
        self._channel_rng = np.random.default_rng(channel_seed)
        self._jammer_rng = np.random.default_rng(jammer_seed)
        self._telemetry_rng = np.random.default_rng(telemetry_seed)
        self._noise_rng = np.random.default_rng(noise_seed)
        self._domain_rng = np.random.default_rng(domain_seed)
        self._traffic_rng = np.random.default_rng(traffic_seed)

    def step(self, action: int | Action) -> StepResult:
        if self._last_observation is None:
            raise RuntimeError("call reset() before step()")
        selected_action = self.catalog.get(action) if isinstance(action, int) else action
        if selected_action not in self.catalog.actions:
            raise ValueError("action must belong to this environment's catalogue")

        arrivals = self._generate_arrivals()
        if self.trace is None:
            self._advance_channel()
        else:
            self._region_quality = np.asarray(self.trace.steps[self._step].channel_quality, dtype=np.float64)
        jammer = self._select_jammer(selected_action)
        throughputs, completed_urllc, completed_latencies, expired, expired_latencies = self._serve_queues(selected_action, jammer)
        delivery = self._delivery_rates(arrivals, throughputs)
        deadline_miss_fraction = expired / max(1, arrivals[Slice.URLLC.value])
        metrics = _Metrics(
            queues=(self._embb_queue, float(len(self._urllc_arrivals)), self._mmtc_queue),
            hol_delay_ms=(0.0, self._urllc_hol_delay_ms(), 0.0),
            throughputs=throughputs,
            delivery_rates=delivery,
            deadline_miss_fraction=deadline_miss_fraction,
            interference=jammer.power if jammer.active else 0.0,
            region_interference=tuple(
                jammer.power if jammer.active and jammer.region == region else 0.0
                for region in range(self.config.jammer_regions)
            ),
            offered_loads=self._offered_rates,
        )
        self._telemetry.append(metrics)
        reward, costs = self._reward_and_costs(selected_action, metrics, jammer)
        self._last_action = selected_action
        observation = self._impaired_observation()
        self._step += 1
        terminated = self._step >= self.config.episode_steps
        info: dict[str, object] = {
            "action": selected_action,
            "jammer": jammer,
            "arrivals": arrivals,
            "completed_urllc": completed_urllc,
            "expired_urllc": expired,
            "urllc_latencies_ms": tuple((*completed_latencies, *expired_latencies)),
            "true_metrics": metrics,
            "channel_region_quality": tuple(float(quality) for quality in self._region_quality),
            "offered_arrival_rates": self._offered_rates,
            "traffic_regime": self._traffic_regime_name(max(0, self._step - 1)),
            "jammer_family": self._episode_jammer.family.value,
            "episode_jammer_power": self._episode_jammer.power,
            "episode_jammer_energy_budget": self._episode_jammer.energy_budget,
        }
        return StepResult(
            observation=observation,
            reward=reward,
            costs=costs,
            terminated=terminated,
            truncated=False,
            info=info,
        )

    def _generate_arrivals(self) -> dict[str, int]:
        if self.trace is None:
            self._offered_rates = self._current_arrival_rates()
            embb, urllc, mmtc = (int(self._arrival_rng.poisson(rate)) for rate in self._offered_rates)
        else:
            trace_step = self.trace.steps[self._step]
            self._offered_rates = trace_step.offered_arrival_rates
            embb, urllc, mmtc = trace_step.arrivals
        self._embb_queue += embb
        self._mmtc_queue += mmtc
        # Place arrivals deterministically within the period. This preserves
        # fixed-trace replay while resolving URLLC deadlines below 100 ms.
        period_start = self._step * self.config.control_period_ms
        if urllc:
            spacing = self.config.control_period_ms / urllc
            self._urllc_arrivals.extend(period_start + (index + 0.5) * spacing for index in range(urllc))
        return {Slice.EMBB.value: embb, Slice.URLLC.value: urllc, Slice.MMTC.value: mmtc}

    def _current_arrival_rates(self) -> tuple[float, float, float]:
        if self.config.traffic_profile is TrafficProfile.STATIONARY:
            return self.config.traffic_arrival_rates
        phase = self._traffic_schedule[self._step]
        rates = list(self.config.traffic_arrival_rates)
        if phase > 0:
            rates[phase - 1] *= self.config.traffic_surge_multiplier
        return tuple(rates)  # type: ignore[return-value]

    def _traffic_regime_name(self, step: int | None = None) -> str:
        if self.config.traffic_profile is TrafficProfile.STATIONARY:
            return "stationary"
        selected_step = self._step if step is None else step
        if self.trace is not None:
            return self.trace.steps[selected_step].traffic_regime
        return ("baseline", "embb_surge", "urllc_surge", "mmtc_surge")[self._traffic_schedule[selected_step]]

    def _advance_channel(self) -> None:
        """Evolve hidden per-region quality using a mean-reverting fading process."""

        innovation = self._channel_rng.normal(0.0, self.config.channel_fading_std, size=self.config.jammer_regions)
        self._region_quality = np.clip(
            (1.0 - self.config.channel_mean_reversion) * self._region_quality
            + self.config.channel_mean_reversion
            + innovation,
            0.25,
            1.25,
        )

    def _select_jammer(self, action: Action) -> JammerStatus:
        if self.trace is not None:
            event = self.trace.steps[self._step]
            if event.jammer_active:
                self._jammer_energy_used += event.jammer_power
            return JammerStatus(
                active=event.jammer_active,
                region=event.jammer_region,
                power=event.jammer_power,
                energy_used=self._jammer_energy_used,
                energy_remaining=max(0.0, self._episode_jammer.energy_budget - self._jammer_energy_used),
            )
        family = self._episode_jammer.family
        remaining = max(0.0, self._episode_jammer.energy_budget - self._jammer_energy_used)
        if family is JammerFamily.NONE or remaining <= 0.0 or self._episode_jammer.power == 0.0:
            return JammerStatus(False, None, 0.0, self._jammer_energy_used, remaining)
        if family is JammerFamily.CONSTANT:
            region = 0
        elif family is JammerFamily.RANDOM:
            region = int(self._jammer_rng.integers(self.config.jammer_regions))
        elif family is JammerFamily.SWEEP:
            region = self._step % self.config.jammer_regions
        elif family is JammerFamily.REACTIVE:
            region = 0 if action.urllc_percent >= action.embb_percent else 1 % self.config.jammer_regions
        else:  # defensive guard for future enum expansion
            raise ValueError(f"unsupported jammer family: {family}")
        power = min(self._episode_jammer.power, remaining)
        self._jammer_energy_used += power
        return JammerStatus(
            active=True,
            region=region,
            power=power,
            energy_used=self._jammer_energy_used,
            energy_remaining=max(0.0, self._episode_jammer.energy_budget - self._jammer_energy_used),
        )

    def _serve_queues(
        self,
        action: Action,
        jammer: JammerStatus,
    ) -> tuple[tuple[float, float, float], int, tuple[float, ...], int, tuple[float, ...]]:
        efficiency, urllc_boost = self._effective_service(action, jammer)
        capacities = [
            int(self.config.total_service_packets * share / 100.0 * efficiency[index])
            for index, share in enumerate(action.shares)
        ]
        served_embb = min(self._embb_queue, capacities[0])
        self._embb_queue -= served_embb
        urllc_slots = int(capacities[1] * urllc_boost)
        served_urllc = 0
        completed_latencies: list[float] = []
        expired_latencies: list[float] = []
        period_start = self._step * self.config.control_period_ms
        period_end = period_start + self.config.control_period_ms
        deadline = float(self.config.urllc_deadline_ms)
        for slot in range(max(0, urllc_slots)):
            service_time = period_start + (slot + 1) * self.config.control_period_ms / max(1, urllc_slots)
            while self._urllc_arrivals and self._urllc_arrivals[0] + deadline < service_time:
                arrival_time = self._urllc_arrivals.popleft()
                expired_latencies.append(deadline)
            if self._urllc_arrivals and self._urllc_arrivals[0] <= service_time:
                arrival_time = self._urllc_arrivals.popleft()
                completed_latencies.append(float(service_time - arrival_time))
                served_urllc += 1
        while self._urllc_arrivals and self._urllc_arrivals[0] + deadline < period_end:
            self._urllc_arrivals.popleft()
            expired_latencies.append(deadline)
        served_mmtc = min(self._mmtc_queue, capacities[2])
        self._mmtc_queue -= served_mmtc
        return (
            (float(served_embb), float(served_urllc), float(served_mmtc)),
            served_urllc,
            tuple(completed_latencies),
            len(expired_latencies),
            tuple(expired_latencies),
        )

    def _effective_service(self, action: Action, jammer: JammerStatus) -> tuple[tuple[float, float, float], float]:
        region_weights = self._region_weights(action)
        channel_efficiency = float(np.dot(region_weights, self._region_quality))
        jammer_exposure = 0.0 if not jammer.active or jammer.region is None else float(region_weights[jammer.region])
        attenuation = 1.0 - 0.65 * jammer.power * jammer_exposure
        efficiency = [max(0.05, channel_efficiency * attenuation)] * 3
        urllc_boost = 1.0
        if jammer.active:
            if action.mitigation is MitigationMode.AVOID_REGION_LOW and jammer.region == 0:
                efficiency = [value * 0.90 for value in efficiency]
            elif action.mitigation is MitigationMode.AVOID_REGION_HIGH and jammer.region == 1:
                efficiency = [value * 0.90 for value in efficiency]
        if action.mitigation is MitigationMode.URLLC_RESERVE:
            urllc_boost = 1.12
            efficiency[0] *= 0.93
        elif action.mitigation is MitigationMode.ROBUST_SCHEDULER:
            urllc_boost = 1.08
            efficiency[0] *= 0.94
            efficiency[2] *= 0.96
        return (tuple(efficiency), urllc_boost)

    def _region_weights(self, action: Action) -> np.ndarray:
        """Map mitigation profiles to a normalized frequency-region allocation."""

        weights = np.full(self.config.jammer_regions, 1.0 / self.config.jammer_regions, dtype=np.float64)
        excluded_region: int | None = None
        if action.mitigation is MitigationMode.AVOID_REGION_LOW:
            excluded_region = 0
        elif action.mitigation is MitigationMode.AVOID_REGION_HIGH and self.config.jammer_regions > 1:
            excluded_region = 1
        if excluded_region is not None and self.config.jammer_regions > 1:
            weights[excluded_region] = 0.0
            weights /= weights.sum()
        return weights

    def _delivery_rates(self, arrivals: dict[str, int], throughputs: tuple[float, float, float]) -> tuple[float, float, float]:
        rates: list[float] = []
        for slice_id, served in zip((Slice.EMBB, Slice.URLLC, Slice.MMTC), throughputs):
            generated = arrivals[slice_id.value]
            rates.append(1.0 if generated == 0 else min(1.0, served / generated))
        return tuple(rates)  # type: ignore[return-value]

    def _urllc_hol_delay_ms(self) -> float:
        if not self._urllc_arrivals:
            return 0.0
        period_end = (self._step + 1) * self.config.control_period_ms
        return max(0.0, float(period_end - self._urllc_arrivals[0]))

    def _impaired_observation(self) -> Observation:
        source_index = len(self._telemetry) - 1 - self.config.kpm_delay_steps
        if source_index < 0:
            return self._to_observation(self._telemetry[0], valid=False, age_steps=len(self._telemetry))
        if self._telemetry_rng.random() < self.config.kpm_loss_probability:
            assert self._last_observation is not None
            stale = self._last_observation
            observation = Observation(
                step=self._step,
                queues=stale.queues,
                head_of_line_delay_ms=stale.head_of_line_delay_ms,
                throughputs=stale.throughputs,
                delivery_rates=stale.delivery_rates,
                deadline_miss_fraction=stale.deadline_miss_fraction,
                interference=stale.interference,
                previous_action_id=self._last_action.action_id,
                feature_valid=False,
                feature_age_steps=stale.feature_age_steps + 1,
                region_interference=stale.region_interference,
                offered_loads=stale.offered_loads,
            )
        else:
            age = self.config.kpm_delay_steps
            observation = self._to_observation(self._telemetry[source_index], valid=True, age_steps=age)
        self._last_observation = observation
        return observation

    def _to_observation(self, metrics: _Metrics, *, valid: bool, age_steps: int) -> Observation:
        noisy_queues = tuple(max(0.0, value + self._noise_rng.normal(0.0, self.config.kpm_noise_std)) for value in metrics.queues)
        return Observation(
            step=self._step,
            queues=noisy_queues,
            head_of_line_delay_ms=metrics.hol_delay_ms,
            throughputs=metrics.throughputs,
            delivery_rates=metrics.delivery_rates,
            deadline_miss_fraction=metrics.deadline_miss_fraction,
            interference=metrics.interference,
            previous_action_id=self._last_action.action_id,
            feature_valid=valid,
            feature_age_steps=age_steps,
            region_interference=metrics.region_interference,
            offered_loads=metrics.offered_loads,
        )

    def _reward_and_costs(self, action: Action, metrics: _Metrics, jammer: JammerStatus) -> tuple[float, tuple[float, float, float]]:
        embb_utility = min(1.0, metrics.throughputs[0] / max(1.0, self._offered_rates[0]))
        mmtc_utility = metrics.delivery_rates[2]
        urllc_utility = 1.0 - metrics.deadline_miss_fraction
        churn = sum(abs(current - previous) for current, previous in zip(action.shares, self._last_action.shares)) / 200.0
        reservation = max(0.0, (action.urllc_percent - 30) / 70.0)
        reward = 0.40 * embb_utility + 0.35 * urllc_utility + 0.25 * mmtc_utility - 0.05 * churn - 0.03 * reservation
        embb_cost = max(0.0, 0.50 - embb_utility)
        mmtc_cost = max(0.0, 0.80 - mmtc_utility)
        return reward, (metrics.deadline_miss_fraction, embb_cost, mmtc_cost)
