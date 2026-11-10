"""Versioned exogenous traces for deterministic paired evaluation.

Traces contain only events outside the controller: offered traffic and jammer
emissions.  They deliberately do not include queue state, observations, or
actions, so a policy still interacts with the simulator rather than replaying
a precomputed outcome.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from modelling.contracts import EnvironmentConfig, JammerFamily, TrafficProfile
from modelling.domain_randomization import build_traffic_phase_schedule, sample_episode_jammer_domain


TRACE_SCHEMA_VERSION = 4


class ActionDependentTraceError(ValueError):
    """Raised when a selected reactive jammer cannot be frozen into a trace."""


@dataclass(frozen=True, slots=True)
class TraceStep:
    """Exogenous inputs for one controller period."""

    arrivals: tuple[int, int, int]
    jammer_active: bool
    jammer_region: int | None
    jammer_power: float
    channel_quality: tuple[float, ...]
    offered_arrival_rates: tuple[float, float, float]
    traffic_regime: str


@dataclass(frozen=True, slots=True)
class ExogenousTrace:
    """Immutable traffic/jammer realization for policy-paired evaluation."""

    schema_version: int
    seed: int
    jammer_family: JammerFamily
    jammer_power: float
    jammer_energy_budget: float
    steps: tuple[TraceStep, ...]

    @classmethod
    def generate(cls, config: EnvironmentConfig, *, seed: int) -> "ExogenousTrace":
        """Generate a trace using the same split RNG streams as ``SafeSliceEnv``.

        Reactive jammers intentionally cannot be traced because their behavior
        depends on the controller action.  Treating them as a fixed trace would
        silently turn an adaptive attacker into an open-loop one.
        """

        seed_sequence = np.random.SeedSequence(seed)
        arrival_seed, channel_seed, jammer_seed, _, _, domain_seed, traffic_seed = seed_sequence.spawn(7)
        domain = sample_episode_jammer_domain(config, np.random.default_rng(domain_seed))
        if domain.family is JammerFamily.REACTIVE:
            raise ActionDependentTraceError("reactive jammer traces are action-dependent and cannot be pre-generated")
        arrival_rng = np.random.default_rng(arrival_seed)
        channel_rng = np.random.default_rng(channel_seed)
        jammer_rng = np.random.default_rng(jammer_seed)
        schedule = build_traffic_phase_schedule(config, np.random.default_rng(traffic_seed))
        energy_used = 0.0
        channel_quality = np.ones(config.jammer_regions, dtype=np.float64)
        steps: list[TraceStep] = []
        for step in range(config.episode_steps):
            rates = list(config.traffic_arrival_rates)
            phase = schedule[step]
            if config.traffic_profile is TrafficProfile.CYCLIC_SURGES and phase > 0:
                rates[phase - 1] *= config.traffic_surge_multiplier
            arrivals = tuple(int(arrival_rng.poisson(rate)) for rate in rates)
            innovation = channel_rng.normal(0.0, config.channel_fading_std, size=config.jammer_regions)
            channel_quality = np.clip(
                (1.0 - config.channel_mean_reversion) * channel_quality + config.channel_mean_reversion + innovation,
                0.25,
                1.25,
            )
            remaining = max(0.0, domain.energy_budget - energy_used)
            active = (
                domain.family is not JammerFamily.NONE
                and domain.power > 0.0
                and remaining > 0.0
            )
            region: int | None = None
            power = 0.0
            if active:
                if domain.family is JammerFamily.CONSTANT:
                    region = 0
                elif domain.family is JammerFamily.RANDOM:
                    region = int(jammer_rng.integers(config.jammer_regions))
                elif domain.family is JammerFamily.SWEEP:
                    region = step % config.jammer_regions
                else:  # Kept defensive in case JammerFamily grows.
                    raise ValueError(f"unsupported trace jammer family: {domain.family}")
                power = min(domain.power, remaining)
                energy_used += power
            steps.append(
                TraceStep(
                    arrivals,
                    active,
                    region,
                    power,
                    tuple(float(value) for value in channel_quality),
                    tuple(float(value) for value in rates),
                    ("stationary" if config.traffic_profile is TrafficProfile.STATIONARY else
                     ("baseline", "embb_surge", "urllc_surge", "mmtc_surge")[phase]),
                )
            )
        return cls(
            schema_version=TRACE_SCHEMA_VERSION,
            seed=seed,
            jammer_family=domain.family,
            jammer_power=domain.power,
            jammer_energy_budget=domain.energy_budget,
            steps=tuple(steps),
        )

    def validate_for(self, config: EnvironmentConfig) -> None:
        if self.schema_version != TRACE_SCHEMA_VERSION:
            raise ValueError(f"unsupported trace schema version: {self.schema_version}")
        if len(self.steps) < config.episode_steps:
            raise ValueError("trace is shorter than the configured episode")
        allowed_families = config.jammer_population or (config.jammer_family,)
        if self.jammer_family not in allowed_families:
            raise ValueError("trace jammer family is outside the configured population")
        if self.jammer_family is JammerFamily.REACTIVE:
            raise ValueError("a reactive jammer cannot consume an action-independent trace")
        power_bounds = config.jammer_power_range or (config.jammer_power, config.jammer_power)
        budget_bounds = config.jammer_energy_budget_range or (
            config.jammer_energy_budget,
            config.jammer_energy_budget,
        )
        if not power_bounds[0] <= self.jammer_power <= power_bounds[1]:
            raise ValueError("trace jammer power is outside the configured domain")
        if not budget_bounds[0] <= self.jammer_energy_budget <= budget_bounds[1]:
            raise ValueError("trace jammer budget is outside the configured domain")
        for index, step in enumerate(self.steps[: config.episode_steps]):
            if len(step.arrivals) != 3 or any(value < 0 for value in step.arrivals):
                raise ValueError(f"invalid arrivals at trace step {index}")
            if step.jammer_active != (step.jammer_power > 0.0):
                raise ValueError(f"inconsistent jammer state at trace step {index}")
            if step.jammer_active and step.jammer_region is None:
                raise ValueError(f"active jammer lacks a region at trace step {index}")
            if not 0.0 <= step.jammer_power <= self.jammer_power:
                raise ValueError(f"invalid jammer power at trace step {index}")
            if len(step.channel_quality) != config.jammer_regions or any(not 0.25 <= value <= 1.25 for value in step.channel_quality):
                raise ValueError(f"invalid channel quality at trace step {index}")
            if len(step.offered_arrival_rates) != 3 or any(value < 0.0 for value in step.offered_arrival_rates):
                raise ValueError(f"invalid offered rates at trace step {index}")

    def write_json(self, path: str | Path) -> None:
        """Persist a portable trace; callers own the chosen run directory."""

        payload = {
            "schema_version": self.schema_version,
            "seed": self.seed,
            "jammer_family": self.jammer_family.value,
            "jammer_power": self.jammer_power,
            "jammer_energy_budget": self.jammer_energy_budget,
            "steps": [asdict(step) for step in self.steps],
        }
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def read_json(cls, path: str | Path) -> "ExogenousTrace":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        try:
            steps = tuple(
                TraceStep(
                    arrivals=tuple(int(value) for value in step["arrivals"]),
                    jammer_active=bool(step["jammer_active"]),
                    jammer_region=None if step["jammer_region"] is None else int(step["jammer_region"]),
                    jammer_power=float(step["jammer_power"]),
                    channel_quality=tuple(float(value) for value in step["channel_quality"]),
                    offered_arrival_rates=tuple(float(value) for value in step["offered_arrival_rates"]),
                    traffic_regime=str(step["traffic_regime"]),
                )
                for step in payload["steps"]
            )
            return cls(
                schema_version=int(payload["schema_version"]),
                seed=int(payload["seed"]),
                jammer_family=JammerFamily(payload["jammer_family"]),
                jammer_power=float(payload["jammer_power"]),
                jammer_energy_budget=float(payload["jammer_energy_budget"]),
                steps=steps,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid exogenous trace document") from error
