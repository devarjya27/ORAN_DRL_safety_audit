"""Learned next-period URLLC risk ensemble and uncertainty-aware shield.

The predictor is intentionally trained only from policy-visible observations,
the chosen catalogue action, and the resulting URLLC cost. It must never use
the simulator's hidden jammer diagnostics as an input feature.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
import json
from math import sqrt
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from torch import Tensor, nn

from modelling.action_catalog import ActionCatalog
from modelling.contracts import Action, EnvironmentConfig, Observation
from modelling.config import scenario_fingerprint
from modelling.envs.slicing_env import SafeSliceEnv
from modelling.safety.shield import SafetyDecision, ShieldConfig, ShieldResult, SafetyShield
from modelling.features import FEATURE_SCHEMA_VERSION, FeatureEncoder


@dataclass(frozen=True, slots=True)
class RiskEstimate:
    mean: float
    standard_deviation: float
    upper_confidence: float


class RiskPredictor(Protocol):
    def estimate(self, observation: Observation, action: Action) -> RiskEstimate: ...


@dataclass(frozen=True, slots=True)
class RiskSample:
    observation: Observation
    action: Action
    urllc_cost: float


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    threshold: float
    sample_count: int
    accepted_count: int
    observed_unsafe_rate: float
    accepted_fraction: float
    target_met: bool


@dataclass(frozen=True, slots=True)
class RiskArtifactMetadata:
    schema_version: int
    created_at_utc: str
    scenario_fingerprint: str
    catalogue_fingerprint: str
    ensemble: EnsembleConfig
    input_size: int
    feature_schema_version: int


def collect_risk_samples(
    config: EnvironmentConfig,
    policy: "Policy",
    *,
    seeds: list[int],
    exploration_probability: float = 0.0,
    exploration_seed: int = 0,
) -> list[RiskSample]:
    """Collect supervised labels without exposing jammer state to the predictor."""

    if not 0.0 <= exploration_probability <= 1.0:
        raise ValueError("exploration_probability must be in [0, 1]")
    generator = np.random.default_rng(exploration_seed)
    samples: list[RiskSample] = []
    for seed in seeds:
        environment = SafeSliceEnv(config)
        observation, _ = environment.reset(seed=seed)
        while True:
            actor_action = policy.act(observation)
            action = (
                environment.catalog.get(int(generator.integers(environment.action_count)))
                if generator.random() < exploration_probability
                else actor_action
            )
            result = environment.step(action)
            samples.append(RiskSample(observation, action, float(result.costs[0])))
            observation = result.observation
            if result.terminated or result.truncated:
                break
    return samples


class Policy(Protocol):
    def act(self, observation: Observation) -> Action: ...


@dataclass(frozen=True, slots=True)
class EnsembleConfig:
    members: int = 5
    hidden_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 20
    batch_size: int = 64
    uncertainty_weight: float = 2.0
    unsafe_sample_weight: float = 5.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.members < 2 or self.hidden_size <= 0 or self.learning_rate <= 0.0 or self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("invalid risk ensemble configuration")
        if self.uncertainty_weight < 0.0:
            raise ValueError("uncertainty_weight must be non-negative")
        if self.unsafe_sample_weight < 1.0:
            raise ValueError("unsafe_sample_weight must be at least one")
        if self.seed < 0:
            raise ValueError("risk ensemble seed must be non-negative")


class _RiskNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features).squeeze(-1)


class RiskEnsemble(nn.Module):
    """Bootstrap ensemble; its spread is a heuristic uncertainty signal."""

    def __init__(self, catalog: ActionCatalog, config: EnsembleConfig | None = None) -> None:
        super().__init__()
        self.catalog = catalog
        self.config = config or EnsembleConfig()
        self._mitigations = tuple(dict.fromkeys(action.mitigation for action in catalog.actions))
        self.feature_encoder = FeatureEncoder(catalog.config, catalog)
        self.input_size = self.feature_encoder.size + 3 + len(self._mitigations)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.config.seed)
            self.members = nn.ModuleList([_RiskNetwork(self.input_size, self.config.hidden_size) for _ in range(self.config.members)])

    def features(self, observation: Observation, action: Action) -> np.ndarray:
        mitigation = [float(action.mitigation is mode) for mode in self._mitigations]
        return np.asarray(
            (*self.feature_encoder.encode(observation), *(share / 100.0 for share in action.shares), *mitigation),
            dtype=np.float32,
        )

    def fit(self, samples: list[RiskSample], *, seed: int | None = None) -> list[float]:
        """Train each member on a deterministic bootstrap resample of the data."""

        if not samples:
            raise ValueError("at least one risk sample is required")
        features = np.stack([self.features(sample.observation, sample.action) for sample in samples])
        labels = np.asarray([min(1.0, max(0.0, sample.urllc_cost)) for sample in samples], dtype=np.float32)
        feature_tensor = torch.as_tensor(features)
        label_tensor = torch.as_tensor(labels)
        generator = np.random.default_rng(self.config.seed if seed is None else seed)
        losses: list[float] = []
        self.train()
        for member_index, member in enumerate(self.members):
            optimizer = torch.optim.Adam(member.parameters(), lr=self.config.learning_rate)
            bootstrap = generator.integers(0, len(samples), size=len(samples))
            final_loss = 0.0
            for _ in range(self.config.epochs):
                generator.shuffle(bootstrap)
                for start in range(0, len(bootstrap), self.config.batch_size):
                    indices = torch.as_tensor(bootstrap[start : start + self.config.batch_size], dtype=torch.long)
                    prediction = member(feature_tensor[indices])
                    selected_labels = label_tensor[indices]
                    weights = torch.where(
                        selected_labels > 0.0,
                        torch.full_like(selected_labels, self.config.unsafe_sample_weight),
                        torch.ones_like(selected_labels),
                    )
                    loss = (weights * torch.square(prediction - selected_labels)).sum() / weights.sum()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    final_loss = float(loss.detach())
            losses.append(final_loss)
        self.eval()
        return losses

    @torch.no_grad()
    def estimate(self, observation: Observation, action: Action) -> RiskEstimate:
        return self.estimate_actions(observation, (action,))[0]

    @torch.no_grad()
    def estimate_actions(self, observation: Observation, actions: tuple[Action, ...]) -> tuple[RiskEstimate, ...]:
        """Score a catalogue batch in one forward pass per ensemble member."""

        if not actions:
            return ()
        features = torch.as_tensor(np.stack([self.features(observation, action) for action in actions]))
        predictions = torch.stack([member(features) for member in self.members])
        means = predictions.mean(dim=0)
        deviations = predictions.std(dim=0, unbiased=False)
        uppers = torch.clamp(means + self.config.uncertainty_weight * deviations, max=1.0)
        return tuple(
            RiskEstimate(float(mean), float(deviation), float(upper))
            for mean, deviation, upper in zip(means, deviations, uppers)
        )


def calibrate_upper_bound_threshold(
    predictor: RiskPredictor,
    samples: list[RiskSample],
    *,
    target_unsafe_rate: float,
    unsafe_cost_threshold: float = 0.0,
    minimum_accepted_samples: int = 1,
    minimum_accepted_fraction: float = 0.0,
) -> CalibrationResult:
    """Select the least-conservative validated UCB threshold meeting a target.

    Samples whose upper bound is at most the threshold are treated as actions
    the shield would accept. The returned rate is an empirical validation
    statistic, not a guarantee beyond the sampled distribution.
    """

    if not samples:
        raise ValueError("calibration requires at least one validation sample")
    if (
        not 0.0 <= target_unsafe_rate <= 1.0
        or unsafe_cost_threshold < 0.0
        or minimum_accepted_samples <= 0
        or not 0.0 <= minimum_accepted_fraction <= 1.0
    ):
        raise ValueError("invalid calibration thresholds")
    required_accepted = max(minimum_accepted_samples, int(np.ceil(minimum_accepted_fraction * len(samples))))
    estimates = [predictor.estimate(sample.observation, sample.action).upper_confidence for sample in samples]
    candidates = sorted(set([0.0, *estimates]))
    viable: list[CalibrationResult] = []
    for threshold in candidates:
        accepted = [sample for sample, estimate in zip(samples, estimates) if estimate <= threshold]
        if len(accepted) < required_accepted:
            continue
        unsafe_rate = sum(sample.urllc_cost > unsafe_cost_threshold for sample in accepted) / len(accepted)
        if unsafe_rate <= target_unsafe_rate:
            viable.append(
                CalibrationResult(threshold, len(samples), len(accepted), unsafe_rate, len(accepted) / len(samples), True)
            )
    if viable:
        return viable[-1]
    return CalibrationResult(0.0, len(samples), 0, float("nan"), 0.0, False)


def save_risk_ensemble(ensemble: RiskEnsemble, path: Path, environment: EnvironmentConfig) -> RiskArtifactMetadata:
    """Persist a learned risk model beside compatibility metadata."""

    checkpoint_path = _checkpoint_path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": 2, "state_dict": ensemble.state_dict()}, checkpoint_path)
    metadata = RiskArtifactMetadata(
        schema_version=2,
        created_at_utc=datetime.now(UTC).isoformat(),
        scenario_fingerprint=scenario_fingerprint(environment),
        catalogue_fingerprint=_catalogue_fingerprint(ensemble.catalog),
        ensemble=ensemble.config,
        input_size=ensemble.input_size,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
    )
    _metadata_path(checkpoint_path).write_text(json.dumps(asdict(metadata), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def load_risk_ensemble(
    path: Path,
    catalog: ActionCatalog,
    environment: EnvironmentConfig,
    *,
    allow_scenario_mismatch: bool = False,
) -> RiskEnsemble:
    """Load a model with exact action contract; cross-scenario use is explicit."""

    checkpoint_path = _checkpoint_path(path)
    try:
        payload = json.loads(_metadata_path(checkpoint_path).read_text(encoding="utf-8"))
        if payload["schema_version"] != 2:
            raise ValueError("unsupported risk artifact schema")
        if payload.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
            raise ValueError("risk artifact feature schema is unsupported")
        if payload["catalogue_fingerprint"] != _catalogue_fingerprint(catalog):
            raise ValueError("risk artifact action catalogue does not match the requested environment")
        if payload["scenario_fingerprint"] != scenario_fingerprint(environment) and not allow_scenario_mismatch:
            raise ValueError("risk artifact scenario fingerprint does not match the requested environment")
        ensemble = RiskEnsemble(catalog, EnsembleConfig(**payload["ensemble"]))
        if ensemble.input_size != payload["input_size"]:
            raise ValueError("risk artifact feature contract does not match the requested action catalogue")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("schema_version") != 2:
            raise ValueError("unsupported risk artifact checkpoint")
        ensemble.load_state_dict(checkpoint["state_dict"])
        ensemble.eval()
        return ensemble
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid risk artifact: {checkpoint_path}") from error


def _checkpoint_path(path: Path) -> Path:
    return path if path.suffix == ".pt" else path.with_suffix(".pt")


def _metadata_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_suffix(".metadata.json")


def _catalogue_fingerprint(catalog: ActionCatalog) -> str:
    actions = [
        (action.action_id, action.embb_percent, action.urllc_percent, action.mmtc_percent, action.mitigation.value)
        for action in catalog.actions
    ]
    return sha256(json.dumps(actions, separators=(",", ":")).encode("utf-8")).hexdigest()


class LearnedRiskShield:
    """Use an upper-confidence risk estimate while preserving fallback contracts."""

    def __init__(
        self,
        catalog: ActionCatalog,
        environment: EnvironmentConfig,
        predictor: RiskPredictor,
        config: ShieldConfig | None = None,
    ) -> None:
        self.catalog = catalog
        self.environment = environment
        self.predictor = predictor
        self.config = config or ShieldConfig()
        self.fallback_action = SafetyShield(catalog, environment, self.config).fallback_action

    def apply(self, observation: Observation, actor_action: Action) -> ShieldResult:
        if actor_action not in self.catalog.actions:
            raise ValueError("actor_action must belong to the shield catalogue")
        estimates = self._catalogue_estimates(observation)
        actor = estimates[actor_action.action_id]
        if not observation.feature_valid or observation.feature_age_steps > self.config.max_observation_age_steps:
            fallback = estimates[self.fallback_action.action_id]
            return ShieldResult(self.fallback_action, SafetyDecision.STALE_OBSERVATION, fallback.upper_confidence, actor.upper_confidence)
        if actor.upper_confidence <= self.config.maximum_predicted_urllc_risk:
            return ShieldResult(actor_action, SafetyDecision.ACTOR, actor.upper_confidence, actor.upper_confidence)
        safe = [
            candidate
            for candidate in self.catalog.actions
            if estimates[candidate.action_id].upper_confidence <= self.config.maximum_predicted_urllc_risk
        ]
        if not safe:
            fallback = estimates[self.fallback_action.action_id]
            return ShieldResult(self.fallback_action, SafetyDecision.NO_SAFE_CANDIDATE, fallback.upper_confidence, actor.upper_confidence)
        selected = min(safe, key=lambda candidate: _projection_score(candidate, actor_action))
        selected_risk = estimates[selected.action_id].upper_confidence
        return ShieldResult(selected, SafetyDecision.PROJECTED, selected_risk, actor.upper_confidence)

    def _catalogue_estimates(self, observation: Observation) -> tuple[RiskEstimate, ...]:
        batched = getattr(self.predictor, "estimate_actions", None)
        if callable(batched):
            estimates = tuple(batched(observation, self.catalog.actions))
        else:
            estimates = tuple(self.predictor.estimate(observation, action) for action in self.catalog.actions)
        if len(estimates) != len(self.catalog.actions):
            raise RuntimeError("risk predictor returned an incomplete catalogue batch")
        return estimates


def _projection_score(candidate: Action, actor_action: Action) -> tuple[int, int, int]:
    distance = sum(abs(candidate_share - actor_share) for candidate_share, actor_share in zip(candidate.shares, actor_action.shares))
    return (distance, int(candidate.mitigation != actor_action.mitigation), candidate.action_id)
