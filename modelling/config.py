"""Versioned scenario configuration and artifact identity helpers."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import yaml

from modelling.action_catalog import ActionCatalog
from modelling.contracts import EnvironmentConfig, JammerFamily, MitigationMode, TrafficProfile


SCENARIO_SCHEMA_VERSION = 1


def load_environment_config(path: str | Path) -> EnvironmentConfig:
    """Load and validate an ``EnvironmentConfig`` from a versioned YAML file."""

    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("scenario document must be a mapping")
    if payload.get("schema_version") != SCENARIO_SCHEMA_VERSION:
        raise ValueError(f"unsupported scenario schema version: {payload.get('schema_version')!r}")
    environment = payload.get("environment")
    if not isinstance(environment, dict):
        raise ValueError("scenario document requires an environment mapping")
    values: dict[str, Any] = dict(environment)
    if "jammer_family" in values:
        values["jammer_family"] = JammerFamily(values["jammer_family"])
    if "traffic_profile" in values:
        values["traffic_profile"] = TrafficProfile(values["traffic_profile"])
    for field_name in (
        "min_shares",
        "max_shares",
        "traffic_arrival_rates",
        "mitigation_modes",
        "jammer_population",
        "jammer_power_range",
        "jammer_energy_budget_range",
    ):
        if field_name in values:
            values[field_name] = tuple(values[field_name])
    if "mitigation_modes" in values:
        values["mitigation_modes"] = tuple(MitigationMode(mode) for mode in values["mitigation_modes"])
    if "jammer_population" in values:
        values["jammer_population"] = tuple(JammerFamily(family) for family in values["jammer_population"])
    try:
        return EnvironmentConfig(**values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid environment configuration in {source}") from error


def scenario_fingerprint(config: EnvironmentConfig) -> str:
    """Hash semantics that must match among the policy, shield, and simulator."""

    catalog = ActionCatalog.build(config)
    canonical = {
        "scenario_schema_version": SCENARIO_SCHEMA_VERSION,
        "environment": _jsonable(asdict(config)),
        "catalog": [
            {"action_id": action.action_id, "shares": action.shares, "mitigation": action.mitigation.value}
            for action in catalog.actions
        ],
    }
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, (JammerFamily, MitigationMode, TrafficProfile)):
        return value.value
    return value
