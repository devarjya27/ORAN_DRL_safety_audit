from pathlib import Path

import pytest

from modelling.config import load_environment_config, scenario_fingerprint
from modelling.contracts import EnvironmentConfig, JammerFamily


def test_default_scenario_loads_and_is_stably_fingerprinted() -> None:
    config = load_environment_config(Path("configs/default.yaml"))
    assert config.jammer_family is JammerFamily.RANDOM
    assert scenario_fingerprint(config) == scenario_fingerprint(config)


def test_research_scenario_loads_jammer_population_and_ranges() -> None:
    config = load_environment_config(Path("configs/research_dynamic.yaml"))
    assert config.jammer_population == (
        JammerFamily.NONE,
        JammerFamily.CONSTANT,
        JammerFamily.RANDOM,
        JammerFamily.REACTIVE,
    )
    assert config.jammer_power_range == (0.35, 0.75)


def test_fingerprint_changes_when_control_semantics_change() -> None:
    assert scenario_fingerprint(EnvironmentConfig()) != scenario_fingerprint(EnvironmentConfig(jammer_power=0.4))


def test_loader_rejects_unknown_schema(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text("schema_version: 999\nenvironment: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="schema version"):
        load_environment_config(path)
