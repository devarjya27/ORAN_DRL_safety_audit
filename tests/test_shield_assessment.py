import json

from modelling.action_catalog import ActionCatalog
from modelling.baselines import EqualStaticPolicy
from modelling.contracts import EnvironmentConfig
from modelling.safety.assessment import assess_shield
from modelling.safety.shield import SafetyShield


def test_shield_assessment_reports_interventions_and_writes_json(tmp_path) -> None:
    config = EnvironmentConfig(episode_steps=4, kpm_delay_steps=1)
    catalog = ActionCatalog.build(config)
    assessment = assess_shield(config, EqualStaticPolicy(catalog), SafetyShield(catalog, config), seeds=[2, 3])
    assert assessment.periods == 8
    assert assessment.rates()["intervention_rate"] > 0.0
    path = tmp_path / "shield.json"
    assessment.write_json(path)
    assert json.loads(path.read_text(encoding="utf-8"))["rates"]["stale_fallback_rate"] > 0.0
