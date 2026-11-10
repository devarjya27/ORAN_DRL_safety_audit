from modelling.action_catalog import ActionCatalog
from modelling.contracts import EnvironmentConfig, MitigationMode, Observation
from modelling.safety.shield import SafetyDecision, SafetyShield


def _observation(*, valid: bool = True, age: int = 0, queue: float = 0.0, hol_delay_ms: float = 0.0, interference: float = 0.0) -> Observation:
    return Observation(
        step=1,
        queues=(0.0, queue, 0.0),
        head_of_line_delay_ms=(0.0, hol_delay_ms, 0.0),
        throughputs=(0.0, 0.0, 0.0),
        delivery_rates=(1.0, 1.0, 1.0),
        deadline_miss_fraction=0.0,
        interference=interference,
        previous_action_id=0,
        feature_valid=valid,
        feature_age_steps=age,
    )


def test_stale_observation_uses_conservative_fallback() -> None:
    environment = EnvironmentConfig()
    catalog = ActionCatalog.build(environment)
    shield = SafetyShield(catalog, environment)
    result = shield.apply(_observation(valid=False), catalog.get(0))
    assert result.decision is SafetyDecision.STALE_OBSERVATION
    assert result.action == shield.fallback_action


def test_shield_projects_high_pressure_low_urllc_action() -> None:
    environment = EnvironmentConfig()
    catalog = ActionCatalog.build(environment)
    shield = SafetyShield(catalog, environment)
    low_urllc_action = min(catalog.actions, key=lambda action: action.urllc_percent)
    result = shield.apply(_observation(queue=20.0, hol_delay_ms=100.0, interference=0.8), low_urllc_action)
    assert result.decision in (SafetyDecision.PROJECTED, SafetyDecision.NO_SAFE_CANDIDATE)
    assert result.action.urllc_percent >= low_urllc_action.urllc_percent


def test_safe_action_is_not_modified() -> None:
    environment = EnvironmentConfig()
    catalog = ActionCatalog.build(environment)
    shield = SafetyShield(catalog, environment)
    safe_action = next(action for action in catalog.actions if action.urllc_percent >= 60 and action.mitigation is MitigationMode.URLLC_RESERVE)
    result = shield.apply(_observation(), safe_action)
    assert result.decision is SafetyDecision.ACTOR
    assert result.action == safe_action
