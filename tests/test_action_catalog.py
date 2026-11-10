from modelling.action_catalog import ActionCatalog
from modelling.contracts import EnvironmentConfig


def test_catalog_actions_respect_share_bounds_and_sum() -> None:
    config = EnvironmentConfig()
    catalog = ActionCatalog.build(config)

    assert len(catalog) > 0
    assert [action.action_id for action in catalog.actions] == list(range(len(catalog)))
    for action in catalog.actions:
        assert sum(action.shares) == 100
        assert all(low <= share <= high for share, low, high in zip(action.shares, config.min_shares, config.max_shares))


def test_catalog_rejects_unknown_action_id() -> None:
    catalog = ActionCatalog.build(EnvironmentConfig())
    try:
        catalog.get(len(catalog))
    except ValueError as exc:
        assert "unknown action_id" in str(exc)
    else:
        raise AssertionError("catalog accepted an out-of-range action id")
