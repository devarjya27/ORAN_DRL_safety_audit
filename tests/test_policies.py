import pytest

from modelling.action_catalog import ActionCatalog
from modelling.contracts import EnvironmentConfig
from modelling.policies import baseline_policy


def test_named_baseline_factory_returns_catalogue_policy() -> None:
    catalog = ActionCatalog.build(EnvironmentConfig())
    assert baseline_policy("equal_static", catalog).act  # Protocol surface exists.
    assert baseline_policy("joint_static", catalog).act(None).shares == (40, 40, 20)
    with pytest.raises(ValueError, match="unknown baseline"):
        baseline_policy("not-a-policy", catalog)
