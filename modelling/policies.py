"""Named non-learning policies used by runnable experiment workflows."""

from __future__ import annotations

from modelling.action_catalog import ActionCatalog
from modelling.baselines import (
    ConservativeURLLCPolicy,
    DemandProportionalPolicy,
    EqualStaticPolicy,
    JointValidatedStaticPolicy,
    ValidatedStaticPolicy,
)
from modelling.evaluation import Policy


def baseline_policy(name: str, catalog: ActionCatalog) -> Policy:
    """Build a baseline by a stable name used in result manifests and scripts."""

    policies = {
        "equal_static": EqualStaticPolicy(catalog),
        "validated_static": ValidatedStaticPolicy(catalog),
        "joint_static": JointValidatedStaticPolicy(catalog),
        "conservative_urllc": ConservativeURLLCPolicy(catalog),
        "demand_proportional": DemandProportionalPolicy(catalog),
    }
    try:
        return policies[name]
    except KeyError as error:
        raise ValueError(f"unknown baseline {name!r}; expected one of {sorted(policies)}") from error
