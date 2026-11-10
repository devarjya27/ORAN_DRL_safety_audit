"""Finite, validated action catalogue for safe near-real-time slice control."""

from __future__ import annotations

from dataclasses import dataclass

from modelling.contracts import Action, EnvironmentConfig, MitigationMode


@dataclass(frozen=True, slots=True)
class ActionCatalog:
    """Deterministic action catalogue shared by policy, shield, and adapter."""

    actions: tuple[Action, ...]
    config: EnvironmentConfig

    @classmethod
    def build(cls, config: EnvironmentConfig) -> "ActionCatalog":
        increment = config.action_increment_percent
        actions: list[Action] = []
        action_id = 0
        for embb in range(0, 101, increment):
            for urllc in range(0, 101 - embb, increment):
                mmtc = 100 - embb - urllc
                shares = (embb, urllc, mmtc)
                if not all(low <= share <= high for share, low, high in zip(shares, config.min_shares, config.max_shares)):
                    continue
                for mitigation in config.mitigation_modes:
                    actions.append(
                        Action(
                            action_id=action_id,
                            embb_percent=embb,
                            urllc_percent=urllc,
                            mmtc_percent=mmtc,
                            mitigation=MitigationMode(mitigation),
                        )
                    )
                    action_id += 1
        if not actions:
            raise ValueError("configuration produced an empty action catalogue")
        return cls(actions=tuple(actions), config=config)

    def __len__(self) -> int:
        return len(self.actions)

    def get(self, action_id: int) -> Action:
        if action_id < 0 or action_id >= len(self.actions):
            raise ValueError(f"unknown action_id {action_id}; expected [0, {len(self.actions) - 1}]")
        return self.actions[action_id]
