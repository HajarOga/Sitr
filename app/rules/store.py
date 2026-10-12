"""The rules library: load it, check it, look a rule up.

The rules are company policy stored as data (`rules.json`), separate from the model
and separate from the code. That separation is the point: a policy change is a file
edit, and the model can only ever *cite* a rule, never write one.

`validate()` refuses to load a broken library, because a policy that silently fails
to match is worse than one that crashes: the document would go out with the data still
in it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .. import config
from ..pii_detection.schema import Action, PiiType

VALID_TYPES = {t.value for t in PiiType}
VALID_ACTIONS = {a.value for a in Action}


class RulesError(ValueError):
    """The rules library cannot be trusted."""


@dataclass(frozen=True)
class Rule:
    rule_id: str
    receiver: str
    pii_type: str
    action: str
    reason: str
    citation: str = ""

    @property
    def text(self) -> str:
        """How the rule reads when it is shown to a reviewer."""
        return (f"{self.rule_id} - {self.pii_type} for {self.receiver}: "
                f"{self.action}. {self.reason}")


@dataclass(frozen=True)
class Receiver:
    key: str
    label: str
    purpose: str
    description: str = ""


class RulesLibrary:
    def __init__(self, payload: dict):
        self.version = payload.get("version", "unknown")
        self.note = payload.get("note", "")
        self.receivers: dict[str, Receiver] = {
            key: Receiver(key=key, label=value.get("label", key),
                          purpose=value.get("purpose", ""),
                          description=value.get("description", ""))
            for key, value in payload.get("receivers", {}).items()
        }
        self.rules: list[Rule] = [
            Rule(rule_id=r["rule_id"], receiver=r["receiver"], pii_type=r["pii_type"],
                 action=r["action"], reason=r["reason"], citation=r.get("citation", ""))
            for r in payload.get("rules", [])
        ]
        self._index = {(r.receiver, r.pii_type): r for r in self.rules}
        self.validate()

    # --------------------------------------------------------------- validation
    def validate(self) -> None:
        problems: list[str] = []
        seen: set[str] = set()
        for rule in self.rules:
            if rule.rule_id in seen:
                problems.append(f"duplicate rule id {rule.rule_id}")
            seen.add(rule.rule_id)
            if rule.receiver not in self.receivers:
                problems.append(f"{rule.rule_id}: unknown receiver {rule.receiver}")
            if rule.pii_type not in VALID_TYPES:
                problems.append(f"{rule.rule_id}: unknown type {rule.pii_type}")
            if rule.action not in VALID_ACTIONS:
                problems.append(f"{rule.rule_id}: unknown action {rule.action}")
            if not rule.reason.strip():
                problems.append(f"{rule.rule_id}: no reason given")
        # a receiver with no rules at all is allowed (the custom receiver), but we
        # want to know about it, so the app can ask a human instead of guessing
        self.receivers_without_rules = [
            key for key in self.receivers if not any(r.receiver == key for r in self.rules)]
        if problems:
            raise RulesError("rules library is invalid: " + "; ".join(problems))

    # ------------------------------------------------------------------ lookups
    def rule_for(self, receiver: str, pii_type: str) -> Rule | None:
        """The exact rule for this receiver and type, if one exists."""
        return self._index.get((receiver, pii_type))

    def rules_for(self, receiver: str) -> list[Rule]:
        return [r for r in self.rules if r.receiver == receiver]

    def receiver_label(self, key: str) -> str:
        return self.receivers[key].label if key in self.receivers else key

    def purposes(self) -> dict[str, str]:
        return {key: value.purpose for key, value in self.receivers.items()}


@lru_cache(maxsize=1)
def library(path: str | None = None) -> RulesLibrary:
    target = Path(path) if path else config.RULES_PATH
    return RulesLibrary(json.loads(Path(target).read_text(encoding="utf-8")))


def reload_library() -> RulesLibrary:
    """Used after the rules file is edited while the app is running."""
    library.cache_clear()
    return library()
