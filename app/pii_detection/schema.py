"""The record we build for every detected item.

This is the contract from the build instructions, section 3:

    {"original_text": ..., "type": ..., "action": ..., "replacement": ...,
     "confidence": ..., "rule_id": ..., "reason": ...}

plus the fields the application needs to write the change back into the document
(`block_id`, `start`, `end`) and the ones the evaluation needs (`detector`).

Two of these fields are deliberately NOT filled in by the detector:
  * `rule_id` and `reason` come from the rules library, never from the model, so a
    rule can always be pointed at.
  * `action` starts as a suggestion and becomes final either after human review, or
    when the confidence is high enough to auto-decide (config.ASK_THRESHOLD).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum


class PiiType(str, Enum):
    PERSON_NAME = "PERSON_NAME"
    NATIONAL_ID = "NATIONAL_ID"
    PHONE = "PHONE"
    EMAIL = "EMAIL"
    ADDRESS = "ADDRESS"
    BANK_ACCOUNT = "BANK_ACCOUNT"
    IBAN = "IBAN"
    DATE_OF_BIRTH = "DATE_OF_BIRTH"
    PASSPORT_NUMBER = "PASSPORT_NUMBER"
    CUSTOMER_ID = "CUSTOMER_ID"
    EMPLOYEE_ID = "EMPLOYEE_ID"
    COMPANY_IDENTIFIER = "COMPANY_IDENTIFIER"


class Action(str, Enum):
    KEEP = "KEEP"
    REPLACE = "REPLACE"
    REMOVE = "REMOVE"
    ASK_HUMAN = "ASK_HUMAN"


REQUIRED_KEYS = ("original_text", "type", "action", "replacement",
                 "confidence", "rule_id", "reason")


@dataclass
class Detection:
    original_text: str
    type: str                       # PiiType value
    confidence: float
    detector: str                   # pattern | keyword | name | model
    block_id: str = "b0"
    start: int = 0
    end: int = 0
    action: str = Action.ASK_HUMAN.value
    replacement: str | None = None
    rule_id: str | None = None
    reason: str | None = None
    reviewer: str | None = None     # who decided, when a person decided
    meta: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- the contract
    def to_record(self) -> dict:
        """Exactly the keys required by the build instructions, in that order."""
        return {
            "original_text": self.original_text,
            "type": self.type,
            "action": self.action,
            "replacement": self.replacement,
            "confidence": round(float(self.confidence), 2),
            "rule_id": self.rule_id,
            "reason": self.reason,
        }

    def to_full(self) -> dict:
        """Everything, for the UI and the audit log."""
        return asdict(self)

    def is_schema_valid(self) -> bool:
        """Used by the structured-output metric: is this record well formed?"""
        if any(key not in self.to_record() for key in REQUIRED_KEYS):
            return False
        if self.type not in {t.value for t in PiiType}:
            return False
        if self.action not in {a.value for a in Action}:
            return False
        if not 0.0 <= float(self.confidence) <= 1.0:
            return False
        if self.action in (Action.REPLACE.value, Action.REMOVE.value) and not self.rule_id:
            return False               # a removal or a replacement must cite a rule
        # ASK_HUMAN may legitimately have no rule: "no rule found" is a valid answer,
        # and inventing one is exactly what the system must not do.
        if self.action == Action.REPLACE.value and not self.replacement:
            return False               # replace must say what it is replaced with
        if not (self.reason or "").strip():
            return False               # every record carries a reason, rule or not
        return True
