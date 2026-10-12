"""What the policy expects, derived from the policy file.

**This module is not part of the running application.** `app/core.py` never imports it,
and no decision in the dashboard comes from here. It exists for two jobs only:

1. `training/prepare_data.py` needs a target to learn from. The policy is the ground
   truth for what should happen to each item, so the target is derived from it - that is
   how a fine-tuning set is built for a policy-driven task.
2. `evaluation/` needs an expectation to score against, so "did the model get it right"
   has a definition. Using the same derivation for training and for scoring is the point:
   if the two disagreed, the numbers would mean nothing.

The rules enforced in `app/guards.py` are the same ones applied here, so a prediction
that matches an expectation is one the guards would accept - and a prediction that does
not match is exactly the kind of answer the guards have to correct.

Why this is not a hard-coded rule engine: it does not read documents, does not find
anything, and does not run in the app. It answers one question about one already-found
item: *given the policy, what should happen to a NATIONAL_ID for an EXTERNAL_VENDOR?*
The generative part - reading the document, deciding which items exist, which rule
applies in context, writing the reason and inventing the substitute - is the model's, and
is measured against this.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.pii_detection.schema import Action
from app.replacement.generator import FakeGenerator
from app.rules.store import RulesLibrary, library as default_library


@dataclass
class Expectation:
    action: str
    rule_id: str | None
    reason: str
    replacement: str | None = None
    must_not_be_visible: bool = False

    def to_record(self, value: str, pii_type: str, confidence: float = 0.95) -> dict:
        """The record shape the model is asked to produce."""
        return {
            "original_text": value,
            "type": pii_type,
            "action": self.action,
            "replacement": self.replacement,
            "confidence": confidence,
            "rule_id": self.rule_id,
            "reason": self.reason,
        }


def expectation(value: str, pii_type: str, receiver: str,
                library: RulesLibrary | None = None,
                generator: FakeGenerator | None = None) -> Expectation:
    """What the policy says should happen to this item for this receiver."""
    lib = library or default_library()
    rule = lib.rule_for(receiver, pii_type)

    if rule is None:
        # No rule covering this type for this receiver: the policy is silent, and the
        # only honest answer is that a person decides.
        return Expectation(action=Action.ASK_HUMAN.value, rule_id=None,
                           reason=f"No rule in the policy covers {pii_type} for "
                                  f"{lib.receiver_label(receiver)}; a person decides.")

    replacement = None
    if rule.action == Action.REPLACE.value:
        maker = generator or FakeGenerator(receiver=receiver)
        replacement = maker.for_detection(value, pii_type)

    return Expectation(
        action=rule.action,
        rule_id=rule.rule_id,
        reason=rule.reason,
        replacement=replacement,
        must_not_be_visible=rule.action in (Action.REPLACE.value, Action.REMOVE.value))


def expected_findings(items: list[dict], receiver: str,
                       library: RulesLibrary | None = None) -> list[dict]:
    """The full answer for one document and receiver.

    `items` are the labelled values: `[{"text": ..., "type": ...}, ...]`.
    """
    lib = library or default_library()
    generator = FakeGenerator(receiver=receiver)
    return [expectation(item["text"], item["type"], receiver, lib, generator)
            .to_record(item["text"], item["type"])
            for item in items]


def must_not_be_visible(items: list[dict], receiver: str,
                        library: RulesLibrary | None = None) -> list[dict]:
    """The labelled items this receiver must not see in the release."""
    lib = library or default_library()
    out = []
    for item in items:
        rule = lib.rule_for(receiver, item["type"])
        if rule is None or rule.action in (Action.REPLACE.value, Action.REMOVE.value):
            out.append(item)
    return out


def action_expected(pii_type: str, receiver: str,
                    library: RulesLibrary | None = None) -> str | None:
    """Just the action, for scoring an item the model decided."""
    lib = library or default_library()
    rule = lib.rule_for(receiver, pii_type)
    return rule.action if rule else Action.ASK_HUMAN.value
