"""Human review.

The rule from the build instructions (section 6): if confidence is below the
threshold, do NOT guess. The item goes to a reviewer with everything needed to decide
in one place: what was found, what the rule says, what the system proposed, and how
sure it was.

The reviewer's answer is final. `Reject suggestion` means "do not apply what you
proposed" and leaves the item open - it never silently keeps the original value.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..core import AnalysisResult, apply_decisions
from ..pii_detection.schema import Action
from ..rules.store import library as default_library

REVIEW_ACTIONS = (Action.KEEP.value, Action.REPLACE.value,
                  Action.REMOVE.value, Action.ASK_HUMAN.value)


@dataclass
class ReviewItem:
    id: str
    original_text: str
    masked: str
    type: str
    confidence: float
    proposed_action: str
    proposed_replacement: str | None
    rule_id: str | None
    rule_text: str
    reason: str
    ask_reason: str
    block_id: str
    start: int
    end: int

    def to_dict(self) -> dict:
        return {
            "id": self.id, "original_text": self.original_text, "masked": self.masked,
            "type": self.type, "confidence": self.confidence,
            "proposed_action": self.proposed_action,
            "proposed_replacement": self.proposed_replacement,
            "rule_id": self.rule_id, "rule_text": self.rule_text,
            "reason": self.reason, "ask_reason": self.ask_reason,
            "location": {"block_id": self.block_id, "start": self.start, "end": self.end},
            "buttons": list(REVIEW_ACTIONS) + ["REJECT_SUGGESTION"],
        }


def build_queue(analysis: AnalysisResult) -> list[ReviewItem]:
    """The items a person has to decide, in reading order."""
    lib = default_library()
    queue: list[ReviewItem] = []
    for detection in analysis.pending:
        rule = lib.rule_for(analysis.receiver, detection.type) if detection.rule_id else None
        queue.append(ReviewItem(
            id=detection.meta["id"],
            original_text=detection.original_text,
            masked=detection.meta["masked"],
            type=detection.type,
            confidence=detection.confidence,
            proposed_action=detection.meta.get("proposed_action", Action.ASK_HUMAN.value),
            proposed_replacement=detection.meta.get("proposed_replacement"),
            rule_id=detection.rule_id,
            rule_text=rule.text if rule else "No rule in the library covers this item.",
            reason=detection.reason or "",
            ask_reason=detection.meta.get("ask_reason",
                                          "no rule in the library covers this item"
                                          if not detection.rule_id else "low confidence"),
            block_id=detection.block_id, start=detection.start, end=detection.end))
    return queue


def apply_review(analysis: AnalysisResult, decisions: dict[str, str],
                 reviewer: str = "reviewer") -> dict:
    """Apply the reviewer's decisions and produce the released document."""
    clean: dict[str, str] = {}
    for item_id, action in (decisions or {}).items():
        upper = (action or "").upper()
        if upper == "REJECT_SUGGESTION":
            clean[item_id] = Action.ASK_HUMAN.value      # stays open, never kept silently
        elif upper in REVIEW_ACTIONS:
            clean[item_id] = upper
    return apply_decisions(analysis, decisions=clean, reviewer=reviewer)


def review_state(analysis: AnalysisResult, decisions: dict[str, str]) -> dict:
    """How many items are answered, for the progress line in the UI."""
    total = len(analysis.pending)
    answered = sum(1 for d in analysis.pending
                   if decisions.get(d.meta["id"]) not in (None, Action.ASK_HUMAN.value))
    return {"total": total, "answered": answered, "open": total - answered,
            "complete": answered == total}
