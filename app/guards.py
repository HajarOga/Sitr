"""The contract: what the model is allowed to make stick.

The prompt tells the model the rules. This module is what happens when it forgets them.
Every finding the model returns passes through here before it can affect a document:

* the value must actually exist in the document (found by exact match, and if that
  fails, by a normalised match that folds Arabic-Indic digits and repeated spaces, with
  the offsets mapped back to the original characters);
* the type and the action must be from the fixed lists;
* if it claims REPLACE or REMOVE it must cite a rule **that exists, for this receiver,
  for this type** - otherwise the item is not silently applied, it goes to a person;
* when the cited rule disagrees with the model's action, **the rule wins** and the
  disagreement is written down, because the policy is the authority and the model's
  job is to find the rule, not to overrule it;
* a REPLACE must carry a substitute that survives the validators; if it does not, the
  deterministic generator supplies one and the substitution is recorded as repaired;
* anything under the confidence threshold goes to a person.

Every one of those decisions is appended to `notes`, which is what makes the dashboard
and the audit log able to say *why* an item looks the way it does.
"""
from __future__ import annotations

from . import config
from .pii_detection.schema import Action, Detection, PiiType
from .replacement.validators import is_safe_replacement

TYPES = {t.value for t in PiiType}
ACTIONS = {a.value for a in Action}

_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def _normalised(text: str) -> tuple[str, list[int]]:
    """Lower-cased, digit-folded, single-spaced text plus a map back to the original."""
    chars: list[str] = []
    origin: list[int] = []
    previous_space = False
    for index, char in enumerate(text):
        folded = char.translate(_ARABIC_DIGITS)
        if folded.isspace():
            if previous_space:
                continue
            folded, previous_space = " ", True
        else:
            previous_space = False
        chars.append(folded)
        origin.append(index)
    return "".join(chars), origin


def find_occurrences(document, value: str) -> list[tuple[str, int, int]]:
    """Every place this value appears, as (block_id, start, end) in the real text.

    An item the model reports once may appear twice in a document; each occurrence is a
    separate thing that has to be dealt with, so each one gets its own record.
    """
    needle = _normalised((value or "").strip())[0].strip()
    if not needle:
        return []
    found: list[tuple[str, int, int]] = []
    for block in document.blocks:
        haystack, origin = _normalised(block.text)
        cursor = 0
        while True:
            position = haystack.find(needle, cursor)
            if position == -1:
                break
            start = origin[position]
            end = origin[position + len(needle) - 1] + 1
            candidate = (block.id, start, end)
            if candidate not in found:
                found.append(candidate)
            cursor = position + len(needle)
    return found


def _clean(value) -> str:
    return str(value or "").strip()


def confidence_of(value) -> float:
    """A confidence the model gave us, forced into 0.0-1.0."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, number))


def check_finding(finding: dict, document, receiver: str, lib, generator) -> tuple[
        list[Detection], list[str]]:
    """Turn one raw model finding into the detections we are willing to apply."""
    notes: list[str] = []
    value = _clean(finding.get("original_text"))
    if not value:
        return [], ["a finding arrived with no original_text and was dropped"]

    pii_type = _clean(finding.get("type")).upper()
    if pii_type not in TYPES:
        return [], [f"the model returned an unknown type {pii_type!r} for {value!r}; dropped"]
    pii_type = PiiType(pii_type).value

    action = _clean(finding.get("action")).upper()
    if action not in ACTIONS:
        notes.append(f"unknown action {action or '(empty)'!r} - treated as ASK_HUMAN")
        action = Action.ASK_HUMAN.value

    confidence = confidence_of(finding.get("confidence"))
    reason = _clean(finding.get("reason")) or "The model did not give a reason."
    cited = _clean(finding.get("rule_id")) or None
    replacement = _clean(finding.get("replacement")) or None

    rule = None
    if cited:
        for candidate in lib.rules_for(receiver):
            if candidate.rule_id == cited:
                rule = candidate
                break
        if rule is None:
            notes.append(f"the model cited rule {cited!r}, which does not exist for "
                         f"{lib.receiver_label(receiver)} - asking a person instead")
            cited = None
        elif rule.pii_type != pii_type:
            notes.append(f"rule {cited} is about {rule.pii_type}, not {pii_type} - "
                         f"asking a person instead")
            rule, cited = None, None

    if rule is not None and rule.action != action:
        notes.append(f"the model proposed {action} but rule {rule.rule_id} says "
                     f"{rule.action} - the policy wins")
        action = rule.action

    if action in (Action.REPLACE.value, Action.REMOVE.value) and rule is None:
        notes.append(f"{action} proposed for {pii_type} without a valid rule - "
                     f"asking a person instead")
        action = Action.ASK_HUMAN.value

    if action != Action.REPLACE.value:
        replacement = None

    occurrences = find_occurrences(document, value)
    if not occurrences:
        return [], [f"the model reported {value!r} but it is not in the document; dropped"]

    detections: list[Detection] = []
    for block_id, start, end in occurrences:
        item = Detection(original_text=document.block(block_id).text[start:end],
                         type=pii_type, confidence=confidence, detector="llm",
                         block_id=block_id, start=start, end=end,
                         action=action, replacement=replacement,
                         rule_id=rule.rule_id if rule else None,
                         reason=reason)
        item.meta["proposed_action"] = _clean(finding.get("action")).upper() or None
        item.meta["model_confidence"] = confidence
        item.meta["rule_action"] = rule.action if rule else None
        item.meta["proposed_replacement"] = replacement

        if config.REQUIRE_MODEL and not rule and action == Action.KEEP.value:
            # "keep" with nothing to justify it is the one answer we never take on trust:
            # keeping is what leaks. A person decides instead.
            notes.append(f"the model wanted to keep {item.original_text!r} with no rule "
                         f"cited - asking a person instead")
            item.action = Action.ASK_HUMAN.value
            item.meta["ask_reason"] = "no rule was cited for a KEEP decision"

        if item.action == Action.REPLACE.value:
            ok, problems = is_safe_replacement(item.original_text,
                                               item.replacement or "", item.type)
            if not ok:
                item.replacement = generator.for_detection(item.original_text, item.type)
                notes.append(f"the model's substitute for {item.original_text!r} was not "
                             f"acceptable ({'; '.join(problems[:2]) or 'shape mismatch'}); "
                             f"the generator supplied one instead")
                item.meta["replacement_repaired"] = True

        if item.confidence < config.ASK_THRESHOLD and item.action != Action.ASK_HUMAN.value:
            item.meta["ask_reason"] = (f"the model's confidence {item.confidence:.2f} is "
                                       f"below the {config.ASK_THRESHOLD:.2f} threshold")
            item.meta["proposed_action"] = item.action
            item.action = Action.ASK_HUMAN.value
        detections.append(item)
    return detections, notes


def check_findings(findings, document, receiver: str, lib, generator) -> tuple[
        list[Detection], list[str]]:
    """All findings from one model answer, deduplicated and non-overlapping."""
    detections: list[Detection] = []
    notes: list[str] = []
    taken: dict[str, list[tuple[int, int]]] = {}

    for finding in findings if isinstance(findings, list) else []:
        if not isinstance(finding, dict):
            notes.append("a finding was not an object and was dropped")
            continue
        batch, batch_notes = check_finding(finding, document, receiver, lib, generator)
        notes.extend(batch_notes)
        for item in batch:
            spans = taken.setdefault(item.block_id, [])
            if any(item.start < end and start < item.end for start, end in spans):
                notes.append(f"{item.original_text!r} overlapped another item and was "
                             f"reported once")
                continue
            spans.append((item.start, item.end))
            detections.append(item)

    detections.sort(key=lambda d: (d.block_id, d.start))
    return detections, notes
