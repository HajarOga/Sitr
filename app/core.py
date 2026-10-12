"""The pipeline - and in this project the model is inside it, not beside it.

    Document → policy into the prompt → model reads the document and decides
             → guards enforce the contract → second look at what it was unsure about
             → human review → document generation → audit

Compare that with the first prototype, where detection and decision were rules and the
model was an optional helper. Here there is no rules-only path: `analyze()` calls Qwen,
and if Qwen is not installed or answers with something unusable, the call raises. That is
deliberate - a silent fallback would mean the system reports "AI" while a regex does the
work.

What the code still owns, and always will:

* which rules exist and which receiver they belong to (the model gets them in the prompt
  and may not invent any - `app/guards.py`);
* whether a substitute is safe to put in a document (shape, checksum, obviously fake);
* the decision to ask a person, and the rule that an unanswered item leaves the release;
* the mask and the hash that go into the audit log instead of the value.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from . import config, guards
from .document_processing.document import Document
from .llm import engine, prompts
from .pii_detection.schema import Action, Detection
from .replacement.generator import FakeGenerator
from .rules.store import RulesLibrary, library as default_library


class ModelAnswerError(RuntimeError):
    """The model is installed but did not give us anything we could use."""


def mask_value(value: str) -> str:
    """A readable but non-identifying rendering: Ahmed Mohammed -> A***d M*******d."""
    def mask(word: str) -> str:
        if len(word) <= 2:
            return word[0] + "*" if word else ""
        return word[0] + "*" * (len(word) - 2) + word[-1]
    return " ".join(mask(part) for part in value.split())


def value_hash(value: str) -> str:
    return hashlib.sha256(value.strip().encode("utf-8")).hexdigest()[:12]


@dataclass
class AnalysisResult:
    document: Document
    receiver: str
    purpose: str
    detections: list[Detection]
    generator: FakeGenerator
    rules_version: str = ""
    notes: list[str] = field(default_factory=list)
    model: str = ""
    calls: list[dict] = field(default_factory=list)
    summary_text: str | None = None
    processing_seconds: float = 0.0
    model_load_seconds: float = 0.0
    model_generation_seconds: float = 0.0
    json_validation_seconds: float = 0.0
    guard_seconds: float = 0.0
    summary_seconds: float = 0.0

    @property
    def pending(self) -> list[Detection]:
        return [d for d in self.detections if d.action == Action.ASK_HUMAN.value]

    def records(self) -> list[dict]:
        return [d.to_record() for d in self.detections]

    def summary(self) -> dict:
        counts = {action.value: 0 for action in Action}
        for detection in self.detections:
            counts[detection.action] += 1
        seconds = round(sum(c.get("seconds", 0.0) for c in self.calls), 2)
        return {
            "receiver": self.receiver,
            "purpose": self.purpose,
            "items": len(self.detections),
            "by_action": counts,
            "needs_review": len(self.pending),
            "rules_version": self.rules_version,
            "blocked": bool(self.pending),
            "model": self.model,
            "model_seconds": seconds,
            "model_calls": len(self.calls),
            "note": ("The release can be generated, but items marked ASK_HUMAN are left "
                     "out until a person decides." if self.pending else
                     "The model decided every item from the policy."),
        }


def document_text(document: Document) -> str:
    return "\n".join(block.text for block in document.blocks if block.text.strip())


def chunks(text: str, size: int | None = None,
           overlap_lines: int | None = None) -> list[str]:
    """Split on line boundaries, so a value is never cut in half by accident."""
    size = size or config.MAX_DOC_CHARS
    overlap_lines = config.CHUNK_OVERLAP_LINES if overlap_lines is None else overlap_lines
    if len(text) <= size:
        return [text]
    lines = text.split("\n")
    out: list[str] = []
    current: list[str] = []
    length = 0
    for line in lines:
        if length + len(line) + 1 > size and current:
            out.append("\n".join(current))
            current = current[-overlap_lines:] if overlap_lines else []
            length = sum(len(part) + 1 for part in current)
        current.append(line)
        length += len(line) + 1
    if current:
        out.append("\n".join(current))
    return out


def _rules_for_prompt(lib: RulesLibrary, receiver: str) -> list[dict]:
    return [{"rule_id": r.rule_id, "type": r.pii_type, "action": r.action,
             "reason": r.reason} for r in lib.rules_for(receiver)]


def _context_for(document: Document, detection: Detection, width: int = 140) -> str:
    text = document.block(detection.block_id).text
    start = max(0, detection.start - width)
    end = min(len(text), detection.end + width)
    return ("…" if start else "") + text[start:end] + ("…" if end < len(text) else "")


def _second_look(detections: list[Detection], document: Document, receiver: str,
                 lib: RulesLibrary, notes: list[str]) -> None:
    """Ask the model again about the items it was unsure of, one batch at a time."""
    uncertain = [d for d in detections
                 if d.action == Action.ASK_HUMAN.value
                 and d.confidence >= config.CONFIDENCE_FLOOR]
    if not uncertain:
        return

    for start in range(0, len(uncertain), 8):
        batch = uncertain[start:start + 8]
        payload_items = [{
            "id": d.meta.get("id"),
            "value": d.original_text,
            "type": d.type,
            "proposed_action": d.meta.get("proposed_action") or d.action,
            "confidence": round(d.confidence, 2),
            "text_around_it": _context_for(document, d),
            "why_questioned": d.meta.get("ask_reason", "low confidence"),
        } for d in batch]

        payload, meta = engine.generate_json(
            prompts.adjudicate_messages(payload_items, lib.receiver_label(receiver),
                                        _rules_for_prompt(lib, receiver)),
            purpose="adjudicate", max_new_tokens=900)
        if not answered(payload, "decisions"):
            notes.append("the second look did not return usable JSON; those items stay "
                         "with a person")
            continue

        by_id = {d.meta.get("id"): d for d in batch}
        for decision in items_from(payload, "decisions"):
            if not isinstance(decision, dict):
                continue
            item = by_id.get(str(decision.get("id")))
            if item is None:
                continue
            item.meta["second_look"] = {k: decision.get(k) for k in
                                        ("action", "rule_id", "confidence", "reason")}
            rule_id = str(decision.get("rule_id") or "").strip()
            rule = next((r for r in lib.rules_for(receiver) if r.rule_id == rule_id), None)
            confidence = guards.confidence_of(decision.get("confidence"))
            if rule is None or rule.pii_type != item.type:
                item.meta["second_look_note"] = ("no valid rule was cited, so the question "
                                                 "stays with a person")
                continue
            if confidence < config.ASK_THRESHOLD:
                item.meta["second_look_note"] = (f"still not confident enough "
                                                 f"({confidence:.2f})")
                continue
            # The policy decides the action; the model only has to point at the right
            # rule and be sure enough about it.
            item.action = rule.action
            item.rule_id = rule.rule_id
            item.confidence = round(min(config.ADJUDICATED_CEILING, confidence), 2)
            item.reason = (str(decision.get("reason") or "").strip()
                           or rule.reason)
            item.meta["settled_by"] = "second look"
            if item.action == Action.REPLACE.value:
                item.replacement = item.replacement or item.meta.get("proposed_replacement")
            if item.action != Action.REPLACE.value:
                item.replacement = None
            notes.append(f"second look settled {item.original_text!r} as "
                         f"{item.action} under {rule.rule_id}")


def items_from(payload, key: str) -> list[dict]:
    """Read a list of records out of whatever the model actually answered.

    The prompt asks for `{"findings": [...]}`, and that is what usually comes back - but the
    model sometimes answers with a bare `[...]` array. Accepting only the dictionary form
    throws away a correct answer, and a thrown-away answer means those values are still in
    the released document. So both shapes are read, and anything else counts as no answer.

    The distinction that matters: `{"findings": []}` is a valid answer meaning "nothing to
    redact", while `None` or prose is a failure. Both are handled as such.
    """
    if isinstance(payload, dict):
        candidate = payload.get(key)
    elif isinstance(payload, list):
        candidate = payload
    else:
        return []
    if not isinstance(candidate, list):
        return []
    return [item for item in candidate if isinstance(item, dict)]


def answered(payload, key: str) -> bool:
    """True when the answer has the right *shape*, even if the list is empty."""
    if isinstance(payload, list):
        return True
    return isinstance(payload, dict) and isinstance(payload.get(key), list)


def analyze(document: Document, receiver: str, purpose: str = "",
            library: RulesLibrary | None = None, *,
            progress_callback: Callable[[dict[str, Any]], None] | None = None,
            include_summary: bool = True) -> AnalysisResult:
    """The model reads the document and proposes every action. Nothing else does."""
    started = time.perf_counter()

    def report(stage: str, message: str, **details: Any) -> None:
        if progress_callback is not None:
            progress_callback({
                "stage": stage, "message": message,
                "elapsed_seconds": round(time.perf_counter() - started, 1),
                **details})

    lib = library or default_library()
    if receiver not in lib.receivers:
        raise KeyError(f"unknown receiver: {receiver}")

    analysis_calls_start = len(engine.calls())
    engine.ensure_installed()          # mandatory: fail loudly, never fall back
    purpose = purpose or lib.receivers[receiver].purpose
    label = lib.receiver_label(receiver)
    rules = _rules_for_prompt(lib, receiver)
    text = document_text(document)
    notes: list[str] = []
    findings: list[dict] = []
    parsed_any = False
    meta: dict = {}

    parts = chunks(text)
    model_was_loaded_at_start = engine.loaded_info().get("loaded", False)
    report("prepared", f"Document prepared as {len(parts)} analysis "
                       f"{'chunk' if len(parts) == 1 else 'chunks'}.",
           completed_chunks=0, total_chunks=len(parts), document_chars=len(text))
    if len(parts) > 1:
        notes.append(f"the document was analysed in {len(parts)} parts "
                     f"({len(text)} characters)")
    for index, part in enumerate(parts, start=1):
        loaded_before = engine.loaded_info().get("loaded", False)
        report("loading_model" if not loaded_before and engine.provider() == "local"
               else "generating",
               ("Loading the local model and analyzing chunk "
                if not loaded_before and engine.provider() == "local"
                else "Analyzing document chunk ") + f"{index}/{len(parts)}.",
               completed_chunks=index - 1, total_chunks=len(parts),
               current_chunk=index)
        chunk_calls_start = len(engine.calls())
        payload, meta = engine.generate_json(
            prompts.analyze_messages(part, label, purpose, rules),
            purpose=f"analyze part {index}/{len(parts)}")
        new_calls = engine.calls()[chunk_calls_start:]
        report("chunk_complete",
               f"Finished analysis chunk {index}/{len(parts)} "
               f"({sum(int(call.get('tokens_out') or 0) for call in new_calls)} output "
               f"tokens; {meta.get('attempts', 1)} JSON attempt(s)).",
               completed_chunks=index, total_chunks=len(parts), current_chunk=index,
               attempts=meta.get("attempts", 1),
               generation_seconds=meta.get("generation_seconds", 0.0),
               json_validation_seconds=meta.get("json_validation_seconds", 0.0))
        if answered(payload, "findings"):
            parsed_any = True
            findings.extend(items_from(payload, "findings"))
        else:
            notes.append(f"the model did not return usable JSON for part "
                         f"{index}/{len(parts)}: nothing was analysed there")

    if not parsed_any:
        raise ModelAnswerError(
            f"{config.ACTIVE_MODEL} did not return usable JSON for this document. "
            f"Nothing was analysed - this is a failure, not an empty result. Last answer: "
            f"{(meta.get('raw') or '')[:200]!r}")

    generator = FakeGenerator(receiver=receiver)
    report("guards", "Checking model findings against the document and policy guards.",
           completed_chunks=len(parts), total_chunks=len(parts))
    guards_started = time.perf_counter()
    detections, guard_notes = guards.check_findings(findings, document, receiver, lib,
                                                    generator)
    notes.extend(guard_notes)

    for index, detection in enumerate(detections):
        detection.meta.setdefault("id", f"d{index}")
        detection.meta["masked"] = mask_value(detection.original_text)
        detection.meta["value_hash"] = value_hash(detection.original_text)
        detection.meta["detector"] = "llm"

    guard_seconds = time.perf_counter() - guards_started
    uncertain_count = sum(
        d.action == Action.ASK_HUMAN.value and d.confidence >= config.CONFIDENCE_FLOOR
        for d in detections)
    if config.ADJUDICATE and uncertain_count:
        report("adjudicating", f"Taking a second look at {uncertain_count} uncertain "
                               f"finding(s).",
               completed_chunks=len(parts), total_chunks=len(parts),
               uncertain_findings=uncertain_count)
        adjudication_started = time.perf_counter()
        _second_look(detections, document, receiver, lib, notes)
        guard_seconds += time.perf_counter() - adjudication_started

    for index, detection in enumerate(detections):
        detection.meta["id"] = f"d{index}"

    analysis = AnalysisResult(document=document, receiver=receiver, purpose=purpose,
                              detections=detections, generator=generator,
                              rules_version=lib.version, notes=notes,
                              model=config.ACTIVE_MODEL,
                              calls=engine.calls()[analysis_calls_start:],
                              model_load_seconds=(
                                  float(engine.loaded_info().get("load_seconds") or 0)
                                  if not model_was_loaded_at_start
                                  and engine.provider() == "local"
                                  else 0.0))
    analysis.processing_seconds = round(time.perf_counter() - started, 2)
    analysis.guard_seconds = round(guard_seconds, 4)
    analysis.model_generation_seconds = round(sum(
        float(call.get("seconds") or 0) for call in analysis.calls), 2)
    analysis.json_validation_seconds = round(sum(
        float(call.get("json_validation_seconds") or 0) for call in analysis.calls), 4)

    if config.SUMMARY and include_summary and detections:
        summary_started = time.perf_counter()
        report("summarizing", "Writing the short analysis summary.",
               completed_chunks=len(parts), total_chunks=len(parts))
        summary = analysis.summary()
        payload, _ = engine.generate_json(
            prompts.summary_messages(label, purpose, summary["by_action"],
                                     summary["items"], summary["needs_review"],
                                     [d.meta for d in detections]),
            purpose="summary", max_new_tokens=250)
        analysis.summary_seconds = round(time.perf_counter() - summary_started, 4)
        if isinstance(payload, dict):
            analysis.summary_text = str(payload.get("summary") or "").strip() or None
        analysis.calls = engine.calls()[analysis_calls_start:]
        analysis.model_generation_seconds = round(sum(
            float(call.get("seconds") or 0) for call in analysis.calls), 2)
        analysis.json_validation_seconds = round(sum(
            float(call.get("json_validation_seconds") or 0)
            for call in analysis.calls), 4)
        analysis.processing_seconds = round(time.perf_counter() - started, 2)
    report("complete", f"Analysis complete with {len(detections)} findings.",
           completed_chunks=len(parts), total_chunks=len(parts),
           findings=len(detections), total_seconds=analysis.processing_seconds)
    return analysis


def write_summary(analysis: AnalysisResult, applied: dict) -> str | None:
    """The AI's audit note for a release, based on what was actually applied."""
    if not config.SUMMARY:
        return None
    counts = {action: 0 for action in (Action.KEEP.value, Action.REPLACE.value,
                                       Action.REMOVE.value, Action.ASK_HUMAN.value)}
    for change in applied.get("applied", []):
        counts[change.get("action", Action.KEEP.value)] = counts.get(
            change.get("action", Action.KEEP.value), 0) + 1
    payload, _ = engine.generate_json(
        prompts.summary_messages(
            default_library().receiver_label(analysis.receiver), analysis.purpose,
            counts, len(applied.get("applied", [])) + len(applied.get("pending", [])),
            len(applied.get("pending", [])), applied.get("applied", [])),
        purpose="release summary", max_new_tokens=250)
    if isinstance(payload, dict):
        return str(payload.get("summary") or "").strip() or None
    return None


def apply_decisions(analysis: AnalysisResult, decisions: dict[str, str] | None = None,
                    reviewer: str = "") -> dict:
    """Turn the model's proposal plus the reviewer's answers into the safe document.

    `decisions` maps detection id -> action. An item still marked ASK_HUMAN is LEFT OUT
    of the release: when nobody has decided, publish less, never more.
    """
    decisions = decisions or {}
    generator = analysis.generator
    changed: dict[str, str] = {}
    applied: list[dict] = []
    still_pending: list[dict] = []

    # Every item needs an id (the UI and the audit refer to items by it) and the masked
    # value plus hash that go into the log. analyze() sets these; filling them here too
    # means a result built anywhere else still cannot reach a document unprepared.
    for index, detection in enumerate(analysis.detections):
        detection.meta.setdefault("id", f"d{index}")
        detection.meta.setdefault("masked", mask_value(detection.original_text))
        detection.meta.setdefault("value_hash", value_hash(detection.original_text))

    by_block: dict[str, list[Detection]] = {}
    for detection in analysis.detections:
        by_block.setdefault(detection.block_id, []).append(detection)

    for block in analysis.document.blocks:
        text = block.text
        items = sorted(by_block.get(block.id, []), key=lambda d: -d.start)
        for detection in items:
            decision = decisions.get(detection.meta["id"], detection.action)
            if decision == Action.KEEP.value:
                action_taken, new_value = Action.KEEP.value, detection.original_text
            elif decision == Action.REPLACE.value:
                new_value = detection.replacement or generator.for_detection(
                    detection.original_text, detection.type)
                action_taken = Action.REPLACE.value
            elif decision == Action.REMOVE.value:
                action_taken, new_value = Action.REMOVE.value, ""
            else:
                action_taken, new_value = Action.ASK_HUMAN.value, ""

            if decision != detection.action:
                detection.reviewer = reviewer or "reviewer"
            detection.meta["final_action"] = action_taken

            if action_taken == Action.ASK_HUMAN.value:
                text = text[:detection.start] + text[detection.end:]
                still_pending.append({
                    "id": detection.meta["id"], "type": detection.type,
                    "masked": detection.meta["masked"],
                    "value_hash": detection.meta["value_hash"],
                    "rule_id": detection.rule_id, "reason": detection.reason,
                    "confidence": detection.confidence,
                    "block_id": block.id, "start": detection.start,
                    "end": detection.end})
            else:
                text = text[:detection.start] + new_value + text[detection.end:]
                applied.append({
                    "id": detection.meta["id"], "type": detection.type,
                    "action": action_taken, "masked": detection.meta["masked"],
                    "value_hash": detection.meta["value_hash"],
                    "replacement": (new_value
                                    if action_taken == Action.REPLACE.value else None),
                    "rule_id": detection.rule_id, "reason": detection.reason,
                    "confidence": detection.confidence,
                    "decided_by": detection.reviewer or "system",
                    "detector": detection.detector, "block_id": block.id})
        if text != block.text:
            text = "\n".join(line.rstrip() for line in text.split("\n"))
        changed[block.id] = text

    return {"changed": changed, "applied": applied, "pending": still_pending,
            "incomplete": bool(still_pending)}
