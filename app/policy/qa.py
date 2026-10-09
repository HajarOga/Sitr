"""Policy Q&A: ask the policy a question, get the rules that answer it.

Two layers, and the order matters:

1. **Retrieval** (Chroma, `policy_index.py`) - the rules that are closest in meaning to
   the question, with their ids, actions and citations. This layer always runs and
   always produces something a person can read and check.
2. **Answer** (the local model, optional) - a short answer written *from those retrieved
   rules only*, citing the rule ids it used. This layer is a convenience on top: if the
   model is not installed, or answers with unusable JSON, the retrieved rules are still
   returned and the caller says so.

The same discipline as the release pipeline applies, for the same reason. The model may
not invent a rule id, so citations are checked against the retrieved set and anything
invented is dropped and reported. And an empty retrieval is a real answer - "the policy
does not cover this" - which must never be smoothed over into an answer, because the
honest consequence of a missing rule is that a person decides (ASK_HUMAN).

Nothing here reads a document. The Q&A page is about the policy, never about the
document being released.
"""
from __future__ import annotations

import re

from .. import config
from ..llm import engine, prompts
from ..rules.store import RulesLibrary, library as default_library
from .policy_index import (PolicyIndex, PolicyIndexError, asks_about_receiver,
                           asks_about_type, citations, get_index)

# Rule ids look like VENDOR-PII-003, HR-PII-011, ANALYST-PII-004. We read them out of the
# answer text as well as out of the `citations` field, because a model that describes the
# right rule may still mistype the id - and a citation the reader can see is the point.
_RULE_ID = re.compile(r"\b[A-Z][A-Z0-9]{1,15}-[A-Z0-9]{1,10}-[0-9]{1,4}\b")


def _distance(left: str, right: str) -> int:
    """Small edit distance, for spotting a typo'd rule id (VENDOR-PIII-003)."""
    if abs(len(left) - len(right)) > 2:
        return 3
    previous = list(range(len(right) + 1))
    for i, char_left in enumerate(left, start=1):
        current = [i]
        for j, char_right in enumerate(right, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (char_left != char_right)))
        previous = current
    return previous[-1]


def _typo_threshold(right: str) -> int:
    """How far a mistyped id may stray before it stops being a typo.

    A rule id is about 13 characters, so one wrong character is a typo and two are
    usually a different rule. Scaling with the length keeps that true for a short id like
    `HR-1-1` and a long one like `CUSTOMER-SUPPORT-012`.
    """
    return max(1, len(right) // 8)


def resolve_citations(answer: str, payload_citations: list, hits: list[dict]) -> dict:
    """Check every rule id the model produced against the rules it was actually given.

    Three outcomes, and they are kept apart on purpose:

    * **verified** - the id is one of the retrieved rules;
    * **suspect** - not a retrieved id, but within one small edit of exactly one of them,
      so it is almost certainly a typo. It is reported as a typo and never silently
      counted as a citation;
    * **invented** - matches nothing, so it is refused. This is the same contract the
      guards apply to a finding: a citation that is not in the policy cannot be trusted,
      and saying so is more useful than quietly dropping it.
    """
    retrieved = {hit["rule_id"]: hit for hit in hits}
    claimed: list[str] = []
    for candidate in list(payload_citations or []) + _RULE_ID.findall(answer or ""):
        token = str(candidate).strip().strip(".,;:()")
        if token and token not in claimed:
            claimed.append(token)

    verified = [token for token in claimed if token in retrieved]
    invented, suspect = [], []
    for token in claimed:
        if token in retrieved:
            continue
        close = [rule_id for rule_id in retrieved
                 if _distance(token, rule_id) <= _typo_threshold(rule_id)]
        if len(close) == 1:
            suspect.append({"claimed": token, "meant": close[0],
                            "action": retrieved[close[0]]["action"]})
        else:
            # Zero matches, or several equally close ones: either way this token is not a
            # citation we can point at, so it is refused rather than guessed at.
            invented.append(token)
    return {"verified": verified, "suspect": suspect, "invented": invented}


def retrieve(question: str, receiver: str | None = None, pii_type: str | None = None,
             top_k: int | None = None, index: PolicyIndex | None = None) -> list[dict]:
    """The rules that answer `question`, best first. Retrieval only, no model call."""
    return (index or get_index()).query(question, receiver=receiver, pii_type=pii_type,
                                        top_k=top_k)


def grounded_answer(result: dict, hits: list[dict], library: RulesLibrary | None = None
                    ) -> str:
    """The answer plus a code-built list of the rules behind it.

    The model writes prose; the *citations* are assembled here, from the retrieved
    metadata, so a reader always has the exact rule ids, actions and sources even when
    the model mistyped one. Nothing in this block is generated text.
    """
    lib = library or default_library()
    lines = []
    if result.get("answer"):
        lines.append(result["answer"].strip())
    if hits:
        lines.append("")
        lines.append("Per the retrieved policy:" if not result.get("answer")
                     else "According to these rules:")
        for hit in hits:
            label = lib.receiver_label(hit["receiver"])
            source = f" - {hit['citation']}" if hit.get("citation") else ""
            lines.append(f"- {hit['rule_id']} ({label}, {hit['pii_type']}): "
                         f"{hit['action']} - {hit['reason']}{source}")
    return "\n".join(lines).strip()


def ask_policy(question: str, receiver: str | None = None, pii_type: str | None = None,
               top_k: int | None = None, use_model: bool = True,
               index: PolicyIndex | None = None,
               library: RulesLibrary | None = None) -> dict:
    """Retrieve, then (optionally) have the local model answer from what was retrieved.

    Returns a dict the UI can render as-is:
        question, receiver_filter, type_filter, hits, citations, answer, answered_by,
        notes, model, seconds
    """
    lib = library or default_library()
    question = (question or "").strip()
    if not question:
        return {"question": "", "hits": [], "citations": [], "answer": None,
                "answered_by": "none", "notes": ["Ask a question about the policy."],
                "receiver_filter": None, "type_filter": None, "model": "", "seconds": 0.0}

    # The filters are a convenience, and the caller can always widen them: a wrong guess
    # here shows up as an empty result, never as a wrong answer.
    receiver_filter = receiver if receiver is not None else asks_about_receiver(question, lib)
    type_filter = pii_type if pii_type is not None else asks_about_type(question)

    notes: list[str] = []
    try:
        hits = retrieve(question, receiver=receiver_filter, pii_type=type_filter,
                        top_k=top_k, index=index)
    except PolicyIndexError as exc:
        return {"question": question, "hits": [], "citations": [], "answer": None,
                "answered_by": "none", "notes": [str(exc)], "receiver_filter": receiver_filter,
                "type_filter": type_filter, "model": "", "seconds": 0.0,
                "error": "index-unavailable"}

    if receiver_filter and not hits:
        notes.append(f"No rule for {lib.receiver_label(receiver_filter)} answers this. "
                     f"A policy with no rule for the item is not permission - the item "
                     f"reaches a person as ASK_HUMAN.")
    elif not hits:
        notes.append("No rule in the policy matches this question. That is an answer: "
                     "the policy is silent, so the item goes to a person.")

    result = {"question": question, "hits": hits, "citations": citations(hits),
              "answer": None, "answered_by": "retrieval",
              "receiver_filter": receiver_filter, "type_filter": type_filter,
              "notes": notes, "model": "", "seconds": 0.0}

    if not use_model:
        notes.append("Answer-writing is switched off: the rules above are the answer.")
        return result
    if not hits:
        return result

    try:
        engine.ensure_installed()
    except engine.ModelNotInstalled as exc:
        notes.append("The retrieved rules are shown unsummarised because the local model "
                     "is not installed. " + str(exc).split("\n")[0])
        return result

    engine.reset_calls()
    try:
        payload, _meta = engine.generate_json(
            prompts.policy_qa_messages(question, hits),
            purpose="policy q&a", max_new_tokens=400)
    except Exception as exc:                                         # noqa: BLE001
        notes.append(f"The model could not write an answer ({exc}). The retrieved rules "
                     f"are shown instead.")
        return result

    if not isinstance(payload, dict) or not str(payload.get("answer") or "").strip():
        notes.append("The model did not return a usable answer; the retrieved rules are "
                     "shown instead. Nothing was invented on its behalf.")
        return result

    answer = str(payload["answer"]).strip()
    resolved = resolve_citations(answer, payload.get("citations"), hits)
    result["answer"] = answer
    result["cited"] = resolved["verified"]
    result["suspect_citations"] = resolved["suspect"]
    result["invented_citations"] = resolved["invented"]
    result["answered_by"] = "model"
    result["model"] = config.ACTIVE_MODEL
    result["seconds"] = round(sum(call.get("seconds", 0.0)
                                  for call in engine.calls()), 2)

    for item in resolved["suspect"]:
        notes.append(f"The model wrote `{item['claimed']}` where the policy says "
                     f"`{item['meant']}` - a typo, read as the real rule "
                     f"({item['action']}). Check it against the rules below.")
    if resolved["invented"]:
        # Same contract as the guards: an invented rule id is refused and reported, not
        # quietly passed to a reader who would trust it.
        notes.append("The model cited rule id(s) that are not in the retrieved policy: "
                     + ", ".join(resolved["invented"]) + ". They are ignored - treat the "
                     "rules below as the answer.")
    if not resolved["verified"] and not resolved["suspect"]:
        notes.append("The answer cites no retrieved rule. Read the rules below rather "
                     "than the summary.")
    result["grounded"] = grounded_answer(result, hits, lib)
    return result


def coverage(question: str, expected_type: str, receiver: str | None = None,
             top_k: int | None = None, index: PolicyIndex | None = None) -> bool:
    """Does retrieval find a rule for `expected_type`? The retriever's recall, no model.

    Used to score the index against the gold test set (`data/test/*.gold.json`) once the
    policy grows large enough for retrieval to matter. Today the whole policy fits in a
    prompt, so this is a measurement, not a decision input.
    """
    hits = retrieve(question, receiver=receiver, top_k=top_k, index=index)
    return any(hit["pii_type"] == expected_type for hit in hits)
