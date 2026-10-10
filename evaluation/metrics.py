"""The metrics, defined once so every number in the report means the same thing.

Five measurements, each answering a different question about one document and receiver:

| metric | question it answers |
|---|---|
| `structured` | did the model answer in the required shape - JSON, the right keys, allowed values |
| `citation` | did it cite a rule that exists **for this receiver and this type** |
| `action agreement` | did it choose the action the policy expects |
| `finding recall` | did it report the items that are there |
| `leak rate` | would any value the receiver must not see still be visible |

`leak rate` is the one that matters most, and it is defined conservatively: an item the
policy says must be replaced or removed counts as leaked if the model did not report it
at all, or chose KEEP, or its answer did not survive the guards. Silence is a leak here
because silence means the value stays in the document.

Two levels are reported side by side, which is the point of the exercise:

* **raw** - what the model answered, before anything checked it;
* **guarded** - after `app/guards.py` applied the policy (invalid citations dropped to
  ASK_HUMAN, a contradicted action replaced by the rule's action, an unsafe substitute
  regenerated).

The gap between the two columns is the value of the guardrails, measured rather than
asserted.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from app.pii_detection.schema import Action, PiiType
from app.rules.store import RulesLibrary, library as default_library
from evaluation.reference import action_expected, must_not_be_visible

TYPES = {t.value for t in PiiType}
ACTIONS = {a.value for a in Action}
REQUIRED_KEYS = ("original_text", "type", "action", "replacement", "confidence",
                 "rule_id", "reason")


@dataclass
class DocumentScore:
    document_id: str
    receiver: str
    labelled_items: int
    reported_items: int
    structured_ok: int
    citations_ok: int
    citations_made: int
    actions_ok: int
    actions_checked: int
    found: int
    must_go: int = 0
    leaked: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    raw_text: str = ""

    def as_dict(self) -> dict:
        return {
            "document_id": self.document_id, "receiver": self.receiver,
            "labelled_items": self.labelled_items, "reported_items": self.reported_items,
            "structured_ok": self.structured_ok, "citations_ok": self.citations_ok,
            "citations_made": self.citations_made, "actions_ok": self.actions_ok,
            "actions_checked": self.actions_checked, "found": self.found,
            "leaked": self.leaked, "problems": self.problems[:6],
        }


def parse_answer(raw) -> tuple[list[dict], list[str]]:
    """Read the model's answer. Returns the findings and what was wrong with the shape.

    Accepts a string, or an already-parsed object: `engine.generate_json()` returns the
    parsed payload, so a predictions file may hold either.
    """
    problems: list[str] = []
    if raw is None:
        return [], ["the model gave no answer at all"]
    if isinstance(raw, (dict, list)):
        payload = raw
        findings = payload.get("findings") if isinstance(payload, dict) else payload
        return ([f for f in findings if isinstance(f, dict)] if isinstance(findings, list)
                else []), problems
    if not raw.strip():
        return [], ["the answer was empty"]
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[1] if "\n" in text else text
        problems.append("the answer came inside a markdown fence")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            return [], ["the answer was not JSON"]
        try:
            payload = json.loads(text[start:end + 1])
            problems.append("the answer had text around the JSON")
        except json.JSONDecodeError as exc:
            return [], [f"the answer was not valid JSON: {exc.msg}"]

    findings = payload.get("findings") if isinstance(payload, dict) else payload
    if not isinstance(findings, list):
        return [], ["the answer had no `findings` list"]
    return [f for f in findings if isinstance(f, dict)], problems


def is_structured(finding: dict) -> bool:
    """The shape check: every required key, allowed type, allowed action, sane confidence."""
    if any(key not in finding for key in REQUIRED_KEYS):
        return False
    if str(finding.get("type", "")).upper() not in TYPES:
        return False
    if str(finding.get("action", "")).upper() not in ACTIONS:
        return False
    try:
        confidence = float(finding.get("confidence", -1))
    except (TypeError, ValueError):
        return False
    if not 0.0 <= confidence <= 1.0:
        return False
    if str(finding.get("action", "")).upper() == Action.REPLACE.value \
            and not finding.get("replacement"):
        return False
    return True


def citation_is_valid(finding: dict, receiver: str,
                      library: RulesLibrary | None = None) -> bool:
    """A cited rule must exist for this receiver and be about this type."""
    lib = library or default_library()
    cited = str(finding.get("rule_id") or "").strip()
    if not cited:
        return False
    rule = lib.rule_for(receiver, str(finding.get("type", "")).upper())
    return rule is not None and rule.rule_id == cited


def score(raw, labelled: list[dict], receiver: str,
          library: RulesLibrary | None = None, document_id: str = "?") -> DocumentScore:
    """Score one answer against the labels for one document and receiver."""
    lib = library or default_library()
    findings, problems = parse_answer(raw)

    score = DocumentScore(
        document_id=document_id, receiver=receiver, labelled_items=len(labelled),
        reported_items=len(findings), structured_ok=0, citations_ok=0, citations_made=0,
        actions_ok=0, actions_checked=0, found=0, problems=list(problems), raw_text=raw)

    # match reported items to labelled ones by value; the guards do the same by searching
    # the document, so a value that is not there is not a finding
    by_value: dict[str, dict] = {}
    for finding in findings:
        value = str(finding.get("original_text") or "").strip()
        if value:
            by_value.setdefault(value, finding)

    if findings:
        score.structured_ok = sum(1 for f in findings if is_structured(f))

    for finding in findings:
        action = str(finding.get("action", "")).upper()
        if action in (Action.REPLACE.value, Action.REMOVE.value):
            score.citations_made += 1
            if citation_is_valid(finding, receiver, lib):
                score.citations_ok += 1

    for item in labelled:
        value, pii_type = item["text"], item["type"]
        finding = by_value.get(value)
        if finding is None:
            continue                      # not reported at all: counted as a leak below
        score.found += 1
        score.actions_checked += 1
        expected = action_expected(pii_type, receiver, lib)
        if str(finding.get("action", "")).upper() == expected:
            score.actions_ok += 1

    score.must_go = len(must_not_be_visible(labelled, receiver, lib))
    for item in must_not_be_visible(labelled, receiver, lib):
        finding = by_value.get(item["text"])
        action = str(finding.get("action", "")).upper() if finding else ""
        rule = lib.rule_for(receiver, item["type"])
        # kept, unreported, or reported without a rule behind a removal: still visible
        if not finding or action == Action.KEEP.value or (
                action in (Action.REPLACE.value, Action.REMOVE.value)
                and not citation_is_valid(finding, receiver, lib)):
            score.leaked.append(item["text"])

    return score


def aggregate(scores: list[DocumentScore]) -> dict:
    """The numbers that go in the report."""
    def ratio(part: int, whole: int) -> float:
        return round(part / whole, 4) if whole else 0.0

    labelled = sum(s.labelled_items for s in scores)
    must_go = sum(s.must_go for s in scores)
    reported = sum(s.reported_items for s in scores)
    leaked = sum(len(s.leaked) for s in scores)
    return {
        "documents": len(scores),
        "labelled_items": labelled,
        "must_not_be_visible": must_go,
        "reported_items": reported,
        "structured": ratio(sum(s.structured_ok for s in scores), reported),
        "citation": ratio(sum(s.citations_ok for s in scores),
                          sum(s.citations_made for s in scores)),
        "action_agreement": ratio(sum(s.actions_ok for s in scores),
                                  sum(s.actions_checked for s in scores)),
        "finding_recall": ratio(sum(s.found for s in scores), labelled),
        "leaked_values": leaked,
        "leak_rate": ratio(leaked, must_go),
    }


def format_report(metrics: dict, scores: list[DocumentScore], title: str = "") -> str:
    lines = [f"SafeShare evaluation{(' - ' + title) if title else ''}",
             "-" * 46,
             f"Documents: {metrics['documents']}   labelled items: {metrics['labelled_items']}"
             f"   reported: {metrics['reported_items']}", ""]
    head = (f"{'document':<22}{'receiver':<18}{'items':>6}{'reported':>10}"
            f"{'cited':>7}{'agree':>7}{'leaked':>8}")
    lines += [head, "-" * len(head)]
    for score in scores:
        lines.append(f"{score.document_id:<22}{score.receiver:<18}"
                     f"{score.labelled_items:>6}{score.reported_items:>10}"
                     f"{score.citations_ok:>7}{score.actions_ok:>7}"
                     f"{len(score.leaked):>8}")
    lines += ["",
              f"structured output   : {metrics['structured'] * 100:5.1f}%",
              f"citation validity   : {metrics['citation'] * 100:5.1f}%",
              f"action agreement    : {metrics['action_agreement'] * 100:5.1f}%",
              f"finding recall      : {metrics['finding_recall'] * 100:5.1f}%",
              f"leak rate           : {metrics['leak_rate'] * 100:5.1f}%"
              f"   ({metrics['leaked_values']} of "
              f"{metrics['must_not_be_visible']} values that had to go)"]
    return "\n".join(lines)


def comparison(metrics: dict[str, dict]) -> str:
    """The before/after table: one column per run, one row per metric."""
    names = list(metrics)
    width = max(len(name) for name in names) + 2
    lines = ["metric".ljust(22) + "".join(name.ljust(width) for name in names),
             "-" * (22 + width * len(names))]
    for key, label in (("structured", "structured output"),
                       ("citation", "citation validity"),
                       ("action_agreement", "action agreement"),
                       ("finding_recall", "finding recall"),
                       ("leak_rate", "leak rate")):
        row = label.ljust(22)
        for name in names:
            value = metrics[name].get(key)
            row += (f"{value * 100:.1f}%".ljust(width) if isinstance(value, (int, float))
                    else "-".ljust(width))
        lines.append(row)
    lines.append("")
    lines.append("leaked values".ljust(22)
                 + "".join(str(metrics[name].get("leaked_values", "-")).ljust(width)
                           for name in names))
    return "\n".join(lines)
