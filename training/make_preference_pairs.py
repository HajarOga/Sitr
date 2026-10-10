"""Build the alignment data: preferred answer vs rejected answer.

    python -m training.make_preference_pairs

This is the DPO half of the adaptation path. Fine-tuning teaches the *shape* of the
answer; alignment teaches the *judgement*. The pairs are built around the four ways a
helpful model gets this task wrong, which are also the four ways this product could
publish somebody's data:

| kind | the rejected answer | what it teaches |
|---|---|---|
| `confident-instead-of-asking` | decides an item the policy does not cover, with 0.95 confidence | ask rather than guess |
| `invented-rule` | cites a rule id that does not exist for this receiver | the policy is the only authority |
| `policy-overruled` | picks an action the cited rule contradicts | the rule wins, not the model |
| `leaked-by-omission` | reports no items, or keeps what the policy says must go | silence leaks data |
| `unsafe-substitute` | "replaces" a value with something that is not obviously fake | a substitute must be fake and shape-preserving |

Every pair keeps the same prompt, so the model is being taught a preference, not a new
task. Output: `data/train/preference_pairs.jsonl`, with a `kind` on each row so the
alignment set can be audited and extended one lesson at a time.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pii_detection.schema import Action                          # noqa: E402
from app.replacement.generator import FakeGenerator                  # noqa: E402
from app.rules.store import library as default_library               # noqa: E402
from evaluation.reference import expectation                         # noqa: E402

TRAIN_OUT = ROOT / "data" / "train"


def _chosen(record: dict) -> dict:
    return dict(record)


def _reject_confident(records: list[dict], receiver: str) -> list[dict] | None:
    """Decide every undecided item anyway, with a confident tone."""
    out = []
    for record in records:
        record = dict(record)
        if record["action"] == Action.ASK_HUMAN.value:
            record["action"] = Action.REPLACE.value
            record["replacement"] = "REDACTED"
            record["reason"] = "Removed to be safe."
            record["confidence"] = 0.95
        out.append(record)
    return out if out != records else None


def _reject_invented_rule(records: list[dict], receiver: str) -> list[dict] | None:
    """Cite a rule that does not exist - plausible, and unacceptable."""
    out, changed = [], False
    for record in records:
        record = dict(record)
        if record["action"] in (Action.REPLACE.value, Action.REMOVE.value):
            record["rule_id"] = "VENDOR-PII-999"
            changed = True
        out.append(record)
    return out if changed else None


def _reject_policy_overruled(records: list[dict], receiver: str) -> list[dict] | None:
    """Keep something the policy says must go, citing the very rule that forbids it."""
    out, changed = [], False
    for record in records:
        record = dict(record)
        if record["action"] == Action.REMOVE.value:
            record["action"] = Action.KEEP.value
            record["replacement"] = None
            record["reason"] = "Kept for readability."
            changed = True
        out.append(record)
    return out if changed else None


def _reject_omission(records: list[dict], receiver: str) -> list[dict] | None:
    """Report nothing: the document 'looked harmless'."""
    return [] if records else None


def _reject_unsafe_substitute(records: list[dict], receiver: str) -> list[dict] | None:
    """A substitute that is not obviously fake, or that is the original value."""
    generator = FakeGenerator(receiver=receiver)
    out, changed = [], False
    for record in records:
        record = dict(record)
        if record["action"] == Action.REPLACE.value:
            record["replacement"] = record["original_text"]      # not a substitute at all
            changed = True
        out.append(record)
    return out if changed else None


REJECTIONS = {
    "confident-instead-of-asking": _reject_confident,
    "invented-rule": _reject_invented_rule,
    "policy-overruled": _reject_policy_overruled,
    "leaked-by-omission": _reject_omission,
    "unsafe-substitute": _reject_unsafe_substitute,
}


def build(sft_path: Path | None = None) -> dict:
    source = sft_path or (TRAIN_OUT / "sft.jsonl")
    if not source.exists():
        raise SystemExit(f"{source} not found. Run first:\n"
                         f"  python -m training.prepare_data --skip-public")

    pairs, kinds = [], collections.Counter()
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        chosen = json.loads(row["completion"])["findings"]
        receiver = row["receiver"]
        for kind, make_rejected in REJECTIONS.items():
            rejected = make_rejected(chosen, receiver)
            if rejected is None or rejected == chosen:
                continue                       # this document offers no lesson of that kind
            pairs.append({
                "messages": row["messages"],
                "chosen": json.dumps({"findings": [_chosen(r) for r in chosen]},
                                     ensure_ascii=False),
                "rejected": json.dumps({"findings": [dict(r) for r in rejected]},
                                       ensure_ascii=False),
                "document": row["document"], "receiver": receiver, "kind": kind})

    if not pairs:
        raise SystemExit("no preference pairs were built - the rejected answers matched "
                         "the chosen ones")

    TRAIN_OUT.mkdir(parents=True, exist_ok=True)
    out = TRAIN_OUT / "preference_pairs.jsonl"
    out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in pairs),
                   encoding="utf-8")
    return {"pairs": len(pairs), "written": str(out), "by_kind": dict(kinds + collections.Counter(
        row["kind"] for row in pairs))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft", default=None, help="source of prompts (default data/train/sft.jsonl)")
    args = parser.parse_args()
    print(json.dumps(build(Path(args.sft) if args.sft else None), ensure_ascii=False, indent=1))
