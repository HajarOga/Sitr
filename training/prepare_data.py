"""Build the fine-tuning and alignment data.

    python -m training.prepare_data --skip-public          # our own documents only
    python -m training.prepare_data                        # + the two licensed datasets

What it produces, and why in this shape:

* **`data/train/sft.jsonl`** - prompt/completion pairs for LoRA. The prompt is exactly
  what the application sends (the same `prompts.analyze_messages()` and the same policy
  block for that receiver), and the completion is the JSON the guards would accept. The
  model therefore learns this app's task in this app's format - which is what the brief
  calls "LoRA/QLoRA: task, format, domain".
* **`data/train/preference_pairs.jsonl`** - chosen/rejected pairs for DPO alignment,
  built by `training/make_preference_pairs.py`.

Where the target comes from: the policy. `evaluation/reference.py` derives what should
happen to each labelled item for each receiver, and that derivation is also what the
evaluation scores against, so training and measurement agree by construction.

Two rules this script obeys:

1. **It never reads `data/test/`.** Training documents come from
   `data/train/documents/`, built by `make_training_documents.py`, which refuses any
   value that appears in the evaluation set.
2. **The Arabic dataset is sampled, never trained on whole.** `--arabic-limit` defaults
   to 2,500 examples, and the script prints the licence of everything it touched.
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

from app.llm import prompts                                    # noqa: E402
from app.rules.store import library as default_library         # noqa: E402
from evaluation.reference import expected_findings             # noqa: E402

TRAIN_DOCS = ROOT / "data" / "train" / "documents"
TRAIN_OUT = ROOT / "data" / "train"
VALIDATION_OUT = ROOT / "data" / "validation"

RECEIVERS = ["EXTERNAL_VENDOR", "INTERNAL_HR", "ANALYST", "CUSTOMER_SUPPORT", "CUSTOM"]

# third-party label -> our category. Their vocabulary is not ours, so it is mapped, and
# anything unmapped is dropped rather than guessed at.
LABEL_MAP = {
    "givenname": "PERSON_NAME", "surname": "PERSON_NAME", "name": "PERSON_NAME",
    "firstname": "PERSON_NAME", "lastname": "PERSON_NAME", "person": "PERSON_NAME",
    "username": "PERSON_NAME", "prefix": "PERSON_NAME",
    "اسم": "PERSON_NAME", "الاسم": "PERSON_NAME",
    "email": "EMAIL", "mail": "EMAIL", "بريد": "EMAIL",
    "tel": "PHONE", "phone": "PHONE", "telephone": "PHONE", "mobile": "PHONE",
    "هاتف": "PHONE", "جوال": "PHONE",
    "idcard": "NATIONAL_ID", "id_number": "NATIONAL_ID", "nationalid": "NATIONAL_ID",
    "هوية": "NATIONAL_ID", "رقم_الهوية": "NATIONAL_ID",
    "iban": "IBAN", "bank": "IBAN", "account": "BANK_ACCOUNT",
    "address": "ADDRESS", "city": "ADDRESS", "street": "ADDRESS", "location": "ADDRESS",
    "عنوان": "ADDRESS",
    "birthdate": "DATE_OF_BIRTH", "date_of_birth": "DATE_OF_BIRTH", "dob": "DATE_OF_BIRTH",
    "passport": "PASSPORT_NUMBER", "customer_id": "CUSTOMER_ID",
    "employee_id": "EMPLOYEE_ID", "customerid": "CUSTOMER_ID", "employeeid": "EMPLOYEE_ID",
}


def map_label(label: str) -> str | None:
    key = (label or "").strip().lower().replace(" ", "_")
    if key in LABEL_MAP:
        return LABEL_MAP[key]
    for token, pii_type in LABEL_MAP.items():
        if token and token in key:
            return pii_type
    return None


def _pair(document_text: str, items: list[dict], receiver: str, source: str,
          document_id: str) -> dict:
    lib = default_library()
    rules = [{"rule_id": r.rule_id, "type": r.pii_type, "action": r.action,
              "reason": r.reason} for r in lib.rules_for(receiver)]
    messages = prompts.analyze_messages(
        document_text, lib.receiver_label(receiver), lib.receivers[receiver].purpose, rules)
    completion = json.dumps({"findings": expected_findings(items, receiver, lib)},
                            ensure_ascii=False)
    return {"messages": messages, "completion": completion, "document": document_id,
            "receiver": receiver, "source": source}


def from_our_documents() -> list[dict]:
    pairs: list[dict] = []
    for gold_path in sorted(TRAIN_DOCS.glob("*.gold.json")):
        gold = json.loads(gold_path.read_text(encoding="utf-8"))
        text_path = gold_path.with_name(gold_path.name.replace(".gold.json", ".txt"))
        if not text_path.exists():
            continue
        text = text_path.read_text(encoding="utf-8")
        items = [{"text": span["text"], "type": span["type"]} for span in gold["spans"]]
        if not items:
            continue
        # every receiver, because the same document has a different correct answer for
        # each one - that is the whole point of the product
        for receiver in gold.get("receivers", RECEIVERS):
            pairs.append(_pair(text, items, receiver, "our own synthetic document",
                               gold["document_id"]))
    return pairs


def from_public_dataset(source: str, limit: int, field_names: list[str],
                        licence: str) -> tuple[list[dict], dict]:
    """One public dataset -> pairs. Skips cleanly when `datasets` is not installed."""
    try:
        from datasets import load_dataset                        # noqa: PLC0415
    except ImportError:
        return [], {"source": source, "skipped": "the `datasets` package is not installed"}

    try:
        dataset = load_dataset(source, split=f"train[:{limit}]")
    except Exception as exc:                                     # noqa: BLE001
        return [], {"source": source, "skipped": f"could not load: {exc}"}

    pairs: list[dict] = []
    for row in dataset:
        text = ""
        for name in field_names:
            if row.get(name):
                text = str(row[name])
                break
        if not text.strip():
            continue
        raw = row.get("privacy_mask") or row.get("labels") or row.get("entities") or []
        items, seen = [], set()
        if isinstance(raw, list):
            for entry in raw:
                if isinstance(entry, dict):
                    value = entry.get("value") or entry.get("text") or ""
                    label = entry.get("label") or entry.get("type") or ""
                else:
                    value, label = "", str(entry)
                pii_type = map_label(str(label))
                if pii_type and str(value).strip() and str(value) not in seen:
                    seen.add(str(value))
                    items.append({"text": str(value), "type": pii_type})
        if not items:
            continue
        for receiver in ("EXTERNAL_VENDOR", "INTERNAL_HR", "ANALYST"):
            pairs.append(_pair(text, items, receiver,
                               f"{source} ({licence})", f"{source}-{len(pairs)}"))
    return pairs, {"source": source, "licence": licence, "limit": limit,
                   "pairs": len(pairs)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SafeShare training data")
    parser.add_argument("--skip-public", action="store_true",
                        help="only our own documents: no downloads, no `datasets` needed")
    parser.add_argument("--english-source", default="aniket-curlscape/pii-masking-english-1k")
    parser.add_argument("--english-limit", type=int, default=1000)
    parser.add_argument("--arabic-source", default="mabahboh/sitr-arabic-pii")
    parser.add_argument("--arabic-limit", type=int, default=2500,
                        help="sample size for the Arabic set, never the whole dataset")
    parser.add_argument("--validation-split", type=float, default=0.1)
    args = parser.parse_args()

    if not list(TRAIN_DOCS.glob("*.gold.json")):
        raise SystemExit("no training documents. Run first:\n"
                         "  python -m training.make_training_documents")

    pairs = from_our_documents()
    sources = [{"source": "our own synthetic documents", "pairs": len(pairs)}]

    if not args.skip_public:
        english, english_meta = from_public_dataset(
            args.english_source, args.english_limit, ["text", "source_text"], "see the card")
        arabic, arabic_meta = from_public_dataset(
            args.arabic_source, args.arabic_limit, ["text", "source_text"], "Apache-2.0")
        pairs += english + arabic
        sources += [english_meta, arabic_meta]

    if not pairs:
        raise SystemExit("no pairs were built")

    TRAIN_OUT.mkdir(parents=True, exist_ok=True)
    VALIDATION_OUT.mkdir(parents=True, exist_ok=True)
    cut = int(len(pairs) * (1 - args.validation_split))
    for target, subset in ((TRAIN_OUT, pairs[:cut]), (VALIDATION_OUT, pairs[cut:])):
        path = target / "sft.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in subset),
                        encoding="utf-8")

    actions = collections.Counter()
    for pair in pairs:
        payload = json.loads(pair["completion"])
        for finding in payload["findings"]:
            actions[finding["action"]] += 1

    print(f"training pairs   : {cut} -> {TRAIN_OUT / 'sft.jsonl'}")
    print(f"validation pairs : {len(pairs) - cut} -> {VALIDATION_OUT / 'sft.jsonl'}")
    print(f"receivers        : {', '.join(RECEIVERS)}")
    print(f"actions in targets: {dict(actions)}")
    for meta in sources:
        print(f"  source: {meta}")
    print("\nthe evaluation set in data/test/ was not read by this script")


if __name__ == "__main__":
    main()
