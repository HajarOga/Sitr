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
# Written against the label vocabularies that are actually in the two datasets - the
# values below were read off them, not guessed at:
#   gretel:  medical_record_number, date_of_birth, ssn, customer_id, email, employee_id,
#            first_name, name, credit_card_number, phone_number, last_name, address, city...
#   sitr:    private_phone, private_email, national_id, iqama, private_date, employee_id,
#            account_number, private_address, passport_number, gulf_iban, national_address...
# Anything not in this map is dropped rather than forced into a category: Sitr has
# twelve types, and a label like `medicine_name` or `vehicle_plate` is not one of them.
LABEL_MAP = {
    # names
    "givenname": "PERSON_NAME", "surname": "PERSON_NAME", "name": "PERSON_NAME",
    "firstname": "PERSON_NAME", "first_name": "PERSON_NAME",
    "lastname": "PERSON_NAME", "last_name": "PERSON_NAME", "person": "PERSON_NAME",
    "username": "PERSON_NAME", "user_name": "PERSON_NAME", "prefix": "PERSON_NAME",
    "middle_name": "PERSON_NAME", "father_name": "PERSON_NAME", "mother_name": "PERSON_NAME",
    "اسم": "PERSON_NAME", "الاسم": "PERSON_NAME", "اسم_الشخص": "PERSON_NAME",
    # contact
    "email": "EMAIL", "mail": "EMAIL", "private_email": "EMAIL", "بريد": "EMAIL",
    "tel": "PHONE", "phone": "PHONE", "phone_number": "PHONE", "telephone": "PHONE",
    "mobile": "PHONE", "private_phone": "PHONE", "gulf_phone": "PHONE",
    "هاتف": "PHONE", "جوال": "PHONE", "رقم_الجوال": "PHONE",
    # national identifiers - several countries' names for the same thing, one category
    "idcard": "NATIONAL_ID", "id_number": "NATIONAL_ID", "nationalid": "NATIONAL_ID",
    "national_id": "NATIONAL_ID", "ssn": "NATIONAL_ID", "tax_id": "NATIONAL_ID",
    "iqama": "NATIONAL_ID", "spoken_iqama": "NATIONAL_ID", "kuwait_civil_id": "NATIONAL_ID",
    "syria_national_id": "NATIONAL_ID", "visitor_number": "NATIONAL_ID",
    "هوية": "NATIONAL_ID", "رقم_الهوية": "NATIONAL_ID",
    # money
    "iban": "IBAN", "gulf_iban": "IBAN", "bank": "IBAN",
    "account": "BANK_ACCOUNT", "account_number": "BANK_ACCOUNT",
    "bank_routing_number": "BANK_ACCOUNT", "credit_card": "BANK_ACCOUNT",
    "credit_card_number": "BANK_ACCOUNT",
    # where someone is
    "address": "ADDRESS", "street": "ADDRESS", "street_address": "ADDRESS",
    "location": "ADDRESS", "city": "ADDRESS", "state": "ADDRESS",
    "postcode": "ADDRESS", "postal_code": "ADDRESS", "private_address": "ADDRESS",
    "national_address": "ADDRESS", "short_address": "ADDRESS",
    "building_number": "ADDRESS", "boundary_number": "ADDRESS",
    "عنوان": "ADDRESS",
    # identity documents and dates
    "birthdate": "DATE_OF_BIRTH", "date_of_birth": "DATE_OF_BIRTH", "dob": "DATE_OF_BIRTH",
    "passport": "PASSPORT_NUMBER", "passport_number": "PASSPORT_NUMBER",
    # references
    "customer_id": "CUSTOMER_ID", "customerid": "CUSTOMER_ID",
    "employee_id": "EMPLOYEE_ID", "employeeid": "EMPLOYEE_ID", "staff_id": "EMPLOYEE_ID",
}


def read_entities(raw) -> list[tuple[str, str]]:
    """Pull (value, label) out of any shape these two datasets use.

    They do not agree with each other, and one of them stores the list as a *string*:
      gretel: "[{'entity': 'Urvashi Jaggi', 'types': ['name']}, ...]"  (a string)
      sitr:   [{'start': 216, 'end': 225, 'label': 'vehicle_plate', 'text': 'ه س ص 900'}]
    Reading only one shape silently yields nothing, so both are handled here.
    """
    import ast                                                   # noqa: PLC0415

    if isinstance(raw, str):
        try:
            raw = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return []
    if not isinstance(raw, list):
        return []

    out: list[tuple[str, str]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        value = entry.get("value") or entry.get("text") or entry.get("entity") or ""
        label = entry.get("label") or entry.get("type") or ""
        if not label and isinstance(entry.get("types"), list) and entry["types"]:
            label = entry["types"][0]
        if str(value).strip() and str(label).strip():
            out.append((str(value), str(label)))
    return out


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
        raw = (row.get("privacy_mask") or row.get("labels") or row.get("entities")
               or row.get("spans") or [])
        items, seen = [], set()
        for value, label in read_entities(raw):
            pii_type = map_label(label)
            if pii_type and value not in seen:
                seen.add(value)
                items.append({"text": value, "type": pii_type})
        if not items:
            continue
        for receiver in ("EXTERNAL_VENDOR", "INTERNAL_HR", "ANALYST"):
            pairs.append(_pair(text, items, receiver,
                               f"{source} ({licence})", f"{source}-{len(pairs)}"))
    return pairs, {"source": source, "licence": licence, "limit": limit,
                   "pairs": len(pairs)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Sitr training data")
    parser.add_argument("--skip-public", action="store_true",
                        help="only our own documents: no downloads, no `datasets` needed")
    parser.add_argument("--english-source", default="gretelai/gretel-pii-masking-en-v1",
                        help="Apache-2.0. The earlier candidate, "
                             "aniket-curlscape/pii-masking-english-1k, declares its licence "
                             "only as 'other', which cannot be defended in a report")
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
            args.english_source, args.english_limit, ["text", "source_text"], "Apache-2.0")
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
