"""Package the training data so a person can look at it, and so it can be handed over.

    python -m scripts.export_training_data

Writes to `data/export/`:

* `safeshare-training-data.zip` - the whole set: the 24 documents with their labels, and
  both JSONL files that the training scripts read.
* `safeshare-training-set.xlsx` - the same thing as a readable index: one row per training
  example and per preference pair, so the set can be filtered and checked without reading
  JSON. Falls back to CSV if `openpyxl` is not installed.
* `README.txt` - what is inside, in plain words, and where every value came from.

What it does **not** include: the two public datasets (`aniket-curlscape/pii-masking-english-1k`,
`mabahboh/sitr-arabic-pii`). Those are downloaded by `training/prepare_data.py` when it is
run without `--skip-public`, because their licences allow use but the polite thing is to
fetch them from the source rather than redistribute a copy. This script says so in the
README it writes, so nobody assumes the shipped set is the whole story.
"""
from __future__ import annotations

import csv
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXPORT = ROOT / "data" / "export"
TRAIN_DOCS = ROOT / "data" / "train" / "documents"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def counts(payload: str) -> dict:
    try:
        findings = json.loads(payload)["findings"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return {}
    tally: dict[str, int] = {}
    for finding in findings:
        action = str(finding.get("action", "?"))
        tally[action] = tally.get(action, 0) + 1
    return tally


def rows_for_examples(rows: list[dict]) -> list[list[str]]:
    out = [["#", "document", "receiver", "source", "items in the target answer",
            "keep", "replace", "remove", "ask", "prompt characters"]]
    for index, row in enumerate(rows, 1):
        tally = counts(row["completion"])
        prompt_chars = sum(len(m["content"]) for m in row.get("messages", []))
        out.append([str(index), row.get("document", ""), row.get("receiver", ""),
                    row.get("source", ""), str(sum(tally.values())),
                    str(tally.get("KEEP", 0)), str(tally.get("REPLACE", 0)),
                    str(tally.get("REMOVE", 0)), str(tally.get("ASK_HUMAN", 0)),
                    str(prompt_chars)])
    return out


def _difference(chosen_payload: str, rejected_payload: str) -> str:
    """Spell out what the rejected answer does differently - a tally hides the lesson."""
    try:
        chosen = json.loads(chosen_payload)["findings"]
        rejected = json.loads(rejected_payload)["findings"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return ""
    if not rejected:
        return f"reports nothing at all, so all {len(chosen)} values survive"
    by_value = {item.get("original_text"): item for item in rejected}
    notes = []
    for item in chosen:
        other = by_value.get(item.get("original_text"))
        if other is None:
            notes.append(f"drops {item.get('original_text')!r}")
        elif other != item:
            if (item.get("action") == other.get("action")
                    and item.get("rule_id") == other.get("rule_id")):
                # the action is right and only the substitute is wrong - say that, because
                # "REPLACE -> REPLACE" reads like nothing changed
                notes.append(f"{item.get('original_text')!r}: substitute "
                             f"{item.get('replacement')!r} -> {other.get('replacement')!r}")
            else:
                notes.append(f"{item.get('original_text')!r}: {item.get('action')} "
                             f"({item.get('rule_id') or 'no rule'}) -> {other.get('action')} "
                             f"({other.get('rule_id') or 'no rule'})")
    return "; ".join(notes[:3]) + (" ..." if len(notes) > 3 else "")


def rows_for_pairs(rows: list[dict]) -> list[list[str]]:
    out = [["#", "kind of mistake being corrected", "document", "receiver",
            "outcome in the preferred answer", "what the rejected answer does instead"]]
    for index, row in enumerate(rows, 1):
        chosen = counts(row.get("chosen", ""))
        out.append([str(index), row.get("kind", ""), row.get("document", ""),
                    row.get("receiver", ""),
                    ", ".join(f"{k} {v}" for k, v in chosen.items()),
                    _difference(row.get("chosen", ""), row.get("rejected", ""))])
    return out


def rows_for_documents() -> list[list[str]]:
    out = [["document", "document type", "language", "receiver", "labelled values",
            "types"]]
    for gold_path in sorted(TRAIN_DOCS.glob("*.gold.json")):
        gold = json.loads(gold_path.read_text(encoding="utf-8"))
        types = sorted({span["type"] for span in gold["spans"]})
        out.append([gold["document_id"], gold.get("document_type", ""),
                    gold.get("language", ""), gold.get("receiver", ""),
                    str(len(gold["spans"])), ", ".join(types)])
    return out


def write_xlsx(path: Path, sheets: dict[str, list[list[str]]]) -> str:
    try:
        from openpyxl import Workbook                             # noqa: PLC0415
        from openpyxl.styles import Font                           # noqa: PLC0415
    except ImportError:
        path = path.with_suffix(".csv")
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            for name, rows in sheets.items():
                writer.writerow([f"-- {name} --"])
                writer.writerows(rows)
                writer.writerow([])
        return f"{path} (CSV: openpyxl is not installed, so the sheets are stacked in one file)"

    book = Workbook()
    book.remove(book.active)
    for name, rows in sheets.items():
        sheet = book.create_sheet(name[:31])
        for row in rows:
            sheet.append(row)
        for cell in sheet[1]:
            cell.font = Font(bold=True)
        for column in range(1, len(rows[0]) + 1):
            width = max(len(str(row[column - 1])) for row in rows[:80]) if rows else 12
            sheet.column_dimensions[sheet.cell(row=1, column=column).column_letter] \
                 .width = min(max(width + 2, 9), 42)
        sheet.freeze_panes = "A2"
    book.save(path)
    return str(path)


def main() -> None:
    sft = load_jsonl(ROOT / "data" / "train" / "sft.jsonl")
    validation = load_jsonl(ROOT / "data" / "validation" / "sft.jsonl")
    pairs = load_jsonl(ROOT / "data" / "train" / "preference_pairs.jsonl")
    if not sft and not pairs:
        raise SystemExit("no training data yet. Run:\n"
                         "  python -m training.make_training_documents\n"
                         "  python -m training.prepare_data --skip-public\n"
                         "  python -m training.make_preference_pairs")

    kind_counts: dict[str, int] = {}
    for row in pairs:
        kind_counts[row["kind"]] = kind_counts.get(row["kind"], 0) + 1

    EXPORT.mkdir(parents=True, exist_ok=True)

    index_path = write_xlsx(EXPORT / "safeshare-training-set.xlsx", {
        "Training examples": rows_for_examples(sft),
        "Validation examples": rows_for_examples(validation),
        "Preference pairs": rows_for_pairs(pairs),
        "Documents": rows_for_documents(),
        "Data sources": [
            ["source", "what it is", "licence", "used for"],
            ["our own synthetic documents",
             f"{len(rows_for_documents()) - 1} documents, Arabic / English / mixed, values "
             "generated with their labels",
             "ours", "training targets and the fixed evaluation set (data/test, never trained on)"],
            ["aniket-curlscape/pii-masking-english-1k", "1,000 English masking examples",
             "see the data card on Hugging Face", "volume for fine-tuning (optional step)"],
            ["mabahboh/sitr-arabic-pii", "Arabic PII examples",
             "Apache-2.0", "Arabic coverage, sampled at 2,500 maximum"],
            ["ai4privacy/pii-masking-65k", "checked, then rejected: no licence declared, no Arabic",
             "none declared", "only its taxonomy of sensitive types was copied"],
        ],
    })

    zip_path = EXPORT / "safeshare-training-data.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(TRAIN_DOCS.glob("*")):
            archive.write(path, f"documents/{path.name}")
        archive.write(ROOT / "data" / "train" / "sft.jsonl", "sft-train.jsonl")
        archive.write(ROOT / "data" / "validation" / "sft.jsonl", "sft-validation.jsonl")
        archive.write(ROOT / "data" / "train" / "preference_pairs.jsonl",
                      "preference_pairs.jsonl")
        archive.write(ROOT / "docs" / "MODEL_ADAPTATION.md", "MODEL_ADAPTATION.md")

    readme = f"""SafeShare - the training data
=============================

Built from commit {_head()}.

Contents
--------
documents/                 the {len(rows_for_documents()) - 1} documents, each with a .txt and a
                           .gold.json holding the labelled values
sft-train.jsonl            {len(sft)} prompt/completion pairs, the fine-tuning set
sft-validation.jsonl       {len(validation)} held-out pairs, to watch for overfitting
preference_pairs.jsonl     {len(pairs)} chosen/rejected pairs, the alignment set:
{chr(10).join(f'                             {count:>4}  {kind}' for kind, count in sorted(kind_counts.items()))}
MODEL_ADAPTATION.md        where the generative AI is, and what each part does

How a pair is shaped
--------------------
Each row holds the exact `messages` the application sends (system contract, receiver,
purpose, the policy rules for that receiver, then the document) and the `completion` the
guards would accept. The rule in the prompt is retrieved from the policy index at run
time - the model is not trained on any rule, which is why editing a rule changes behaviour
with no retraining.

Where the values come from
--------------------------
Every value in these documents is invented for this project: names, national IDs, phones,
IBANs, addresses. The generator uses reserved test vocabulary (example.com, 0500000000,
1000000000) so a released document cannot be mistaken for real data. `make_training_documents.py`
checks that no value in the training set appears in the evaluation set, and prints the
result of that check every time it runs.

What is NOT in this file
------------------------
The two public datasets named in the brief are fetched from Hugging Face by
`python -m training.prepare_data` (without `--skip-public`) rather than redistributed here:
  aniket-curlscape/pii-masking-english-1k   see the data card
  mabahboh/sitr-arabic-pii                  Apache-2.0, sampled at 2,500 maximum
So the full training run uses these files plus those two sources. This file is the part
that is ours.
"""
    (EXPORT / "README.txt").write_text(readme, encoding="utf-8")

    print(f"zip  : {zip_path}  ({zip_path.stat().st_size / 1024:.0f} KB)")
    print(f"index: {index_path}")
    print(f"readme: {EXPORT / 'README.txt'}")
    print(f"\ninside: {len(sft)} training + {len(validation)} validation pairs, "
          f"{len(pairs)} preference pairs, {len(rows_for_documents()) - 1} documents")


def _head() -> str:
    import subprocess                                             # noqa: PLC0415
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:                                             # noqa: BLE001
        return "unknown"


if __name__ == "__main__":
    main()
