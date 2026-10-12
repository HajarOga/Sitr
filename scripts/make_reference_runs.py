"""Write reference runs so the metrics can be demonstrated and tested without a GPU.

    python -m scripts.make_reference_runs

These are **not model output**. They are three fixed answers to the same test set, used
to show that the metrics discriminate and to keep that behaviour under test:

* `reference-perfect` - exactly what the policy expects, item by item;
* `reference-naive`   - notices every item and redacts it, but cites a rule that does not
  exist. A plausible answer that would still publish all 49 values;
* `reference-silent`  - reports nothing at all, which is the same leak with less work.

Real numbers come from `python -m evaluation.evaluate --live`, which runs the configured
model instead of these stand-ins.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.reference import expected_findings            # noqa: E402

OUT = ROOT / "predictions"


def main() -> None:
    cases = []
    for gold_path in sorted((ROOT / "data" / "test").glob("*.gold.json")):
        gold = json.loads(gold_path.read_text(encoding="utf-8"))
        cases.append({"document_id": gold["document_id"], "receiver": gold["receiver"],
                      "items": [{"text": span["text"], "type": span["type"]}
                                for span in gold["spans"]]})
    if not cases:
        raise SystemExit("no test cases. Run: python -m scripts.make_test_set")

    OUT.mkdir(exist_ok=True)
    perfection, naive, silent = [], [], []
    for case in cases:
        findings = expected_findings(case["items"], case["receiver"])
        perfection.append({**case, "answer": json.dumps({"findings": findings},
                                                        ensure_ascii=False)})
        naive.append({**case, "answer": json.dumps({"findings": [
            {**finding, "action": "REPLACE", "rule_id": "VENDOR-PII-999",
             "replacement": "REDACTED", "reason": "Redacted."} for finding in findings]},
            ensure_ascii=False)})
        silent.append({**case, "answer": '{"findings": []}'})

    for name, rows in (("reference-perfect", perfection), ("reference-naive", naive),
                       ("reference-silent", silent)):
        path = OUT / f"{name}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                        encoding="utf-8")
        print(f"{len(rows):>3} cases -> {path}")

    print("\nscore them with:")
    print("  python -m evaluation.evaluate --compare "
          "predictions/reference-perfect.jsonl predictions/reference-naive.jsonl "
          "predictions/reference-silent.jsonl")


if __name__ == "__main__":
    main()
