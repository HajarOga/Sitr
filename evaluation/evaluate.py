"""The evaluation harness: before/after numbers for the adaptation path.

    # 1. score the model as it is (the baseline), on the fixed test set
    python -m evaluation.evaluate --live --out predictions/base.jsonl

    # 2. after fine-tuning, and after alignment
    SAFESHARE2_ADAPTER=training/lora_adapter  python -m evaluation.evaluate --live --out predictions/lora.jsonl
    SAFESHARE2_ADAPTER=training/dpo_adapter   python -m evaluation.evaluate --live --out predictions/dpo.jsonl

    # 3. the table for the report - no model needed, works anywhere
    python -m evaluation.evaluate --compare predictions/base.jsonl predictions/lora.jsonl predictions/dpo.jsonl

Three modes, chosen because the numbers have to be producible in more than one place:

* `--live`   runs the real prompt through the configured model (the API, or a local model,
             with or without an adapter) and writes a predictions file.
* `--score`  scores a predictions file. Runs on any machine, including a laptop with no
             GPU and no key - so a run on Colab can be scored and compared at home.
* `--compare` puts several predictions files side by side.

What it never does: read or write `data/train/`. The evaluation set is `data/test/`, it
is fixed, and it is the only thing this script reads.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import config                                            # noqa: E402
from app.llm import engine, prompts                               # noqa: E402
from app.rules.store import library as default_library            # noqa: E402
from evaluation import metrics as M                               # noqa: E402

TEST_DIR = ROOT / "data" / "test"


def test_cases(directory: Path | None = None) -> list[dict]:
    """The fixed test set: one case per document, with its labels."""
    cases = []
    for gold_path in sorted((directory or TEST_DIR).glob("*.gold.json")):
        gold = json.loads(gold_path.read_text(encoding="utf-8"))
        text_path = gold_path.with_name(gold_path.name.replace(".gold.json", ".txt"))
        if not text_path.exists():
            continue
        cases.append({"document_id": gold["document_id"], "receiver": gold["receiver"],
                      "text": text_path.read_text(encoding="utf-8"),
                      "items": [{"text": s["text"], "type": s["type"]}
                                for s in gold["spans"]]})
    if not cases:
        raise SystemExit(f"no test cases in {directory or TEST_DIR}. Run:\n"
                         f"  python -m scripts.make_test_set\n"
                         f"  python -m scripts.make_no_rule_set   (for data/test_no_rules)")
    return cases


# ------------------------------------------------------------------------------- live


def run_live(out_path: Path, document_ids: list[str] | None = None,
             directory: Path | None = None) -> dict:
    """Ask the configured model for every test document and save the raw answers."""
    lib = default_library()
    state = engine.status()
    if not state.get("installed"):
        raise SystemExit("the model is not configured. Set a key in .env, or install the "
                         "local model with `python -m scripts.setup_model`.")

    cases = test_cases(directory)
    if document_ids:
        cases = [c for c in cases if c["document_id"] in document_ids]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for case in cases:
            rules = [{"rule_id": r.rule_id, "type": r.pii_type, "action": r.action,
                      "reason": r.reason} for r in lib.rules_for(case["receiver"])]
            messages = prompts.analyze_messages(
                case["text"], lib.receiver_label(case["receiver"]),
                lib.receivers[case["receiver"]].purpose, rules)
            engine.reset_calls()
            try:
                payload, meta = engine.generate_json(messages, purpose="evaluation")
            except engine.ModelCallFailed as exc:
                print(f"  {case['document_id']}: the model refused the call ({exc})")
                continue
            # `generate_json` returns the parsed payload, not text. Store it as text so the
            # predictions file is readable, and count what actually came back - counting the
            # keys of a dict once cost this project a whole baseline run.
            answer = "" if payload is None else json.dumps(payload, ensure_ascii=False)
            findings = (payload or {}).get("findings", []) if isinstance(payload, dict) else \
                (payload or [])
            handle.write(json.dumps({
                "document_id": case["document_id"], "receiver": case["receiver"],
                "answer": answer, "parsed": bool(payload), "findings": len(findings),
                "attempts": meta.get("attempts") if isinstance(meta, dict) else None,
                "model": state.get("model"), "adapter": config.ADAPTER,
                "seconds": meta.get("seconds") if isinstance(meta, dict) else None,
                "items": case["items"]}, ensure_ascii=False) + "\n")
            written += 1
            if payload is None:
                print(f"  {case['document_id']:<22} {len(case['items'])} labels, "
                      f"NO USABLE ANSWER after {meta.get('attempts')} attempts")
            else:
                print(f"  {case['document_id']:<22} {len(case['items'])} labels, "
                      f"{len(findings)} findings, {meta.get('seconds')}s")

    print(f"\n{written} predictions -> {out_path}")
    print("nothing was written to data/train/ or data/test/")
    return {"written": written, "path": str(out_path)}


# ------------------------------------------------------------------------------ score


def score_file(path: Path) -> tuple[list[M.DocumentScore], dict]:
    lib = default_library()
    scores = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        scores.append(M.score(row["answer"], row.get("items", []), row["receiver"],
                              library=lib, document_id=row["document_id"]))
    return scores, M.aggregate(scores)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Sitr against its test set")
    parser.add_argument("--live", action="store_true",
                        help="run the configured model on the test set")
    parser.add_argument("--out", default=str(ROOT / "predictions" / "run.jsonl"),
                        help="where --live writes its predictions")
    parser.add_argument("--score", default=None, help="score this predictions file")
    parser.add_argument("--compare", nargs="+", default=None,
                        help="two or more predictions files, side by side")
    parser.add_argument("--docs", nargs="+", default=None,
                        help="only these documents (with --live)")
    parser.add_argument("--test-dir", default=None,
                        help="which case directory to run: data/test (default) or "
                             "data/test_no_rules, for the receiver with no policy")
    parser.add_argument("--json", default=None, help="write the metrics here")
    args = parser.parse_args()

    if args.compare:
        tables, metrics = {}, {}
        for path in args.compare:
            scores, summary = score_file(Path(path))
            name = Path(path).stem
            tables[name] = scores
            metrics[name] = summary
            print(M.format_report(summary, scores, title=name))
            print()
        print(M.comparison(metrics))
        if args.json:
            Path(args.json).write_text(json.dumps(metrics, indent=1), encoding="utf-8")
        return 0

    if args.live:
        run_live(Path(args.out), args.docs,
                 Path(args.test_dir) if args.test_dir else None)

    path = Path(args.score or args.out)
    if not path.exists():
        raise SystemExit(f"{path} not found. Use --live to produce it, or --score FILE.")
    scores, summary = score_file(path)
    print(M.format_report(summary, scores, title=path.stem))
    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
