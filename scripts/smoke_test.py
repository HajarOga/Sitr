"""End-to-end smoke test - what a person would do, in order.

    python -m scripts.smoke_test

Run this after `python -m scripts.setup_model`. It checks the whole chain: the model is
installed and loads, it reads the sample document, the guards accept the answer, a
release document is written, the audit log records both the decisions and the model's
calls. Then it renders all three dashboard pages and, if the API is installed, posts a
document through it.

Everything that needs the model is skipped - and reported as skipped, never as passed -
when the weights are not on this machine, so the script is still useful on a fresh clone.
"""
from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import config, core, guards                          # noqa: E402
from app.audit import store as audit                          # noqa: E402
from app.document_processing import readers                   # noqa: E402
from app.document_processing.document import Block, Document  # noqa: E402
from app.llm import engine                                    # noqa: E402
from app.pii_detection.schema import Action                   # noqa: E402
from app.replacement.generator import FakeGenerator           # noqa: E402
from app.rules.store import library as default_library        # noqa: E402

PASSED: list[str] = []
FAILED: list[tuple[str, str]] = []
SKIPPED: list[tuple[str, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSED.append(name)
        print(f"PASS  {name}" + (f"  [{detail}]" if detail else ""))
    else:
        FAILED.append((name, detail))
        print(f"FAIL  {name}" + (f"  [{detail}]" if detail else ""))


def skip(name: str, why: str) -> None:
    SKIPPED.append((name, why))
    print(f"SKIP  {name}  [{why}]")


def sample_document() -> Document:
    path = sorted((config.DATA_DIR / "synthetic").glob("*.txt"))[0]
    stored = readers.store_upload(path.name, path.read_bytes())
    try:
        return readers.read(stored, path.name)
    finally:
        readers.discard(stored)


def run_gate_checks(state: dict) -> None:
    print("\n-- the model gate " + "-" * 40)
    check("the interpreter running this file has the libraries",
          True, sys.executable)
    if state["installed"]:
        detail = (f"{state['model']} at {state.get('base_url', '')}"
                  if state.get("provider") == "api"
                  else f"{state['model']} {state['size_gb']:.2f} GB")
        check("the model is installed", True, detail)
    else:
        skip("the model is installed", "run: python -m scripts.setup_model")
        return
    try:
        engine.load()
        info = engine.loaded_info()
        check("the model loads", True, f"{info['device']}/{info['dtype']}")
    except Exception as exc:                                   # noqa: BLE001
        check("the model loads", False, f"{type(exc).__name__}: {exc}")


def run_pipeline_checks(state: dict) -> dict:
    print("\n-- the pipeline " + "-" * 42)
    lib = default_library()
    check("the policy library loads", len(lib.rules) >= 48, f"{len(lib.rules)} rules")

    if not state["installed"]:
        skip("the model analyses a real document", "model not installed")
        return {}

    engine.reset_calls()
    document = sample_document()
    analysis = core.analyze(document, "EXTERNAL_VENDOR",
                            "a vendor is fixing the line")
    check("the model found items", len(analysis.detections) > 0,
          f"{len(analysis.detections)} items")
    check("every change cites a rule or is a question",
          all(d.rule_id or d.action == Action.ASK_HUMAN.value
              for d in analysis.detections))
    check("no replacement is left unvalidated",
          all(d.replacement for d in analysis.detections
              if d.action == Action.REPLACE.value))
    check("the model calls were recorded", len(analysis.calls) > 0,
          f"{len(analysis.calls)} calls, {analysis.summary()['model_seconds']}s")
    check("the model wrote an audit summary", bool(analysis.summary_text))

    applied = core.apply_decisions(analysis)
    released = applied["changed"]
    leaks = [d.original_text for d in analysis.detections
             if d.action in (Action.REMOVE.value, Action.REPLACE.value)
             and d.original_text in released.get(d.block_id, "")]
    check("no removed or replaced value is still in the release", not leaks, str(leaks[:3]))

    written = __import__("app.document_processing.writers",
                         fromlist=["write_release"]).write_release(
        document, config.TMP_DIR / "releases", Path(document.filename).stem, released)
    check("a release document was written", Path(written["path"]).exists(),
          str(written["path"]))

    with tempfile.TemporaryDirectory() as folder:
        db = Path(folder) / "audit.sqlite3"
        run_id = audit.record_run(analysis, applied, db_path=db,
                                  ai_summary=analysis.summary_text)
        calls = audit.model_calls(run_id, db_path=db)
        check("the audit log recorded the model calls", len(calls) == len(analysis.calls),
              f"{len(calls)} rows")
        check("the audit log does not store an original value",
              all(d.original_text.encode("utf-8") not in db.read_bytes()
                  for d in analysis.detections if len(d.original_text) > 6))
    return {"analysis": analysis, "applied": applied}


def run_ui_checks(analysis) -> None:
    print("\n-- the dashboard " + "-" * 41)
    try:
        from streamlit.testing.v1 import AppTest
    except ImportError:                                        # pragma: no cover
        skip("the dashboard renders", "streamlit is not installed")
        return

    ui = str(ROOT / "app" / "ui_streamlit.py")

    upload = AppTest.from_file(ui, default_timeout=90)
    upload.run()
    check("the upload page renders without an exception", not upload.exception,
          str(upload.exception[0].value) if upload.exception else "")

    if analysis is None:
        skip("the review page renders", "no analysis (model not installed)")
        skip("the audit page renders", "no analysis (model not installed)")
        return

    # Navigate the way a person does - through the sidebar radio. Setting
    # `session_state["page"]` looks like it works and does nothing, because the radio
    # owns that key: that is how this check passed while rendering the wrong page.
    review = AppTest.from_file(ui, default_timeout=90)
    review.session_state["analysis"] = analysis
    review.session_state["decisions"] = {}
    review.run()
    review.radio[0].set_value("Review").run()
    check("the review page renders without an exception", not review.exception,
          str(review.exception[0].value) if review.exception else "")
    check("the review page shows a decision per item",
          len(review.dataframe) >= 1 and len(review.metric) >= 5,
          f"{len(review.dataframe)} table(s), {len(review.metric)} metrics")

    audit_page = AppTest.from_file(ui, default_timeout=90)
    audit_page.run()
    audit_page.radio[0].set_value("Audit log").run()
    check("the audit page renders without an exception", not audit_page.exception,
          str(audit_page.exception[0].value) if audit_page.exception else "")


def run_policy_page_check() -> None:
    """The Policy page renders - it does not need the model, only the index."""
    print("\n-- the policy page " + "-" * 40)
    try:
        from streamlit.testing.v1 import AppTest
    except ImportError:                                        # pragma: no cover
        skip("the policy page renders", "streamlit is not installed")
        return
    try:
        from app.policy.policy_index import SentenceTransformerEmbedder

        if not SentenceTransformerEmbedder().is_installed():
            skip("the policy page renders", "the embedder is not installed")
            return
    except Exception:                                          # noqa: BLE001
        skip("the policy page renders", "the policy index is not available")
        return

    policy_page = AppTest.from_file(str(ROOT / "app" / "ui_streamlit.py"), default_timeout=120)
    policy_page.run()
    policy_page.radio[0].set_value("Policy").run()
    check("the policy page renders without an exception", not policy_page.exception,
          str(policy_page.exception[0].value) if policy_page.exception else "")
    # ask a question through the page's own widgets, with the model switched off
    if policy_page.text_input:
        policy_page.text_input[0].set_value("May an external vendor see a phone number?")
    for toggle in policy_page.toggle:
        if "model" in str(getattr(toggle, "label", "")).lower():
            toggle.set_value(False)
    if policy_page.button:
        policy_page.button[0].click().run()
    check("the policy page answers a question from its own widgets",
          not policy_page.exception and any(
              "VENDOR-PII" in str(block.value) for block in policy_page.code),
          str(policy_page.exception[0].value) if policy_page.exception else "")


def run_policy_checks() -> None:
    """The policy index, questions and the edit path - none of which needs the 3B model."""
    print("\n-- the policy index " + "-" * 38)
    from app.policy import qa as policy_qa
    from app.policy.policy_editor import PolicyEditor
    from app.policy.policy_index import (PolicyIndexError, SentenceTransformerEmbedder,
                                         get_index)

    embedder = SentenceTransformerEmbedder()
    if not embedder.is_installed():
        skip("the embedding model is installed", "run: python -m scripts.setup_model")
        skip("the policy index is in sync", "the embedder is not installed")
        return
    try:
        index = get_index()
        report = index.sync()
        status = index.status()
    except PolicyIndexError as exc:
        check("the policy index builds", False, str(exc)[:120])
        return
    check("the policy index builds", status["in_sync"],
          f"{status['documents']} rules in {status['collection']}")
    check("the index is level with the policy file",
          report["embedded"] == 0 or report["rules"] == status["rules_in_policy"],
          f"{report['embedded']} embedded this run")

    top = index.query("May an external vendor see a phone number?", receiver="EXTERNAL_VENDOR",
                      pii_type="PHONE", top_k=1)
    check("a question finds the rule that answers it",
          bool(top) and top[0]["pii_type"] == "PHONE", top[0]["rule_id"] if top else "none")

    result = policy_qa.ask_policy("May an external vendor see a phone number?",
                                  use_model=False, index=index)
    check("a policy question comes back with citations", bool(result["citations"]),
          ", ".join(citation["rule_id"] for citation in result["citations"][:3]))

    with tempfile.TemporaryDirectory() as folder:
        target = Path(folder) / "rules.json"
        target.write_text(config.RULES_PATH.read_text(encoding="utf-8"), encoding="utf-8")
        editor = PolicyEditor(path=target, db_path=Path(folder) / "audit.sqlite3")
        try:
            edit = editor.set_rule("VENDOR-PII-003", action="KEEP",
                                   reason="Vendors need the callback number.",
                                   editor="smoke test", sync_index=False)
            check("a policy edit is validated and version-bumped", edit["changed"]
                  and edit["version"] != edit["version_before"],
                  f"{edit['version_before']} -> {edit['version']}")
            check("the policy edit is recorded in the history",
                  bool(editor.history(limit=1)), "")
        except Exception as exc:                               # noqa: BLE001
            check("a policy edit is validated and version-bumped", False,
                  f"{type(exc).__name__}: {exc}")


def run_api_check(state: dict) -> None:
    print("\n-- the API " + "-" * 46)
    if not state["installed"]:
        skip("the API answers /api/health", "model not installed")
        return
    try:
        from fastapi.testclient import TestClient
    except ImportError:                                        # pragma: no cover
        skip("the API answers /api/health", "fastapi/httpx is not installed")
        return
    from app.main import app
    client = TestClient(app)
    health = client.get("/api/health")
    check("the API answers /api/health", health.status_code == 200,
          f"{health.status_code} {health.text[:80]}")
    status = client.get("/api/model")
    check("the API reports the model", status.status_code == 200
          and status.json().get("installed") is True, status.text[:80])


def main() -> int:
    print("Sitr smoke test")
    state = engine.install_state()
    if not state["installed"]:
        print("\nThe model is not installed yet, so only the parts that do not need it "
              "will run.\nInstall it with:  python -m scripts.setup_model\n")
    try:
        run_gate_checks(state)
        result = run_pipeline_checks(state)
        run_policy_checks()
        run_ui_checks(result.get("analysis"))
        run_policy_page_check()
        run_api_check(state)
    except Exception:                                          # noqa: BLE001
        traceback.print_exc()
        FAILED.append(("smoke test crashed", "see traceback above"))

    print("\n" + "=" * 60)
    print(f"{len(PASSED)} passed, {len(FAILED)} failed, {len(SKIPPED)} skipped")
    for name, why in FAILED:
        print(f"  failed: {name} ({why})")
    for name, why in SKIPPED:
        print(f"  skipped: {name} ({why})")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
