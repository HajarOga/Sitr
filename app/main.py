"""FastAPI backend.

    uvicorn app.main:app --reload --port 8000     then open http://127.0.0.1:8000/docs

The dashboard calls the pipeline directly, so this server is optional - it exists so the
same system can be scripted or driven by a different front end without touching the
pipeline. The routes are the same shape as the first prototype's, with two additions:
`/api/model` reports whether the AI is installed (it is mandatory here), and every
analysis response carries the model's own footprint - how many calls, how many seconds.

Privacy: an uploaded document is written to the temporary folder only for as long as it
takes to read it, then deleted. Sessions live in memory and disappear with the process.
Whether the document text leaves this machine depends on the configured provider: with
`SAFESHARE2_PROVIDER=local` it does not, with `api` the prompt (which contains the
document) is sent to the configured endpoint. `/api/model` reports which one is active.
"""
from __future__ import annotations

import uuid
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel

from . import config
from .audit import store as audit
from .core import analyze, apply_decisions, write_summary
from .document_processing import readers
from .document_processing.writers import safe_text, write_release
from .llm import engine
from .policy import policy_audit
from .policy import qa as policy_qa
from .policy.policy_index import PolicyIndexError, get_index
from .review.review_queue import apply_review, build_queue, review_state
from .rules.store import library as default_library

app = FastAPI(title="SafeShare2", version="0.1.0",
              description="Create a safe version of a document for each receiver, with "
                          "a local language model making the decisions.")

# in-memory only: a session is a document under review. Nothing is persisted here; the
# decisions that matter go to the audit log.
SESSIONS: dict[str, dict] = {}
RELEASE_DIR = config.TMP_DIR / "releases"


def _public_blocks(document, changed: dict[str, str] | None = None) -> list[dict]:
    blocks = []
    for block in document.blocks:
        if not block.text.strip():
            blocks.append({"id": block.id, "kind": block.kind, "text": ""})
            continue
        blocks.append({"id": block.id, "kind": block.kind, "section": block.section,
                       "text": (changed or {}).get(block.id, block.text)})
    return blocks


# ---------------------------------------------------------------------- meta routes


@app.get("/api/health")
def health() -> dict:
    state = engine.status()
    return {"status": "ok",
            "model": {k: state[k] for k in ("model", "installed", "partial", "size_gb",
                                            "device", "dtype", "path")},
            "rules_version": default_library().version,
            "temporary_files": str(config.TMP_DIR),
            "uploads_kept": config.KEEP_UPLOADS}


@app.get("/api/model")
def model_status() -> dict:
    """Whether the AI is installed, and the command that installs it if not."""
    return engine.status()


@app.get("/api/receivers")
def receivers() -> list[dict]:
    lib = default_library()
    return [{"key": key, "label": value.label, "purpose": value.purpose,
             "description": value.description, "rules": len(lib.rules_for(key))}
            for key, value in lib.receivers.items()]


@app.get("/api/rules")
def rules(receiver: str | None = None) -> list[dict]:
    lib = default_library()
    selected = lib.rules_for(receiver) if receiver else lib.rules
    return [asdict(rule) for rule in selected]


# ------------------------------------------------------------------- policy routes
# Read and ask, but not edit: a policy change is made by a person in the dashboard, where
# they see the rule, the action and the confirmation, and where it is audited with a name
# attached. Opening a write route here would put "who changed company policy" at the mercy
# of a token, which is a trade this project deliberately does not make.


class PolicyQuestion(BaseModel):
    question: str
    receiver: str | None = None
    pii_type: str | None = None
    top_k: int | None = None
    use_model: bool = True


@app.get("/api/policy")
def policy_status() -> dict:
    """The policy, its version, and whether the search index is level with it."""
    lib = default_library()
    try:
        index = get_index().status()
    except PolicyIndexError as exc:
        index = {"in_sync": False, "error": str(exc)}
    return {"version": lib.version, "rules": len(lib.rules),
            "receivers": {key: {"label": value.label, "purpose": value.purpose,
                                "rules": len(lib.rules_for(key))}
                          for key, value in lib.receivers.items()},
            "index": index}


@app.post("/api/policy/ask")
def policy_ask(request: PolicyQuestion) -> dict:
    """Answer a policy question from retrieved rules. Optional local model, checked cites."""
    try:
        return policy_qa.ask_policy(request.question, receiver=request.receiver,
                                    pii_type=request.pii_type, top_k=request.top_k,
                                    use_model=request.use_model)
    except PolicyIndexError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/policy/history")
def policy_history(limit: int = 50) -> list[dict]:
    """Every recorded change to the policy, newest first."""
    return policy_audit.policy_edits(limit=limit)


# ------------------------------------------------------------------ analyze routes


def _analyze_document(document, receiver: str, purpose: str) -> dict:
    try:
        analysis = analyze(document, receiver, purpose)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except engine.ModelNotInstalled as exc:
        # 503: the service exists but its mandatory dependency does not.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:                                   # noqa: BLE001
        raise HTTPException(status_code=502,
                            detail=f"the model did not produce a usable answer: {exc}") from exc

    session_id = uuid.uuid4().hex[:12]
    SESSIONS[session_id] = {"analysis": analysis}
    return {"session_id": session_id,
            "model": analysis.model,
            "model_calls": analysis.calls,
            "summary": analysis.summary(),
            "ai_summary": analysis.summary_text,
            "records": analysis.records(),
            "review_queue": [item.to_dict() for item in build_queue(analysis)],
            "guard_notes": analysis.notes,
            "blocks": _public_blocks(document)}


@app.post("/api/analyze")
async def analyze_upload(file: UploadFile = File(...), receiver: str = Form(...),
                         purpose: str = Form("")) -> dict:
    content = await file.read()
    try:
        document = readers.read_bytes(file.filename or "upload.txt", content)
    except readers.UnsupportedDocument as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _analyze_document(document, receiver, purpose)


class TextRequest(BaseModel):
    text: str
    filename: str = "pasted.txt"
    receiver: str = "EXTERNAL_VENDOR"
    purpose: str = ""


@app.post("/api/analyze-text")
def analyze_text(request: TextRequest) -> dict:
    """Same pipeline, for text pasted into a form (and for the tests)."""
    path = readers.store_upload(request.filename, request.text.encode("utf-8"))
    try:
        document = readers.read(path, request.filename)
    finally:
        readers.discard(path)
    return _analyze_document(document, request.receiver, request.purpose)


# ------------------------------------------------------------------ release routes


class ReleaseRequest(BaseModel):
    session_id: str
    decisions: dict[str, str] = {}
    reviewer: str = "reviewer"


def _session(session_id: str) -> dict:
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="unknown or expired session")
    return SESSIONS[session_id]


@app.post("/api/release")
def release(request: ReleaseRequest) -> dict:
    session = _session(request.session_id)
    analysis = session["analysis"]
    applied = apply_review(analysis, request.decisions, reviewer=request.reviewer)

    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    written = write_release(analysis.document, RELEASE_DIR, request.session_id,
                            applied["changed"])
    summary_text = write_summary(analysis, applied)
    run_id = audit.record_run(analysis, applied, ai_summary=summary_text)
    session["last"] = {"applied": applied, "written": written, "run_id": run_id}

    return {"session_id": request.session_id, "run_id": run_id,
            "model": analysis.model,
            "safe_text": safe_text(analysis.document, applied["changed"]),
            "blocks": _public_blocks(analysis.document, applied["changed"]),
            "applied": applied["applied"], "pending": applied["pending"],
            "incomplete": applied["incomplete"],
            "download": written,
            "ai_summary": summary_text,
            "review_state": review_state(analysis, request.decisions)}


@app.get("/api/release/download/{session_id}")
def download(session_id: str):
    session = _session(session_id)
    written = (session.get("last") or {}).get("written")
    if not written:
        raise HTTPException(status_code=404, detail="generate the release first")
    path = Path(written["path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="the release file is no longer there")
    return FileResponse(str(path), filename=path.name)


@app.get("/api/compare")
def compare(session_id: str, receivers: str | None = None) -> dict:
    """The same document, one version per receiver - each prepared by the model.

    This is the expensive route on purpose: every receiver is a fresh model call, because
    the decision genuinely depends on the receiver. `receivers` narrows it.
    """
    analysis = _session(session_id)["analysis"]
    lib = default_library()
    wanted = ([key.strip() for key in receivers.split(",") if key.strip()]
              if receivers else list(lib.receivers))
    out = {}
    for key in wanted:
        if key not in lib.receivers:
            continue
        other = analyze(analysis.document, key)
        applied = apply_decisions(other)
        out[key] = {"label": lib.receivers[key].label,
                    "summary": other.summary(),
                    "safe_text": safe_text(analysis.document, applied["changed"]),
                    "applied": applied["applied"], "pending": applied["pending"]}
    return {"document": analysis.document.filename, "versions": out}


# -------------------------------------------------------------------- audit routes


@app.get("/api/audit")
def audit_recent(limit: int = 50) -> list[dict]:
    return audit.recent(limit=limit)


@app.get("/api/audit/runs")
def audit_runs(limit: int = 20) -> list[dict]:
    return audit.runs(limit=limit)


@app.get("/api/audit/model-calls/{run_id}")
def audit_model_calls(run_id: str) -> list[dict]:
    """The AI's footprint for one run: which jobs, how long, how many tokens."""
    return audit.model_calls(run_id)


@app.get("/api/audit/summary")
def audit_summary() -> dict:
    return audit.summary()


@app.get("/api/audit/export.csv")
def audit_export(limit: int = 1000) -> PlainTextResponse:
    return PlainTextResponse(audit.to_csv(audit.recent(limit=limit)),
                             media_type="text/csv",
                             headers={"Content-Disposition":
                                      "attachment; filename=safeshare2-audit.csv"})
