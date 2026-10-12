"""The audit record.

Every run writes one row per decision: what was found (type, masked value, hash),
what was done, under which rule, why, with what confidence, and who decided.

What is NOT written: the original value, and the released document. The audit proves
the decision was made and reviewed; it is not a second copy of the sensitive data.
"""
from __future__ import annotations

import csv
import io
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from contextlib import contextmanager

from .. import config

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


@contextmanager
def _session(db_path: str | Path | None = None):
    """A connection that is closed on the way out - on every platform.

    `with sqlite3.connect(...)` commits and rolls back, but it does NOT close the
    connection. On macOS and Linux that goes unnoticed, because deleting a file that is
    still open is allowed; on Windows the open handle locks the file, so a test that
    cleans up its temporary database fails with WinError 32. Closing explicitly is the
    difference between "passes on my machine" and "passes on the user's machine".
    """
    connection = _connect(db_path)
    try:
        with connection:                      # keeps the commit / rollback behaviour
            yield connection
    finally:
        connection.close()


def _connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    path = Path(db_path or config.DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path))
    connection.row_factory = sqlite3.Row
    connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    _migrate(connection)
    return connection


def _migrate(connection: sqlite3.Connection) -> None:
    """Bring an older database up to the current schema, in place.

    The log is append-only and it lives next to the app, so it gets upgraded rather than
    recreated: `decisions.rules_version` was added when the policy became editable, and a
    database written before that still has to open and still has to be readable.
    """
    future = {
        "decisions": {"rules_version": "TEXT"},
        "runs": {
            "processing_seconds": "REAL DEFAULT 0",
            "model_load_seconds": "REAL DEFAULT 0",
            "json_validation_seconds": "REAL DEFAULT 0",
        },
        "model_calls": {
            "tokens_in": "INTEGER",
            "tokens_per_second": "REAL",
            "prompt_chars": "INTEGER",
            "output_chars": "INTEGER",
            "json_attempt": "INTEGER",
            "json_valid": "INTEGER",
            "json_validation_seconds": "REAL DEFAULT 0",
        },
    }
    for table, columns in future.items():
        present = {row["name"] for row in
                   connection.execute(f"PRAGMA table_info({table})").fetchall()}
        for name, kind in columns.items():
            if name not in present:
                connection.execute(f"ALTER TABLE {table} ADD COLUMN {name} {kind}")
    connection.commit()


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def record_run(analysis, applied: dict, run_id: str | None = None,
               db_path: str | Path | None = None,
               ai_summary: str | None = None) -> str:
    """Write the run summary, one row per decision, and one row per model call."""
    run_id = run_id or uuid.uuid4().hex[:12]
    counts = {"REPLACE": 0, "REMOVE": 0, "KEEP": 0}
    for item in applied["applied"]:
        if item["action"] in counts:
            counts[item["action"]] += 1

    calls = list(getattr(analysis, "calls", []) or [])
    model_seconds = round(sum(float(c.get("seconds") or 0) for c in calls), 2)
    json_validation_seconds = round(sum(
        float(c.get("json_validation_seconds") or 0) for c in calls), 4)
    document_name = analysis.document.filename
    with _session(db_path) as connection:
        connection.execute(
            "INSERT INTO runs (run_id, created_at, document, receiver, purpose, items,"
            " replaced, removed, kept, pending, incomplete, rules_version, model,"
            " model_seconds, model_calls, processing_seconds, model_load_seconds,"
            " json_validation_seconds, ai_summary)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, _now(), document_name, analysis.receiver, analysis.purpose,
             len(analysis.detections), counts["REPLACE"], counts["REMOVE"], counts["KEEP"],
             len(applied["pending"]), int(applied["incomplete"]), analysis.rules_version,
             getattr(analysis, "model", "") or "", model_seconds, len(calls),
             float(getattr(analysis, "processing_seconds", 0.0) or 0.0),
             float(getattr(analysis, "model_load_seconds", 0.0) or 0.0),
             json_validation_seconds,
             ai_summary or getattr(analysis, "summary_text", None)))

        rows = []
        for item in applied["applied"]:
            detection = next((d for d in analysis.detections
                              if d.meta["id"] == item["id"]), None)
            rows.append((run_id, _now(), document_name, analysis.receiver, analysis.purpose,
                         item["type"], item["masked"], item["value_hash"], item["action"],
                         item["replacement"], item["rule_id"], item["reason"],
                         item["confidence"],
                         detection.meta.get("retrieval_score") if detection else None,
                         analysis.rules_version,
                         detection.detector if detection else None,
                         item["decided_by"]))
        for item in applied["pending"]:
            rows.append((run_id, _now(), document_name, analysis.receiver, analysis.purpose,
                         item["type"], item["masked"], item.get("value_hash", ""),
                         "ASK_HUMAN", None, item["rule_id"], item["reason"],
                         item["confidence"], None, analysis.rules_version, None, "unresolved"))
        connection.executemany(
            "INSERT INTO decisions (run_id, created_at, document, receiver, purpose,"
            " item_type, value_masked, value_hash, action, replacement, rule_id, reason,"
            " confidence, retrieval_score, rules_version, detector, decided_by)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)

        if calls:
            connection.executemany(
                "INSERT INTO model_calls (run_id, created_at, purpose, model, device,"
                " tokens_in, tokens_out, tokens_per_second, prompt_chars, output_chars,"
                " seconds, json_attempt, json_valid, json_validation_seconds, prompt_sha)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [(run_id, _now(), str(call.get("purpose", "")),
                  str(call.get("model", "")), str(call.get("device", "")),
                  int(call["tokens_in"]) if call.get("tokens_in") is not None else None,
                  int(call.get("tokens_out") or 0),
                  float(call["tokens_per_second"])
                  if call.get("tokens_per_second") is not None else None,
                  int(call["prompt_chars"]) if call.get("prompt_chars") is not None else None,
                  int(call["output_chars"]) if call.get("output_chars") is not None else None,
                  float(call.get("seconds") or 0),
                  int(call["json_attempt"]) if call.get("json_attempt") is not None else None,
                  int(bool(call["json_valid"])) if call.get("json_valid") is not None else None,
                  float(call.get("json_validation_seconds") or 0),
                  str(call.get("prompt_sha", ""))) for call in calls])
        connection.commit()
    return run_id


def model_calls(run_id: str, db_path: str | Path | None = None) -> list[dict]:
    """Every model call made for one run: the AI's own footprint."""
    with _session(db_path) as connection:
        rows = connection.execute(
            "SELECT * FROM model_calls WHERE run_id = ? ORDER BY call_id",
            (run_id,)).fetchall()
        return [dict(row) for row in rows]


def recent(limit: int = 50, db_path: str | Path | None = None) -> list[dict]:
    with _session(db_path) as connection:
        rows = connection.execute(
            "SELECT * FROM decisions ORDER BY decision_id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(row) for row in rows]


def runs(limit: int = 20, db_path: str | Path | None = None) -> list[dict]:
    with _session(db_path) as connection:
        rows = connection.execute(
            "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(row) for row in rows]


def decisions_for_run(run_id: str, db_path: str | Path | None = None) -> list[dict]:
    with _session(db_path) as connection:
        rows = connection.execute(
            "SELECT * FROM decisions WHERE run_id = ? ORDER BY decision_id", (run_id,)).fetchall()
        return [dict(row) for row in rows]


def summary(db_path: str | Path | None = None) -> dict:
    with _session(db_path) as connection:
        total = connection.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
        by_action = dict(connection.execute(
            "SELECT action, COUNT(*) FROM decisions GROUP BY action").fetchall())
        by_type = dict(connection.execute(
            "SELECT item_type, COUNT(*) FROM decisions GROUP BY item_type").fetchall())
        runs_count = connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    return {"runs": runs_count, "decisions": total,
            "by_action": by_action, "by_type": by_type}


def to_csv(rows: list[dict] | None = None, db_path: str | Path | None = None) -> str:
    rows = rows if rows is not None else recent(limit=1000, db_path=db_path)
    buffer = io.StringIO()
    if not rows:
        return ""
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def to_json(rows: list[dict] | None = None, db_path: str | Path | None = None) -> str:
    return json.dumps(rows if rows is not None else recent(limit=200, db_path=db_path),
                      ensure_ascii=False, indent=1)
