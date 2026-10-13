"""The policy's change history.

Release decisions and policy edits are recorded in the same SQLite file but in different
tables, on purpose. A decision is "this value in this document was replaced"; a policy
edit is "the rule that would decide such a value now says REMOVE instead of KEEP". Mixing
them would make the audit log answer two different questions with one row shape, and it
would let an administrative change hide among document decisions.

What is stored: the rule before and after, who changed it, when, and which version string
the change produced. What is not stored: any document or personal data - a policy edit
never touches a document.

Keeping this next to `app/audit/store.py` rather than inside it means the audit log's
"one row per decision" rule is untouched, while the policy still gets an append-only
history that a reviewer can read.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..audit.store import _now, _session


def _payload(edit) -> dict:
    return {"editor": edit.editor, "kind": edit.kind, "rule_id": edit.rule_id,
            "version_before": edit.version_before, "version_after": edit.version_after,
            "before": edit.before, "after": edit.after,
            "created_at": edit.created_at or _now(),
            "changes": edit.changes()}


def record_policy_edit(edit, db_path: str | Path | None = None) -> int:
    """Append one policy change. Returns the row id."""
    payload = _payload(edit)
    with _session(db_path) as connection:
        cursor = connection.execute(
            "INSERT INTO policy_edits (created_at, editor, kind, rule_id, version_before,"
            " version_after, changed_fields, before_json, after_json, note)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            (payload["created_at"], payload["editor"], payload["kind"], payload["rule_id"],
             payload["version_before"], payload["version_after"],
             json.dumps(payload["changes"], ensure_ascii=False),
             json.dumps(payload["before"], ensure_ascii=False) if payload["before"] else None,
             json.dumps(payload["after"], ensure_ascii=False) if payload["after"] else None,
             ""))
        connection.commit()
        return int(cursor.lastrowid or 0)


def policy_edits(limit: int = 50, rule_id: str | None = None,
                 db_path: str | Path | None = None) -> list[dict]:
    """The history, newest first, with the JSON blobs decoded for the UI."""
    query = "SELECT * FROM policy_edits"
    params: list = []
    if rule_id:
        query += " WHERE rule_id = ?"
        params.append(rule_id)
    query += " ORDER BY edit_id DESC LIMIT ?"
    params.append(limit)
    with _session(db_path) as connection:
        rows = [dict(row) for row in connection.execute(query, params).fetchall()]
    for row in rows:
        for source, target in (("before_json", "before"), ("after_json", "after"),
                               ("changed_fields", "changes")):
            raw = row.get(source)
            try:
                row[target] = json.loads(raw) if raw else ([] if source == "changed_fields"
                                                           else None)
            except (json.JSONDecodeError, TypeError):
                row[target] = None
    return rows


def latest_version(db_path: str | Path | None = None) -> str:
    with _session(db_path) as connection:
        row = connection.execute(
            "SELECT version_after FROM policy_edits ORDER BY edit_id DESC LIMIT 1"
        ).fetchone()
    return row["version_after"] if row else ""
