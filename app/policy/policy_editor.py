"""The only code allowed to write `rules.json`.

Why this exists
---------------
`app/rules/rules.json` is the authority the whole release pipeline cites, and
`app/rules/store.py` refuses to load a library that is broken, because a policy that
silently fails to match is worse than one that crashes. That rule has a consequence: if
the app is going to let a person edit the policy, the write has to be **at least as
careful as the read**. So every mutation here goes through:

1. **Edit a copy, validate the whole library, then commit.** The new payload is parsed
   back into a `RulesLibrary` before anything touches the file. Nothing that would fail
   to load is ever written.
2. **Atomic write.** The new file is written next to the old one and `os.replace`d into
   place, so an interrupted write cannot leave a half-written policy.
3. **A version bump and an audit row.** The version is today's date plus a revision
   count, and the change (who, when, before, after) is appended to the `policy_edits`
   table. Past runs keep the `rules_version` they were decided under.
4. **A cache clear.** `rules.store.reload_library()` invalidates the `lru_cache`, so the
   next analysis reads the new policy. There is no restart step.

What is deliberately *not* editable
-----------------------------------
A rule's `rule_id` and `pii_type` are immutable. Everything downstream joins on them:
the audit log cites the id, the guards check the id against the type, and past runs
reference the id. Changing either in place would silently rewrite history. To change
what a rule *says* - the action and the reason - is an edit; to change what it *is* is a
new rule and a deleted old one, and the UI does it as exactly that pair.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from .. import config
from ..pii_detection.schema import Action, PiiType
from ..rules.store import Rule, RulesError, RulesLibrary, reload_library
from . import policy_audit as audit
from .policy_index import PolicyIndex, get_index

VALID_TYPES = {t.value for t in PiiType}
VALID_ACTIONS = {a.value for a in Action}

# One writer at a time inside the process. Streamlit reruns are single-threaded, but the
# API and a background script are not, and a lost update to the policy is not acceptable.
_WRITE_LOCK = threading.Lock()

EDITABLE_FIELDS = ("action", "reason", "citation")


class PolicyEditError(ValueError):
    """The proposed change would produce a policy that cannot be trusted."""


@dataclass
class PolicyEdit:
    """One recorded change, ready for the audit table."""

    kind: str                      # update | add | delete | revert
    rule_id: str
    editor: str = "reviewer"
    before: dict | None = None
    after: dict | None = None
    version_before: str = ""
    version_after: str = ""
    created_at: str = ""

    def changes(self) -> list[str]:
        """The fields that actually differ, as 'action: KEEP -> REMOVE' lines."""
        if self.kind == "add":
            return [f"{key}: {value!r}" for key, value in (self.after or {}).items()]
        if self.kind == "delete":
            return [f"deleted: {key}={value!r}" for key, value in (self.before or {}).items()
                    if key in EDITABLE_FIELDS]
        out = []
        for key in EDITABLE_FIELDS:
            old = (self.before or {}).get(key)
            new = (self.after or {}).get(key)
            if old != new:
                out.append(f"{key}: {old!r} -> {new!r}")
        return out


class PolicyEditor:
    """Read the policy, and change it safely.

    `path` and `db_path` are injectable so a test can edit a copy of the policy without
    touching the real one or the real audit log.
    """

    def __init__(self, path: str | Path | None = None, db_path: str | Path | None = None,
                 index: PolicyIndex | None = None):
        self.path = Path(path or config.RULES_PATH)
        self.db_path = db_path
        self._index = index

    # ------------------------------------------------------------------- reading
    def payload(self) -> dict:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def library(self) -> RulesLibrary:
        return RulesLibrary(self.payload())

    def rules(self, receiver: str | None = None) -> list[Rule]:
        rules = self.library().rules
        return [r for r in rules if receiver in (None, r.receiver)]

    def rule(self, rule_id: str) -> Rule | None:
        return next((r for r in self.library().rules if r.rule_id == rule_id), None)

    def receivers(self) -> list[str]:
        return list(self.library().receivers)

    def history(self, limit: int = 50) -> list[dict]:
        return audit.policy_edits(limit=limit, db_path=self.db_path)

    # ------------------------------------------------------------------- writing
    def set_rule(self, rule_id: str, *, action: str | None = None, reason: str | None = None,
                 citation: str | None = None, editor: str = "reviewer",
                 bump_version: bool = True, sync_index: bool = True,
                 index: PolicyIndex | None = None) -> dict:
        """Change an existing rule's action, reason or citation."""
        with _WRITE_LOCK:
            payload = self.payload()
            entry = self._find(payload, rule_id)
            if entry is None:
                raise PolicyEditError(f"no rule with id {rule_id!r}")
            before = dict(entry)
            if action is not None:
                self._check_action(action)
                entry["action"] = str(action).strip().upper()
            if reason is not None:
                self._check_reason(reason)
                entry["reason"] = str(reason).strip()
            if citation is not None:
                entry["citation"] = str(citation).strip()
            if before == dict(entry):
                return {"changed": False, "rule_id": rule_id, "version": payload["version"],
                        "changes": [], "note": "nothing to change"}
            return self._commit(payload, PolicyEdit(
                kind="update", rule_id=rule_id, editor=editor, before=before,
                after=dict(entry)), bump_version=bump_version, sync_index=sync_index,
                index=index)

    def add_rule(self, *, rule_id: str, receiver: str, pii_type: str, action: str,
                 reason: str, citation: str = "", editor: str = "reviewer",
                 bump_version: bool = True, sync_index: bool = True,
                 index: PolicyIndex | None = None) -> dict:
        """Add a rule. Refuses an id that already exists, or a duplicate receiver+type."""
        with _WRITE_LOCK:
            payload = self.payload()
            rule_id = (rule_id or "").strip()
            if not rule_id:
                raise PolicyEditError("a rule needs an id")
            if self._find(payload, rule_id) is not None:
                raise PolicyEditError(f"rule id {rule_id!r} already exists")
            receiver = (receiver or "").strip()
            if receiver not in payload.get("receivers", {}):
                raise PolicyEditError(
                    f"unknown receiver {receiver!r}; known: "
                    f"{', '.join(payload.get('receivers', {}))}")
            pii_type = (pii_type or "").strip().upper()
            if pii_type not in VALID_TYPES:
                raise PolicyEditError(f"unknown type {pii_type!r}; known: "
                                      f"{', '.join(sorted(VALID_TYPES))}")
            self._check_action(action)
            self._check_reason(reason)
            # get_or_create? No: a second rule for the same receiver+type would be
            # unreachable, because guards.rule_for() indexes on exactly that pair.
            clash = next((r for r in payload["rules"]
                          if r["receiver"] == receiver and r["pii_type"] == pii_type), None)
            if clash is not None:
                raise PolicyEditError(
                    f"{receiver} already has a rule for {pii_type} ({clash['rule_id']}). "
                    f"Edit that one, or delete it first.")
            entry = {"rule_id": rule_id, "receiver": receiver, "pii_type": pii_type,
                     "action": str(action).strip().upper(), "reason": str(reason).strip(),
                     "citation": str(citation).strip()}
            payload["rules"].append(entry)
            return self._commit(payload, PolicyEdit(
                kind="add", rule_id=rule_id, editor=editor, before=None,
                after=dict(entry)), bump_version=bump_version, sync_index=sync_index,
                index=index)

    def delete_rule(self, rule_id: str, *, editor: str = "reviewer",
                    bump_version: bool = True, sync_index: bool = True,
                    index: PolicyIndex | None = None) -> dict:
        """Remove a rule. The receiver then has no rule for that type: ASK_HUMAN, never KEEP."""
        with _WRITE_LOCK:
            payload = self.payload()
            entry = self._find(payload, rule_id)
            if entry is None:
                raise PolicyEditError(f"no rule with id {rule_id!r}")
            before = dict(entry)
            payload["rules"] = [r for r in payload["rules"] if r["rule_id"] != rule_id]
            return self._commit(payload, PolicyEdit(
                kind="delete", rule_id=rule_id, editor=editor, before=before,
                after=None), bump_version=bump_version, sync_index=sync_index, index=index)

    # -------------------------------------------------------------------- plumbing
    @staticmethod
    def _find(payload: dict, rule_id: str) -> dict | None:
        return next((r for r in payload.get("rules", [])
                     if str(r.get("rule_id", "")).strip() == str(rule_id).strip()), None)

    @staticmethod
    def _check_action(action: str) -> None:
        if str(action).strip().upper() not in VALID_ACTIONS:
            raise PolicyEditError(f"unknown action {action!r}; known: "
                                  f"{', '.join(sorted(VALID_ACTIONS))}")

    @staticmethod
    def _check_reason(reason: str) -> None:
        if not str(reason or "").strip():
            raise PolicyEditError("a rule without a reason cannot be defended; write one")

    def _next_version(self, current: str) -> str:
        """Today's date, or the same date with the revision bumped: 2026-10-08.2."""
        today = date.today().isoformat()
        head = str(current or "").split(".")[0]
        if head != today:
            return today
        try:
            revision = int(str(current).split(".")[1])
        except (IndexError, ValueError):
            revision = 0
        return f"{today}.{revision + 1}"

    def _serialise(self, payload: dict) -> str:
        return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"

    def _commit(self, payload: dict, edit: PolicyEdit, *, bump_version: bool,
                sync_index: bool, index: PolicyIndex | None) -> dict:
        """Validate, write atomically, clear the cache, record the edit, resync the index."""
        edit.version_before = str(payload.get("version", ""))
        if bump_version:
            payload["version"] = self._next_version(edit.version_before)
        edit.version_after = str(payload.get("version", ""))
        edit.created_at = datetime.now(timezone.utc).astimezone().isoformat(
            timespec="seconds")

        text = self._serialise(payload)
        # 1. It must load. This is the same validator the pipeline uses, so the file can
        #    never be left in a state the pipeline would refuse.
        try:
            RulesLibrary(json.loads(text))
        except RulesError as exc:
            raise PolicyEditError(f"refusing to write an invalid policy: {exc}") from exc

        # 2. Atomic replace.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, self.path)

        # 3. Make the change visible to the running app.
        reload_library()

        # 4. Record it. A failure here is reported but does not undo the write; the file
        #    is the authority and the history is a courtesy.
        audit_error = ""
        try:
            audit.record_policy_edit(edit, db_path=self.db_path)
        except Exception as exc:                                     # noqa: BLE001
            audit_error = str(exc)

        # 5. Keep the vector index level with the file.
        index_report: dict = {}
        index_error = ""
        if sync_index:
            try:
                index_report = (index or self._index or get_index()).sync()
            except Exception as exc:                                 # noqa: BLE001
                # The policy file is already correct; the index can be rebuilt at will.
                index_error = str(exc)

        return {"changed": True, "rule_id": edit.rule_id, "kind": edit.kind,
                "version": edit.version_after, "version_before": edit.version_before,
                "changes": edit.changes(), "editor": edit.editor,
                "created_at": edit.created_at, "index": index_report,
                "audit_error": audit_error, "index_error": index_error}


def describe_rule(rule: Rule) -> dict:
    """A rule as the UI and the API hand it around."""
    return {"rule_id": rule.rule_id, "receiver": rule.receiver, "pii_type": rule.pii_type,
            "action": rule.action, "reason": rule.reason, "citation": rule.citation,
            "text": rule.text}
