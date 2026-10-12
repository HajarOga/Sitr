-- SafeShare2 audit log.
--
-- Three design decisions worth stating:
--
-- 1. The ORIGINAL VALUE IS NOT STORED. Each row keeps a masked rendering
--    ("A***d M*******d") and a short hash of the value. The audit can still prove
--    which item was changed, and two rows about the same value can be linked by the
--    hash, but the log itself is not a copy of the sensitive data.
-- 2. Decisions are append-only. A reviewer's answer is written as a new row with
--    decided_by set; nothing is edited in place, so the history can be read back.
-- 3. THE MODEL IS AUDITED TOO (table `model_calls`). Every call to the language model
--    is recorded - which job it was doing, how many tokens came back, how long it took,
--    and a hash of the prompt. The prompt itself is not stored, because it contains
--    document text. This is what lets the audit say "the AI decided this, here is when
--    and how expensive it was" rather than asking anyone to take it on trust.

CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL,
    document    TEXT NOT NULL,
    receiver    TEXT NOT NULL,
    purpose     TEXT,
    items       INTEGER NOT NULL,
    replaced    INTEGER NOT NULL DEFAULT 0,
    removed     INTEGER NOT NULL DEFAULT 0,
    kept        INTEGER NOT NULL DEFAULT 0,
    pending     INTEGER NOT NULL DEFAULT 0,
    incomplete  INTEGER NOT NULL DEFAULT 0,
    rules_version TEXT,
    model       TEXT,
    model_seconds REAL DEFAULT 0,
    model_calls INTEGER DEFAULT 0,
    processing_seconds REAL DEFAULT 0,
    model_load_seconds REAL DEFAULT 0,
    json_validation_seconds REAL DEFAULT 0,
    ai_summary  TEXT
);

CREATE TABLE IF NOT EXISTS decisions (
    decision_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    document      TEXT NOT NULL,
    receiver      TEXT NOT NULL,
    purpose       TEXT,
    item_type     TEXT NOT NULL,
    value_masked  TEXT NOT NULL,
    value_hash    TEXT NOT NULL,
    action        TEXT NOT NULL,
    replacement   TEXT,
    rule_id       TEXT,
    reason        TEXT,
    confidence    REAL,
    retrieval_score REAL,
    rules_version TEXT,
    detector      TEXT,
    decided_by    TEXT,
    FOREIGN KEY (run_id) REFERENCES runs (run_id)
);

-- One row per model call. No prompt text, no document text: a hash and the sizes only.
CREATE TABLE IF NOT EXISTS model_calls (
    call_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    purpose     TEXT NOT NULL,
    model       TEXT NOT NULL,
    device      TEXT,
    tokens_in   INTEGER,
    tokens_out  INTEGER,
    tokens_per_second REAL,
    prompt_chars INTEGER,
    output_chars INTEGER,
    seconds     REAL,
    json_attempt INTEGER,
    json_valid  INTEGER,
    json_validation_seconds REAL DEFAULT 0,
    prompt_sha  TEXT,
    FOREIGN KEY (run_id) REFERENCES runs (run_id)
);

-- A run records a version string; this table is what a version string means. Policy edits
-- are recorded separately from release decisions on purpose: editing the policy is an
-- administrative act with its own history, and it must not look like a decision about a
-- document. `before_json`/`after_json` are the rule, not the document: no personal data.
CREATE TABLE IF NOT EXISTS policy_edits (
    edit_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at    TEXT NOT NULL,
    editor        TEXT NOT NULL,
    kind          TEXT NOT NULL,
    rule_id       TEXT NOT NULL,
    version_before TEXT,
    version_after TEXT,
    changed_fields TEXT,
    before_json   TEXT,
    after_json    TEXT,
    note          TEXT
);

CREATE INDEX IF NOT EXISTS idx_policy_edits_rule ON policy_edits (rule_id);
CREATE INDEX IF NOT EXISTS idx_decisions_run ON decisions (run_id);
CREATE INDEX IF NOT EXISTS idx_decisions_receiver ON decisions (receiver);
CREATE INDEX IF NOT EXISTS idx_decisions_type ON decisions (item_type);
CREATE INDEX IF NOT EXISTS idx_model_calls_run ON model_calls (run_id);
