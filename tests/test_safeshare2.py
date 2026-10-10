"""SafeShare tests - the contract, not the model.

    python3 tests/test_safeshare2.py

Everything here runs without the language model, in about a second, and that is the
point: the *decisions* are the model's, but the rules about what is allowed to reach a
document are code, and code can be tested. The one test that needs the model skips
itself when the weights are not installed, so a fresh clone can still run the suite.

What is covered:

* the policy wins when the model and the rules disagree;
* an invented rule id is never applied;
* a value the model reports that is not in the document is dropped;
* low confidence and an un-cited KEEP both end up with a human;
* Arabic-Indic digits are located correctly;
* an unsafe substitute is repaired by the generator;
* JSON with fences or trailing commas is still parsed;
* chunking never loses a line;
* solving: unanswered items leave the release;
* the audit log records the model's calls and stores no original value;
* the policy index finds rules by meaning, filters by receiver, and stays level with the
  policy file;
* a policy question cites real rules, and an invented citation from the model is dropped;
* a policy edit is validated, written atomically, version-bumped and audited, and a bad
  edit is refused without touching the file.
"""
from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import itertools
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import config, core, guards                              # noqa: E402
from app.audit import store as audit                              # noqa: E402
from app.document_processing.document import Block, Document      # noqa: E402
from app.llm import engine, prompts                               # noqa: E402
from app.pii_detection.schema import Action, Detection            # noqa: E402
from app.policy import qa as policy_qa                            # noqa: E402
from app.replacement.generator import FakeGenerator               # noqa: E402
from app.rules.store import library as default_library            # noqa: E402

LETTER = (
    "Customer Complaint\n"
    "Name: Abdullah Mohammed Alharbi\n"
    "الاسم: عبدالله محمد الحربي\n"
    "National ID: 1045837299\n"
    "Mobile: +966 55 123 4567            الجوال: ٠٥٥١٢٣٤٥٦٧\n"
    "Email: a.alharbi@example.test\n"
    "Ticket ref.: TKT-88213\n"
)


def document(text: str = LETTER) -> Document:
    return Document(filename="letter.txt", blocks=[Block(id="b0", text=text)])


def finding(**kwargs) -> dict:
    """A raw model finding, with sane defaults, so each test states only its own point."""
    base = {"original_text": "1045837299", "type": "NATIONAL_ID", "action": "REMOVE",
            "replacement": None, "confidence": 0.95, "rule_id": "VENDOR-PII-002",
            "reason": "A national ID is not shared with a vendor."}
    base.update(kwargs)
    return base


def checked(findings: list[dict], receiver: str = "EXTERNAL_VENDOR",
            text: str = LETTER):
    return guards.check_findings(findings, document(text), receiver,
                                 default_library(), FakeGenerator(receiver=receiver))


# ------------------------------------------------------------------ guards: the rules


def test_the_policy_wins_over_the_model():
    detections, notes = checked([finding(original_text="a.alharbi@example.test",
                                         type="EMAIL", action="REPLACE",
                                         replacement="b.person@example.test",
                                         rule_id="VENDOR-PII-004")])
    assert len(detections) == 1
    assert detections[0].action == Action.REMOVE.value, detections[0].action
    assert any("the policy wins" in note for note in notes), notes


def test_an_invented_rule_is_never_applied():
    detections, notes = checked([finding(rule_id="VENDOR-PII-999")])
    assert detections[0].action == Action.ASK_HUMAN.value
    assert detections[0].rule_id is None
    assert any("does not exist" in note for note in notes), notes


def test_a_rule_for_another_type_is_refused():
    detections, notes = checked([finding(rule_id="VENDOR-PII-004")])   # email rule, ID item
    assert detections[0].action == Action.ASK_HUMAN.value
    assert any("is about EMAIL" in note for note in notes), notes


def test_a_value_that_is_not_in_the_document_is_dropped():
    detections, notes = checked([finding(original_text="9999999999")])
    assert detections == []
    assert any("not in the document" in note for note in notes), notes


def test_low_confidence_goes_to_a_person():
    detections, _ = checked([finding(confidence=0.62)])
    assert detections[0].action == Action.ASK_HUMAN.value
    assert "ask_reason" in detections[0].meta


def test_keep_without_a_rule_goes_to_a_person():
    detections, notes = checked([finding(original_text="TKT-88213", type="CUSTOMER_ID",
                                         action="KEEP", rule_id=None)])
    assert detections[0].action == Action.ASK_HUMAN.value
    assert any("wanted to keep" in note for note in notes), notes


def test_an_unknown_type_is_refused():
    detections, notes = checked([finding(type="FAVOURITE_COLOUR")])
    assert detections == []
    assert any("unknown type" in note for note in notes), notes


def test_an_unsafe_substitute_is_repaired():
    detections, notes = checked([finding(original_text="Abdullah Mohammed Alharbi",
                                         type="PERSON_NAME", action="REPLACE",
                                         replacement="Abdullah Mohammed Alharbi",
                                         rule_id="VENDOR-PII-001", confidence=0.9)])
    assert detections[0].action == Action.REPLACE.value
    assert detections[0].replacement != "Abdullah Mohammed Alharbi"
    assert detections[0].meta.get("replacement_repaired") is True
    assert any("not" in note and "acceptable" in note for note in notes), notes


def test_arabic_indic_digits_are_found():
    occurrences = guards.find_occurrences(document(), "٠٥٥١٢٣٤٥٦٧")
    assert len(occurrences) == 1
    block_id, start, end = occurrences[0]
    assert document().block(block_id).text[start:end] == "٠٥٥١٢٣٤٥٦٧"


def test_every_occurrence_of_a_value_is_reported():
    text = "Call 0500000000 today. Backup line: 0500000000."
    detections, _ = checked([finding(original_text="0500000000", type="PHONE",
                                     action="REMOVE", rule_id="VENDOR-PII-003")],
                            text=text)
    assert len(detections) == 2, [d.start for d in detections]


def test_overlapping_items_are_reported_once():
    text = "Name: Abdullah Mohammed Alharbi"
    detections, notes = checked([
        finding(original_text="Abdullah Mohammed", type="PERSON_NAME", action="REPLACE",
                replacement="Salem Al-Harbi", rule_id="VENDOR-PII-001", confidence=0.9),
        finding(original_text="Abdullah Mohammed Alharbi", type="PERSON_NAME",
                action="REPLACE", replacement="Salem Al-Harbi",
                rule_id="VENDOR-PII-001", confidence=0.9)], text=text)
    assert len(detections) == 1
    assert any("overlapped" in note for note in notes), notes


# ------------------------------------------------------------------ engine: the JSON


def test_a_bare_list_answer_is_used_not_thrown_away():
    """The model sometimes answers with a bare [...] array instead of {"findings": [...]}.

    Dropping that answer is not a cosmetic bug: it releases the document with those values
    still in it. Both shapes have to be read, and an empty list is a valid "nothing to
    redact" - not a failure.
    """
    items = [finding()]
    assert core.answered(items, "findings") is True
    assert core.items_from(items, "findings") == items
    assert core.items_from({"findings": items}, "findings") == items
    assert core.answered({"findings": []}, "findings") is True
    assert core.items_from({"findings": []}, "findings") == []
    assert core.answered(None, "findings") is False
    assert core.answered("prose, no JSON", "findings") is False
    assert core.items_from({"decisions": []}, "findings") == []
    # the second look answers with "decisions", and gets the same treatment
    assert core.answered([{"id": "d0"}], "decisions") is True
    assert core.items_from([{"id": "d0"}], "decisions") == [{"id": "d0"}]


def test_analyze_uses_a_bare_list_answer_end_to_end():
    """The same thing, through analyze(): a bare array must still become detections."""
    original_json = engine.generate_json
    original_install = engine.ensure_installed
    original_adjudicate, original_summary = config.ADJUDICATE, config.SUMMARY
    try:
        engine.ensure_installed = lambda: {"installed": True}
        config.ADJUDICATE = False
        config.SUMMARY = False
        # a bare list, exactly as the model answered on Colab
        engine.generate_json = lambda *a, **k: ([finding()], {"parsed": True, "attempts": 1})
        analysis = core.analyze(document(), "EXTERNAL_VENDOR")
        assert analysis.detections, "the bare-list answer was thrown away"
        assert analysis.detections[0].original_text == "1045837299"
        assert analysis.detections[0].action == Action.REMOVE.value
    finally:
        engine.generate_json = original_json
        engine.ensure_installed = original_install
        config.ADJUDICATE, config.SUMMARY = original_adjudicate, original_summary
    return "verified: a bare list becomes a real decision"


def test_json_with_a_fence_is_parsed():
    payload = engine.extract_json('```json\n{"findings": []}\n```')
    assert payload == {"findings": []}


def test_json_with_trailing_commas_is_parsed():
    payload = engine.extract_json('{"findings": [{"type": "PHONE",},],}')
    assert payload["findings"][0]["type"] == "PHONE"


def test_json_inside_prose_is_parsed():
    payload = engine.extract_json('Sure! Here you go: {"findings": []} hope that helps')
    assert payload == {"findings": []}


def test_prose_without_json_is_not_parsed():
    assert engine.extract_json("I could not find anything in this document.") is None


def test_the_chunker_keeps_every_line():
    text = "\n".join(f"line {i} with some content" for i in range(400))
    parts = core.chunks(text, size=500, overlap_lines=2)
    assert len(parts) > 1
    for line in ("line 0 with some content", "line 399 with some content"):
        assert any(line in part for part in parts), f"{line} was lost"


# ------------------------------------------------------------------- the model gate


def test_the_model_is_mandatory():
    """With no weights, analyze() stops instead of quietly doing something else."""
    state = engine.install_state()
    if state["installed"]:
        return "skipped: the model is installed on this machine"
    try:
        core.analyze(document(), "EXTERNAL_VENDOR")
    except engine.ModelNotInstalled as exc:
        assert "setup_model" in str(exc)
        return "verified"
    raise AssertionError("analyze() ran without the model installed")


def test_the_prompts_state_the_contract():
    messages = prompts.analyze_messages("doc", "External Vendor", "why",
                                        [{"rule_id": "VENDOR-PII-002",
                                          "type": "NATIONAL_ID", "action": "REMOVE",
                                          "reason": "x"}])
    system = messages[0]["content"]
    assert "JSON only" in system
    assert "Never invent a rule" in system
    for pii_type in ("NATIONAL_ID", "PHONE", "PERSON_NAME"):
        assert pii_type in system
    assert "VENDOR-PII-002" in messages[1]["content"]


# ------------------------------------------------------------------- release & audit


def test_unanswered_items_leave_the_release():
    detections, _ = checked([finding(confidence=0.55)])          # goes to a person
    analysis = core.AnalysisResult(document=document(), receiver="EXTERNAL_VENDOR",
                                   purpose="test", detections=detections,
                                   generator=FakeGenerator(receiver="EXTERNAL_VENDOR"))
    applied = core.apply_decisions(analysis)
    assert applied["incomplete"] is True
    assert "1045837299" not in applied["changed"]["b0"]


def test_the_audit_records_the_model_and_stores_no_value():
    detections, _ = checked([finding(), finding(original_text="TKT-88213",
                                                type="CUSTOMER_ID", action="KEEP",
                                                rule_id="VENDOR-PII-010")])
    for index, detection in enumerate(detections):
        detection.meta.setdefault("id", f"d{index}")
        detection.meta["masked"] = core.mask_value(detection.original_text)
        detection.meta["value_hash"] = core.value_hash(detection.original_text)
    analysis = core.AnalysisResult(
        document=document(), receiver="EXTERNAL_VENDOR", purpose="test",
        detections=detections, generator=FakeGenerator(receiver="EXTERNAL_VENDOR"),
        rules_version="test", model="Qwen/Qwen2.5-3B-Instruct",
        calls=[{"purpose": "analyze part 1/1", "model": "Qwen/Qwen2.5-3B-Instruct",
                "device": "cpu", "tokens_out": 120, "seconds": 4.5,
                "prompt_sha": "abc123"}],
        summary_text="The model removed two identifiers.")

    applied = core.apply_decisions(analysis)
    with tempfile.TemporaryDirectory() as folder:
        db = Path(folder) / "audit.sqlite3"
        run_id = audit.record_run(analysis, applied, db_path=db)
        calls = audit.model_calls(run_id, db_path=db)
        assert len(calls) == 1 and calls[0]["tokens_out"] == 120, calls
        row = audit.runs(limit=1, db_path=db)[0]
        assert row["model_seconds"] == 4.5 and row["model"] == analysis.model, row
        assert row["ai_summary"] == "The model removed two identifiers."
        blob = db.read_bytes()
        assert b"1045837299" not in blob, "the audit log stored the original value"
        assert b"a.alharbi@example.test" not in blob


def test_the_audit_database_handle_is_released():
    """The connection must be CLOSED, not merely committed.

    `with sqlite3.connect(...)` commits and rolls back but leaves the connection open.
    macOS and Linux do not care, so this passed for a week; Windows locks the file and
    every test that cleans up its temporary database failed with WinError 32. Renaming
    the file is the portable way to assert the handle is gone.
    """
    detections, _ = checked([finding()])
    for index, detection in enumerate(detections):
        detection.meta.setdefault("id", f"d{index}")
        detection.meta["masked"] = core.mask_value(detection.original_text)
        detection.meta["value_hash"] = core.value_hash(detection.original_text)
    analysis = core.AnalysisResult(
        document=document(), receiver="EXTERNAL_VENDOR", purpose="test",
        detections=detections, generator=FakeGenerator(receiver="EXTERNAL_VENDOR"))
    applied = core.apply_decisions(analysis)

    with tempfile.TemporaryDirectory() as folder:
        db = Path(folder) / "audit.sqlite3"
        run_id = audit.record_run(analysis, applied, db_path=db)
        assert audit.runs(limit=1, db_path=db)[0]["run_id"] == run_id
        moved = db.with_name("audit-moved.sqlite3")
        db.rename(moved)                       # WinError 32 before this fix
        assert moved.exists()
    return "the database file could be renamed after use"


def test_a_release_document_is_written():
    detections, _ = checked([finding()])
    for index, detection in enumerate(detections):
        detection.meta.setdefault("id", f"d{index}")
        detection.meta["masked"] = core.mask_value(detection.original_text)
        detection.meta["value_hash"] = core.value_hash(detection.original_text)
    analysis = core.AnalysisResult(document=document(), receiver="EXTERNAL_VENDOR",
                                   purpose="test", detections=detections,
                                   generator=FakeGenerator(receiver="EXTERNAL_VENDOR"))
    applied = core.apply_decisions(analysis)
    with tempfile.TemporaryDirectory() as folder:
        written = __import__("app.document_processing.writers",
                             fromlist=["write_release"]).write_release(
            analysis.document, folder, "letter", applied["changed"])
        text = Path(written["path"]).read_text(encoding="utf-8")
    assert "1045837299" not in text
    assert "National ID" in text


# --------------------------------------------------------------- the policy index
# These use a hashing embedder and an in-memory Chroma collection, so the suite stays
# download-free and fast. The real multilingual embedder is checked only for the
# faithful-wiring property (it must be able to produce one vector per document).


_INDEX_SEQUENCE = itertools.count()


def indexed(library=None):
    """A throwaway index: an in-memory Chroma with a name no other test shares.

    Chroma's in-memory client is process-wide (two `EphemeralClient`s see each other's
    collections), so a shared collection name would leak state between tests - one test's
    filled index would look "already built" to the next. The name is varied, not the
    assertions.
    """
    from app.policy.policy_index import HashingEmbedder, PolicyIndex

    return PolicyIndex(embedder=HashingEmbedder(), persist=False,
                       library=library or default_library(),
                       collection_name=f"test-policy-{next(_INDEX_SEQUENCE)}")


def test_a_rule_embeds_its_type_action_receiver_and_reason():
    from app.policy import policy_index

    rule = default_library().rules[0]
    text = policy_index.rule_document(rule, default_library().receivers)
    assert rule.pii_type in text and rule.action in text and rule.reason in text
    assert policy_index.rule_hash(rule, default_library().receivers) == \
        policy_index.rule_hash(rule, default_library().receivers)


def test_the_index_holds_one_entry_per_rule_and_is_idempotent():
    index = indexed()
    report = index.sync()
    assert report["added"] == len(default_library().rules)
    assert report["embedded"] == len(default_library().rules)
    second = index.sync()
    assert second["embedded"] == 0 and second["removed"] == 0, second
    status = index.status()
    assert status["in_sync"] and status["documents"] == len(default_library().rules)


def test_a_receiver_filter_can_only_return_that_receivers_rules():
    index = indexed()
    for receiver in default_library().receivers:
        for hit in index.query("a name and a phone number", receiver=receiver, top_k=8):
            assert hit["receiver"] == receiver, hit


def test_a_filter_that_matches_nothing_returns_nothing():
    """CUSTOM has no rules. An empty result must stay empty - never a fallback to all."""
    index = indexed()
    assert index.query("anything at all", receiver="CUSTOM", top_k=5) == []


def test_the_collection_name_carries_the_policy_version():
    """An edited policy must land on a new collection, so no stale vector can be mixed in."""
    from app.policy import policy_index
    from app.rules.store import RulesLibrary

    index = indexed()
    before = index.collection_name()
    payload = json.loads(config.RULES_PATH.read_text(encoding="utf-8"))
    payload["rules"][0]["action"] = ("REMOVE" if payload["rules"][0]["action"] != "REMOVE"
                                     else "KEEP")
    changed = RulesLibrary(payload)
    from app.policy.policy_index import HashingEmbedder, PolicyIndex

    other = PolicyIndex(embedder=HashingEmbedder(), persist=False, library=changed)
    assert other.collection_name() != before
    assert policy_index.policy_fingerprint(changed, "hashing:x") != \
        policy_index.policy_fingerprint(default_library(), "hashing:x")


def test_the_real_embedder_would_produce_one_vector_per_document():
    """Faithful wiring, no download: skip unless the embedder is already installed."""
    from app.policy.policy_index import SentenceTransformerEmbedder

    embedder = SentenceTransformerEmbedder()
    if not embedder.is_installed():
        return "skipped: run `python -m scripts.setup_model` to fetch the embedder"
    vectors = embedder.encode(["سياسة مشاركة البيانات", "data sharing policy"])
    assert len(vectors) == 2 and len(vectors[0]) == len(vectors[1]) > 0


# ------------------------------------------------------------ policy Q&A and edits


def test_a_policy_question_returns_cited_rules():
    result = policy_qa.ask_policy("What may a vendor see?", use_model=False,
                                  index=indexed(), library=default_library())
    assert result["answered_by"] == "retrieval"
    assert result["citations"], result
    assert all(citation["rule_id"] for citation in result["citations"])


def test_an_invented_citation_from_the_qa_model_is_dropped():
    """The model may only cite rules it was given - the same contract as the guards."""
    original_installed, original_json = engine.ensure_installed, engine.generate_json
    engine.ensure_installed = lambda: {"installed": True}
    engine.generate_json = lambda *a, **k: (
        {"answer": "Names may not be shared (VENDOR-PII-001). See ALSO-999-001.",
         "citations": ["VENDOR-PII-001", "FAKE-999"]},
        {"attempts": 1, "parsed": True, "seconds": 0.0})
    try:
        result = policy_qa.ask_policy("May a vendor see a name?", use_model=True,
                                      index=indexed(), library=default_library())
    finally:
        engine.ensure_installed, engine.generate_json = original_installed, original_json
    assert result["answered_by"] == "model"
    assert result["cited"] == ["VENDOR-PII-001"], result.get("cited")
    assert "FAKE-999" in result["invented_citations"], result
    assert "ALSO-999-001" in result["invented_citations"], result
    assert any("FAKE-999" in note for note in result["notes"]), result["notes"]


def test_a_mistyped_rule_id_is_reported_as_a_typo_not_as_a_citation():
    """The real model wrote VENDOR-PIII-003. A reader must see that, not silence."""
    hits = [{"rule_id": "VENDOR-PII-003", "action": "REMOVE"},
            {"rule_id": "VENDOR-PII-001", "action": "REPLACE"}]
    resolved = policy_qa.resolve_citations(
        "A phone number must be removed (VENDOR-PIII-003).", [], hits)
    assert resolved["verified"] == []
    assert resolved["suspect"] == [{"claimed": "VENDOR-PIII-003", "meant": "VENDOR-PII-003",
                                    "action": "REMOVE"}], resolved
    assert resolved["invented"] == []


def test_a_different_rule_id_is_not_mistaken_for_a_typo():
    hits = [{"rule_id": "VENDOR-PII-003", "action": "REMOVE"},
            {"rule_id": "VENDOR-PII-001", "action": "REPLACE"}]
    resolved = policy_qa.resolve_citations("See VENDOR-PII-004.", [], hits)
    assert resolved["suspect"] == []
    assert resolved["invented"] == ["VENDOR-PII-004"]


def test_the_grounded_answer_block_is_built_from_the_retrieved_rules():
    result = policy_qa.ask_policy("May a vendor see a phone number?", use_model=False,
                                  index=indexed(), library=default_library())
    text = policy_qa.grounded_answer(result, result["hits"], default_library())
    assert "VENDOR-PII-003" in text and "REMOVE" in text
    assert "External Vendor" in text


def test_a_question_with_no_matching_receiver_says_the_policy_is_silent():
    result = policy_qa.ask_policy("May the custom receiver see a name?", use_model=False,
                                  index=indexed(), library=default_library())
    assert result["hits"] == []
    assert any("ASK_HUMAN" in note for note in result["notes"]), result["notes"]


def test_editing_a_rule_validates_writes_bumps_and_audits():
    from app.policy.policy_editor import PolicyEditor

    with tempfile.TemporaryDirectory() as folder:
        target = Path(folder) / "rules.json"
        target.write_text(config.RULES_PATH.read_text(encoding="utf-8"), encoding="utf-8")
        db = Path(folder) / "audit.sqlite3"
        editor = PolicyEditor(path=target, db_path=db)
        before = editor.rule("VENDOR-PII-003")
        report = editor.set_rule("VENDOR-PII-003", action="KEEP",
                                 reason="Vendors need the callback number.",
                                 editor="hajar", sync_index=False)
        assert report["changed"] and "action" in " ".join(report["changes"])

        # the file on disk is the new policy, and it still loads
        after = PolicyEditor(path=target).rule("VENDOR-PII-003")
        assert after.action == "KEEP" and after.action != before.action
        assert after.rule_id == before.rule_id and after.pii_type == before.pii_type

        # the version moved, and the change is in the history with a before and after
        original_version = json.loads(
            config.RULES_PATH.read_text(encoding="utf-8"))["version"]
        history = editor.history(limit=5)
        assert history and history[0]["rule_id"] == "VENDOR-PII-003"
        assert history[0]["editor"] == "hajar"
        assert history[0]["version_before"] == original_version
        assert history[0]["before"]["action"] == before.action
        assert history[0]["after"]["action"] == "KEEP"


def test_a_bad_edit_is_refused_and_the_file_is_untouched():
    from app.policy.policy_editor import PolicyEditError, PolicyEditor

    with tempfile.TemporaryDirectory() as folder:
        target = Path(folder) / "rules.json"
        original = config.RULES_PATH.read_text(encoding="utf-8")
        target.write_text(original, encoding="utf-8")
        editor = PolicyEditor(path=target, db_path=Path(folder) / "audit.sqlite3")

        for attempt in (lambda: editor.set_rule("VENDOR-PII-003", action="DELETE_EVERYTHING"),
                        lambda: editor.set_rule("VENDOR-PII-999", action="KEEP"),
                        lambda: editor.set_rule("VENDOR-PII-003", reason="   "),
                        lambda: editor.add_rule(rule_id="VENDOR-PII-003", receiver="ANALYST",
                                                pii_type="PHONE", action="KEEP", reason="x"),
                        lambda: editor.add_rule(rule_id="NEW-1", receiver="ANALYST",
                                                pii_type="PHONE", action="KEEP", reason="x")):
            try:
                attempt()
            except PolicyEditError:
                continue
            raise AssertionError("a bad edit was accepted")
        assert target.read_text(encoding="utf-8") == original, "the policy file was changed"


def test_an_edited_policy_is_what_the_release_pipeline_reads():
    """The point of reload_library(): the next analysis uses the new action, no restart."""
    from app.policy.policy_editor import PolicyEditor
    from app.rules.store import RulesLibrary

    with tempfile.TemporaryDirectory() as folder:
        target = Path(folder) / "rules.json"
        target.write_text(config.RULES_PATH.read_text(encoding="utf-8"), encoding="utf-8")
        editor = PolicyEditor(path=target, db_path=Path(folder) / "audit.sqlite3")
        editor.set_rule("VENDOR-PII-002", action="KEEP",
                        reason="The vendor needs the ID to verify the warranty.",
                        sync_index=False)

        # a library built from that file has the new action...
        reloaded = RulesLibrary(json.loads(target.read_text(encoding="utf-8")))
        assert reloaded.rule_for("EXTERNAL_VENDOR", "NATIONAL_ID").action == "KEEP"
        # ...and the guards now enforce what the policy says, not what it said
        detections, _ = guards.check_findings(
            [finding(action="REPLACE", replacement="1000000000")],
            document(), "EXTERNAL_VENDOR", reloaded, FakeGenerator(receiver="EXTERNAL_VENDOR"))
        assert detections[0].action == Action.KEEP.value, detections[0].action


# ------------------------------------------------------------- with the real model


def test_end_to_end_with_the_model():
    """The real thing: the model reads a document and the guards accept the result."""
    state = engine.install_state()
    if not state["installed"]:
        return "skipped: run `python -m scripts.setup_model` then run the suite again"
    engine.reset_calls()
    try:
        analysis = core.analyze(document(), "EXTERNAL_VENDOR", "a vendor is fixing the line")
    except engine.ModelCallFailed as exc:
        # a key that has been rotated or expired is an environment problem, not a code
        # failure, so it is reported as a skip that names the cause
        return f"skipped: the configured model refused the call ({exc}) - " \
               f"update the key in .env, or set SAFESHARE2_PROVIDER=off to run without it"
    assert analysis.detections, "the model found nothing in a document full of PII"
    for detection in analysis.detections:
        assert detection.action in {a.value for a in Action}
        if detection.action in (Action.REPLACE.value, Action.REMOVE.value):
            assert detection.rule_id, "a change without a cited rule reached the release"
    assert analysis.calls, "no model calls were recorded"
    assert analysis.summary_text, "the model did not write the audit summary"
    return f"verified: {len(analysis.detections)} items, {len(analysis.calls)} model calls"


# ------------------------------------------------------- the adaptation metrics


def test_the_metrics_are_perfect_for_a_perfect_answer():
    """A model that answers exactly what the policy expects must score 100%."""
    from evaluation import metrics as M
    from evaluation.reference import expected_findings

    items = [{"text": "1045837299", "type": "NATIONAL_ID"},
             {"text": "a.alharbi@example.com", "type": "EMAIL"},
             {"text": "TKT-88213", "type": "CUSTOMER_ID"}]
    receiver = "EXTERNAL_VENDOR"
    answer = json.dumps({"findings": expected_findings(items, receiver)}, ensure_ascii=False)
    score = M.score(answer, items, receiver, document_id="perfect")
    summary = M.aggregate([score])
    assert summary["structured"] == 1.0, summary
    assert summary["citation"] == 1.0, summary
    assert summary["action_agreement"] == 1.0, summary
    assert summary["finding_recall"] == 1.0, summary
    assert summary["leak_rate"] == 0.0, score.leaked


def test_the_metrics_catch_a_model_that_reports_nothing():
    """Silence is a leak, and the leak rate must say so."""
    from evaluation import metrics as M

    items = [{"text": "1045837299", "type": "NATIONAL_ID"},
             {"text": "a.alharbi@example.com", "type": "EMAIL"}]
    score = M.score('{"findings": []}', items, "EXTERNAL_VENDOR", document_id="silent")
    summary = M.aggregate([score])
    assert summary["finding_recall"] == 0.0
    assert summary["leak_rate"] == 1.0, score.leaked
    assert len(score.leaked) == 2


def test_the_metrics_catch_an_invented_rule_and_a_kept_value():
    from evaluation import metrics as M

    items = [{"text": "1045837299", "type": "NATIONAL_ID"}]
    invented = json.dumps({"findings": [{
        "original_text": "1045837299", "type": "NATIONAL_ID", "action": "REMOVE",
        "replacement": None, "confidence": 0.95, "rule_id": "VENDOR-PII-999",
        "reason": "Looks sensitive."}]})
    score = M.score(invented, items, "EXTERNAL_VENDOR", document_id="invented")
    summary = M.aggregate([score])
    assert summary["citation"] == 0.0, "an invented rule was accepted"
    # the guard turns an ungrounded removal into a question, and the app leaves the item out
    # of the release until a person answers - not published, not resolved
    assert summary["leak_rate"] == 0.0, "the value would have been published"
    assert summary["deferred_rate"] == 1.0, "an ungrounded removal must reach a person"

    kept = json.dumps({"findings": [{
        "original_text": "1045837299", "type": "NATIONAL_ID", "action": "KEEP",
        "replacement": None, "confidence": 0.95, "rule_id": "VENDOR-PII-002",
        "reason": "Kept."}]})
    kept_score = M.score(kept, items, "EXTERNAL_VENDOR", document_id="kept")
    assert M.aggregate([kept_score])["leak_rate"] == 1.0
    assert M.aggregate([kept_score])["action_agreement"] == 0.0


def test_a_value_in_another_digit_style_is_the_same_value():
    """The model may answer 0551234567 for a document that says ٠٥٥١٢٣٤٥٦٧.

    The application folds digits and Arabic letter variants before comparing, so the
    evaluation must too - comparing exactly scored a correct answer as missed and produced
    a "0.0% recall, 100% leak" baseline from a run whose citations were 96.7% valid.
    """
    from evaluation import metrics as M

    assert M.same_value("٠٥٥١٢٣٤٥٦٧", "0551234567")
    assert M.same_value("0551234567", "٠٥٥١٢٣٤٥٦٧")
    assert M.same_value("عبدالله محمد الحربي", "عبدالله محمد الحربي")
    assert M.same_value("أحمد الغامدي", "احمد الغامدي")      # hamza variant
    assert M.same_value(" \"1045837299\" ", "1045837299")     # quoted, padded
    assert not M.same_value("0551234567", "0509999999")
    assert not M.same_value("", "0551234567")

    # VENDOR-PII-003 is REMOVE, not REPLACE: a vendor may not receive a customer's phone
    # number at all. Using the action the policy actually names is the point of the test.
    labelled = [{"text": "٠٥٥١٢٣٤٥٦٧", "type": "PHONE"}]
    answer = json.dumps({"findings": [
        {"original_text": "0551234567", "type": "PHONE", "action": "REMOVE",
         "replacement": None, "confidence": 0.95, "rule_id": "VENDOR-PII-003",
         "reason": "Vendors do not receive customer phone numbers."}]})
    summary = M.aggregate([M.score(answer, labelled, "EXTERNAL_VENDOR")])
    assert summary["finding_recall"] == 1.0, summary
    assert summary["action_agreement"] == 1.0, summary
    assert summary["leak_rate"] == 0.0, summary


def test_a_labelled_line_covers_its_value_but_a_document_echo_does_not():
    """The shapes a real model actually produced, scored the way the app would score them.

    From the first GPU baseline: most documents came back with the labelled line
    ("National ID: 1045837299"), which the app accepts because the value is inside the
    finding; one document came back as the entire document text with no type, which the app
    drops. Scoring must treat those two differently - the first is a detection, the second
    is not a detection at all.
    """
    from evaluation import metrics as M

    assert M.covers("National ID: 1045837299", "1045837299")
    assert M.covers("Mobile: ٠٥٥١٢٣٤٥٦٧", "0551234567")
    assert M.covers('"1045837299"', "1045837299")
    assert not M.covers("0509999999", "0551234567")

    document = ("Customer Complaint Report\nName: Abdullah Mohammed Alharbi\n"
                "National ID: 1045837299\nMobile: 0551234567\n"
                "Email: a.alharbi@example.com\nAddress: Al Narjis district, Street 12\n"
                "Customer no.: CUST-4471\nThe washing machine stopped during the spin cycle.")
    assert not M.covers(document, "1045837299"), "a document-sized answer is not a detection"

    labelled = [{"text": "1045837299", "type": "NATIONAL_ID"}]

    # the echo: one finding, the whole document, no type - the app drops it, so the value leaks
    echo = json.dumps({"findings": [{"original_text": document, "type": None,
                                     "action": None, "replacement": None,
                                     "confidence": 0.9, "rule_id": None,
                                     "reason": "found it"}]})
    echo_score = M.aggregate([M.score(echo, labelled, "EXTERNAL_VENDOR")])
    assert echo_score["finding_recall"] == 0.0, echo_score
    assert echo_score["leak_rate"] == 1.0, echo_score

    # the labelled line: a real detection, because the app would remove it from the release
    line = json.dumps({"findings": [{"original_text": "National ID: 1045837299",
                                     "type": "NATIONAL_ID", "action": "REMOVE",
                                     "replacement": None, "confidence": 0.95,
                                     "rule_id": "VENDOR-PII-002",
                                     "reason": "A national ID is not shared with a vendor."}]})
    line_score = M.aggregate([M.score(line, labelled, "EXTERNAL_VENDOR")])
    assert line_score["finding_recall"] == 1.0, line_score
    assert line_score["leak_rate"] == 0.0, line_score


def test_trainer_settings_are_filtered_to_what_is_actually_installed():
    """A renamed trainer argument must not be able to stop a half-hour GPU run.

    TRL added and removed `warmup_ratio` from DPOConfig between releases, and the failure
    arrived after the model had loaded - the last line before training starts. The settings are
    offered as a dictionary and filtered against the installed class instead.
    """
    from training.train_dpo import supported_kwargs

    class StrictlyTyped:                       # like this Colab's DPOConfig
        def __init__(self, beta, output_dir):
            self.beta, self.output_dir = beta, output_dir

    kept, dropped = supported_kwargs(
        StrictlyTyped, {"beta": 0.1, "output_dir": "out", "warmup_ratio": 0.1})
    assert kept == {"beta": 0.1, "output_dir": "out"}, kept
    assert dropped == {"warmup_ratio": 0.1}, dropped

    class AcceptsAnything:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    kept, dropped = supported_kwargs(AcceptsAnything, {"anything": 1})
    assert kept == {"anything": 1} and dropped == {}, (kept, dropped)
    return "verified: unsupported arguments are dropped and reported"


def test_the_no_rule_set_exists_and_the_metric_notices_the_difference():
    """The alignment half of the adaptation path needs a set it can move.

    Every document in the frozen set is addressed to a receiver that has rules, so it can only
    measure whether the model applies a policy it was given. `data/test_no_rules` is the same
    documents addressed to a receiver whose policy is silent, where the one correct answer for
    every value is ASK_HUMAN. Without it, "ask instead of guessing" has no number.
    """
    from evaluation import metrics as M
    from evaluation.evaluate import test_cases
    from evaluation.reference import action_expected

    cases = test_cases(ROOT / "data" / "test_no_rules")
    assert cases, "run: python -m scripts.make_no_rule_set"

    lib = default_library()
    for case in cases:
        expected = {action_expected(item["type"], case["receiver"], lib)
                    for item in case["items"]}
        assert expected == {"ASK_HUMAN"}, (case["document_id"], expected)

    case = cases[0]
    asked = json.dumps({"findings": [
        {"original_text": item["text"], "type": item["type"], "action": "ASK_HUMAN",
         "replacement": None, "confidence": 0.9, "rule_id": None,
         "reason": "No rule covers this receiver."} for item in case["items"]]})
    decided = json.dumps({"findings": [
        {"original_text": item["text"], "type": item["type"], "action": "KEEP",
         "replacement": None, "confidence": 0.9, "rule_id": None,
         "reason": "Looks harmless."} for item in case["items"]]})

    asked_score = M.aggregate([M.score(asked, case["items"], case["receiver"])])
    decided_score = M.aggregate([M.score(decided, case["items"], case["receiver"])])
    assert asked_score["action_agreement"] == 1.0, asked_score
    assert asked_score["leak_rate"] == 0.0, asked_score
    assert decided_score["action_agreement"] == 0.0, decided_score
    assert decided_score["leak_rate"] == 1.0, decided_score
    return "verified: asking scores 100%, deciding anyway leaks 100%"


def test_three_outcomes_are_counted_separately():
    """resolved / deferred / leaked - the three things the release can do with a value.

    The application leaves a deferred item OUT of the release, so it is not published; but
    nothing has resolved it, and the human workload is a real product cost. Folding it into
    either of the other two would misstate the product: as "resolved" it flatters the model,
    as "leaked" it overstates the danger.
    """
    from evaluation import metrics as M

    items = [{"text": "1045837299", "type": "NATIONAL_ID"}]
    base = {"original_text": "1045837299", "type": "NATIONAL_ID", "replacement": None,
            "confidence": 0.95, "reason": "..."}

    resolved = json.dumps({"findings": [{**base, "action": "REMOVE",
                                         "rule_id": "VENDOR-PII-002"}]})
    summary = M.aggregate([M.score(resolved, items, "EXTERNAL_VENDOR")])
    assert (summary["leak_rate"], summary["deferred_rate"]) == (0.0, 0.0), summary

    deferred = json.dumps({"findings": [{**base, "action": "ASK_HUMAN", "rule_id": None}]})
    summary = M.aggregate([M.score(deferred, items, "EXTERNAL_VENDOR")])
    assert (summary["leak_rate"], summary["deferred_rate"]) == (0.0, 1.0), summary

    leaked = json.dumps({"findings": [{**base, "action": "KEEP",
                                       "rule_id": "VENDOR-PII-002"}]})
    summary = M.aggregate([M.score(leaked, items, "EXTERNAL_VENDOR")])
    assert (summary["leak_rate"], summary["deferred_rate"]) == (1.0, 0.0), summary

    silent = json.dumps({"findings": []})
    summary = M.aggregate([M.score(silent, items, "EXTERNAL_VENDOR")])
    assert (summary["leak_rate"], summary["deferred_rate"]) == (1.0, 0.0), summary


def test_a_parsed_answer_scores_the_same_as_its_text():
    """engine.generate_json returns a parsed object; a predictions file may hold either.

    This is not hypothetical: the first Colab baseline stored the object, `parse_answer`
    assumed text, and the whole run crashed after the model had worked correctly.
    """
    from evaluation import metrics as M

    items = [{"text": "1045837299", "type": "NATIONAL_ID"}]
    payload = {"findings": [{"original_text": "1045837299", "type": "NATIONAL_ID",
                             "action": "REMOVE", "replacement": None, "confidence": 0.95,
                             "rule_id": "VENDOR-PII-002", "reason": "Not to vendors."}]}
    from_text = M.aggregate([M.score(json.dumps(payload), items, "EXTERNAL_VENDOR")])
    from_object = M.aggregate([M.score(payload, items, "EXTERNAL_VENDOR")])
    assert from_text == from_object, (from_text, from_object)
    assert from_text["finding_recall"] == 1.0


def test_no_answer_at_all_is_a_leak_not_a_crash():
    from evaluation import metrics as M

    items = [{"text": "1045837299", "type": "NATIONAL_ID"}]
    score = M.score(None, items, "EXTERNAL_VENDOR", document_id="none")
    assert score.leaked == ["1045837299"]
    assert M.aggregate([score])["leak_rate"] == 1.0


def test_a_fenced_answer_still_counts_as_structured():
    """A markdown fence is a formatting wobble, not a broken answer - but it is noted."""
    from evaluation import metrics as M

    answer = '```json\n{"findings": []}\n```'
    findings, problems = M.parse_answer(answer)
    assert findings == []
    assert any("fence" in problem for problem in problems), problems


# ---------------------------------------------------------------------------- runner


def main() -> int:
    tests = [(name, value) for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    failures = []
    for name, test in tests:
        try:
            note = test()
        except Exception as exc:                              # noqa: BLE001
            failures.append((name, exc))
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"PASS  {name}" + (f"  [{note}]" if isinstance(note, str) else ""))
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
