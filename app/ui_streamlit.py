"""SafeShare - the dashboard.

    streamlit run app/ui_streamlit.py

Three pages, the same layout as the first prototype, because the interface was never the
problem:

  1. Upload  - pick a document, choose the receiver, say why it is being shared.
               The model panel shows whether the AI is installed and on what hardware,
               because in this version there is no analysis without it.
  2. Review  - original and safe version side by side, one card per item, and on every
               card: the rule the model cited, the model's own confidence, and - when the
               model was not trusted - the exact reason it was sent to you instead.
  3. Audit   - every decision with its rule and reason, the run's AI summary, and the
               model's footprint: which jobs it was asked to do, how long each took, how
               many tokens came back.

Two behaviours are deliberate and worth knowing when you read this file:

* if the model is not installed, the Upload page does not offer an analysis at all - it
  tells you the exact command to install it. There is no rules-only path to fall back to,
  and pretending otherwise would misrepresent what the system is;
* the model is only called when you press Analyze (or the per-receiver button). Nothing
  here runs a model call on a page load, so opening the dashboard never costs a minute.
"""
from __future__ import annotations

import html
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import config                                      # noqa: E402
from app.audit import store as audit                        # noqa: E402
from app.core import analyze, apply_decisions, write_summary  # noqa: E402
from app.document_processing import readers                 # noqa: E402
from app.document_processing.writers import safe_text, write_release  # noqa: E402
from app.llm import engine                                  # noqa: E402
from app.pii_detection.schema import Action, PiiType        # noqa: E402
from app.policy import qa as policy_qa                      # noqa: E402
from app.policy.policy_editor import PolicyEditError, PolicyEditor  # noqa: E402
from app.policy.policy_index import PolicyIndexError, get_index  # noqa: E402
from app.review.review_queue import apply_review, build_queue, review_state  # noqa: E402
from app.rules.store import library as default_library      # noqa: E402

SAMPLE_DIR = config.DATA_DIR / "synthetic"

ACTION_STYLE = {
    "KEEP": ("#0f5132", "#d1e7dd", "kept"),
    "REPLACE": ("#664d03", "#fff3cd", "replaced"),
    "REMOVE": ("#842029", "#f8d7da", "removed"),
    "ASK_HUMAN": ("#3d2b8f", "#e5e0ff", "needs human review"),
}
PENDING_STYLE = ACTION_STYLE["ASK_HUMAN"]


# --------------------------------------------------------------------- rendering


def _mark(text: str, style: tuple, title: str) -> str:
    """A piece of the released text, highlighted, with the reason on hover."""
    return (f"<span title='{html.escape(title)}' style='background:{style[1]};"
            f"color:{style[0]};padding:0 2px;border-radius:3px;"
            f"border-bottom:2px solid {style[0]}'>{html.escape(text)}</span>")


def _gap(style: tuple, title: str) -> str:
    """Where a value was taken out of the release.

    A mark, not characters: what was removed is genuinely absent from the safe version, and
    this only shows where it was. It adds nothing to the text.
    """
    return (f"<span title='{html.escape(title)}' style='display:inline-block;width:3px;"
            f"height:0.9em;margin:0 4px;vertical-align:-0.1em;border-radius:1px;"
            f"background:{style[0]}'></span>")


def highlight_block(text: str, items: list[dict]) -> str:
    """One block of the original, with every item the model reported marked.

    Always the original characters: this is the left column, whose job is to show what was
    found. The safe version is `released_block_html()` below.
    """
    if not text:
        return "<div style='height:0.6rem'></div>"
    marks = []
    for item in items:
        start, end = item.get("start", 0), item.get("end", 0)
        if end <= start or end > len(text):
            continue
        style = PENDING_STYLE if item.get("action") == Action.ASK_HUMAN.value else \
            ACTION_STYLE.get(item.get("action", "KEEP"), ACTION_STYLE["KEEP"])
        title = f"{item['type']} - {style[2]}" + (
            f" -> {item['replacement']}" if item.get("replacement") else "")
        marks.append((start, end, style, title))
    marks.sort(key=lambda m: m[0])

    out, cursor = [], 0
    for start, end, style, title in marks:
        if start < cursor:
            continue
        out.append(html.escape(text[cursor:start]))
        out.append(_mark(text[start:end], style, title))
        cursor = end
    out.append(html.escape(text[cursor:]))
    return f"<div style='margin:0 0 0.35rem 0;line-height:1.6'>{''.join(out)}</div>"


def released_block_html(block_text: str, items: list[dict],
                        applied: dict[str, dict]) -> str:
    """One block as it will actually be released, with every change marked.

    The safe version, not the original: the substituted value is what is shown for a
    REPLACE, a removed value leaves only a mark where it stood, and a kept value is
    highlighted in place. An item that is in neither list (`applied` has every decided
    item, one entry per id) is still waiting for a person and is absent from the release.

    Built by replaying the block in reading order rather than by highlighting the original
    text, because the released text is a different string: the offsets move.
    """
    out: list[str] = []
    cursor = 0
    for item in items:
        start, end = item.get("start", 0), item.get("end", 0)
        if end <= start or end > len(block_text) or start < cursor:
            continue
        out.append(html.escape(block_text[cursor:start]))
        decision = applied.get(item.get("id") or "")
        if decision is None:
            out.append(_gap(PENDING_STYLE, f"{item['type']} - left out until reviewed"))
        elif decision.get("action") == Action.REPLACE.value:
            out.append(_mark(decision.get("replacement") or "", ACTION_STYLE["REPLACE"],
                             f"{item['type']} - replaced (was {item['masked']})"))
        elif decision.get("action") == Action.REMOVE.value:
            out.append(_gap(ACTION_STYLE["REMOVE"],
                            f"{item['type']} - removed ({item['masked']})"))
        else:
            out.append(_mark(block_text[start:end], ACTION_STYLE["KEEP"],
                             f"{item['type']} - kept ({item['masked']})"))
        cursor = end
    out.append(html.escape(block_text[cursor:]))
    return f"<div style='margin:0 0 0.35rem 0;line-height:1.6'>{''.join(out)}</div>"


def items_for_block(analysis, block_id: str) -> list[dict]:
    """Every item the model reported in this block, in reading order."""
    return [{"start": d.start, "end": d.end, "type": d.type, "action": d.action,
             "replacement": d.replacement, "id": d.meta.get("id"),
             "masked": d.meta.get("masked", "")}
            for d in sorted((d for d in analysis.detections
                             if d.block_id == block_id), key=lambda d: d.start)]


def badge(text: str, action: str) -> str:
    colour, background, _ = ACTION_STYLE.get(action, ACTION_STYLE["KEEP"])
    return (f"<span style='background:{background};color:{colour};padding:2px 8px;"
            f"border-radius:10px;font-size:0.78rem;font-weight:600'>{html.escape(text)}</span>")


# ------------------------------------------------------------------ page 1: upload


def page_upload() -> None:
    st.title("SafeShare")
    lib = default_library()
    state = engine.status()
    st.caption("Create a safe version of a document for every receiver.")

    upload = st.file_uploader("Upload a document", type=["pdf", "docx", "txt"])
    samples = sorted(SAMPLE_DIR.glob("*.docx")) + sorted(SAMPLE_DIR.glob("*.txt"))
    sample = st.selectbox("...or open a sample document",
                          ["none"] + [p.name for p in samples])

    receiver_keys = list(lib.receivers)
    default = "EXTERNAL_VENDOR" if "EXTERNAL_VENDOR" in receiver_keys else receiver_keys[0]
    receiver = st.selectbox(
        "Who is receiving it?", receiver_keys,
        index=receiver_keys.index(default),
        format_func=lambda key: f"{lib.receivers[key].label} - {lib.receivers[key].purpose}")
    # The purpose is stored with the audit record, so a release can be justified later.
    purpose = lib.receivers[receiver].purpose
    go = st.button("Analyze", type="primary", disabled=not state["installed"])

    if not state["installed"]:
        st.divider()
        st.info("SafeShare is not configured yet — the Analyze button appears once it is."
                + (" Add your API key to `.env` and reload this page."
                   if state.get("provider") == "api"
                   else " Run `python -m scripts.setup_model`, then reload this page."))
        return
    if not go:
        return

    document = None
    if upload is not None:
        try:
            document = readers.read_bytes(upload.name, upload.getvalue())
        except readers.UnsupportedDocument as exc:
            st.error(str(exc))
            return
    elif sample != "none":
        path = SAMPLE_DIR / sample
        stored = readers.store_upload(path.name, path.read_bytes())
        try:
            document = readers.read(stored, path.name)
        finally:
            readers.discard(stored)
    else:
        st.warning("Upload a document or choose a sample first.")
        return

    engine.reset_calls()
    with st.spinner("The model is reading the document against the policy. "
                    + ("This takes a few seconds per page of text on a laptop."
                       if state.get("provider") == "local"
                       else "Usually a few seconds.")):
        try:
            analysis = analyze(document, receiver, purpose)
        except engine.ModelNotInstalled as exc:
            st.error(str(exc))
            return
        except Exception as exc:                                   # noqa: BLE001
            st.error(f"The model did not produce a usable answer, so nothing was "
                     f"analysed: {exc}")
            return

    st.session_state["analysis"] = analysis
    st.session_state["decisions"] = {}
    st.session_state["page"] = "Review"
    st.success(f"{len(analysis.detections)} item(s) found by {analysis.model} in "
               f"{analysis.summary()['model_seconds']}s of model time.")
    st.rerun()


# ------------------------------------------------------------------ page 2: review


def page_review() -> None:
    analysis = st.session_state.get("analysis")
    if analysis is None:
        st.info("Upload a document first.")
        return

    lib = default_library()
    decisions: dict[str, str] = st.session_state.setdefault("decisions", {})
    queue = build_queue(analysis)
    state = review_state(analysis, decisions)
    summary = analysis.summary()

    st.title("Review")
    st.caption(f"{analysis.document.filename} - for {lib.receiver_label(analysis.receiver)}"
               f" ({analysis.purpose}) - decided by {analysis.model}")

    columns = st.columns(5)
    columns[0].metric("Items found", summary["items"])
    columns[1].metric("Replaced", summary["by_action"]["REPLACE"])
    columns[2].metric("Removed", summary["by_action"]["REMOVE"])
    columns[3].metric("Kept", summary["by_action"]["KEEP"])
    columns[4].metric("Needs review", summary["needs_review"])

    st.caption(f"The model made {summary['model_calls']} call(s), "
               f"{summary['model_seconds']}s in total, for this document.")

    if summary["needs_review"]:
        st.warning(f"{summary['needs_review']} item(s) were not decided with enough "
                   f"confidence, or lacked a rule the model could cite. They stay out of "
                   f"the release until you decide. ({state['answered']}/{state['total']} "
                   f"answered)")

    if analysis.notes:
        with st.expander(f"What the guards did to the model's answer "
                         f"({len(analysis.notes)})"):
            for note in analysis.notes:
                st.markdown(f"- {note}")

    tab_compare, tab_items = st.tabs(["Original and safe version", "Decisions"])

    with tab_compare:
        # The right column shows the released text, so it is built with the same call the
        # "Generate release" button makes: the model's proposal plus anything you have
        # answered so far, with every undecided item left OUT of the release.
        preview = apply_review(analysis, decisions)
        applied = {item["id"]: item for item in preview["applied"]}
        pending_by_block: dict[str, int] = {}
        for item in preview["pending"]:
            pending_by_block[item["block_id"]] = pending_by_block.get(item["block_id"], 0) + 1

        left, right = st.columns(2)
        left.markdown("**Original document**")
        right.markdown("**Safe version** — as it will be released")
        for block in analysis.document.blocks:
            if not block.text.strip():
                continue
            items = items_for_block(analysis, block.id)
            with left:
                st.markdown(highlight_block(block.text, items), unsafe_allow_html=True)
            with right:
                waiting = pending_by_block.get(block.id, 0)
                if waiting:
                    st.markdown(
                        f"<div style='margin:0 0 0.35rem 0;color:{PENDING_STYLE[0]};"
                        f"background:{PENDING_STYLE[1]};padding:2px 6px;border-radius:3px'>"
                        f"{waiting} item(s) left out until you decide</div>",
                        unsafe_allow_html=True)
                st.markdown(released_block_html(block.text, items, applied),
                            unsafe_allow_html=True)
        legend = " &nbsp; ".join(badge(f"{action} = {style[2]}", action)
                                 for action, style in ACTION_STYLE.items())
        st.markdown(f"<div style='margin-top:1rem'>{legend}</div>", unsafe_allow_html=True)
        st.caption("Left: what the model found, marked on the original text. Right: the "
                   "released text itself — a highlighted value is what goes into the "
                   "document (yellow = a substitute, green = kept), and a thin mark shows "
                   "where a value was removed, because in the release it is gone. Nothing "
                   "still waiting for you is in the release.")

    with tab_items:
        decided = [d for d in analysis.detections if d.action != Action.ASK_HUMAN.value]
        if decided:
            st.markdown("**Decided by the model and the policy**")
            st.dataframe(pd.DataFrame([{
                "type": d.type, "action": d.action, "rule": d.rule_id or "-",
                "confidence": round(d.confidence, 2),
                "value": d.meta.get("masked", ""),
                "replacement": d.replacement or "",
            } for d in decided]), width="stretch", hide_index=True)

        if not queue:
            st.success("Every item was decided from the policy. Nothing needs a person.")
        else:
            st.markdown("**Waiting for you**")
        for item in queue:
            decision = decisions.get(item.id)
            with st.container(border=True):
                head, meta = st.columns([3, 2])
                with head:
                    st.markdown(f"**{item.type}** &nbsp; {badge('Model was unsure', 'ASK_HUMAN')}",
                                unsafe_allow_html=True)
                    st.markdown(f"`{item.original_text}` → **{item.proposed_action}**"
                                + (f" → `{item.proposed_replacement}`"
                                   if item.proposed_replacement else ""))
                    st.markdown(f"**Rule the model cited:** "
                                f"`{item.rule_id or 'none'}`  \n{item.rule_text}")
                    st.markdown(f"**Model's reason:** {item.reason}")
                with meta:
                    st.metric("Model confidence", f"{item.confidence:.2f}")
                    st.caption(f"Asked because {item.ask_reason}.")
                    if decision:
                        st.success(f"Your decision: {decision}")
                buttons = st.columns(4)
                for index, (label, action) in enumerate(
                        (("Keep", Action.KEEP.value), ("Replace", Action.REPLACE.value),
                         ("Remove", Action.REMOVE.value))):
                    if buttons[index].button(label, key=f"{label.lower()}-{item.id}",
                                             width="stretch"):
                        decisions[item.id] = action
                        st.rerun()
                if buttons[3].button("Reject suggestion", key=f"reject-{item.id}",
                                     width="stretch",
                                     help="Apply nothing the model proposed. The item "
                                          "stays open and out of the release."):
                    decisions[item.id] = "REJECT_SUGGESTION"
                    st.rerun()

        if queue:
            st.divider()
            if st.toggle("Accept the model's proposal for the remaining items",
                         value=False,
                         help="Faster for a demo. Each answer is still recorded as a "
                              "human decision in the audit log."):
                for item in queue:
                    decisions.setdefault(item.id, item.proposed_action)
                st.rerun()

    st.divider()
    left, right = st.columns([1, 3])
    if left.button("Generate release", type="primary"):
        applied = apply_review(analysis, decisions)
        out_dir = config.TMP_DIR / "releases"
        written = write_release(analysis.document, out_dir,
                                Path(analysis.document.filename).stem, applied["changed"])
        with st.spinner("Asking the model for the audit summary..."):
            summary_text = write_summary(analysis, applied)
        run_id = audit.record_run(analysis, applied, ai_summary=summary_text)
        st.session_state["last"] = {"applied": applied, "written": written,
                                    "run_id": run_id, "summary": summary_text}

    last = st.session_state.get("last")
    if last:
        applied, written = last["applied"], last["written"]
        st.caption(f"Format: {written['format']} - layout preserved: "
                   f"{'yes' if written['layout_preserved'] else 'no'}"
                   + (f" ({written.get('note', '')})" if written.get("note") else ""))
        if applied["incomplete"]:
            st.error(f"{len(applied['pending'])} item(s) still need a person and were left "
                     f"out of the release.")
        info = st.columns(3)
        info[0].success(f"{len(applied['applied'])} change(s) applied")
        info[1].info(f"run id {last['run_id']}")
        info[2].download_button("Download the safe version",
                                data=Path(written["path"]).read_bytes(),
                                file_name=Path(written["path"]).name)
        if last.get("summary"):
            st.info(f"**Audit note written by the model:** {last['summary']}")
        with st.expander("Safe version as text"):
            st.code(safe_text(analysis.document, applied["changed"]), language=None)


# ------------------------------------------------------------------- page 3: audit


def page_audit() -> None:
    st.title("Audit log")
    st.caption("Every decision with its rule and reason, and the model's own footprint. "
               "The log stores a masked value and a hash, never the original text.")

    summary = audit.summary()
    columns = st.columns(4)
    columns[0].metric("Runs", summary["runs"])
    columns[1].metric("Decisions", summary["decisions"])
    columns[2].metric("Replacements", summary["by_action"].get("REPLACE", 0))
    columns[3].metric("Removals", summary["by_action"].get("REMOVE", 0))

    runs = audit.runs(limit=20)
    if runs:
        st.subheader("Runs")
        st.dataframe(pd.DataFrame(runs), width="stretch", hide_index=True)
        chosen_run = st.selectbox("Model calls for run",
                                  ["none"] + [row["run_id"] for row in runs])
        if chosen_run != "none":
            calls = audit.model_calls(chosen_run)
            if calls:
                st.dataframe(pd.DataFrame(calls), width="stretch", hide_index=True)
                st.caption(f"{len(calls)} model call(s), "
                           f"{sum(c['seconds'] for c in calls):.2f}s of model time. "
                           f"`prompt_sha` is a hash of the prompt - the prompt itself "
                           f"contains document text and is not stored.")

    rows = audit.recent(limit=500)
    if not rows:
        st.info("No decisions recorded yet. Generate a release from the Review page.")
        return

    frame = pd.DataFrame(rows)
    chosen = st.selectbox("Receiver", ["all"] + sorted(frame["receiver"].unique().tolist()))
    if chosen != "all":
        frame = frame[frame["receiver"] == chosen]

    st.subheader("Decisions")
    st.dataframe(frame, width="stretch", hide_index=True)
    st.download_button("Download CSV", data=audit.to_csv(rows),
                       file_name="safeshare2-audit.csv", mime="text/csv")


# ----------------------------------------------------------------- page 4: policy

POLICY_QUICK_QUESTIONS = [
    "May an external vendor see a phone number?",
    "What may HR keep in an employee record?",
    "هل يمكن مشاركة رقم الهوية مع مورد خارجي؟",
    "What does the analyst receive?",
    "What happens if no rule covers an item?",
]


def _action_badge(action: str) -> str:
    return badge(action, action)


def page_policy() -> None:
    """Ask the policy a question (Chroma + the local model), and edit the rules.

    Two things about this page are deliberate:

    * the **question answering** is a reading aid. It retrieves rules and cites them;
      it never sees a document, and nothing it says is applied to one. If the model is
      not installed the retrieved rules are still shown, unsummarised.
    * the **editing** path is the only way the app writes `rules.json`. It validates the
      whole library, writes atomically, bumps the version and records who changed what -
      see `app/policy/policy_editor.py`.
    """
    st.title("Policy")
    lib = default_library()
    editor = PolicyEditor()

    try:
        status = get_index().status()
        index_error = status.get("error", "")
    except PolicyIndexError as exc:
        status, index_error = {}, str(exc)

    columns = st.columns(4)
    columns[0].metric("Rules", len(lib.rules))
    columns[1].metric("Receivers", len(lib.receivers))
    columns[2].metric("Version", lib.version)
    columns[3].metric("Indexed rules", status.get("documents", 0))

    if index_error:
        st.error(index_error)
    elif not status.get("installed"):
        st.warning(
            f"The embedding model `{status.get('model', '')}` is not on this machine, so "
            f"questions cannot be searched yet. The rules below are fully editable and the "
            f"release pipeline is unaffected.")
        st.code("python -m scripts.setup_model", language="bash")

    tab_ask, tab_rules, tab_history = st.tabs(["Ask the policy", "Browse and edit",
                                               "Change history"])
    with tab_ask:
        policy_ask_tab(lib, status)
    with tab_rules:
        policy_rules_tab(lib, editor)
    with tab_history:
        policy_history_tab(editor)


def policy_ask_tab(lib, status: dict) -> None:
    st.caption("Ask in Arabic or English. The answer comes from the rules that match the "
               "question, and every statement points at a rule id - the retrieved rules "
               "are always shown, so you can check the answer against the policy itself.")

    left, right = st.columns([2, 1])
    with left:
        question = st.text_input("Your question",
                                 placeholder="May an external vendor see a phone number?",
                                 key="policy_question")
        quick = st.selectbox("...or pick one", ["none"] + POLICY_QUICK_QUESTIONS)
        if quick != "none":
            question = quick
    with right:
        receiver_keys = ["any"] + list(lib.receivers)
        receiver = st.selectbox(
            "Receiver filter", receiver_keys, key="policy_receiver",
            format_func=lambda key: "any receiver" if key == "any"
            else f"{lib.receivers[key].label} ({len(lib.rules_for(key))} rules)")
        type_filter = st.selectbox("Type filter", ["any"] + [t.value for t in PiiType],
                                   key="policy_type")
        use_model = st.toggle(
            "Let the model write the answer", value=True, key="policy_use_model",
            disabled=not status.get("installed"),
            help="Off: the retrieved rules are the answer. On: the local model writes a "
                 "short answer from those rules only, and its citations are checked.")

    if st.button("Ask", type="primary", key="policy_ask"):
        with st.spinner("Searching the policy..."):
            try:
                result = policy_qa.ask_policy(
                    question, receiver=None if receiver == "any" else receiver,
                    pii_type=None if type_filter == "any" else type_filter,
                    use_model=use_model)
            except PolicyIndexError as exc:
                st.error(str(exc))
                return
        st.session_state["policy_answer"] = result

    result = st.session_state.get("policy_answer")
    if not result:
        return
    st.divider()
    for note in result.get("notes", []):
        st.info(note)

    if result.get("answer"):
        st.markdown("**Answer**")
        st.markdown(result["answer"])
        if result.get("cited"):
            st.caption("Verified citations: " + ", ".join(f"`{rule_id}`"
                                                          for rule_id in result["cited"]))
        for suspect in result.get("suspect_citations", []):
            st.warning(f"The model wrote `{suspect['claimed']}`; the policy id is "
                       f"`{suspect['meant']}` ({suspect['action']}). The list below is "
                       f"authoritative.")
        if result.get("invented_citations"):
            st.error("The model cited rule id(s) that are not in the policy: "
                     + ", ".join(f"`{rule_id}`" for rule_id in result["invented_citations"])
                     + ". They are refused - read the rules below.")
        st.caption(f"Written by `{result.get('model') or 'the model'}` from the retrieved "
                   f"rules in {result.get('seconds', 0)}s. Guidance for a person, not a "
                   f"decision - the reviewer and the policy decide.")
    elif result.get("hits"):
        st.markdown("**The rules that match** (the model did not write an answer)")

    shown = {"receiver": result.get("receiver_filter") or "any",
             "type": result.get("type_filter") or "any"}
    st.caption(f"Filters used: receiver `{shown['receiver']}`, type `{shown['type']}` - "
               f"{len(result.get('hits', []))} rule(s) retrieved.")

    for hit in result.get("hits", []):
        with st.expander(f"{hit['rule_id']} - {hit['pii_type']} for "
                         f"{lib.receiver_label(hit['receiver'])}: {hit['action']} "
                         f"(relevance {hit.get('relevance')})"):
            st.markdown(f"**Action:** {_action_badge(hit['action'])}", unsafe_allow_html=True)
            st.markdown(f"**Reason:** {hit['reason']}")
            if hit.get("citation"):
                st.markdown(f"**Source:** {hit['citation']}")
            st.caption(f"Embedded text: {hit.get('document', '').replace(chr(10), ' | ')}")

    if result.get("hits"):
        with st.expander("Answer with the rule ids spelled out (built from the index, "
                         "not by the model)"):
            st.code(result.get("grounded") or policy_qa.grounded_answer(
                result, result["hits"], lib), language=None)
            st.caption("This block is assembled from the retrieved rule metadata, so the "
                       "ids and actions are exact even when the model mistyped one.")
        with st.expander("Send this to the Review page"):
            st.caption("A policy answer is context for a reviewer, never an instruction to "
                       "the release pipeline. Nothing here changes a decision or a document.")
            st.code(result.get("grounded") or result.get("answer") or "", language=None)


def policy_rules_tab(lib, editor: PolicyEditor) -> None:
    receiver_keys = ["all"] + list(lib.receivers)
    left, right = st.columns([1, 3])
    with left:
        receiver = st.selectbox("Receiver", receiver_keys, key="policy_edit_receiver",
                                format_func=lambda key: "all receivers" if key == "all"
                                else f"{lib.receivers[key].label} "
                                     f"({len(lib.rules_for(key))} rules)")
    rules = editor.rules(None if receiver == "all" else receiver)
    with right:
        st.caption(f"{len(rules)} rule(s). The `rule_id` and `pii_type` are fixed: the "
                   f"audit log, the guards and past runs all join on them. Change what a "
                   f"rule says (action, reason, source) or add a new rule instead.")

    action_counts = {a.value: 0 for a in Action}
    for rule in rules:
        action_counts[rule.action] = action_counts.get(rule.action, 0) + 1
    st.caption("Actions in this selection: " + ", ".join(
        f"{value} {count}" for value, count in action_counts.items()))

    if rules:
        st.dataframe(pd.DataFrame([{
            "rule_id": rule.rule_id, "receiver": rule.receiver, "type": rule.pii_type,
            "action": rule.action, "reason": rule.reason, "source": rule.citation,
        } for rule in rules]), width="stretch", hide_index=True)

    st.divider()
    st.subheader("Change a rule")
    rule_ids = [rule.rule_id for rule in rules]
    if not rule_ids:
        st.info("This receiver has no rules. Every sensitive item it receives comes back "
                "as ASK_HUMAN - add a rule below if that is not what you want.")
    else:
        chosen_id = st.selectbox("Rule", rule_ids, key="policy_edit_rule")
        chosen = editor.rule(chosen_id)
        with st.form("policy-edit-form", clear_on_submit=False):
            columns = st.columns([1, 2])
            with columns[0]:
                st.text_input("Rule id (fixed)", value=chosen.rule_id, disabled=True)
                st.text_input("Type (fixed)", value=chosen.pii_type, disabled=True)
                st.text_input("Receiver (fixed)",
                              value=lib.receiver_label(chosen.receiver), disabled=True)
            with columns[1]:
                actions = [action.value for action in Action]
                new_action = st.selectbox(
                    "Action", actions, index=actions.index(chosen.action),
                    help="KEEP = this receiver may see it, REPLACE = a fake substitute, "
                         "REMOVE = take it out, ASK_HUMAN = a person decides.")
                new_reason = st.text_area("Reason", value=chosen.reason, height=90,
                                          help="Shown in the prompt and in the audit log. "
                                               "A rule without a reason cannot be defended.")
                new_citation = st.text_input("Policy source", value=chosen.citation)
            editor_name = st.text_input("Your name (recorded in the change history)",
                                        value="reviewer")
            confirmed = st.checkbox(
                f"I understand this changes the policy for **every** future analysis, "
                f"not just the one on screen.")
            submitted = st.form_submit_button("Save the new policy", type="primary",
                                              disabled=not confirmed)
        if submitted:
            try:
                report = editor.set_rule(chosen_id, action=new_action, reason=new_reason,
                                         citation=new_citation, editor=editor_name)
            except PolicyEditError as exc:
                st.error(f"Refused: {exc}")
            else:
                if not report.get("changed"):
                    st.info("Nothing to change - the rule already says that.")
                else:
                    st.success(f"{chosen_id} updated. Policy version "
                               f"{report['version_before']} → **{report['version']}**.")
                    for change in report.get("changes", []):
                        st.markdown(f"- `{change}`")
                    if report.get("index", {}).get("orphaned"):
                        st.caption(f"The index moved to a new collection for this version; "
                                   f"{report['index']['orphaned']} older collection(s) are "
                                   f"left behind. Remove them with "
                                   f"`python -m scripts.setup_model --rebuild-index`.")
                    if report.get("audit_error"):
                        st.warning(f"The change was saved but not recorded in the audit "
                                   f"log: {report['audit_error']}")
                    if report.get("index_error"):
                        st.warning(f"The change was saved; the search index will catch up "
                                   f"on the next question: {report['index_error']}")
                    st.caption("Reloaded: the next analysis reads the new policy "
                               "immediately - no restart.")

        with st.expander("Delete this rule"):
            st.caption("Deleting a rule means the receiver has no rule for that type. The "
                       "item then reaches a person as ASK_HUMAN - never a silent KEEP.")
            typed = st.text_input(f"Type {chosen_id} to confirm", key="policy_delete_confirm")
            if st.button("Delete the rule", key="policy_delete",
                         disabled=typed.strip() != chosen_id):
                try:
                    report = editor.delete_rule(chosen_id, editor="reviewer")
                except PolicyEditError as exc:
                    st.error(f"Refused: {exc}")
                else:
                    st.success(f"{chosen_id} deleted. Policy version "
                               f"{report['version_before']} → **{report['version']}**.")

    with st.expander("Add a rule"):
        st.caption("A receiver can have only one rule per type - that is the pair the "
                   "guards look up. To change an existing pair, edit it above.")
        with st.form("policy-add-form", clear_on_submit=True):
            new_id = st.text_input("New rule id", placeholder="VENDOR-PII-013")
            columns = st.columns(3)
            new_receiver = columns[0].selectbox("Receiver", list(lib.receivers),
                                                format_func=lib.receiver_label)
            new_type = columns[1].selectbox("Type", [t.value for t in PiiType])
            new_action = columns[2].selectbox("Action", [a.value for a in Action])
            new_reason = st.text_area("Reason", height=80)
            new_citation = st.text_input("Policy source")
            confirm_add = st.checkbox("I understand this adds a rule to the live policy.")
            added = st.form_submit_button("Add the rule", type="primary",
                                          disabled=not confirm_add)
        if added:
            try:
                report = editor.add_rule(rule_id=new_id, receiver=new_receiver,
                                         pii_type=new_type, action=new_action,
                                         reason=new_reason, citation=new_citation,
                                         editor="reviewer")
            except PolicyEditError as exc:
                st.error(f"Refused: {exc}")
            else:
                st.success(f"{new_id} added under policy version **{report['version']}**. "
                           f"It is in the next prompt the model reads.")


def policy_history_tab(editor: PolicyEditor) -> None:
    st.caption("Every change to the policy, newest first: who, when, and what the rule "
               "said before and after. Release decisions record the version they used, so "
               "an old run can still be read against the policy it was decided under.")
    history = editor.history(limit=50)
    if not history:
        st.info("No policy changes recorded yet. Edits made on the previous tab appear "
                "here.")
        return
    st.dataframe(pd.DataFrame([{
        "when": row.get("created_at", ""), "editor": row.get("editor", ""),
        "kind": row.get("kind", ""), "rule": row.get("rule_id", ""),
        "version": f"{row.get('version_before', '')} → {row.get('version_after', '')}",
        "changes": "; ".join(row.get("changes") or []),
    } for row in history]), width="stretch", hide_index=True)

    chosen = st.selectbox("Inspect one change", [row["edit_id"] for row in history],
                          format_func=lambda edit_id: next(
                              (f"#{row['edit_id']} {row['rule_id']} "
                               f"{row.get('created_at', '')}" for row in history
                               if row["edit_id"] == edit_id), str(edit_id)))
    row = next(row for row in history if row["edit_id"] == chosen)
    left, right = st.columns(2)
    with left:
        st.markdown("**Before**")
        st.json(row.get("before") or {})
    with right:
        st.markdown("**After**")
        st.json(row.get("after") or {})


# --------------------------------------------------------------------------- main


def main() -> None:
    st.set_page_config(page_title="SafeShare", page_icon="🔒", layout="wide")
    lib = default_library()
    state = engine.status()

    with st.sidebar:
        st.markdown("### SafeShare")
        pages = ["Upload", "Review", "Audit log", "Policy"]
        page = st.radio("Page", pages,
                        index=pages.index(st.session_state.get("page", "Upload")))
        st.session_state["page"] = page
        st.divider()
        st.markdown("**Policy**")
        st.caption(f"{len(lib.rules)} rules across {len(lib.receivers)} receivers, "
                   f"version {lib.version}")
        st.markdown("**Policy search**")
        try:
            index_state = get_index().status()
        except PolicyIndexError:
            index_state = {}
        if index_state.get("in_sync"):
            st.caption(f"Chroma: {index_state['documents']} rules indexed\n\n"
                       f"`{index_state.get('model', '')}`")
        elif index_state.get("installed"):
            st.caption("Chroma is installed; the index builds on the first question.")
        else:
            st.caption("Chroma index not built — run `python -m scripts.setup_model`.")

    {"Upload": page_upload, "Review": page_review, "Audit log": page_audit,
     "Policy": page_policy}[page]()


if __name__ == "__main__":
    main()
