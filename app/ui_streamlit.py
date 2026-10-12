"""SafeShare - the dashboard.

    streamlit run app/ui_streamlit.py

Four focused pages share one privacy-first dashboard:

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
* the model is only called when you press Analyze. Opening the dashboard never triggers
  document analysis.
"""
from __future__ import annotations

import html
import queue
import re
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import config                                      # noqa: E402
from app.audit import store as audit                        # noqa: E402
from app.core import analyze, write_summary                    # noqa: E402
from app.document_processing import readers                 # noqa: E402
from app.document_processing.writers import safe_text, write_release  # noqa: E402
from app.llm import engine                                  # noqa: E402
from app.pii_detection.schema import Action, PiiType        # noqa: E402
from app.policy import qa as policy_qa                      # noqa: E402
from app.policy import document_qa                         # noqa: E402
from app.policy.policy_editor import PolicyEditError, PolicyEditor  # noqa: E402
from app.policy.policy_index import PolicyIndexError, get_index  # noqa: E402
from app.review.review_queue import apply_review, build_queue, review_state  # noqa: E402
from app.rules.store import library as default_library      # noqa: E402

SAMPLE_DIR = config.DATA_DIR / "synthetic"
_ANALYSIS_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="safeshare-analysis")

ACTION_STYLE = {
    "KEEP": ("#0f5132", "#d1e7dd", "kept"),
    "REPLACE": ("#664d03", "#fff3cd", "replaced"),
    "REMOVE": ("#842029", "#f8d7da", "removed"),
    "ASK_HUMAN": ("#3d2b8f", "#e5e0ff", "needs human review"),
}
PENDING_STYLE = ACTION_STYLE["ASK_HUMAN"]
CHART_COLORS = {
    "KEEP": "#40d6ae", "REPLACE": "#ffb454", "REMOVE": "#ff647c",
    "ASK_HUMAN": "#a78bfa", "OTHER": "#7290ad",
}


def _apply_theme() -> None:
    """Premium dark dashboard styling; charts and content remain local."""
    st.markdown("""
    <style>
    :root { color-scheme: dark; }
    .stApp { background:
      radial-gradient(ellipse at 85% 0%, rgba(255,91,112,.10), transparent 34%),
      radial-gradient(ellipse at 0% 10%, rgba(69,116,190,.12), transparent 35%),
      #080d18; color:#e9eff9; font-family:Inter,Segoe UI,sans-serif; }
    [data-testid="stHeader"] { background:rgba(8,13,24,.72); }
    [data-testid="stSidebar"] { background:linear-gradient(180deg,#0d1422,#0a101b); border-right:1px solid #1b293b; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label { padding:.48rem .65rem; border-radius:10px; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) { background:#192337; }
    h1,h2,h3,[data-testid="stMetricValue"] { font-family:Inter,Segoe UI,sans-serif !important; letter-spacing:-.035em; }
    h1 { font-size:2.35rem !important; }
    h2 { font-size:1.55rem !important; }
    h3 { font-size:1.12rem !important; }
    [data-testid="stMetric"] { background:linear-gradient(145deg,rgba(21,32,49,.95),rgba(14,22,35,.96)); border:1px solid #24344a; border-radius:16px; padding:17px 18px; }
    [data-testid="stMetricLabel"] { color:#94a5bb !important; }
    [data-testid="stMetricValue"] { color:#f3f6fc !important; font-size:1.7rem !important; }
    [data-testid="stMetricDelta"] { font-size:.8rem; }
    [data-testid="stVerticalBlockBorderWrapper"] { border-color:#24344a !important; border-radius:16px !important; background:rgba(16,25,39,.72); }
    [data-testid="stFileUploader"] section { background:linear-gradient(135deg,rgba(21,33,52,.9),rgba(15,23,36,.9)); border:1px dashed #445875; border-radius:18px; }
    [data-testid="stFileUploader"] section:hover { border-color:#ff7180; }
    .hero { padding:1.45rem 1.6rem; margin:.2rem 0 1.25rem; border:1px solid #27374c; border-radius:20px;
      background:linear-gradient(115deg,rgba(25,39,59,.97),rgba(17,27,42,.82) 62%,rgba(60,31,43,.55)); }
    .hero-kicker { text-transform:uppercase; letter-spacing:.16em; font-size:.72rem; font-weight:700; color:#ff8791; }
    .hero-title { margin:.35rem 0 .35rem; font:600 1.9rem Inter,Segoe UI,sans-serif; color:#f5f7fb; }
    .hero-copy { margin:0; color:#9eafc5; font-size:.95rem; }
    .section-label { color:#92a4bb; text-transform:uppercase; letter-spacing:.12em; font-size:.72rem; font-weight:700; margin:.3rem 0 .55rem; }
    .pipeline { display:flex; gap:.45rem; align-items:center; flex-wrap:wrap; padding:1rem 1.1rem; margin:.7rem 0 1rem;
      border:1px solid #202f43; border-radius:15px; background:rgba(13,21,34,.8); }
    .pipe-step { padding:.48rem .72rem; border:1px solid #304158; border-radius:10px; color:#bdcadd; font-size:.78rem; background:#121e2e; }
    .pipe-arrow { color:#ff7482; font-weight:700; }
    .card-title { color:#f2f5fb; font-weight:700; font-size:.98rem; }
    .muted { color:#94a5bb; font-size:.85rem; }
    .rule-card { padding:1rem 1.05rem; border:1px solid #26374c; border-radius:15px; background:linear-gradient(145deg,#141f30,#101927); margin:.35rem 0 .6rem; }
    .rule-id { color:#ff929b; font:600 .78rem Inter,Segoe UI,sans-serif; letter-spacing:.04em; }
    .rule-reason { color:#d0d9e6; margin:.45rem 0; }
    .score-chip { color:#9eb5d2; font-size:.76rem; }
    .action-chip { display:inline-block; padding:.19rem .55rem; border-radius:999px; font-size:.7rem; font-weight:700; background:#202d40; color:#d6e0ef; }
    div.stButton > button[kind="primary"] { background:linear-gradient(110deg,#ff687b,#f45168); border:0; color:white; font-weight:700; border-radius:11px; box-shadow:0 8px 24px rgba(244,81,104,.17); }
    div.stButton > button { border-radius:10px; border-color:#34465e; }
    div[data-baseweb="tab-list"] { gap:.35rem; border-bottom:1px solid #26374c; }
    button[data-baseweb="tab"] { border-radius:10px 10px 0 0; }
    [data-testid="stExpander"] { border-color:#26374c; border-radius:13px; background:rgba(16,25,39,.5); }
    [data-testid="stDataFrame"] { border:1px solid #26374c; border-radius:13px; overflow:hidden; }
    .stProgress > div > div { background:linear-gradient(90deg,#ff647c,#ffad72); }
    @media (max-width: 800px) { .hero-title {font-size:1.55rem;} .hero {padding:1.1rem;} }
    </style>
    """, unsafe_allow_html=True)


def _hero(kicker: str, title: str, copy: str) -> None:
    st.markdown(
        f'<div class="hero"><div class="hero-kicker">{html.escape(kicker)}</div>'
        f'<div class="hero-title">{html.escape(title)}</div>'
        f'<p class="hero-copy">{html.escape(copy)}</p></div>',
        unsafe_allow_html=True)


def _chart(fig, key: str, height: int = 300) -> None:
    fig.update_layout(
        template="plotly_dark", height=height, margin=dict(l=8, r=8, t=28, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, sans-serif", color="#bac8da", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
        hoverlabel=dict(bgcolor="#172337", bordercolor="#344963"))
    fig.update_xaxes(gridcolor="rgba(104,129,160,.13)", zerolinecolor="rgba(104,129,160,.13)")
    fig.update_yaxes(gridcolor="rgba(104,129,160,.13)", zerolinecolor="rgba(104,129,160,.13)")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False},
                    key=key)


def _mask_audit_value(value: object) -> str:
    """Mask again at the UI boundary, including short or malformed stored values."""
    text = str(value or "")
    masked = []
    for token in text.split():
        if len(token) <= 2:
            masked.append("•••")
        else:
            masked.append(token[0] + "*" * (len(token) - 2) + token[-1])
    return " ".join(masked)


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


_ARABIC_LETTER = re.compile(r"[\u0600-\u06FF]")
_LATIN_LETTER = re.compile(r"[A-Za-z]")
_RULE_ID = re.compile(r"\b[A-Z]{2,}(?:-[A-Z0-9]+)+\b")           # VENDOR-PII-002


def _line_html(line: str) -> str:
    """One line of model text, with the direction that line actually has."""
    letters = _RULE_ID.sub("", line)
    rtl = len(_ARABIC_LETTER.findall(letters)) > len(_LATIN_LETTER.findall(letters))
    body = html.escape(line)
    body = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", body)
    body = _RULE_ID.sub(lambda m: f"<bdi dir='ltr'>{m.group(0)}</bdi>", body)
    return (f"<div dir='{'rtl' if rtl else 'ltr'}' style='text-align:"
            f"{'right' if rtl else 'left'};line-height:1.9;margin:0 0 0.25rem 0'>{body}</div>")


def bidi_html(text: str) -> str:
    """Model text as HTML that reads in the right order, Arabic or English or both.

    Streamlit lays markdown out left-to-right. An Arabic answer with a Latin rule id inside
    it - "... (VENDOR-PII-002). ..." - is then reordered by the browser's bidirectional
    algorithm and the sentences come out jumbled. The fix is to give each line its real
    direction (right-to-left when it is mostly Arabic) and to keep every rule id as one
    left-to-right unit, so the punctuation around it stays where the sentence put it. The
    direction is decided per line, so an English line under an Arabic one is not flipped.
    The text is escaped first, so nothing the model writes can inject markup.
    """
    return "".join(_line_html(line) if line.strip() else "<div style='height:0.5rem'></div>"
                   for line in (text or "").splitlines())


def show_text(text: str) -> None:
    """Show model-written text with the right reading direction."""
    st.markdown(bidi_html(text), unsafe_allow_html=True)


def _analysis_progress_text(event: dict, elapsed: float) -> str:
    """Describe observed analysis work without inventing a completion percentage."""
    minutes, seconds = divmod(int(elapsed), 60)
    clock = f"{minutes:02d}:{seconds:02d}"
    completed = event.get("completed_chunks", 0)
    total = event.get("total_chunks", 0)
    detail = f"{completed}/{total} chunks complete" if total else ""
    if event.get("stage") == "chunk_complete":
        detail = (f"{detail} · {event.get('generation_seconds', 0):.1f}s generation"
                  f" · {event.get('attempts', 1)} JSON attempt(s)")
    return f"{event.get('message', 'Preparing analysis.')} · {detail} · elapsed {clock}"


# ------------------------------------------------------------------ page 1: upload


def page_upload() -> None:
    _hero("SAFE SHARE WORKSPACE", "Privacy, by design.",
          "Prepare a receiver-specific release with an auditable AI and policy guardrail.")
    lib = default_library()
    state = engine.status()

    installed = bool(state["installed"])
    provider_label = "Hosted API" if state.get("provider") == "api" else "On-device model"
    model_name = state.get("model") or "Model unavailable"
    info = st.columns(3)
    info[0].metric("AI engine", provider_label)
    info[1].metric("Active model", model_name)
    info[2].metric("Connection", "Ready" if installed else "Needs setup")

    st.markdown('<div class="section-label">Processing pipeline</div>'
                '<div class="pipeline"><span class="pipe-step">01 · Upload</span>'
                '<span class="pipe-arrow">›</span><span class="pipe-step">02 · Detect PII</span>'
                '<span class="pipe-arrow">›</span><span class="pipe-step">03 · Load receiver policy</span>'
                '<span class="pipe-arrow">›</span><span class="pipe-step">04 · AI decisions</span>'
                '<span class="pipe-arrow">›</span><span class="pipe-step">05 · Guards</span>'
                '<span class="pipe-arrow">›</span><span class="pipe-step">06 · Safe release</span></div>',
                unsafe_allow_html=True)

    st.markdown('<div class="section-label">01 / Choose the recipient</div>',
                unsafe_allow_html=True)
    receiver_keys = list(lib.receivers)
    default = st.session_state.get("selected_receiver", "EXTERNAL_VENDOR")
    if default not in receiver_keys:
        default = receiver_keys[0]
    receiver = default
    for start in range(0, len(receiver_keys), 3):
        row_keys = receiver_keys[start:start + 3]
        receiver_columns = st.columns(len(row_keys))
        for column, key in zip(receiver_columns, row_keys):
            receiver_info = lib.receivers[key]
            with column:
                selected = key == default
                with st.container(border=True):
                    st.markdown(f"**{html.escape(receiver_info.label)}**")
                    st.caption(f"{len(lib.rules_for(key))} policy rules")
                    if st.button("Selected" if selected else "Choose",
                                 key=f"receiver-{key}", use_container_width=True,
                                 type="primary" if selected else "secondary"):
                        st.session_state["selected_receiver"] = key
                        st.rerun()
    receiver = st.session_state.get("selected_receiver", default)
    purpose = lib.receivers[receiver].purpose
    st.caption(f"**Purpose:** {purpose}  ·  {lib.receivers[receiver].description}")

    st.markdown('<div class="section-label">02 / Add a document</div>',
                unsafe_allow_html=True)
    upload_col, sample_col = st.columns([1.2, 1])
    with upload_col:
        upload = st.file_uploader("Drop a document or browse",
                                  type=["pdf", "docx", "txt"],
                                  help="PDF, DOCX and TXT are supported.")
    with sample_col:
        samples = sorted(SAMPLE_DIR.glob("*.docx")) + sorted(SAMPLE_DIR.glob("*.txt"))
        sample = st.selectbox("Or use a synthetic sample",
                              ["none"] + [p.name for p in samples])
        if sample != "none":
            st.caption("Sample data only · local files stay on this machine.")

    # The purpose is stored with the audit record, so a release can be justified later.
    left, right = st.columns([1, 3])
    go = left.button("Analyze document  →", type="primary",
                     disabled=not installed, use_container_width=True)
    if state.get("provider") == "api":
        right.caption(f"Document text is sent to **{config.API_BASE_URL}** for analysis.")
    else:
        right.caption("Local inference · document text remains on this device.")

    if not installed:
        st.warning("The analysis provider is not configured yet — the API needs a key in "
                   "`.env`, or install the local model. "
                   + ("Add your API key to `.env` and reload this page."
                      if state.get("provider") == "api" else
                      "Run `python -m scripts.setup_model`, then reload this page."))
        return
    task = st.session_state.get("analysis_task")
    if go and task is None:
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
        events: queue.Queue[dict] = queue.Queue()
        started = time.perf_counter()
        future = _ANALYSIS_EXECUTOR.submit(
            analyze, document, receiver, purpose,
            progress_callback=events.put, include_summary=False)
        task = {"future": future, "events": events, "started": started}
        st.session_state["analysis_task"] = task

    if task is None:
        return

    future: Future = task["future"]
    events: queue.Queue[dict] = task["events"]
    started = task["started"]
    latest = {
        "message": "Preparing document analysis.",
        "completed_chunks": 0,
        "total_chunks": 0,
    }
    with st.status("Analyzing document against the receiver policy…",
                   expanded=True) as status_box:
        progress_line = status_box.empty()
        last_update = 0.0
        while not future.done():
            received = False
            while True:
                try:
                    latest = events.get_nowait()
                    received = True
                except queue.Empty:
                    break
            elapsed = time.perf_counter() - started
            if received or elapsed - last_update >= 1.0:
                progress_line.write(_analysis_progress_text(latest, elapsed))
                last_update = elapsed
            time.sleep(0.25)

        while True:
            try:
                latest = events.get_nowait()
            except queue.Empty:
                break
            progress_line.write(_analysis_progress_text(
                latest, time.perf_counter() - started))
        try:
            analysis = future.result()
        except engine.ModelNotInstalled as exc:
            status_box.update(label="Model unavailable", state="error")
            st.error(str(exc))
            st.session_state.pop("analysis_task", None)
            return
        except Exception as exc:                                   # noqa: BLE001
            status_box.update(label="Analysis could not be completed", state="error")
            st.error(f"The model did not produce a usable answer, so nothing was "
                     f"analysed: {exc}")
            st.session_state.pop("analysis_task", None)
            return
        elapsed = time.perf_counter() - started
        status_box.update(
            label=f"Analysis complete · {len(analysis.detections)} findings checked",
            state="complete", expanded=False)

    st.session_state.pop("analysis_task", None)
    st.session_state["analysis"] = analysis
    st.session_state["decisions"] = {}
    st.session_state["page"] = "Review"
    st.session_state["analysis_elapsed"] = analysis.processing_seconds or elapsed
    st.success(f"{len(analysis.detections)} items detected by {analysis.model} in "
               f"{elapsed:.2f}s. Opening the privacy review.")
    st.rerun()


# ------------------------------------------------------------------ page 2: review


def page_review() -> None:
    analysis = st.session_state.get("analysis")
    if analysis is None:
        _hero("PRIVACY REVIEW", "Nothing to review yet.",
              "Run an analysis to compare the original with a receiver-safe release.")
        if st.button("Go to upload", type="primary"):
            st.session_state["page"] = "Upload"
            st.rerun()
        return

    lib = default_library()
    decisions: dict[str, str] = st.session_state.setdefault("decisions", {})
    queue = build_queue(analysis)
    state = review_state(analysis, decisions)
    summary = analysis.summary()

    _hero("REVIEW WORKSPACE", analysis.document.filename,
          f"{lib.receiver_label(analysis.receiver)} · {analysis.purpose} · model {analysis.model}")

    columns = st.columns(5)
    columns[0].metric("Detected", summary["items"])
    columns[1].metric("Replaced", summary["by_action"]["REPLACE"])
    columns[2].metric("Removed", summary["by_action"]["REMOVE"])
    columns[3].metric("Retained", summary["by_action"]["KEEP"])
    columns[4].metric("Human review", summary["needs_review"])

    timing_col, fields_col, protect_col = st.columns(3)
    timing_col.caption(f"⏱ End-to-end analysis · "
                       f"{st.session_state.get('analysis_elapsed', summary['model_seconds']):.2f}s")
    fields_col.caption(f"◈ Protected fields · "
                       f"{summary['by_action']['REPLACE'] + summary['by_action']['REMOVE']}")
    protect_col.caption(f"◉ Model calls · {summary['model_calls']} · "
                        f"{summary['model_seconds']:.2f}s inference")

    if summary["needs_review"]:
        st.warning(f"{summary['needs_review']} item(s) were not decided with enough "
                   f"confidence, or lacked a rule the model could cite. They stay out of "
                   f"the release until you decide. ({state['answered']}/{state['total']} "
                   f"answered)")

    if analysis.notes:
        with st.expander(f"Guardrail activity · {len(analysis.notes)} recorded notes"):
            for note in analysis.notes:
                st.warning(note, icon="⚠️")

    total = summary["items"]
    protected = summary["by_action"]["REPLACE"] + summary["by_action"]["REMOVE"]
    coverage = (protected / total) if total else 0
    st.markdown('<div class="section-label">Privacy exposure comparison</div>',
                unsafe_allow_html=True)
    exp_left, exp_right = st.columns([1, 2])
    exp_left.metric("Items protected", f"{protected} / {total}",
                    help="Count of items marked REPLACE or REMOVE by the final decisions.")
    exp_right.progress(coverage, text=f"{coverage:.0%} of detected items replaced or removed")

    tab_compare, tab_items = st.tabs(["◫  Original vs. safe release", "◈  Decision center"])

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
        left.markdown("#### Original")
        right.markdown("#### Safe release preview")
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
        st.caption("Coral marks indicate findings. On the right, amber is a substitute, "
                   "green is retained by policy, and a red marker shows a removed span. "
                   "Left: what the model found, marked on the original text. Right: the "
                   "released text itself — a highlighted value is what goes into the "
                   "document (yellow = a substitute, green = kept), and a thin mark shows "
                   "where a value was removed, because in the release it is gone. Nothing "
                   "still waiting for you is in the release.")

    with tab_items:
        filter_cols = st.columns(3)
        type_options = ["All types"] + sorted({d.type for d in analysis.detections})
        action_options = ["All actions"] + sorted({d.action for d in analysis.detections})
        selected_type = filter_cols[0].selectbox("PII type", type_options, key="review_type")
        selected_action = filter_cols[1].selectbox("Action", action_options,
                                                    key="review_action")
        confidence_floor = filter_cols[2].slider("Minimum confidence", 0.0, 1.0, 0.0,
                                                  0.05, key="review_confidence")
        filtered_decisions = [
            d for d in analysis.detections
            if d.action != Action.ASK_HUMAN.value
            and (selected_type == "All types" or d.type == selected_type)
            and (selected_action == "All actions" or d.action == selected_action)
            and d.confidence >= confidence_floor]
        if filtered_decisions:
            st.markdown("#### Model and policy decisions")
            for detection in filtered_decisions:
                color = CHART_COLORS.get(detection.action, CHART_COLORS["OTHER"])
                with st.container(border=True):
                    card_left, card_right = st.columns([3, 1])
                    with card_left:
                        st.markdown(
                            f"**{detection.type}**　<span class='action-chip'>"
                            f"{html.escape(detection.action)}</span>　"
                            f"<span style='color:{color}'>●</span> "
                            f"`{html.escape(detection.meta.get('masked', ''))}`",
                            unsafe_allow_html=True)
                        st.caption(f"Policy · {detection.rule_id or 'No valid rule'}"
                                   + (f" · {detection.reason}" if detection.reason else ""))
                        if detection.action == Action.REPLACE.value and detection.replacement:
                            st.caption(f"Validated synthetic substitute · `{detection.replacement}`")
                    with card_right:
                        st.metric("Confidence", f"{detection.confidence:.0%}")
                        st.progress(max(0.0, min(1.0, detection.confidence)))
        elif analysis.detections:
            st.info("No findings match these filters.")

        filtered_queue = [
            item for item in queue
            if (selected_type == "All types" or item.type == selected_type)
            and (selected_action == "All actions"
                 or selected_action == Action.ASK_HUMAN.value)
            and item.confidence >= confidence_floor]
        if not queue:
            st.success("Every item was decided from the policy. Nothing needs a person.")
        elif filtered_queue:
            st.markdown("**Waiting for your decision**")
        else:
            st.info("No open review items match these filters.")
        for item in filtered_queue:
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
    _hero("SECURITY ANALYTICS", "Audit intelligence",
          "Evidence from recorded release runs, policy decisions and actual model calls.")
    runs = audit.runs(limit=1000)
    rows = audit.recent(limit=5000)
    if not runs and not rows:
        st.info("No audit events yet. Generate a release from the Review page to populate analytics.")
        return

    run_frame = pd.DataFrame(runs)
    decision_frame = pd.DataFrame(rows)
    for frame in (run_frame, decision_frame):
        if not frame.empty and "created_at" in frame:
            frame["event_time"] = pd.to_datetime(frame["created_at"], errors="coerce", utc=True)

    filter_cols = st.columns(4)
    receivers = sorted(set(
        (run_frame["receiver"].dropna().astype(str).tolist() if not run_frame.empty else [])
        + (decision_frame["receiver"].dropna().astype(str).tolist()
           if not decision_frame.empty else [])))
    receiver_filter = filter_cols[0].selectbox(
        "Receiver", ["All receivers"] + receivers, key="audit_receiver")
    documents = sorted(set(
        (run_frame["document"].dropna().astype(str).tolist() if not run_frame.empty else [])
        + (decision_frame["document"].dropna().astype(str).tolist()
           if not decision_frame.empty else [])))
    document_labels = {
        label: filename for index, filename in enumerate(documents, start=1)
        for label in [f"Document {index:02d}"]
    }
    document_filter = filter_cols[1].selectbox(
        "Document", ["All documents"] + list(document_labels), key="audit_document")
    actions = (sorted(decision_frame["action"].dropna().astype(str).unique().tolist())
               if not decision_frame.empty else [])
    action_filter = filter_cols[2].selectbox("Action", ["All actions"] + actions,
                                              key="audit_action")

    date_values = []
    for frame in (run_frame, decision_frame):
        if not frame.empty and "event_time" in frame:
            date_values.extend(frame["event_time"].dropna().dt.date.tolist())
    if date_values:
        date_range = filter_cols[3].date_input(
            "Date range", value=(min(date_values), max(date_values)),
            min_value=min(date_values), max_value=max(date_values), key="audit_dates")
        if isinstance(date_range, tuple) and len(date_range) == 2:
            date_start, date_end = date_range
        elif isinstance(date_range, date):
            date_start = date_end = date_range
        else:
            date_start, date_end = min(date_values), max(date_values)
    else:
        date_start = date_end = None

    filtered_runs = run_frame.copy()
    filtered_decisions = decision_frame.copy()
    for frame in (filtered_runs, filtered_decisions):
        if frame.empty:
            continue
        if receiver_filter != "All receivers":
            frame.drop(frame.index[frame["receiver"] != receiver_filter], inplace=True)
        if document_filter != "All documents":
            frame.drop(frame.index[
                frame["document"] != document_labels[document_filter]], inplace=True)
        if date_start and date_end and "event_time" in frame:
            event_dates = frame["event_time"].dt.date
            frame.drop(frame.index[(event_dates < date_start) | (event_dates > date_end)],
                       inplace=True)
    if not filtered_runs.empty and not filtered_decisions.empty:
        selected_run_ids = set(filtered_runs["run_id"].astype(str))
        filtered_decisions = filtered_decisions[
            filtered_decisions["run_id"].astype(str).isin(selected_run_ids)]
    if action_filter != "All actions" and not filtered_decisions.empty:
        filtered_decisions = filtered_decisions[
            filtered_decisions["action"] == action_filter]
    if not filtered_runs.empty:
        matching_run_ids = (set(filtered_decisions["run_id"].astype(str))
                            if not filtered_decisions.empty else set())
        filtered_runs = filtered_runs[
            filtered_runs["run_id"].astype(str).isin(matching_run_ids)]

    decisions_count = len(filtered_decisions)
    replacements = int((filtered_decisions["action"] == Action.REPLACE.value).sum()) \
        if decisions_count else 0
    removals = int((filtered_decisions["action"] == Action.REMOVE.value).sum()) \
        if decisions_count else 0
    review_count = int((filtered_decisions["action"] == Action.ASK_HUMAN.value).sum()) \
        if decisions_count else 0
    kpis = st.columns(5)
    kpis[0].metric("Runs", len(filtered_runs))
    kpis[1].metric("PII decisions", decisions_count)
    kpis[2].metric("Replacements", replacements)
    kpis[3].metric("Removals", removals)
    kpis[4].metric("Review requests", review_count)

    if filtered_decisions.empty:
        st.info("No recorded decisions match these filters.")
    else:
        action_counts = (filtered_decisions["action"].value_counts()
                         .rename_axis("action").reset_index(name="count"))
        left, right = st.columns(2)
        with left:
            st.markdown("#### Decision distribution")
            donut = px.pie(action_counts, names="action", values="count", hole=.68,
                           color="action", color_discrete_map=CHART_COLORS)
            donut.update_traces(textposition="inside", textinfo="percent+label",
                                marker=dict(line=dict(color="#0b1220", width=3)))
            _chart(donut, "audit-action-donut", 310)
        with right:
            st.markdown("#### Protected data categories")
            by_type = (filtered_decisions.groupby(["item_type", "action"], as_index=False)
                       .size().rename(columns={"size": "count"}))
            bars = px.bar(by_type, x="item_type", y="count", color="action",
                          barmode="stack", color_discrete_map=CHART_COLORS)
            bars.update_layout(xaxis_title="", yaxis_title="Recorded decisions")
            bars.update_xaxes(tickangle=-25)
            _chart(bars, "audit-pii-categories", 310)

        left, right = st.columns(2)
        with left:
            st.markdown("#### Activity timeline")
            if not filtered_runs.empty:
                timeline = filtered_runs.dropna(subset=["event_time"]).copy()
                if not timeline.empty:
                    timeline["day"] = timeline["event_time"].dt.floor("D")
                    daily = (timeline.groupby("day", as_index=False)
                             .agg(runs=("run_id", "count"), items=("items", "sum")))
                    figure = px.area(daily, x="day", y="runs", markers=True,
                                     color_discrete_sequence=["#ff7180"])
                    figure.add_scatter(x=daily["day"], y=daily["items"], mode="lines+markers",
                                       name="Decisions", line=dict(color="#718fe6"))
                    figure.update_layout(yaxis_title="Events", xaxis_title="")
                    figure.update_xaxes(tickformat="%b %d")
                    _chart(figure, "audit-timeline", 290)
                else:
                    st.caption("Run timestamps are not available for this selection.")
            else:
                st.caption("No matching run-level activity.")
        with right:
            st.markdown("#### Activity by receiver")
            receiver_activity = (filtered_decisions.groupby(
                ["receiver", "action"], as_index=False).size()
                .rename(columns={"size": "decisions"}))
            by_receiver = px.bar(receiver_activity, x="receiver", y="decisions",
                                 color="action", barmode="stack",
                                 color_discrete_map=CHART_COLORS)
            by_receiver.update_layout(xaxis_title="", yaxis_title="Decisions")
            by_receiver.update_xaxes(tickangle=-15)
            _chart(by_receiver, "audit-receiver-activity", 290)

    calls = []
    if not filtered_runs.empty:
        for run_id in filtered_runs["run_id"].astype(str).tolist():
            calls.extend(audit.model_calls(run_id))
    elapsed_values = [float(call.get("seconds") or 0) for call in calls
                      if call.get("seconds") is not None]
    latency = (sum(elapsed_values) / len(elapsed_values)) if elapsed_values else None
    processing_values = [
        float(run.get("processing_seconds") or 0)
        for _, run in filtered_runs.iterrows()
        if run.get("processing_seconds") is not None
    ] if not filtered_runs.empty else []
    mean_processing = (
        sum(processing_values) / len(processing_values) if processing_values else None)
    perf = st.columns(4)
    perf[0].metric("Model calls", len(calls))
    perf[1].metric("Mean call latency", f"{latency:.2f}s" if latency is not None else "—")
    perf[2].metric("Recorded inference time",
                   f"{sum(elapsed_values):.2f}s" if elapsed_values else "—")
    perf[3].metric("Mean document processing",
                   f"{mean_processing:.2f}s" if mean_processing is not None else "—")
    if calls:
        with st.expander("Model call telemetry · recorded metadata only"):
            st.dataframe(pd.DataFrame([{
                "purpose": call.get("purpose"), "model": call.get("model"),
                "seconds": call.get("seconds"), "tokens_in": call.get("tokens_in"),
                "tokens_out": call.get("tokens_out"),
                "tokens/s": call.get("tokens_per_second"),
                "JSON attempt": call.get("json_attempt"),
                "JSON validation (s)": call.get("json_validation_seconds"),
                "device": call.get("device"), "prompt_sha": call.get("prompt_sha"),
            } for call in calls]), use_container_width=True, hide_index=True)
            st.caption("Prompts and document text are not stored or sent to charting services.")

    st.markdown("#### Release activity")
    if not filtered_runs.empty:
        run_summaries = []
        for _, run in filtered_runs.iterrows():
            run_summaries.append({
                "Run": run.get("run_id"), "Time": run.get("created_at"),
                "Document": next((label for label, name in document_labels.items()
                                  if name == run.get("document")), "Document"),
                "Receiver": run.get("receiver"),
                "PII decisions": run.get("items"), "Policy version": run.get("rules_version"),
                "Model": run.get("model"), "Model calls": run.get("model_calls"),
                "Inference seconds": run.get("model_seconds"),
                "Model load seconds": run.get("model_load_seconds"),
                "Document processing seconds": run.get("processing_seconds"),
            })
        st.dataframe(pd.DataFrame(run_summaries), use_container_width=True, hide_index=True)

    if not filtered_decisions.empty:
        filtered_decisions = filtered_decisions.copy()
        filtered_decisions["document_label"] = filtered_decisions["document"].map(
            {filename: label for label, filename in document_labels.items()})
        safe_columns = [("created_at", "Time"), ("document_label", "Document"),
                        ("receiver", "Receiver"), ("item_type", "PII type"),
                        ("value_masked", "Masked value"), ("action", "Action"),
                        ("rule_id", "Policy rule"), ("rules_version", "Policy version"),
                        ("confidence", "Confidence"), ("decided_by", "Decision source")]
        shown = filtered_decisions[[column for column, _ in safe_columns
                                    if column in filtered_decisions.columns]].rename(
                                        columns=dict(safe_columns))
        if "Masked value" in shown:
            shown["Masked value"] = shown["Masked value"].map(_mask_audit_value)
        with st.expander(f"Decision-level audit trail · {len(shown)} records"):
            st.caption("Values are masked. The table omits raw values, model explanations, "
                       "and replacement text.")
            st.dataframe(shown, use_container_width=True, hide_index=True)
            csv_data = shown.to_csv(index=False).encode("utf-8")
            st.download_button("Download filtered, masked audit CSV", data=csv_data,
                               file_name="safeshare-audit-masked.csv", mime="text/csv")
            st.markdown("**Policy and decision trace**")
            st.caption("Historical per-run guard notes are not stored in the current audit "
                       "schema; this trace shows the recorded action, policy and reviewer.")
            for _, row in filtered_decisions.iterrows():
                st.markdown(
                    f"- `{_mask_audit_value(row.get('value_masked', ''))}` · "
                    f"{row.get('item_type', '')} "
                    f"→ **{row.get('action', '')}** · policy `{row.get('rule_id') or 'none'}` "
                    f"(version {row.get('rules_version') or 'unknown'}) · "
                    f"{row.get('decided_by') or 'system'}")


# ----------------------------------------------------------------- page 4: policy

POLICY_QUICK_QUESTIONS = [
    "May an external vendor see a phone number?",
    "What may HR keep in an employee record?",
    "هل يمكن مشاركة رقم الهوية مع مورد خارجي؟",
    "What does the analyst receive?",
    "What happens if no rule covers an item?",
]


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
    lib = default_library()
    editor = PolicyEditor()

    try:
        status = get_index().status()
        index_error = status.get("error", "")
    except PolicyIndexError as exc:
        status, index_error = {}, str(exc)

    _hero("POLICY INTELLIGENCE", "One policy. Every release.",
          "Explore receiver coverage, ask grounded questions, and audit policy changes.")
    columns = st.columns(4)
    columns[0].metric("Active rules", len(lib.rules))
    columns[1].metric("Receivers", len(lib.receivers))
    columns[2].metric("Policy version", lib.version)
    columns[3].metric("Indexed rules", status.get("documents", 0))
    coverage_entries = list(lib.receivers.items())
    for start in range(0, len(coverage_entries), 3):
        coverage = st.columns(min(3, len(coverage_entries) - start))
        for column, (key, receiver_info) in zip(coverage, coverage_entries[start:start + 3]):
            rule_count = len(lib.rules_for(key))
            with column:
                with st.container(border=True):
                    st.markdown(f"**{html.escape(receiver_info.label)}**")
                    st.metric("Coverage", f"{rule_count} rules")
                    st.caption(receiver_info.description)

    if index_error:
        st.error(index_error)
    elif not status.get("installed"):
        st.warning(
            f"The embedding model `{status.get('model', '')}` is not on this machine, so "
            f"questions cannot be searched yet. The rules below are fully editable and the "
            f"release pipeline is unaffected.")
        st.code("python -m scripts.setup_model", language="bash")

    tab_ask, tab_rules, tab_history, tab_document = st.tabs(
        ["Ask the policy", "Browse and edit", "Change history", "Released document"])
    with tab_ask:
        policy_ask_tab(lib, status)
    with tab_rules:
        policy_rules_tab(lib, editor)
    with tab_history:
        policy_history_tab(editor)
    with tab_document:
        policy_document_tab()


def policy_ask_tab(lib, status: dict) -> None:
    st.markdown("### ◈  Ask the policy")
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
        show_text(result["answer"])
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
        with st.container(border=True):
            head, relevance = st.columns([4, 1])
            head.markdown(
                f"<span class='rule-id'>{html.escape(hit['rule_id'])}</span>　"
                f"<span class='action-chip'>{html.escape(hit['action'])}</span>　"
                f"**{html.escape(hit['pii_type'])}** · "
                f"{html.escape(lib.receiver_label(hit['receiver']))}",
                unsafe_allow_html=True)
            relevance.metric("Relevance", f"{float(hit.get('relevance') or 0):.2f}")
            st.markdown(hit["reason"])
            if hit.get("citation"):
                st.caption(f"Source · {hit['citation']}")

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
        st.markdown("### Policy map")
        groups = {key: [rule for rule in rules if rule.receiver == key]
                  for key in (lib.receivers if receiver == "all" else [receiver])}
        group_columns = st.columns(min(3, max(1, len(groups))))
        for index, (key, group_rules) in enumerate(groups.items()):
            with group_columns[index % len(group_columns)]:
                st.markdown(f"#### {lib.receiver_label(key)}")
                for rule in group_rules:
                    color = CHART_COLORS.get(rule.action, CHART_COLORS["OTHER"])
                    st.markdown(
                        f"<div class='rule-card'><div class='rule-id'>"
                        f"{html.escape(rule.rule_id)}</div><div style='margin:.4rem 0'>"
                        f"<span class='action-chip' style='color:{color}'>"
                        f"{html.escape(rule.action)}</span>　"
                        f"<span class='muted'>{html.escape(rule.pii_type)}</span></div>"
                        f"<div class='rule-reason'>{html.escape(rule.reason)}</div>"
                        f"<div class='muted'>{html.escape(rule.citation or 'No source citation')}</div>"
                        f"</div>", unsafe_allow_html=True)

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
    history_frame = pd.DataFrame([{
        "when": row.get("created_at", ""), "editor": row.get("editor", ""),
        "kind": row.get("kind", ""), "rule": row.get("rule_id", ""),
        "version": f"{row.get('version_before', '')} → {row.get('version_after', '')}",
        "changes": "; ".join(row.get("changes") or []),
    } for row in history])
    if not history_frame.empty:
        history_frame["event_time"] = pd.to_datetime(history_frame["when"],
                                                       errors="coerce", utc=True)
        timeline = history_frame.dropna(subset=["event_time"]).copy()
        if not timeline.empty:
            timeline["policy event"] = timeline["kind"].astype(str) + " · " + timeline["rule"]
            figure = px.scatter(timeline, x="event_time", y="policy event",
                                color="kind", hover_data=["editor", "version"],
                                color_discrete_sequence=["#ff7180", "#718fe6",
                                                         "#40d6ae", "#ffb454"])
            figure.update_layout(xaxis_title="", yaxis_title="")
            _chart(figure, "policy-change-timeline", 280)
    with st.expander("Change records"):
        st.dataframe(history_frame.drop(columns=["event_time"], errors="ignore"),
                     use_container_width=True, hide_index=True)

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


def policy_document_tab() -> None:
    """Summarise, or ask questions about, the document that was just released.

    It reads the released text only (the same string "Safe version as text" shows on the
    Review page), never the original. It cannot change a decision or the release.
    """
    analysis = st.session_state.get("analysis")
    last = st.session_state.get("last")
    if analysis is None or not last:
        st.info("Generate a release on the Review page first. This tab then summarises it "
                "or answers questions about it.")
        return

    st.subheader("🤖 Safe Document RAG Assistant")
    released = safe_text(analysis.document, last["applied"]["changed"])
    run_id = last["run_id"]
    if st.session_state.get("doc_qa_run") != run_id:       # a new release: a new conversation
        st.session_state["doc_qa_run"] = run_id
        st.session_state["doc_qa_history"] = []
    history: list[dict] = st.session_state["doc_qa_history"]

    st.caption(f"{analysis.document.filename} - release {run_id} - {len(released):,} "
               f"characters. The model sees the released text only, never the original.")
    if config.resolve_provider() == "api":
        st.caption(f"The hosted model is used, so this released text is sent to "
                   f"`{config.API_BASE_URL}` in the prompt.")
    if last["applied"]["incomplete"]:
        st.warning(f"{len(last['applied']['pending'])} item(s) were still undecided and are "
                   f"not in this document, so answers may be missing them.")

    ready = engine.status().get("installed", False)
    if not ready:
        st.info("The model is not configured, so questions are unavailable.")

    question = st.text_input("Ask about the document", key="doc_qa_question",
                             placeholder="Who signed the agreement? / من الطرف الثاني في العقد؟")
    ask_col, sum_col, clear_col = st.columns([1, 1, 3])
    ask = ask_col.button("Ask", type="primary", key="doc_qa_ask",
                         disabled=not (ready and question.strip()))
    summarize = sum_col.button("Summarize", key="doc_qa_summarize", disabled=not ready)
    if clear_col.button("Clear conversation", key="doc_qa_clear", disabled=not history):
        history.clear()
        st.rerun()

    if ask or summarize:
        with st.spinner("The model is reading the released document..."):
            try:
                history.append(document_qa.summarize(released) if summarize
                               else document_qa.ask(released, question))
            except document_qa.DocumentQAError as exc:
                st.error(str(exc))
            except engine.ModelNotInstalled as exc:
                st.error(str(exc))
            except Exception as exc:                                   # noqa: BLE001
                st.error(f"The model did not produce a usable answer: {exc}")

    for entry in reversed(history):
        with st.container(border=True):
            st.markdown("**Summary**" if entry["kind"] == "summary"
                        else f"**Q:** {entry['question']}")
            show_text(entry["answer"])
            for note in entry["notes"]:
                st.caption(note)
            st.caption(f"{entry['model']} - {entry['seconds']}s")


# --------------------------------------------------------------------------- main


def main() -> None:
    st.set_page_config(page_title="SafeShare | Privacy workspace",
                       page_icon="🔒", layout="wide")
    _apply_theme()

    with st.sidebar:
        st.markdown(
            "<div style='padding:.55rem .2rem 1rem'>"
            "<div style='color:#ff7785;font-size:.72rem;font-weight:800;"
            "letter-spacing:.16em'>SAFESHARE</div>"
            "<div style='font-size:1.3rem;font-weight:700;margin-top:.3rem'>"
            "Privacy workspace</div>"
            "<div style='color:#8294ad;font-size:.8rem;margin-top:.3rem'>"
            "AI governance · secure release</div></div>",
            unsafe_allow_html=True)
        pages = ["Upload", "Review", "Audit log", "Policy"]
        current_page = st.session_state.get("page", "Upload")
        if current_page not in pages:
            current_page = "Upload"
        page = st.radio("Workspace", pages, index=pages.index(current_page),
                        label_visibility="collapsed")
        st.session_state["page"] = page
        st.divider()
        state = engine.status()
        st.markdown("**AI provider**")
        st.caption(f"{'●' if state.get('installed') else '○'} "
                   f"{state.get('model') or 'Model unavailable'}")
        if state.get("provider") == "api":
            st.caption("API endpoint")
        else:
            quantization = " · NF4 4-bit" if state.get("load_in_4bit") else ""
            st.caption(f"{state.get('device', 'local')} · {state.get('dtype', '')}"
                       f"{quantization}")
        st.caption("Documents are sent to the configured API."
                   if state.get("provider") == "api"
                   else "Local inference · document text stays on-device.")
        st.divider()
        st.caption("POLICY VERSION")
        st.caption(default_library().version)
        st.caption("Audit values are masked. Chart data stays local.")

    {"Upload": page_upload, "Review": page_review, "Audit log": page_audit,
     "Policy": page_policy}[page]()


if __name__ == "__main__":
    main()