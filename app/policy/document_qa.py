"""Summarise, or answer questions about, the document SafeShare released.

This is a reading aid for the person who just generated a release. It is given the
**released text only** - what `safe_text(document, applied["changed"])` returns - never the
original, so the model sees exactly what the receiver would see: substitutes instead of
real values, removed values absent, undecided items left out.

Like the policy Q&A next to it, it never changes a decision or a document. It reads.

Retrieval is deliberately small and dependency-free. A document that fits in one prompt
(`config.MAX_DOC_CHARS`) is sent whole, which is the common case for this product. A longer
one is split on line boundaries and, for a question, only the passages that share the most
character trigrams with it are sent (trigrams, not words, so Arabic prefixes and spelling
variants still match). A summary of a long document is built per part and then combined.
"""
from __future__ import annotations

import re
import time

from .. import config
from ..llm import engine

PART_SIZE = 1800            # characters per passage when a document is too long for one prompt
TOP_PASSAGES = 4            # passages sent for a question about a long document
MAX_SUMMARY_PARTS = 8       # beyond this a summary says it covered only the start

_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_DIACRITICS = re.compile("[\u0610-\u061a\u064b-\u065f\u0670\u0640]")


class DocumentQAError(RuntimeError):
    """The model answered, but not with anything we can show."""


_SYSTEM = (
    "You answer questions about one document. The document below is the RELEASED, safe "
    "version of a company document: personal data in it was already replaced with fake "
    "substitute values or taken out before you saw it.\n"
    "Rules:\n"
    "- Use only the document text. If it does not contain the answer, say so plainly.\n"
    "- A substitute value is synthetic. Present it as it appears; never call it real and "
    "never try to guess, rebuild or hint at an original value that was replaced or removed.\n"
    "- If the question asks for something that looks removed, say that the released "
    "document does not contain it.\n"
    "- Answer in the language of the user's request (Arabic or English).\n"
    "- Be concise and specific.\n"
    'Reply with JSON only: {"answer": "<your answer>"}'
)


def _messages(task: str, document_text: str) -> list[dict]:
    return [{"role": "system", "content": _SYSTEM},
            {"role": "user", "content": f"DOCUMENT:\n\"\"\"\n{document_text}\n\"\"\"\n\n{task}"}]


def _normalise(text: str) -> str:
    text = _DIACRITICS.sub("", text.translate(_ARABIC_DIGITS).lower())
    return re.sub(r"\s+", " ", text).strip()


def _trigrams(text: str) -> set[str]:
    text = _normalise(text)
    return {text[i:i + 3] for i in range(max(0, len(text) - 2))}


def split_passages(text: str, size: int = PART_SIZE) -> list[str]:
    """Line-boundary passages of about `size` characters; a long line is cut at `size`."""
    passages: list[str] = []
    current: list[str] = []
    length = 0
    for line in text.split("\n"):
        while len(line) > size:
            if current:
                passages.append("\n".join(current))
                current, length = [], 0
            passages.append(line[:size])
            line = line[size:]
        if length + len(line) + 1 > size and current:
            passages.append("\n".join(current))
            current, length = [], 0
        current.append(line)
        length += len(line) + 1
    if current:
        passages.append("\n".join(current))
    return [p for p in passages if p.strip()]


def pick_passages(text: str, question: str, top: int = TOP_PASSAGES) -> list[str]:
    """The passages most like the question, returned in reading order."""
    passages = split_passages(text)
    wanted = _trigrams(question)
    if not wanted or len(passages) <= top:
        return passages[:top] if len(passages) > top else passages
    scored = sorted(range(len(passages)),
                    key=lambda i: len(wanted & _trigrams(passages[i])) / len(wanted),
                    reverse=True)[:top]
    return [passages[i] for i in sorted(scored)]


def _ask_model(task: str, document_text: str, purpose: str, max_new_tokens: int) -> str:
    payload, _meta = engine.generate_json(_messages(task, document_text), purpose=purpose,
                                          max_new_tokens=max_new_tokens)
    answer = None
    if isinstance(payload, dict):
        answer = payload.get("answer") or payload.get("summary")
    elif isinstance(payload, str):
        answer = payload
    answer = str(answer or "").strip()
    if not answer:
        raise DocumentQAError(f"{config.ACTIVE_MODEL} did not return an answer I can show.")
    return answer


def _result(kind: str, answer: str, started: float, notes: list[str], **extra) -> dict:
    return {"kind": kind, "answer": answer, "model": config.ACTIVE_MODEL,
            "seconds": round(time.perf_counter() - started, 2), "notes": notes, **extra}


def ask(released_text: str, question: str) -> dict:
    """Answer one question from the released document."""
    question = (question or "").strip()
    if not question:
        raise DocumentQAError("write a question first")
    if not released_text.strip():
        raise DocumentQAError("the released document is empty")

    started = time.perf_counter()
    notes: list[str] = []
    if len(released_text) <= config.MAX_DOC_CHARS:
        context, whole = released_text, True
    else:
        context = "\n[...]\n".join(pick_passages(released_text, question))
        whole = False
        notes.append("The document is long, so the model was shown only the passages "
                     "closest to your question. If the answer looks incomplete, ask it "
                     "more specifically.")
    answer = _ask_model(f"QUESTION: {question}", context, "document question", 700)
    return _result("question", answer, started, notes, question=question,
                   whole_document=whole, context_chars=len(context))


def summarize(released_text: str) -> dict:
    """A short summary of the released document."""
    if not released_text.strip():
        raise DocumentQAError("the released document is empty")

    started = time.perf_counter()
    notes: list[str] = []
    task = ("Summarise this document in a short paragraph and then 3-6 bullet points of the "
            "main facts. Use the language the document is mainly written in.")

    if len(released_text) <= config.MAX_DOC_CHARS:
        answer = _ask_model(task, released_text, "document summary", 600)
        return _result("summary", answer, started, notes, whole_document=True,
                       context_chars=len(released_text))

    parts = split_passages(released_text, config.MAX_DOC_CHARS // 2)
    if len(parts) > MAX_SUMMARY_PARTS:
        notes.append(f"The document is long ({len(parts)} parts); the summary covers the "
                     f"first {MAX_SUMMARY_PARTS}.")
        parts = parts[:MAX_SUMMARY_PARTS]
    partial = [_ask_model("Summarise this part of a longer document in 3-5 sentences.",
                          part, f"document summary part {i}/{len(parts)}", 350)
               for i, part in enumerate(parts, start=1)]
    answer = _ask_model(task + " The text below is a set of summaries of consecutive parts; "
                        "combine them into one.",
                        "\n\n".join(f"Part {i}: {s}" for i, s in enumerate(partial, 1)),
                        "document summary combine", 600)
    return _result("summary", answer, started, notes, whole_document=False,
                   context_chars=len(released_text))