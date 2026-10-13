"""Reading PDF, DOCX and TXT into blocks.

Privacy: nothing here sends the file anywhere. TXT and DOCX are read in memory.
A PDF is opened from disk because PyMuPDF needs a file, so the upload is written to
the temporary directory first and deleted immediately afterwards (see `config.TMP_DIR`
and `store_upload`). Set SAFESHARE_KEEP_UPLOADS=1 only while debugging.

Files are never stored permanently: the audit log keeps the decisions, not the document.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from .. import config
from .document import Block, Document

SUPPORTED = {".txt", ".md", ".docx", ".pdf"}


class UnsupportedDocument(Exception):
    pass


def store_upload(filename: str, content: bytes) -> Path:
    """Write an upload to the temporary folder and return its path."""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED:
        raise UnsupportedDocument(
            f"{suffix or 'this file type'} is not supported. Use PDF, DOCX or TXT.")
    config.TMP_DIR.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(dir=config.TMP_DIR, suffix=suffix, delete=False)
    handle.write(content)
    handle.close()
    return Path(handle.name)


def discard(path: str | Path | None) -> None:
    """Delete a temporary upload. Called in a finally block, always."""
    if not path or config.KEEP_UPLOADS:
        return
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass


# --------------------------------------------------------------------------- txt


def read_txt(path: str | Path, filename: str | None = None) -> Document:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    blocks = [Block(id=f"b{i}", text=line.rstrip("\n"), kind="line")
              for i, line in enumerate(text.splitlines())]
    return Document(blocks=blocks, filename=filename or Path(path).name,
                    kind="txt", path=str(path))


# -------------------------------------------------------------------------- docx


def read_docx(path: str | Path, filename: str | None = None) -> Document:
    from docx import Document as DocxDocument          # noqa: PLC0415
    from docx.table import Table                        # noqa: PLC0415
    from docx.text.paragraph import Paragraph           # noqa: PLC0415

    docx = DocxDocument(str(path))
    blocks: list[Block] = []
    counter = 0
    section = None
    table_index = 0

    for child in docx.element.body.iterchildren():
        if child.tag.endswith("}p"):
            paragraph = Paragraph(child, docx)
            text = paragraph.text
            style = (paragraph.style.name or "").lower()
            kind = "heading" if style.startswith("heading") or style == "title" else "paragraph"
            if kind == "heading" and text.strip():
                section = text.strip()
            blocks.append(Block(id=f"b{counter}", text=text, kind=kind, section=section,
                                ref={"kind": "paragraph", "index": counter},
                                meta={"style": style}))
            counter += 1
        elif child.tag.endswith("}tbl"):
            table = Table(child, docx)
            for row_index, row in enumerate(table.rows):
                for col_index, cell in enumerate(row.cells):
                    for para_index, paragraph in enumerate(cell.paragraphs):
                        blocks.append(Block(
                            id=f"b{counter}", text=paragraph.text, kind="table_cell",
                            section=section,
                            ref={"kind": "cell", "table": table_index, "row": row_index,
                                 "col": col_index, "para": para_index},
                            meta={"table": table_index}))
                        counter += 1
            table_index += 1

    return Document(blocks=blocks, filename=filename or Path(path).name,
                    kind="docx", path=str(path))


# --------------------------------------------------------------------------- pdf


def read_pdf(path: str | Path, filename: str | None = None) -> Document:
    """Text extraction only.

    Layout preservation for PDFs is out of scope for the prototype (the build
    instructions allow this). We read the text so detection and review work, and the
    release is written out as DOCX or TXT with a note saying so.
    """
    import fitz                                          # noqa: PLC0415

    blocks: list[Block] = []
    counter = 0
    with fitz.open(str(path)) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            for line in page.get_text("text").splitlines():
                if not line.strip():
                    continue
                blocks.append(Block(id=f"b{counter}", text=line.rstrip(), kind="page_line",
                                    ref={"kind": "pdf_line", "page": page_number},
                                    meta={"page": page_number}))
                counter += 1
    return Document(blocks=blocks, filename=filename or Path(path).name,
                    kind="pdf", path=str(path), meta={"layout_preserved": False})


def read(path: str | Path, filename: str | None = None) -> Document:
    """Pick the reader from the file extension."""
    suffix = Path(path).suffix.lower()
    if suffix in {".txt", ".md"}:
        return read_txt(path, filename)
    if suffix == ".docx":
        return read_docx(path, filename)
    if suffix == ".pdf":
        return read_pdf(path, filename)
    raise UnsupportedDocument(f"{suffix} is not supported. Use PDF, DOCX or TXT.")


def read_bytes(filename: str, content: bytes) -> Document:
    """Read an uploaded file, then delete the temporary copy straight away.

    The bytes are kept in memory so the released DOCX can still be written from the
    original structure - nothing needs to stay on disk, which is the point.
    """
    path = store_upload(filename, content)
    try:
        document = read(path, filename)
    finally:
        discard(path)
    if Path(filename).suffix.lower() == ".docx":
        document.meta["original_bytes"] = content
    return document


def read_sample_copy(path: str | Path) -> Document:
    """Read a file that is already on disk (used by the demo and the evaluator)."""
    return read(path)


def describe_storage() -> dict:
    """What the UI shows on the privacy line."""
    return {"temporary_files": str(config.TMP_DIR),
            "kept_after_processing": config.KEEP_UPLOADS,
            "note": "uploads are deleted after extraction unless SAFESHARE_KEEP_UPLOADS=1"}
