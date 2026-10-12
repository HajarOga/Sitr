"""The document model.

A document is a list of **blocks**: paragraphs, headings, table cells and lines.
Detections are found inside a block and carry that block's id plus offsets within
the block's text. Keeping the structure this way is what lets the released document
still be a table, with its headings and numbers in place - the alternative, turning
everything into one string, throws the layout away.

Each block also carries a `ref`: the information the writer needs to put the changed
text back where it came from.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Block:
    id: str
    text: str
    kind: str = "paragraph"          # paragraph | heading | table_cell | line | page_line
    section: str | None = None       # nearest heading above, when we know it
    ref: dict = field(default_factory=dict)   # writer locator: {"para": 12} or {"table":0,"row":2,"col":1}
    meta: dict = field(default_factory=dict)  # page number for PDFs, style name, ...

    def replace_span(self, start: int, end: int, new_text: str) -> None:
        self.text = self.text[:start] + new_text + self.text[end:]


@dataclass
class Document:
    blocks: list[Block]
    filename: str = "document"
    kind: str = "txt"                # txt | docx | pdf
    path: str | None = None          # original file, when it came from disk
    meta: dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n".join(block.text for block in self.blocks)

    def block(self, block_id: str) -> Block:
        for block in self.blocks:
            if block.id == block_id:
                return block
        raise KeyError(block_id)

    def text_offsets(self) -> dict[str, int]:
        """Where each block starts in `text` (blocks are joined with "\n")."""
        offsets, cursor = {}, 0
        for block in self.blocks:
            offsets[block.id] = cursor
            cursor += len(block.text) + 1
        return offsets

    def global_span(self, block_id: str, start: int, end: int) -> tuple[int, int]:
        """Convert a block-local span to document offsets."""
        base = self.text_offsets().get(block_id, 0)
        return base + start, base + end

    def visible_text(self) -> str:
        """The text a reader would see, used by the leak-rate metric."""
        return "\n".join(block.text for block in self.blocks if block.text.strip())
