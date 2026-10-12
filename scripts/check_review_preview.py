"""Check the Review page's two columns: the right one must be the released text."""
import html as html_mod
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import config
from app.core import apply_decisions, analyze
from app.document_processing import readers
from app.document_processing.writers import safe_text
from app.pii_detection.schema import Action
from app.review.review_queue import apply_review
from app.ui_streamlit import highlight_block, items_for_block, released_block_html


def strip_tags(fragment: str) -> str:
    text = re.sub(r"<[^>]+>", "", fragment)
    return html_mod.unescape(text).strip()


def norm(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.split("\n")).strip()


path = sorted((config.DATA_DIR / "synthetic").glob("*.txt"))[0]
stored = readers.store_upload(path.name, path.read_bytes())
try:
    document = readers.read(stored, path.name)
finally:
    readers.discard(stored)

analysis = analyze(document, "EXTERNAL_VENDOR", "a vendor is fixing the line")

# exactly what the "Generate release" button does, with no reviewer answers yet
preview = apply_review(analysis, {})
applied = {item["id"]: item for item in preview["applied"]}

print(f"items: {len(analysis.detections)} | applied {len(preview['applied'])} "
      f"| pending {len(preview['pending'])}")

problems = []
original_leaks = []
page_blocks = []
for block in document.blocks:
    if not block.text.strip():
        continue
    items = items_for_block(analysis, block.id)

    left = strip_tags(highlight_block(block.text, items))
    right = strip_tags(released_block_html(block.text, items, applied))
    released = norm(preview["changed"][block.id])

    if norm(right) != released:
        problems.append(f"block {block.id}: right column != released text\n"
                        f"  page:     {norm(right)!r}\n  released: {released!r}")
    # a KEEP changes nothing, so identical columns are only wrong when something in the
    # block was replaced or removed
    if left == right and any(d.block_id == block.id and d.action != Action.KEEP.value
                             for d in analysis.detections):
        problems.append(f"block {block.id}: the two columns are identical")
    if right.strip():
        page_blocks.append(right)

    for d in analysis.detections:
        if d.block_id != block.id:
            continue
        if d.action in (Action.REMOVE.value, Action.REPLACE.value) \
                and d.original_text in right:
            original_leaks.append(f"{d.type} {d.original_text!r} still in the safe column")
        if d.action == Action.REPLACE.value and d.replacement \
                and d.replacement not in right:
            problems.append(f"block {block.id}: substitute {d.replacement!r} missing "
                            f"from the safe column")
        if d.action == Action.KEEP.value and d.original_text not in right:
            problems.append(f"block {block.id}: kept value {d.original_text!r} missing")

whole_page = "\n".join(page_blocks)
released_doc = safe_text(document, preview["changed"])
same = "\n".join(norm(p) for p in page_blocks) == "\n".join(
    norm(x) for x in released_doc.split("\n"))
print(f"\nwhole-page text == released document text: "
      f"{norm(whole_page) == norm(released_doc)}")

print("\n--- what the Safe version column now reads ---")
print(whole_page)
print("--- end ---")

print("\nleaks (original value visible in the safe column):", original_leaks or "none")
print("problems:", *problems, sep="\n  " if problems else " ")
raise SystemExit(1 if (problems or original_leaks) else 0)
