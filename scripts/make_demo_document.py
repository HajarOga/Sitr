"""Build the demo document.

One document, both languages, everything the final demo needs to show:

  Arabic name, English name, Saudi phone in Arabic-Indic digits, email,
  National ID, home address, company registration and VAT numbers,
  and useful non-sensitive business information (invoice total, ticket, fault).

Written with python-docx so the document has a heading, paragraphs and a table - the
release has to keep that structure, so the demo document must actually have it.

    python -m scripts.make_demo_document
"""
from __future__ import annotations

from pathlib import Path

from docx import Document as DocxDocument
from docx.shared import Pt

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "synthetic"

# ------------------------------------------------------------------ the contents
# Deliberately mixed: Arabic and English in the same document, and digits written
# both ways - this is what a Saudi company's documents actually look like.
LINES = [
    ("heading", "Customer Complaint - Ufuq Digital Services"),
    ("heading", "شكوى عميل - أفق للخدمات الرقمية"),
    ("plain", "Report no.: CMP-2026-0417            Date: 2026-10-07"),
    ("heading", "Customer details"),
    ("plain", "Name: Abdullah Mohammed Alharbi"),
    ("plain", "الاسم: عبدالله محمد الحربي"),
    ("plain", "National ID: 1045837299"),
    ("plain", "Mobile: +966 55 123 4567            الجوال: ٠٥٥١٢٣٤٥٦٧"),
    ("plain", "Email: a.alharbi@example.com"),
    ("plain", "Address: Al Narjis district, Street 12, Villa 34"),
    ("plain", "العنوان: حي النرجس، شارع ١٢، فيلا ٣٤"),
    ("plain", "Customer no.: CUST-4471"),
    ("heading", "Complaint"),
    ("plain", "My internet has been down for three nights. My neighbour Saad Al-Anzi "
              "has the same problem on the same street. Please fix it quickly."),
    ("plain", "Ticket ref.: TKT-88213            Service: fibre 500 Mbps"),
    ("heading", "Invoice"),
    ("plain", "Total: 1,250.00 SAR            VAT number: 310123456700003"),
    ("plain", "Card: **** **** **** 4412 - expires 07/29"),
    ("heading", "Handling agent"),
    ("plain", "Name: Sarah Al-Otaibi            Employee ID: EMP-33217"),
]

TABLE_ROWS = [
    ("Item", "Amount (SAR)"),
    ("Line rental - October", "750.00"),
    ("Router replacement", "500.00"),
    ("Total", "1,250.00"),
]

# Gold truth for the demo document: what is sensitive, and what must survive.
SENSITIVE = [
    ("Abdullah Mohammed Alharbi", "PERSON_NAME"),
    ("عبدالله محمد الحربي", "PERSON_NAME"),
    ("1045837299", "NATIONAL_ID"),
    ("+966 55 123 4567", "PHONE"),
    ("٠٥٥١٢٣٤٥٦٧", "PHONE"),
    ("a.alharbi@example.com", "EMAIL"),
    ("Al Narjis district, Street 12, Villa 34", "ADDRESS"),
    ("حي النرجس، شارع ١٢، فيلا ٣٤", "ADDRESS"),
    ("CUST-4471", "CUSTOMER_ID"),
    ("Saad Al-Anzi", "PERSON_NAME"),
    ("Sarah Al-Otaibi", "PERSON_NAME"),
    ("EMP-33217", "EMPLOYEE_ID"),
    ("310123456700003", "COMPANY_IDENTIFIER"),
]

MUST_KEEP = [
    "1,250.00 SAR", "TKT-88213", "fibre 500 Mbps", "three nights",
    "Line rental - October", "Router replacement", "CMP-2026-0417",
]


def build(out_dir: Path = OUT_DIR) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    document = DocxDocument()

    for kind, text in LINES:
        if kind == "heading":
            document.add_heading(text, level=2)
            continue
        paragraph = document.add_paragraph(text)
        if text.startswith("Total:"):
            paragraph.runs[0].bold = True

    document.add_paragraph()
    table = document.add_table(rows=0, cols=2)
    table.style = "Light Grid Accent 1"
    for label, amount in TABLE_ROWS:
        cells = table.add_row().cells
        cells[0].text = label
        cells[1].text = amount
        cells[1].paragraphs[0].runs[0].font.size = Pt(10)

    docx_path = out_dir / "COMPLAINT-2026-0417.docx"
    document.save(docx_path)

    # a plain-text version of the same document, for anyone without Word
    text_path = out_dir / "COMPLAINT-2026-0417.txt"
    text_path.write_text("\n".join(item[1] for item in LINES), encoding="utf-8")

    return {"docx": str(docx_path), "txt": str(text_path),
            "sensitive_items": len(SENSITIVE), "must_keep": len(MUST_KEEP)}


if __name__ == "__main__":
    import json
    print(json.dumps(build(), ensure_ascii=False, indent=1))
