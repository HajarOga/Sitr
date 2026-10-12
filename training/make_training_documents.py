"""Build our own training documents, with labels.

    python -m training.make_training_documents            # 24 documents
    python -m training.make_training_documents --count 48

Why this exists: the fixed evaluation set in `data/test/` must never be trained on, so
training needs its own documents. These are generated from the same business templates
but with a **different value pool**, and the script refuses to write anything that shares
a labelled value with the test set - so "we did not train on the test set" is a check that
runs, not a promise in a README.

Every document records its labels *by construction*: the generator writes the value and
records where it put it, so the labels cannot drift from the text.

Output: `data/train/documents/<id>.txt` and `<id>.gold.json`.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TEST_DIR = ROOT / "data" / "test"
OUT_DIR = ROOT / "data" / "train" / "documents"

# a different pool from the evaluation set, on purpose (no shared values)
AR_NAMES = ["ماجد سعود العنزي", "هند ناصر الشهري", "بندر تركي المطيري", "لمى عادل الغامدي",
            "فيصل حمد البلوي", "عبير سلطان الجهني", "طلال مشعل السبيعي", "منى راشد الخالدي"]
EN_NAMES = ["Majed Saud Al-Anazi", "Hind Nasser Al-Shahri", "Bandar Turki Al-Mutairi",
            "Lama Adel Al-Ghamdi", "Faisal Hamad Al-Balawi", "Abeer Sultan Al-Juhani",
            "Talal Mishal Al-Subaie", "Mona Rashid Al-Khalidi"]
PHONES_EN = ["0533221100", "+966 53 445 5667", "0567788990"]
PHONES_AR = ["٠٥٤٤٧٧٨٨٩٩", "٠٥٦١٢٣٤٥٦٧"]
EMAILS = ["m.anazi@example.com", "h.shahri@example.com", "b.mutairi@example.com",
          "l.ghamdi@example.com"]
IDS = ["1055667788", "1077889900", "1066554433", "1088990011"]      # Luhn-valid, as real IDs are
ADDRESSES_EN = ["Al Yasmin district, Street 7, Villa 22",
                "Al Rehab district, Street 14, Building 3",
                "Al Nakheel district, King Abdullah Road"]
ADDRESSES_AR = ["حي الياسمين، شارع ٧، فيلا ٢٢", "حي الرحاب، شارع ١٤، مبنى ٣",
                "حي النخيل، طريق الملك عبدالله"]
IBANS = ["SA45 8000 0000 6080 1016 9999", "SA21 8000 0000 6080 1016 7519",
         "SA88 8000 0000 6080 1016 1123", "SA12 8000 0000 6080 1016 4560"]
CUSTOMER_IDS = ["CUST-7201", "CUST-7255", "CUST-7388", "CUST-7412"]
EMPLOYEE_IDS = ["EMP-51044", "EMP-52119", "EMP-53307", "EMP-54882"]
DOBS = ["18/05/1992", "07/09/1988", "30/01/1995", "14/06/1983"]
PASSPORTS = ["C55667788", "D44332211", "E77889900"]

BUSINESS = ["Total: 2,480.00 SAR", "Invoice no.: INV-2026-7712", "Contract term: 12 months",
            "Monthly volume: 8,940 units", "Region: Eastern", "VAT number: 310987654300003",
            "CR number: 1010778899", "Care plan: standard", "Subtotal: 1,900.00 SAR",
            "Total: 63,200.00 SAR", "Subject: renewing the supply contract for 12 months",
            "Payment terms: 45 days from invoice date", "الموضوع: تجديد عقد التوريد",
            "الراتب الأساسي: 11,500.00 ريال", "مدة الخدمة: ثلاث سنوات"]

# template, language, receiver — the same eight business situations as the evaluation set
TEMPLATES = [
    {"template": "complaint", "receiver": "EXTERNAL_VENDOR", "language": "en",
     "title": "Service Complaint Report",
     "body": ["Customer name: {EN_NAME}", "National ID: {ID}", "Mobile: {PHONE}",
              "Email: {EMAIL}", "Address: {ADDRESS}", "Customer no.: {CUSTOMER_ID}",
              "The air conditioning unit stopped cooling and the customer reported a "
              "loud noise before it shut down.",
              "Technician visit on 06/10/2026 confirmed the compressor failed.",
              "Parts required: compressor, capacitor."]},
    {"template": "complaint", "receiver": "EXTERNAL_VENDOR", "language": "ar",
     "title": "تقرير شكوى خدمة",
     "body": ["الاسم: {AR_NAME}", "رقم الهوية: {ID}", "الجوال: {AR_PHONE}",
              "البريد الإلكتروني: {EMAIL}", "العنوان: {AR_ADDRESS}",
              "رقم العميل: {CUSTOMER_ID}",
              "توقف المكيف عن التبريد وذكر العميل سماع صوت مرتفع قبل توقفه.",
              "تمت الزيارة بتاريخ ٠٦/١٠/٢٠٢٦ وتأكد تلف الضاغط."]},
    {"template": "hr-letter", "receiver": "INTERNAL_HR", "language": "en",
     "title": "Salary Certificate",
     "body": ["Employee name: {EN_NAME}", "Employee ID: {EMPLOYEE_ID}",
              "National ID: {ID}", "Mobile: {PHONE}", "Email: {EMAIL}",
              "Address: {ADDRESS}", "Date of birth: {DOB}",
              "This certificate is issued at the employee's request for a bank "
              "application and confirms the basic salary stated in the payroll system."]},
    {"template": "hr-record", "receiver": "INTERNAL_HR", "language": "ar",
     "title": "ملف موظف محدث",
     "body": ["اسم الموظف: {AR_NAME}", "الرقم الوظيفي: {EMPLOYEE_ID}",
              "رقم الهوية: {ID}", "الجوال: {AR_PHONE}", "تاريخ الميلاد: {DOB}",
              "العنوان: {AR_ADDRESS}", "IBAN: {IBAN}",
              "تم تحديث الملف بعد ترقية الموظف إلى الدرجة الثانية."]},
    {"template": "kyc", "receiver": "EXTERNAL_VENDOR", "language": "mixed",
     "title": "نموذج تحديث بيانات - KYC update",
     "body": ["الاسم: {AR_NAME}", "Name: {EN_NAME}", "رقم الهوية: {ID}",
              "رقم الجواز: {PASSPORT}", "الجوال: {AR_PHONE}", "Mobile: {PHONE}",
              "العنوان: {AR_ADDRESS}", "IBAN: {IBAN}", "النشاط: خدمات لوجستية"]},
    {"template": "analysis", "receiver": "ANALYST", "language": "en",
     "title": "Quarterly Handling Analysis",
     "body": ["Customer no.: {CUSTOMER_ID}", "Employee ID: {EMPLOYEE_ID}",
              "Name: {EN_NAME}", "National ID: {ID}", "Address: {ADDRESS}",
              "Date of birth: {DOB}", "Region: Eastern",
              "Average resolution time: 2.8 days", "Escalations: 7%"]},
    {"template": "support", "receiver": "CUSTOMER_SUPPORT", "language": "ar",
     "title": "تذكرة دعم فني",
     "body": ["الاسم: {AR_NAME}", "رقم العميل: {CUSTOMER_ID}", "الجوال: {AR_PHONE}",
              "البريد الإلكتروني: {EMAIL}", "رقم الهوية: {ID}",
              "العنوان: {AR_ADDRESS}",
              "تم إرسال رسالة نصية للعميل بموعد الزيارة، بانتظار التأكيد."]},
    {"template": "supplier", "receiver": "EXTERNAL_VENDOR", "language": "mixed",
     "title": "خطاب إلى مورد - Letter to a supplier",
     "body": ["السادة: شركة النماذج للتوريدات", "المرجع: SUP-2026-902",
              "اسم الممثل: {AR_NAME}", "Contact person: {EN_NAME}", "Mobile: {PHONE}",
              "البريد الإلكتروني: {EMAIL}", "IBAN: {IBAN}",
              "Subject: renewing the supply contract for 12 months",
              "Payment terms: 45 days from invoice date"]},
]

PLACEHOLDERS = {"EN_NAME", "AR_NAME", "PHONE", "AR_PHONE", "EMAIL", "ID", "ADDRESS",
                "AR_ADDRESS", "CUSTOMER_ID", "EMPLOYEE_ID", "DOB", "PASSPORT", "IBAN"}

TYPE_OF = {"EN_NAME": "PERSON_NAME", "AR_NAME": "PERSON_NAME", "PHONE": "PHONE",
           "AR_PHONE": "PHONE", "EMAIL": "EMAIL", "ID": "NATIONAL_ID", "ADDRESS": "ADDRESS",
           "AR_ADDRESS": "ADDRESS", "CUSTOMER_ID": "CUSTOMER_ID",
           "EMPLOYEE_ID": "EMPLOYEE_ID", "DOB": "DATE_OF_BIRTH",
           "PASSPORT": "PASSPORT_NUMBER", "IBAN": "IBAN"}

BUSINESS_TOKENS = ["Total:", "VAT number", "CR number", "Invoice no", "Contract term",
                   "Monthly volume", "Region", "Subtotal", "Care plan", "Subject:",
                   "Payment terms", "الموضوع", "الراتب الأساسي", "مدة الخدمة"]


def _values(index: int, rng: random.Random) -> dict[str, str]:
    pick = lambda pool: pool[index % len(pool)]                       # noqa: E731
    return {"EN_NAME": pick(EN_NAMES), "AR_NAME": pick(AR_NAMES),
            "PHONE": pick(PHONES_EN), "AR_PHONE": pick(PHONES_AR),
            "EMAIL": pick(EMAILS), "ID": pick(IDS), "ADDRESS": pick(ADDRESSES_EN),
            "AR_ADDRESS": pick(ADDRESSES_AR), "CUSTOMER_ID": pick(CUSTOMER_IDS),
            "EMPLOYEE_ID": pick(EMPLOYEE_IDS), "DOB": pick(DOBS),
            "PASSPORT": pick(PASSPORTS), "IBAN": pick(IBANS)}


def _build(spec: dict, index: int, rng: random.Random) -> tuple[str, list[dict]]:
    values = _values(index, rng)
    lines = [spec["title"], f"Reference: {spec['template'].upper()}-2026-{index + 100:03d}"]
    inserted: list[tuple[str, str]] = []
    for template in spec["body"]:
        line = template
        for field in [f for f in PLACEHOLDERS if "{" + f + "}" in template]:
            line = line.replace("{" + field + "}", values[field])
            inserted.append((values[field], TYPE_OF[field]))
        lines.append(line)
    lines.append(BUSINESS[index % len(BUSINESS)])
    text = "\n".join(lines)

    spans, seen = [], {}
    for value, pii_type in inserted:
        occurrence = seen.get(value, 0)
        position = _find_nth(text, value, occurrence)
        seen[value] = occurrence + 1
        if position is not None:
            spans.append({"text": value, "type": pii_type, "start": position,
                          "end": position + len(value)})
    return text, spans


def _find_nth(text: str, value: str, occurrence: int) -> int | None:
    start, position = 0, None
    for _ in range(occurrence + 1):
        position = text.find(value, start)
        if position == -1:
            return None
        start = position + 1
    return position


def _test_values() -> set[str]:
    """Every labelled value in the evaluation set - training must not contain one."""
    values: set[str] = set()
    for path in TEST_DIR.glob("*.gold.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        values |= {span["text"] for span in payload.get("spans", [])}
    return values


def build(count: int = 24, out_dir: Path = OUT_DIR, seed: int = 7) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    forbidden = _test_values()
    written, collisions = [], []

    for index in range(count):
        spec = TEMPLATES[index % len(TEMPLATES)]
        text, spans = _build(spec, index, rng)

        clash = [span["text"] for span in spans if span["text"] in forbidden]
        if clash:
            collisions.append({"document": index, "values": clash})
            continue

        document_id = f"train-{spec['template']}-{index:02d}"
        (out_dir / f"{document_id}.txt").write_text(text, encoding="utf-8")
        gold = {"document_id": document_id, "receiver": spec["receiver"],
                "language": spec["language"],
                "source": "generated by training/make_training_documents.py - synthetic",
                "spans": spans,
                "must_keep": [line.strip() for line in text.splitlines()
                          if any(token in line for token in BUSINESS_TOKENS)][:1],
            "receivers": ["EXTERNAL_VENDOR", "INTERNAL_HR", "ANALYST", "CUSTOMER_SUPPORT",
                          "CUSTOM"],
                "note": "training document; the evaluation set in data/test is never used here"}
        (out_dir / f"{document_id}.gold.json").write_text(
            json.dumps(gold, ensure_ascii=False, indent=1), encoding="utf-8")
        written.append(document_id)

    return {"documents": len(written), "dir": str(out_dir),
            "rejected_for_overlapping_the_test_set": collisions,
            "note": "every labelled value is distinct from the evaluation set"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--out", default=str(OUT_DIR))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    print(json.dumps(build(args.count, Path(args.out), args.seed), ensure_ascii=False, indent=1))
