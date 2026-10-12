"""Build the evaluation test set.

These documents are generated, and their gold labels are known **by construction**:
the generator writes a document, and while writing it, it records every sensitive
value it inserted together with its position. So the labels cannot drift from the
document - which is what makes the metrics meaningful.

Rules that keep this honest:

* The test set lives in `data/test/` and is never used for training. `prepare_data.py`
  reads `data/train/` only, and the evaluator never writes to `data/test/`.
* The values are fictional (reserved example domains, no real number ranges).
* Receivers are varied across documents, so the same evaluator covers HR, vendor,
  analyst and support.

    python -m scripts.make_test_set          # writes data/test/*.txt + *.gold.json
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / "data" / "test"

AR_NAMES = ["عبدالله محمد الحربي", "نورة سعد القحطاني", "خالد إبراهيم العتيبي",
            "ريم فهد الدوسري", "سلطان عبدالعزيز الشمري"]
EN_NAMES = ["Abdullah Mohammed Alharbi", "Noura Saad Al-Qahtani", "Khalid Ibrahim Al-Otaibi",
            "Reem Fahad Al-Dosari", "Sultan Abdulaziz Al-Shammari"]
PHONES = ["0551234567", "+966 55 987 6543", "٠٥٥٤٤٥٥٦٦٧", "0501112233"]
EMAILS = ["a.alharbi@example.com", "n.qahtani@example.com", "k.otaibi@example.com"]
# Luhn-valid, as a real Saudi ID is: the checksum is checked by the detector
IDS = ["1045837299", "1099887760", "1033445568", "1088776651"]
ADDRESSES = ["Al Narjis district, Street 12, Villa 34", "حي النرجس، شارع ١٢، فيلا ٣٤",
             "Al Malqa district, Street 5, Building 9", "حي الملقا، شارع ٥، مبنى ٩"]
# mod-97 valid, as a real IBAN is
IBS = ["SA70 8000 0000 6080 1016 9999", "SA03 8000 0000 6080 1016 7519",
       "SA35 8000 0000 6080 1016 1123", "SA65 8000 0000 6080 1016 4560"]
CUSTOMER_IDS = ["CUST-4471", "CUST-8890", "CUST-1123"]
EMPLOYEE_IDS = ["EMP-33217", "EMP-90011", "EMP-10542"]
DOBS = ["12/03/1990", "01/07/1985", "22/11/1979"]
PASSPORTS = ["A12345678", "B99887766"]

BUSINESS = [
    "Total: 1,250.00 SAR", "Invoice no.: INV-2026-8891",
    "Contract term: 24 months", "Monthly volume: 14,320 units",
    "Region: Central", "VAT number: 310123456700003",
    "CR number: 1010123456", "Care plan: premium", "Subtotal: 980.00 SAR",
]

DOCUMENTS = [
    {"id": "complaint-en", "receiver": "EXTERNAL_VENDOR", "language": "en",
     "title": "Customer Complaint Report",
     "body": ["Name: {EN_NAME}", "National ID: {ID}", "Mobile: {PHONE}",
              "Email: {EMAIL}", "Address: {ADDRESS}", "Customer no.: {CUSTOMER_ID}",
              "The washing machine stopped during the spin cycle and the customer "
              "reported a burning smell.",
              "Our technician visited on 05/10/2026 and confirmed the drum bearing failed.",
              "Parts required: drum bearing kit, door seal."]},
    {"id": "complaint-ar", "receiver": "EXTERNAL_VENDOR", "language": "ar",
     "title": "تقرير شكوى عميل",
     "body": ["الاسم: {AR_NAME}", "رقم الهوية: {ID}", "الجوال: {AR_PHONE}",
              "البريد الإلكتروني: {EMAIL}", "العنوان: {AR_ADDRESS}",
              "رقم العميل: {CUSTOMER_ID}",
              "توقفت الغسالة أثناء دورة العصر وذكر العميل ظهور رائحة احتراق.",
              "تمت الزيارة بتاريخ ٠٥/١٠/٢٠٢٦ وتأكد تلف رولمان البلي."]},
    {"id": "hr-letter-en", "receiver": "INTERNAL_HR", "language": "en",
     "title": "Employment Confirmation Letter",
     "body": ["Employee name: {EN_NAME}", "Employee ID: {EMPLOYEE_ID}",
              "National ID: {ID}", "Mobile: {PHONE}", "Email: {EMAIL}",
              "Address: {ADDRESS}", "Date of birth: {DOB}",
              "This letter confirms that the employee has completed the probation period "
              "and is entitled to the standard annual leave of 21 working days."]},
    {"id": "hr-record-ar", "receiver": "INTERNAL_HR", "language": "ar",
     "title": "ملف موظف",
     "body": ["اسم الموظف: {AR_NAME}", "الرقم الوظيفي: {EMPLOYEE_ID}",
              "رقم الهوية: {ID}", "الجوال: {AR_PHONE}", "تاريخ الميلاد: {DOB}",
              "العنوان: {AR_ADDRESS}",
              "الراتب الأساسي: 12,000.00 ريال",
              "مدة الخدمة: سنتان"]},
    {"id": "kyc-form-ar", "receiver": "EXTERNAL_VENDOR", "language": "mixed",
     "title": "نموذج فتح حساب - KYC form",
     "body": ["الاسم: {AR_NAME}", "Name: {EN_NAME}", "رقم الهوية: {ID}",
              "رقم الجواز: {PASSPORT}", "الجوال: {AR_PHONE}", "Mobile: {PHONE}",
              "العنوان: {AR_ADDRESS}", "IBAN: {IBAN}",
              "النشاط: تجارة تجزئة"]},
    {"id": "analysis-export", "receiver": "ANALYST", "language": "en",
     "title": "Regional Service Analysis",
     "body": ["Customer no.: {CUSTOMER_ID}", "Employee ID: {EMPLOYEE_ID}",
              "Name: {EN_NAME}", "National ID: {ID}", "Address: {ADDRESS}",
              "Date of birth: {DOB}", "Region: Central",
              "Average handling time: 3.4 days", "Repeat visits: 12%"]},
    {"id": "support-ticket-ar", "receiver": "CUSTOMER_SUPPORT", "language": "ar",
     "title": "تذكرة دعم",
     "body": ["الاسم: {AR_NAME}", "رقم العميل: {CUSTOMER_ID}", "الجوال: {AR_PHONE}",
              "البريد الإلكتروني: {EMAIL}", "رقم الهوية: {ID}",
              "العنوان: {AR_ADDRESS}",
              "تم الاتصال بالعميل مرتين دون إجابة، وسيتم الاتصال مرة أخرى صباح الغد."]},
    {"id": "vendor-invoice", "receiver": "EXTERNAL_VENDOR", "language": "mixed",
     "title": "فاتورة مورد - Vendor Invoice",
     "body": ["الاسم: {AR_NAME}", "Contact person: {EN_NAME}", "Mobile: {PHONE}",
              "Email: {EMAIL}", "IBAN: {IBAN}", "العنوان: {AR_ADDRESS}",
              "VAT number: 310123456700003",
              "Service: annual maintenance for 12 lifts",
              "Total: 48,500.00 SAR"]},
    {"id": "medical-referral", "receiver": "EXTERNAL_VENDOR", "language": "en",
     "title": "Medical Referral Note",
     "body": ["Patient name: {EN_NAME}", "National ID: {ID}", "Mobile: {PHONE}",
              "Date of birth: {DOB}", "Address: {ADDRESS}",
              "Referred for a physiotherapy assessment after a workplace injury.",
              "Light duties recommended for four weeks."]},
    {"id": "supplier-letter-ar", "receiver": "EXTERNAL_VENDOR", "language": "ar",
     "title": "خطاب إلى مورد",
     "body": ["السادة: شركة الأمثلة للتوريدات", "المرجع: SUP-2026-331",
              "اسم الممثل: {AR_NAME}", "الجوال: {AR_PHONE}", "البريد الإلكتروني: {EMAIL}",
              "IBAN: {IBAN}",
              "الموضوع: تجديد عقد التوريد لمدة 24 شهرًا",
              "شروط الدفع: 30 يومًا من تاريخ الفاتورة"]},
    {"id": "analysis-ar", "receiver": "ANALYST", "language": "ar",
     "title": "تحليل إقليمي",
     "body": ["الاسم: {AR_NAME}", "رقم العميل: {CUSTOMER_ID}", "رقم الهوية: {ID}",
              "الجوال: {AR_PHONE}", "المنطقة: الوسطى",
              "متوسط مدة المعالجة: 3.4 يوم", "نسبة الزيارات المتكررة: 12%"]},
    {"id": "contract-en", "receiver": "CUSTOMER_SUPPORT", "language": "en",
     "title": "Service Contract Summary",
     "body": ["Account holder: {EN_NAME}", "National ID: {ID}", "Mobile: {PHONE}",
              "Email: {EMAIL}", "Address: {ADDRESS}", "IBAN: {IBAN}",
              "Contract term: 24 months", "Monthly volume: 14,320 units",
              "Care plan: premium"]},
]

# which field fills which placeholder, per receiver-independent document
FIELDS = {
    "EN_NAME", "AR_NAME", "PHONE", "AR_PHONE", "EMAIL", "ID", "ADDRESS", "AR_ADDRESS",
    "CUSTOMER_ID", "EMPLOYEE_ID", "DOB", "PASSPORT", "IBAN",
}
BUSINESS_TOKENS = ["Total:", "VAT number", "CR number", "Invoice no", "Contract term",
                   "Monthly volume", "Region", "Subtotal", "Care plan", "الموضوع",
                   "الراتب الأساسي", "مدة الخدمة", "شروط الدفع"]


def _pick(values: list[str], index: int) -> str:
    return values[index % len(values)]


def _build_document(spec: dict, index: int) -> tuple[str, list[dict]]:
    values = {
        "EN_NAME": _pick(EN_NAMES, index), "AR_NAME": _pick(AR_NAMES, index + 1),
        "PHONE": _pick(PHONES, index), "AR_PHONE": _pick(PHONES, index + 2),
        "EMAIL": _pick(EMAILS, index), "ID": _pick(IDS, index),
        "ADDRESS": _pick(ADDRESSES, index), "AR_ADDRESS": _pick(ADDRESSES, index + 1),
        "CUSTOMER_ID": _pick(CUSTOMER_IDS, index), "EMPLOYEE_ID": _pick(EMPLOYEE_IDS, index),
        "DOB": _pick(DOBS, index), "PASSPORT": _pick(PASSPORTS, index),
        "IBAN": _pick(IBS, index),
    }
    lines = [spec["title"], f"Reference: {spec['id'].upper()}-{2026}-{index + 1:03d}"]
    inserted: list[tuple[str, str]] = []          # (value, type) in document order
    for template in spec["body"]:
        placeholders = [f for f in FIELDS if "{" + f + "}" in template]
        line = template
        for field in placeholders:
            line = line.replace("{" + field + "}", values[field])
            inserted.append((values[field], _type_of(field)))
        lines.append(line)

    # one shared business line per document, from a fixed list, so it is stable
    lines.append(BUSINESS[index % len(BUSINESS)])
    text = "\n".join(lines)

    spans = []
    seen: dict[str, int] = {}
    for value, pii_type in inserted:
        occurrence = seen.get(value, 0)
        position = _find_nth(text, value, occurrence)
        seen[value] = occurrence + 1
        if position is None:
            continue
        spans.append({"text": value, "type": pii_type, "start": position,
                      "end": position + len(value)})
    return text, spans


def _type_of(field: str) -> str:
    return {
        "EN_NAME": "PERSON_NAME", "AR_NAME": "PERSON_NAME",
        "PHONE": "PHONE", "AR_PHONE": "PHONE", "EMAIL": "EMAIL",
        "ID": "NATIONAL_ID", "ADDRESS": "ADDRESS", "AR_ADDRESS": "ADDRESS",
        "CUSTOMER_ID": "CUSTOMER_ID", "EMPLOYEE_ID": "EMPLOYEE_ID",
        "DOB": "DATE_OF_BIRTH", "PASSPORT": "PASSPORT_NUMBER", "IBAN": "IBAN",
    }[field]


def _find_nth(text: str, value: str, occurrence: int) -> int | None:
    start = 0
    for _ in range(occurrence + 1):
        position = text.find(value, start)
        if position == -1:
            return None
        start = position + 1
    return position


def must_keep(text: str, spans: list[dict]) -> list[str]:
    """Business information that has to survive the release."""
    keep = []
    for token in BUSINESS_TOKENS:
        for line in text.splitlines():
            if token in line:
                keep.append(line.strip())
                break
    return keep


def build(test_dir: Path = TEST_DIR) -> dict:
    test_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, spec in enumerate(DOCUMENTS):
        text, spans = _build_document(spec, index)
        doc_path = test_dir / f"{spec['id']}.txt"
        doc_path.write_text(text, encoding="utf-8")

        gold = {
            "document_id": spec["id"],
            "receiver": spec["receiver"],
            "language": spec["language"],
            "source": "generated by scripts/make_test_set.py - synthetic, no real data",
            "spans": spans,
            "must_keep": must_keep(text, spans),
            "note": "labels are exact by construction: the generator recorded every value it wrote",
        }
        (test_dir / f"{spec['id']}.gold.json").write_text(
            json.dumps(gold, ensure_ascii=False, indent=1), encoding="utf-8")
        manifest.append({"id": spec["id"], "receiver": spec["receiver"],
                         "language": spec["language"], "spans": len(spans)})

    (test_dir / "MANIFEST.json").write_text(
        json.dumps({"documents": manifest, "count": len(manifest)}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    return {"documents": len(manifest),
            "spans": sum(item["spans"] for item in manifest), "dir": str(test_dir)}


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=1))
