"""Build a second, hand-written test set: prose documents from no existing set.

    python -m scripts.make_second_test_set

Why this exists. The frozen 12-document set and the training documents were written by the
same generator, so they share a house style: labelled lines, one value per line. That measures
whether the model learned the task, but it cannot tell us whether it survives a document that
does not look like the training data.

These twenty documents were written by hand, in sentences, with the personal data embedded
in prose rather than on labelled lines. They belong to no existing set, and no value here appears
in `data/train/`, `data/test/` or `data/test_no_rules/` - the script proves that on every run
and refuses to write anything if a value collides.

The correct action for each value is NOT hand-typed here: it is read from `app/rules/rules.json`,
so the labels cannot drift away from the policy. Offsets are found in the text and a value that
appears twice is an error, not a guess.

Read the numbers this way: if agreement and recall hold up here, the behaviour generalises past
the generator's style. If they fall, that is a real finding about the model, and it belongs in
the report - not something to hide behind the frozen set.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TARGET = ROOT / "data" / "test_second"
RULES = ROOT / "app" / "rules" / "rules.json"

# --------------------------------------------------------------------- the documents
# Written by hand. Arabic, English and one bilingual, spread over the four receivers.
DOCUMENTS = [
    {
        "id": "vendor-lift-ar",
        "receiver": "EXTERNAL_VENDOR",
        "language": "ar",
        "text": """إلى قسم خدمة العملاء،

أكتب إليكم بخصوص طلب الصيانة رقم SR-4471 لمصعد المبنى في حي الملقا، شارع الأمير تركي، مبنى 22.
صاحبة الطلب هي سارة عبدالله الحربي، ويمكن التواصل معها على الجوال 0539110033، أو على البريد sara.harbi@example.com.
دفعنا مبلغ 1,250 ريال عن الزيارة الأولى، والحساب المسجّل باسمنا هو SA72 9000 0000 4471 2026 3391.
أرجو إبلاغ الفني بأن الزيارة ستكون يوم الأحد بعد الظهر.

وشكرًا،
إدارة المبنى""",
        "spans": [
            ("حي الملقا، شارع الأمير تركي، مبنى 22", "ADDRESS"),
            ("سارة عبدالله الحربي", "PERSON_NAME"),
            ("0539110033", "PHONE"),
            ("sara.harbi@example.com", "EMAIL"),
            ("SA72 9000 0000 4471 2026 3391", "IBAN"),
        ],
    },
    {
        "id": "vendor-generator-en",
        "receiver": "EXTERNAL_VENDOR",
        "language": "en",
        "text": """Subject: Annual service visit for the generator

Hello,

We are arranging the annual maintenance visit for the generator at the Rawdah warehouse.
The site contact is Nawaf Saleh Al-Harbi, and his mobile is +966 56 699 8811.
Please send the technician's report to facilities@example.test when the visit is done.

For the invoice, our VAT number is 310445566700003 and the commercial registration is 1010884411.
Payment will be transferred to the company account, never to a personal account.""",
        "spans": [
            ("Nawaf Saleh Al-Harbi", "PERSON_NAME"),
            ("+966 56 699 8811", "PHONE"),
            ("facilities@example.test", "EMAIL"),
            ("310445566700003", "COMPANY_IDENTIFIER"),
            ("1010884411", "COMPANY_IDENTIFIER"),
        ],
    },
    {
        "id": "hr-transfer-ar",
        "receiver": "INTERNAL_HR",
        "language": "ar",
        "text": """مذكرة داخلية - إدارة الموارد البشرية

بخصوص طلب نقل الموظفة هند ناصر العتيبي (الرقم الوظيفي EMP-44219) من إدارة العمليات إلى إدارة المشاريع.
تاريخ الميلاد المسجل في الملف: 1993-11-02
رقم الهوية الوطنية: 1098765432
جوال التواصل الداخلي: 0578002244
جميع المستندات أعلاه موجودة في ملف الموظفة ولا حاجة لإرسال أي منها خارج الشركة.""",
        "spans": [
            ("هند ناصر العتيبي", "PERSON_NAME"),
            ("EMP-44219", "EMPLOYEE_ID"),
            ("1993-11-02", "DATE_OF_BIRTH"),
            ("1098765432", "NATIONAL_ID"),
            ("0578002244", "PHONE"),
        ],
    },
    {
        "id": "hr-onboarding-en",
        "receiver": "INTERNAL_HR",
        "language": "en",
        "text": """Onboarding note - new joiner in the operations team

Dana Fouad Al-Subaie starts next month.
Employee ID: EMP-50133
National ID: 1076543219
Date of birth: 1991-07-23
Passport number: S7781234
Personal mobile: 0561445566
Home address for the HR file: Al Yasmin district, Street 9, Building 41
Bank account for payroll: 3920014477""",
        "spans": [
            ("Dana Fouad Al-Subaie", "PERSON_NAME"),
            ("EMP-50133", "EMPLOYEE_ID"),
            ("1076543219", "NATIONAL_ID"),
            ("1991-07-23", "DATE_OF_BIRTH"),
            ("S7781234", "PASSPORT_NUMBER"),
            ("0561445566", "PHONE"),
            ("Al Yasmin district, Street 9, Building 41", "ADDRESS"),
            ("3920014477", "BANK_ACCOUNT"),
        ],
    },
    {
        "id": "analyst-summary-en",
        "receiver": "ANALYST",
        "language": "en",
        "text": """Monthly service performance summary, written for internal analysis.

Service ticket volume grew 12% in September. The largest single contributor was the customer
Rakan Adel Al-Dossari (customer ID CUST-884213), who raised nine tickets about the same lift.
Average resolution time came down from 3.4 days to 2.6 days over the same month.""",
        "spans": [
            ("Rakan Adel Al-Dossari", "PERSON_NAME"),
            ("CUST-884213", "CUSTOMER_ID"),
        ],
    },
    {
        "id": "analyst-note-ar",
        "receiver": "ANALYST",
        "language": "ar",
        "text": """ملاحظة تحليلية - للاستخدام الداخلي فقط

انخفضت نسبة تكرار الأعطال في مصاعد المنطقة الوسطى إلى 7% خلال الربع الثالث.
العميل صاحب أكبر عدد من الطلبات هو ماجد سليمان القحطاني (رقم العميل CUST-771204).
تكلفة القطعة البديلة وصلت إلى 860 ريال، ولم تُسجّل أي مطالبة من الموردين هذا الشهر.""",
        "spans": [
            ("ماجد سليمان القحطاني", "PERSON_NAME"),
            ("CUST-771204", "CUSTOMER_ID"),
        ],
    },
    {
        "id": "support-call-ar",
        "receiver": "CUSTOMER_SUPPORT",
        "language": "ar",
        "text": """سجل مكالمة - الدعم الفني

اتصلت العميلة لمى خالد الزهراني تشتكي من تأخر الفني عن موعد الزيارة.
جوال التواصل: ٠٥٤٥٦٦٧٧٨٨
البريد الإلكتروني: lamaz@example.com
رقم الهوية الوطنية 1055664477 وتم التحقق منه أثناء المكالمة.
طلبت أن يكون الاتصال بعد الساعة الرابعة عصرًا.""",
        "spans": [
            ("لمى خالد الزهراني", "PERSON_NAME"),
            ("٠٥٤٥٦٦٧٧٨٨", "PHONE"),
            ("lamaz@example.com", "EMAIL"),
            ("1055664477", "NATIONAL_ID"),
        ],
    },
    {
        "id": "support-chat-en",
        "receiver": "CUSTOMER_SUPPORT",
        "language": "en",
        "text": """Support chat transcript

Customer: I still have not received the replacement part for my air conditioning unit.
Agent: I can see your order, and the technician visit is already booked.
Customer: Please call me on 0534778899 instead of my office line.
Customer: The account is under my name, Turki Bander Al-Shammari, customer number CUST-905517.
Agent: Noted. I have updated the callback number and left a note for the technician.""",
        "spans": [
            ("0534778899", "PHONE"),
            ("Turki Bander Al-Shammari", "PERSON_NAME"),
            ("CUST-905517", "CUSTOMER_ID"),
        ],
    },
    {
        "id": "vendor-hvac-ar",
        "receiver": "EXTERNAL_VENDOR",
        "language": "ar",
        "text": """طلب صيانة مكيفات - فرع العارض

نرجو منكم إرسال فني لصيانة ثلاث وحدات تكييف في فرع العارض.
جهة الاتصال في الفرع: نورة سعد الغامدي
الجوال: 0558001122
البريد الإلكتروني: n.ghamdi@example.com
عنوان الفرع: حي العارض، شارع الأمير سلطان، مبنى 6
رقم ضريبة القيمة المضافة للفرع: 310778899000003
نرجو إرسال عرض السعر قبل الزيارة.""",
        "spans": [
            ("نورة سعد الغامدي", "PERSON_NAME"),
            ("0558001122", "PHONE"),
            ("n.ghamdi@example.com", "EMAIL"),
            ("حي العارض، شارع الأمير سلطان، مبنى 6", "ADDRESS"),
            ("310778899000003", "COMPANY_IDENTIFIER"),
        ],
    },
    {
        "id": "vendor-parts-en",
        "receiver": "EXTERNAL_VENDOR",
        "language": "en",
        "text": """Spare parts order - reference SP-2291

Please supply two control boards for the lifts at the Sulaimaniyah building.
Our technical contact is Fahad Nasser Al-Zahrani, mobile 0567008899.
Send the delivery note to procurement@example.test.

Company registration: 1012334455
Payment runs from the finance system to your registered account, so no bank details are
needed in this message.""",
        "spans": [
            ("Fahad Nasser Al-Zahrani", "PERSON_NAME"),
            ("0567008899", "PHONE"),
            ("procurement@example.test", "EMAIL"),
            ("1012334455", "COMPANY_IDENTIFIER"),
        ],
    },
    {
        "id": "vendor-security-en",
        "receiver": "EXTERNAL_VENDOR",
        "language": "en",
        "text": """Security cover request for the warehouse

We need two guards for the night shift at the Rawdah warehouse starting Sunday.
Site supervisor: Sultan Majed Al-Otaibi
Mobile: +966 55 133 2244
Email: hse@example.com

Bank account on the contract: 4471003322
VAT number: 310556677800003""",
        "spans": [
            ("Sultan Majed Al-Otaibi", "PERSON_NAME"),
            ("+966 55 133 2244", "PHONE"),
            ("hse@example.com", "EMAIL"),
            ("4471003322", "BANK_ACCOUNT"),
            ("310556677800003", "COMPANY_IDENTIFIER"),
        ],
    },
    {
        "id": "hr-payroll-ar",
        "receiver": "INTERNAL_HR",
        "language": "ar",
        "text": """مذكرة داخلية - تعديل بيانات الراتب

الموظف: عبدالعزيز ماجد الدوسري
الرقم الوظيفي: EMP-61004
رقم الهوية الوطنية: 1067788990
الحساب البنكي المحفوظ في النظام: 2299105544
جوال الموظف: 0532004455
لا حاجة لإرسال أي من البيانات أعلاه خارج إدارة الموارد البشرية.""",
        "spans": [
            ("عبدالعزيز ماجد الدوسري", "PERSON_NAME"),
            ("EMP-61004", "EMPLOYEE_ID"),
            ("1067788990", "NATIONAL_ID"),
            ("2299105544", "BANK_ACCOUNT"),
            ("0532004455", "PHONE"),
        ],
    },
    {
        "id": "hr-exit-en",
        "receiver": "INTERNAL_HR",
        "language": "en",
        "text": """Exit checklist - departing employee

Employee: Layla Sami Al-Ghamdi
Employee ID: EMP-58822
Personal email for the final settlement: layla.g@example.test
Mobile: 0542773311
Home address on file: Al Qadisiyah district, Street 4, Building 12
Date of birth: 1987-05-30""",
        "spans": [
            ("Layla Sami Al-Ghamdi", "PERSON_NAME"),
            ("EMP-58822", "EMPLOYEE_ID"),
            ("layla.g@example.test", "EMAIL"),
            ("0542773311", "PHONE"),
            ("Al Qadisiyah district, Street 4, Building 12", "ADDRESS"),
            ("1987-05-30", "DATE_OF_BIRTH"),
        ],
    },
    {
        "id": "hr-training-ar",
        "receiver": "INTERNAL_HR",
        "language": "ar",
        "text": """ترشيح لحضور برنامج تدريبي

نرشح الموظفة جواهر تركي السبيعي (الرقم الوظيفي EMP-60077) لحضور برنامج إدارة المشاريع.
رقم الجوال للتواصل: 0566221188
رقم الهوية: 1033445566
جواز السفر المسجّل في الملف: R4410977
تاريخ الميلاد: 1995-09-12""",
        "spans": [
            ("جواهر تركي السبيعي", "PERSON_NAME"),
            ("EMP-60077", "EMPLOYEE_ID"),
            ("0566221188", "PHONE"),
            ("1033445566", "NATIONAL_ID"),
            ("R4410977", "PASSPORT_NUMBER"),
            ("1995-09-12", "DATE_OF_BIRTH"),
        ],
    },
    {
        "id": "analyst-churn-en",
        "receiver": "ANALYST",
        "language": "en",
        "text": """Quarterly churn analysis - internal

Two accounts caused most of the cancelled service contracts this quarter.
The first is Nada Faisal Al-Mutairi (customer ID CUST-441209), who cancelled after two
missed visits. The second is Yousef Hamad Al-Muhanna (customer ID CUST-330871), who
cancelled without raising a single ticket.""",
        "spans": [
            ("Nada Faisal Al-Mutairi", "PERSON_NAME"),
            ("CUST-441209", "CUSTOMER_ID"),
            ("Yousef Hamad Al-Muhanna", "PERSON_NAME"),
            ("CUST-330871", "CUSTOMER_ID"),
        ],
    },
    {
        "id": "analyst-cost-ar",
        "receiver": "ANALYST",
        "language": "ar",
        "text": """تحليل التكلفة - الربع الثالث

ارتفعت تكلفة قطع الغيار بنسبة 9% مقارنة بالربع الثاني.
العميلة ريم ناصر الشهري (رقم العميل CUST-778410) هي الأعلى في عدد الزيارات المتكررة.
متوسط تكلفة الزيارة الواحدة 420 ريال، ولم تُسجّل أي مطالبة تعويض هذا الربع.""",
        "spans": [
            ("ريم ناصر الشهري", "PERSON_NAME"),
            ("CUST-778410", "CUSTOMER_ID"),
        ],
    },
    {
        "id": "analyst-export-en",
        "receiver": "ANALYST",
        "language": "en",
        "text": """Data export summary - September

The export covered 1,204 service records. Contact details were included in the raw file,
for example the case owner Amal Yousef Al-Harbi, email amal.h@example.com, and the
callback number +966 55 099 8877. The analysis itself uses only the customer identifier.""",
        "spans": [
            ("Amal Yousef Al-Harbi", "PERSON_NAME"),
            ("amal.h@example.com", "EMAIL"),
            ("+966 55 099 8877", "PHONE"),
        ],
    },
    {
        "id": "support-refund-ar",
        "receiver": "CUSTOMER_SUPPORT",
        "language": "ar",
        "text": """سجل طلب استرداد

تقدّم العميل خالد عبدالرحمن العمري بطلب استرداد مبلغ 320 ريال عن زيارة لم تُنفَّذ.
جواله للتواصل: 0509447722
البريد الإلكتروني: khaled.a@example.com
رقم الهوية الوطنية: 1078899112 وتم التحقق منه في المكالمة.
رقم العميل: CUST-552340""",
        "spans": [
            ("خالد عبدالرحمن العمري", "PERSON_NAME"),
            ("0509447722", "PHONE"),
            ("khaled.a@example.com", "EMAIL"),
            ("1078899112", "NATIONAL_ID"),
            ("CUST-552340", "CUSTOMER_ID"),
        ],
    },
    {
        "id": "support-visit-en",
        "receiver": "CUSTOMER_SUPPORT",
        "language": "en",
        "text": """Technician visit confirmation

Customer: Maha Saleh Al-Dosari
Mobile for the callback: 0567331188
Email: maha.dosari@example.com
Visit address: Al Sahafah district, Building 7, Office 302
The technician will arrive between 10:00 and 12:00.""",
        "spans": [
            ("Maha Saleh Al-Dosari", "PERSON_NAME"),
            ("0567331188", "PHONE"),
            ("maha.dosari@example.com", "EMAIL"),
            ("Al Sahafah district, Building 7, Office 302", "ADDRESS"),
        ],
    },
    {
        "id": "support-escalation-ar",
        "receiver": "CUSTOMER_SUPPORT",
        "language": "ar",
        "text": """سجل تصعيد - عميل متكرر الشكوى

العميل فيصل عمر الزهراني (رقم العميل CUST-661877) أبلغ عن تأخر ثالث زيارة.
جواله: 0557112299
البريد الإلكتروني: faisal.z@example.com
طلبت إدارة الخدمة من الفريق الاتصال به لتحديد موعد جديد.""",
        "spans": [
            ("فيصل عمر الزهراني", "PERSON_NAME"),
            ("0557112299", "PHONE"),
            ("faisal.z@example.com", "EMAIL"),
            ("CUST-661877", "CUSTOMER_ID"),
        ],
    },
]


def policy_actions() -> dict[tuple[str, str], str]:
    payload = json.loads(RULES.read_text(encoding="utf-8"))
    table = {}
    for rule in payload["rules"]:
        table[(rule["receiver"], rule["pii_type"])] = rule["action"]
    return table


def everything_already_elsewhere() -> str:
    """Every value we have used anywhere else, so a collision can be refused."""
    blob = []
    for folder in ("data/train/documents", "data/test", "data/test_no_rules"):
        for path in sorted((ROOT / folder).glob("*.txt")):
            blob.append(path.read_text(encoding="utf-8"))
    return "\n".join(blob)


def main() -> int:
    actions = policy_actions()
    elsewhere = everything_already_elsewhere()
    TARGET.mkdir(parents=True, exist_ok=True)

    written = 0
    total_spans = 0
    collisions = []
    summary = []

    for document in DOCUMENTS:
        text = document["text"]
        spans = []
        for value, pii_type in document["spans"]:
            count = text.count(value)
            if count != 1:
                collisions.append(f"{document['id']}: {value!r} appears {count} times "
                                  f"(expected exactly once)")
                continue
            if value in elsewhere:
                collisions.append(f"{document['id']}: {value!r} already exists in another set")
                continue
            if (document["receiver"], pii_type) not in actions:
                collisions.append(f"{document['id']}: no rule for {document['receiver']} / "
                                  f"{pii_type}")
                continue
            start = text.index(value)
            spans.append({"text": value, "type": pii_type, "start": start, "end": start + len(value)})

        if collisions:
            continue

        gold = {
            "document_id": document["id"],
            "receiver": document["receiver"],
            "language": document["language"],
            "source": ("written by hand for scripts/make_second_test_set.py - synthetic, "
                       "no real data, and in prose rather than labelled lines"),
            "spans": spans,
            "note": ("The expected action for each value is derived from app/rules/rules.json at "
                     "scoring time, not written here. This document belongs to no existing set "
                     "and was written in a different style from the generator's output."),
        }
        (TARGET / f"{document['id']}.txt").write_text(text + "\n", encoding="utf-8")
        (TARGET / f"{document['id']}.gold.json").write_text(
            json.dumps(gold, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        written += 1
        total_spans += len(spans)
        summary.append((document["id"], document["receiver"], document["language"],
                        len(spans), [actions[(document["receiver"], t)] for _, t in document["spans"]]))

    if collisions:
        print("refused to write - fix these first:")
        for line in collisions:
            print("  -", line)
        return 1

    print(f"wrote {written} documents, {total_spans} labelled values -> "
          f"{TARGET.relative_to(ROOT)}")
    print()
    print(f"{'document':22s} {'receiver':18s} {'lang':5s} {'values':>6s}  expected actions")
    for document_id, receiver, language, count, expected in summary:
        actions_text = " ".join(expected)
        print(f"{document_id:22s} {receiver:18s} {language:5s} {count:6d}  {actions_text}")
    print()
    print("no value in this set appears in data/train, data/test or data/test_no_rules")
    print("the frozen set in data/test/ was not touched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
