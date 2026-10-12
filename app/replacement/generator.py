"""Controlled fake values.

Two rules from the build instructions (section 5):

* **Never generate real-looking information casually.** Every substitute here is
  obviously synthetic: reserved test values, `example.test` addresses, and repeats.
  `0500000000` cannot belong to anybody. Nothing is drawn from a real number range.
* **Preserve the format where possible.** A phone written `+966 55 123 4567` becomes
  `+966 50 000 0000`, not `500000000` - the receiver still sees a phone number in the
  same shape, and any downstream format check still passes.

Determinism matters too: the same name in two places in one document must become the
same fake name, or the receiver cannot tell it is the same person. The generator keeps
a per-document map, and a value it has already replaced always gets the same answer.
"""
from __future__ import annotations

import hashlib

from ..pii_detection.normalize import digits_only, language_of, looks_arabic
from ..pii_detection.schema import PiiType

# --------------------------------------------------------------------- vocabularies
# Split by gender, because "Saad Al-Anzi -> Layla Mansour" looks like a mistake to a
# reader even though both names are fake. When the original's given name is one we
# recognise, the substitute stays in the same gender; otherwise the seed decides.
FAKE_NAMES_MALE_EN = ["Omar Hassan", "Khalid Nasser", "Yousef Adel", "Tariq Saleh",
                      "Bilal Farouk", "Sami Rashid", "Nasser Kamal", "Rashid Mansour"]
FAKE_NAMES_FEMALE_EN = ["Nadia Kamal", "Layla Mansour", "Huda Nasser", "Reem Adel",
                        "Mona Saleh", "Abeer Rashid"]
FAKE_NAMES_MALE_AR = ["عمر حسن", "خالد ناصر", "يوسف عادل", "طارق صالح", "سامي راشد"]
FAKE_NAMES_FEMALE_AR = ["نادية كمال", "ليلى منصور", "هدى ناصر", "ريم عادل"]

FEMALE_GIVEN = {"noura", "nora", "reem", "reema", "sara", "sarah", "layla", "laila",
                "mona", "huda", "hind", "lama", "abeer", "jawaher", "amal", "ruba",
                "sana", "hajar", "yasmin", "yasmeen", "nadia", "maria", "maryam",
                "نوره", "نورة", "ريم", "ساره", "سارة", "ليلى", "منى", "هند", "لمى",
                "عبير", "جواهر", "أمل", "امال", "رنا", "سنا", "هاجر", "ياسمين", "ندى"}


def gender_of(name: str) -> str | None:
    """'female', 'male', or None when the given name is not one we know."""
    tokens = [token.strip(".,'-").lower() for token in (name or "").split()]
    for token in tokens[:2]:                     # the given name comes first
        if token in FEMALE_GIVEN:
            return "female"
    return "male" if tokens else None

FAKE_EMAILS = ["customer001@example.test", "customer002@example.test",
               "customer003@example.test"]
FAKE_PLACES_EN = ["Riyadh, Example District", "Jeddah, Example District",
                  "Dammam, Example District"]
FAKE_PLACES_AR = ["الرياض، حي المثال", "جدة، حي المثال", "الدمام، حي المثال"]

NATIONAL_ID_FAKE = "1000000000"
IBAN_FAKE = "SA00 0000 0000 0000 0000 0000"
BANK_ACCOUNT_FAKE = "0000000000000"
PHONE_FAKE = "0500000000"
DOB_FAKE = "01/01/1990"
PASSPORT_FAKE = "A0000000"
CUSTOMER_ID_FAKE = "CUST-0001"
EMPLOYEE_ID_FAKE = "EMP-00001"
COMPANY_ID_FAKE = "0000000000"


def _seed(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16)


def _shape_of_phone(original: str) -> str:
    """Rebuild the separator style of the original phone number."""
    digits = digits_only(original)
    if original.strip().startswith("+"):
        return f"+966 50 000 0000"
    if digits.startswith("00966"):
        return "00966 50 000 0000"
    if digits.startswith("966"):
        return "966 50 000 0000"
    if " " in original.strip():
        # keep the same group sizes: "055 123 4567" -> "050 000 0000"
        groups, index = [], 0
        compact = digits.lstrip("0") or "0"
        filler = (PHONE_FAKE[1:] * 3)
        for chunk in original.split():
            groups.append(filler[index:index + len(chunk)].ljust(len(chunk), "0"))
            index += len(chunk)
        return " ".join(groups)
    if "-" in original:
        return "050-000-0000"
    return PHONE_FAKE


def _shape_of_iban(original: str) -> str:
    return IBAN_FAKE if " " in original else IBAN_FAKE.replace(" ", "")


class FakeGenerator:
    """Produces the replacement for one document. Keep one instance per document."""

    def __init__(self, receiver: str = ""):
        self.receiver = receiver
        self._map: dict[str, str] = {}     # original value -> replacement
        self.counters: dict[str, int] = {}
        self.name_counters: dict[tuple, int] = {}

    # ------------------------------------------------------------------ internals
    def _remember(self, original: str, replacement: str) -> str:
        self._map[original] = replacement
        return replacement

    def _next_index(self, kind: str) -> int:
        self.counters[kind] = self.counters.get(kind, 0) + 1
        return self.counters[kind]

    # --------------------------------------------------------------------- public
    def for_detection(self, original: str, pii_type: str) -> str:
        if original in self._map:
            return self._map[original]           # same value -> same substitute
        return self._remember(original, self._make(original, pii_type))

    def _make(self, original: str, pii_type: str) -> str:
        seed = _seed(original)

        if pii_type == PiiType.PERSON_NAME.value:
            arabic = looks_arabic(original) or language_of(original) == "ar"
            female = gender_of(original) == "female"
            if arabic:
                pool = FAKE_NAMES_FEMALE_AR if female else FAKE_NAMES_MALE_AR
            else:
                pool = FAKE_NAMES_FEMALE_EN if female else FAKE_NAMES_MALE_EN
            # Hand out names in order rather than by hash: hashing gave two different
            # people in one document the same substitute, which makes the release
            # unreadable ("why is Rashid Mansour mentioned twice?").
            key = ("ar" if arabic else "en", female)
            index = self.name_counters.get(key, 0)
            self.name_counters[key] = index + 1
            return pool[index % len(pool)]

        if pii_type == PiiType.PHONE.value:
            return _shape_of_phone(original)

        if pii_type == PiiType.EMAIL.value:
            return FAKE_EMAILS[self._next_index("email") % len(FAKE_EMAILS)]

        if pii_type == PiiType.NATIONAL_ID.value:
            return NATIONAL_ID_FAKE          # 10 digits, starting 1, as required

        if pii_type == PiiType.IBAN.value:
            return _shape_of_iban(original)

        if pii_type == PiiType.BANK_ACCOUNT.value:
            digits = digits_only(original) or "1"
            return BANK_ACCOUNT_FAKE[:max(len(digits), 8)]

        if pii_type == PiiType.ADDRESS.value:
            if looks_arabic(original):
                return FAKE_PLACES_AR[seed % len(FAKE_PLACES_AR)]
            return FAKE_PLACES_EN[seed % len(FAKE_PLACES_EN)]

        if pii_type == PiiType.DATE_OF_BIRTH.value:
            return DOB_FAKE

        if pii_type == PiiType.PASSPORT_NUMBER.value:
            prefix = "".join(ch for ch in original if ch.isalpha())[:1] or "A"
            return prefix + "0000000"

        if pii_type == PiiType.CUSTOMER_ID.value:
            return f"CUST-{self._next_index('customer'):04d}"

        if pii_type == PiiType.EMPLOYEE_ID.value:
            return f"EMP-{self._next_index('employee'):05d}"

        if pii_type == PiiType.COMPANY_IDENTIFIER.value:
            return COMPANY_ID_FAKE

        return "[REDACTED]"                  # an unknown type never leaks the original

    # -------------------------------------------------------------------- helpers
    def replacements(self) -> dict[str, str]:
        return dict(self._map)

    def is_consistent(self) -> bool:
        """Every original maps to exactly one replacement (used by the tests)."""
        return len(set(self._map.values())) == len(self._map)
