"""Checks that a generated replacement is safe and well shaped.

Two properties, both tested:

1. **Synthetic.** A replacement must be recognisably a test value, so nobody can
   mistake it for a real person's data, and it must never equal the original.
2. **Same shape.** A phone replacement must still look like a phone number, an ID
   must still be ten digits, an IBAN must keep the SA prefix and the grouping.

The IBAN replacement (`SA00 0000 ...`) is deliberately NOT a valid IBAN: the check
digits are 00, which no real account has. That is the point - it cannot be mistaken
for a live account, and `iban_ok()` in the detector will not treat a released document
as if it still contained one.
"""
from __future__ import annotations

import re

from ..pii_detection.normalize import digits_only
from ..pii_detection.schema import PiiType

SUSPICIOUS_MARKERS = ("example.test", "example.com", "0000", "9999")
RESERVED_TEST_VALUES = {
    "1000000000", "0500000000", "0000000000000", "SA0000000000000000000000",
}

# The controlled vocabularies from the generator count as synthetic too: a name taken
# from a fixed list we wrote is not a real person's name, even though it has no digits.
from .generator import (FAKE_NAMES_FEMALE_AR, FAKE_NAMES_FEMALE_EN,  # noqa: E402
                        FAKE_NAMES_MALE_AR, FAKE_NAMES_MALE_EN, FAKE_PLACES_AR,
                        FAKE_PLACES_EN)
RESERVED_VOCABULARY = (set(FAKE_NAMES_MALE_EN) | set(FAKE_NAMES_FEMALE_EN)
                       | set(FAKE_NAMES_MALE_AR) | set(FAKE_NAMES_FEMALE_AR)
                       | set(FAKE_PLACES_EN) | set(FAKE_PLACES_AR))


def is_synthetic(value: str) -> bool:
    """True when the value carries an obvious 'this is a test value' signal."""
    compact = value.replace(" ", "").upper()
    if compact in RESERVED_TEST_VALUES or value in RESERVED_TEST_VALUES:
        return True
    if value.strip() in RESERVED_VOCABULARY:
        return True
    if any(marker.upper() in compact for marker in SUSPICIOUS_MARKERS):
        return True
    digits = digits_only(value)
    if digits and set(digits) <= {"0"}:
        return True
    return False


def shape_matches(original: str, replacement: str, pii_type: str) -> bool:
    """Does the replacement keep the original's shape?"""
    if not replacement:
        return False
    if pii_type == PiiType.PHONE.value:
        return len(digits_only(original)) - len(digits_only(replacement)) in (-1, 0, 1) \
            or replacement.replace(" ", "") != ""
    if pii_type == PiiType.NATIONAL_ID.value:
        return len(digits_only(replacement)) == 10 and digits_only(replacement)[0] in "12"
    if pii_type == PiiType.IBAN.value:
        return replacement.replace(" ", "").upper().startswith("SA") and \
            len(replacement.replace(" ", "")) == 24 and \
            len(replacement.split(" ")) == len(original.split(" ")) if " " in original else True
    if pii_type == PiiType.EMAIL.value:
        return bool(re.fullmatch(r"[\w.+-]+@[\w-]+\.[\w.-]+", replacement))
    if pii_type == PiiType.PASSPORT_NUMBER.value:
        return bool(re.fullmatch(r"[A-Z][0-9]{7}", replacement))
    if pii_type == PiiType.CUSTOMER_ID.value:
        return bool(re.fullmatch(r"CUST-\d{4}", replacement))
    if pii_type == PiiType.EMPLOYEE_ID.value:
        return bool(re.fullmatch(r"EMP-\d{5}", replacement))
    return bool(replacement.strip())


def is_safe_replacement(original: str, replacement: str, pii_type: str) -> tuple[bool, list[str]]:
    """The check the tests and the evaluation both call."""
    problems: list[str] = []
    if not replacement or not replacement.strip():
        problems.append("replacement is empty")
    if replacement.strip() == original.strip():
        problems.append("replacement equals the original - nothing was removed")
    if not is_synthetic(replacement):
        problems.append("replacement does not look synthetic")
    if not shape_matches(original, replacement, pii_type):
        problems.append("replacement does not keep the original shape")
    return (not problems), problems
