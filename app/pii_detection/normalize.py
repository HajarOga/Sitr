"""Text folding for Arabic and English.

Two jobs:

1. **Digits.** Saudi documents mix Western digits (0551234567) with Arabic-Indic
   ones (٠٥٥١٢٣٤٥٦٧) and sometimes the Extended Arabic-Indic set (۰۵۵۱...). A phone
   number written in Arabic-Indic digits is still a phone number.

2. **Letter variants.** The same name is written أ / إ / آ / ا, and ة / ه, and ى / ي.
   Without folding, one name looks like several.

Both folds happen on a copy of the text, and `fold()` also returns the index map, so
every match found in the folded copy can be reported at its ORIGINAL position. That
matters because the release is written back into the original document - the offsets
must line up with the text the user uploaded.

Length note: Arabic-Indic digits and letter variants are one character each, so
folding them changes nothing about positions; the diacritics (tashkeel) are removed,
which is why the explicit index map is needed rather than assuming 1:1.
"""
from __future__ import annotations

_ARABIC_INDIC = {chr(0x0660 + i): str(i) for i in range(10)}      # ٠-٩
_EXT_ARABIC_INDIC = {chr(0x06F0 + i): str(i) for i in range(10)}  # ۰-۹
_DIGITS = {**_ARABIC_INDIC, **_EXT_ARABIC_INDIC}

_LETTERS = {
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
    "ة": "ه", "ى": "ي", "ئ": "ي", "ؤ": "و",
    "ﻻ": "لا",
}

# tashkeel and tatweel are removed before matching
_DIACRITICS = set(
    "\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652\u0653\u0654\u0655\u0670\u0640"
)


def to_latin_digits(text: str) -> str:
    """٠٥٥١٢٣٤٥٦٧ -> 0551234567 (same length)."""
    return "".join(_DIGITS.get(ch, ch) for ch in text)


def fold(text: str) -> tuple[str, list[int]]:
    """Return (folded_text, index_map).

    index_map[i] is the position in `text` of folded_text[i], so a match on the
    folded text can be mapped back to the original document.
    """
    out: list[str] = []
    index_map: list[int] = []
    for i, ch in enumerate(text):
        if ch in _DIACRITICS:
            continue
        ch = _DIGITS.get(ch, ch)
        ch = _LETTERS.get(ch, ch)
        out.append(ch)
        index_map.append(i)
    return "".join(out), index_map


def original_span(folded_start: int, folded_end: int, index_map: list[int]) -> tuple[int, int]:
    """Map a folded span back to original coordinates."""
    if not index_map:
        return 0, 0
    start = index_map[max(0, min(folded_start, len(index_map) - 1))]
    end = index_map[min(folded_end, len(index_map)) - 1] + 1 if folded_end > 0 else start
    return start, end


def digits_only(text: str) -> str:
    """Digits of a value, Arabic-Indic included - used for validators."""
    return "".join(ch for ch in to_latin_digits(text) if ch.isdigit())


def looks_arabic(text: str) -> bool:
    return any("\u0600" <= ch <= "\u06FF" for ch in text)


def language_of(text: str) -> str:
    """'ar', 'en', or 'mixed' - used to pick the right fake names."""
    arabic = sum(1 for ch in text if "\u0600" <= ch <= "\u06FF")
    latin = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    if arabic and latin:
        return "mixed" if min(arabic, latin) > 5 else ("ar" if arabic > latin else "en")
    return "ar" if arabic else "en"
