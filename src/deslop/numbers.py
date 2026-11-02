# ABOUTME: Number words as digits ("three hundred" -> "300", "fifty" -> "50", "tenth" -> "10"),
# ABOUTME: so the lexical lock and the primary-fact check treat the two spellings as equal.
from __future__ import annotations

import re

UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19,
}  # fmt: skip
TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90,
}  # fmt: skip
SCALES = {"hundred": 100, "thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}
ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7,
    "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13,
    "fourteenth": 14, "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18,
    "nineteenth": 19, "twentieth": 20, "thirtieth": 30, "fortieth": 40, "fiftieth": 50,
    "sixtieth": 60, "seventieth": 70, "eightieth": 80, "ninetieth": 90, "hundredth": 100,
    "thousandth": 1_000, "millionth": 1_000_000,
}  # fmt: skip
WORDS = {**UNITS, **TENS, **SCALES, **ORDINALS, "and": -1}
PHRASE_RE = re.compile(
    r"\b(?:"
    + "|".join(sorted(WORDS, key=len, reverse=True))
    + r")(?:[\s-]+(?:"
    + "|".join(sorted(WORDS, key=len, reverse=True))
    + r")\b)*",
    re.IGNORECASE,
)


def phrase_value(phrase: str) -> int | None:
    """The value of a run of number words, or None when it is not one ("and" alone)."""
    total = 0
    current = 0
    seen = False
    for word in re.split(r"[\s-]+", phrase.strip().lower()):
        if word == "and":
            continue
        value = WORDS.get(word)
        if value is None:
            return None
        seen = True
        if word in SCALES:
            current = max(current, 1) * value
            if value >= 1_000:
                total += current
                current = 0
        else:
            current += value
    return total + current if seen else None


def number_phrases(text: str) -> list[tuple[int, int, str]]:
    """Every run of number words in `text` as (start, end, digits)."""
    out = []
    for m in PHRASE_RE.finditer(text):
        phrase = m.group(0)
        # trim a leading or trailing "and", which is not a number by itself
        core = re.sub(r"^(?:and[\s-]+)+|(?:[\s-]+and)+$", "", phrase, flags=re.IGNORECASE)
        if not core:
            continue
        value = phrase_value(core)
        if value is None:
            continue
        start = m.start() + phrase.lower().find(core.lower())
        end = start + len(core)
        first = re.split(r"[\s-]+", core.lower())[0]
        if (
            first in SCALES
            and text[max(start - 2, 0) : start].lower() == "a "
            and (start < 2 or not text[start - 3].isalpha())
        ):
            start -= 2  # "a hundred and five" is one number
        out.append((start, end, str(value)))
    return out


def words_to_digits(text: str) -> str:
    """`text` with every run of number words replaced by its digits."""
    pieces = []
    pos = 0
    for start, end, digits in number_phrases(text):
        pieces.append(text[pos:start])
        pieces.append(digits)
        pos = end
    pieces.append(text[pos:])
    return "".join(pieces)
