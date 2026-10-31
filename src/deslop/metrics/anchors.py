# ABOUTME: Anchor metrics: positive features of grounded prose such as citations,
# ABOUTME: quotations, concrete tokens, reader address, first person, committed claims.
from __future__ import annotations

import re

from deslop.metrics.base import mean, metric
from deslop.nlp import Analysis

# Verse refs (Gen 1:1, Genesis 12:3-5), Stephanus (Rep. 514a, 331c), page/line
# refs (p. 12, pp. 12-14, l. 5, ll. 5-9), (Author 1996), (Author 1996, 12).
CITATION_PATTERNS = [
    re.compile(r"\b[1-3]?\s?[A-Z][a-z]+\.?\s\d{1,3}:\d{1,3}(?:[-–]\d{1,3})?\b"),
    re.compile(r"\b\d{1,3}:\d{1,3}(?:[-–]\d{1,3})?\b"),
    re.compile(r"\b\d{2,3}[a-e]\b(?=[\s,.;)])"),
    re.compile(r"\b(?:pp?|ll?)\.\s?\d+(?:[-–]\d+)?\b"),
    re.compile(
        r"\([A-Z][A-Za-z'\-]+(?:\s(?:and|&)\s[A-Z][A-Za-z'\-]+)?,?\s\d{4}[a-z]?(?:[,:]\s?\d+(?:[-–]\d+)?)?\)"
    ),
    re.compile(r"\b(?:Act|act)\s[IVX1-5]+,?\s(?:[Ss]c(?:ene|\.)\s?[ivx0-9]+)"),
    re.compile(r"\b[IVX]+\.\s?[ivx]+\.\s?\d+\b"),
]

QUOTE_RE = re.compile(r"[\"“]([^\"”]{1,600})[\"”]")
NUMBER_RE = re.compile(r"\b\d[\d,\.]*\b")
NON_LATIN_RE = re.compile(r"[Ͱ-Ͽ֐-׿؀-ۿ一-鿿]+")
ITALIC_RE = re.compile(r"(?<!\w)[*_]([^*_\n]{2,60})[*_](?!\w)")

READER_PRONOUN_RE = re.compile(r"\b(?:you|your|yours|yourself)\b", re.IGNORECASE)
FIRST_PERSON_RE = re.compile(r"\b(?:I|me|my|mine|myself|I'm|I've|I'd|I'll)\b")

COMMITTED_CLAIM_RE = re.compile(
    r"\b(?:is|are|was|were|seems|seem)\s+(?:plainly\s+|simply\s+|surely\s+|clearly\s+)?"
    r"(?:wrong|right|mistaken|correct|false|true|absurd|untenable|unconvincing|convincing|"
    r"the\s+better\s+reading|the\s+right\s+reading|the\s+wrong\s+reading)\b"
    r"|\bI\s+(?:take|read|think|believe|doubt|suspect|hold|maintain|contend|argue|prefer)\b"
    r"|\b(?:the|a)\s+(?:better|worse|stronger|weaker|more\s+likely|less\s+likely|"
    r"more\s+plausible|less\s+plausible)\s+(?:reading|view|account|explanation|interpretation)\b"
    r"|\b(?:must|cannot)\s+be\s+(?:right|wrong|rejected|accepted|the\s+case)\b",
    re.IGNORECASE,
)


def _per_100w(a: Analysis, count: int) -> float:
    return 100.0 * count / max(a.word_count, 1)


def _per_1k(a: Analysis, count: int) -> float:
    return count / a.words_per_1k


def count_citations(text: str) -> int:
    seen: set[tuple[int, int]] = set()
    for pattern in CITATION_PATTERNS:
        for m in pattern.finditer(text):
            if not any(s <= m.start() < e or s < m.end() <= e for s, e in seen):
                seen.add((m.start(), m.end()))
    return len(seen)


@metric(
    "citation_per_100w",
    group="anchors",
    slop_side="low",
    description="Citations (verse, Stephanus, page/line, author-year) per 100 words",
)
def citation_per_100w(a: Analysis) -> float:
    return _per_100w(a, count_citations(a.text))


def count_quotes(text: str) -> int:
    return sum(1 for m in QUOTE_RE.finditer(text) if len(m.group(1).split()) >= 4)


@metric(
    "quote_per_para",
    group="anchors",
    slop_side="low",
    description="Quoted spans of four or more words per paragraph",
)
def quote_per_para(a: Analysis) -> float:
    return mean([float(count_quotes(p.text)) for p in a.paragraphs])


def count_concrete(a: Analysis) -> int:
    count = len(NUMBER_RE.findall(a.text))
    count += len(NON_LATIN_RE.findall(a.text))
    count += len(ITALIC_RE.findall(a.text))
    for p in a.paragraphs:
        count += sum(1 for t in p.doc if t.pos_ == "PROPN")
    return count


@metric(
    "concrete_per_100w",
    group="anchors",
    slop_side="low",
    description="Numbers, proper nouns, non-Latin script, italic terms per 100 words",
)
def concrete_per_100w(a: Analysis) -> float:
    return _per_100w(a, count_concrete(a))


def count_reader_address(a: Analysis) -> int:
    count = len(READER_PRONOUN_RE.findall(a.text))
    for s in a.sentences:
        root = s.span.root
        first = next((t for t in s.span if t.is_alpha), None)
        if (
            first is not None
            and first.i == root.i
            and root.tag_ == "VB"
            and not any(c.dep_ == "nsubj" for c in root.children)
        ):
            count += 1
    return count


@metric(
    "reader_address_per_1k",
    group="anchors",
    slop_side="none",
    description="Second-person pronouns and reader imperatives per 1k words",
)
def reader_address_per_1k(a: Analysis) -> float:
    return _per_1k(a, count_reader_address(a))


@metric(
    "first_person_per_1k",
    group="anchors",
    slop_side="low",
    description="First-person singular pronouns per 1k words",
)
def first_person_per_1k(a: Analysis) -> float:
    return _per_1k(a, len(FIRST_PERSON_RE.findall(a.text)))


@metric(
    "committed_claim_per_para",
    group="anchors",
    slop_side="low",
    description="Evaluative predicates asserting a reading right or wrong, per paragraph",
)
def committed_claim_per_para(a: Analysis) -> float:
    return mean([float(len(COMMITTED_CLAIM_RE.findall(p.text))) for p in a.paragraphs])
