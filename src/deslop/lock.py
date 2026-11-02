# ABOUTME: Content lock: a document-scoped allowlist of citations, quotes, numbers and names.
# ABOUTME: Removals from a unit are violations; additions pass when the document already has them.
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from deslop.metrics.anchors import CITATION_PATTERNS, NUMBER_RE
from deslop.nlp import load_nlp
from deslop.numbers import number_phrases

LOCK_CLASSES = ("citations", "quotes", "numbers", "proper_nouns")
ANY_QUOTE_RE = re.compile(r"[\"“]([^\"”]{1,600})[\"”]")
# Markdown structure is not content: headings, rules, list markers and emphasis marks are
# dropped before anything is inventoried.
HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s.*$", re.M)
RULE_RE = re.compile(r"^\s{0,3}([-*_])(\s*\1){2,}\s*$", re.M)
LIST_MARKER_RE = re.compile(r"^(\s*)(?:\d{1,3}[.)]|[-*+•])\s+", re.M)
EMPHASIS_RE = re.compile(r"(\*\*|__|\*|_)(?=\S)(.+?)(?<=\S)\1")
# Citations to the primary text under discussion may always be added: verse references,
# bare chapter:verse, Stephanus pages, page and line numbers, act and scene.
PRIMARY_CITATION_PATTERNS = (
    CITATION_PATTERNS[0],
    CITATION_PATTERNS[1],
    CITATION_PATTERNS[2],
    CITATION_PATTERNS[3],
    CITATION_PATTERNS[5],
    CITATION_PATTERNS[6],
)
LOCKED_NUMBER_WORD_MIN = 10  # a number word below this, on its own, is prose
NAME_LABELS = {"PERSON", "ORG", "GPE", "LOC", "NORP", "WORK_OF_ART", "EVENT", "FAC", "LANGUAGE"}


@dataclass
class Inventory:
    citations: Counter[str] = field(default_factory=Counter)
    quotes: Counter[str] = field(default_factory=Counter)
    numbers: Counter[str] = field(default_factory=Counter)
    proper_nouns: Counter[str] = field(default_factory=Counter)

    def by_class(self, name: str) -> Counter[str]:
        counter: Counter[str] = getattr(self, name)
        return counter

    def is_empty(self) -> bool:
        return not any(self.by_class(c) for c in LOCK_CLASSES)

    def describe(self, classes: tuple[str, ...] = LOCK_CLASSES) -> list[str]:
        lines = []
        for cls in classes:
            items = self.by_class(cls)
            if items:
                lines.append(f"{cls}: " + "; ".join(sorted(items.elements())))
        return lines


def _normalize_quote(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip())


def strip_markup(text: str) -> str:
    """The words of a markdown text without its structure: headings and horizontal rules are
    removed, list markers and numbering dropped, emphasis marks unwrapped."""
    text = HEADING_RE.sub("", text)
    text = RULE_RE.sub("", text)
    text = LIST_MARKER_RE.sub(r"\1", text)
    for _ in range(2):  # nested emphasis
        text = EMPHASIS_RE.sub(r"\2", text)
    return text


def is_primary_citation(text: str) -> bool:
    item = text.strip()
    for pattern in PRIMARY_CITATION_PATTERNS:
        m = pattern.match(item + " ")  # trailing space satisfies end-of-token lookaheads
        if m and m.end() == len(item):
            return True
    return False


def document_names(text: str) -> set[str]:
    """Names the document establishes: entity and proper-noun tokens that appear capitalised
    somewhere other than a sentence or item start. A word capitalised only where a sentence
    starts (whatever the tagger calls it), or one the document also writes in lower case, is
    not a name."""
    doc = load_nlp(ner=True)(strip_markup(text))
    lowercase = {tok.text.lower() for tok in doc if tok.is_alpha and tok.text.islower()}
    names: set[str] = set()
    for ent in doc.ents:
        if ent.label_ in NAME_LABELS:
            names.update(t.text for t in ent if t.text[:1].isupper() and not t.is_sent_start)
    for tok in doc:
        if tok.pos_ == "PROPN" and tok.text[:1].isupper() and not tok.is_sent_start:
            names.add(tok.text)
    return {n for n in names if n.lower() not in lowercase}


def inventory(text: str, names: set[str] | None = None) -> Inventory:
    """Locked items in `text`. With `names` (from `document_names`), a sentence-initial
    capitalised token counts as a proper noun only when the document names it elsewhere."""
    inv = Inventory()
    text = strip_markup(text)
    seen: list[tuple[int, int]] = []
    for pattern in CITATION_PATTERNS:
        for m in pattern.finditer(text):
            if not any(s <= m.start() < e or s < m.end() <= e for s, e in seen):
                seen.append((m.start(), m.end()))
                inv.citations[m.group(0).strip()] += 1
    for m in ANY_QUOTE_RE.finditer(text):
        inv.quotes[_normalize_quote(m.group(1))] += 1
    quoted = [(m.start(), m.end()) for m in ANY_QUOTE_RE.finditer(text)]
    found = [(m.start(), m.end(), m.group(0)) for m in NUMBER_RE.finditer(text)]
    # "three hundred" counts as 300: the two spellings are one. Small single words ("one
    # man", "two things", "the first") are prose, not figures, and are not locked.
    found += [
        (a, b, value)
        for a, b, value in number_phrases(text)
        if int(value) >= LOCKED_NUMBER_WORD_MIN or " " in text[a:b] or "-" in text[a:b]
    ]
    for start, _end, value in found:
        if any(s <= start < e for s, e in seen):
            continue  # part of a citation
        if any(s <= start < e for s, e in quoted):
            continue  # protected by the quote already
        inv.numbers[value] += 1
    doc = load_nlp(ner=True)(text)
    entity_tokens = {t.i for ent in doc.ents if ent.label_ in NAME_LABELS for t in ent}
    lowercase = {tok.text.lower() for tok in doc if tok.is_alpha and tok.text.islower()}
    for tok in doc:
        if not tok.text[:1].isupper():
            continue
        is_name = tok.pos_ == "PROPN" or tok.i in entity_tokens
        if not is_name:
            continue
        known = names is not None and tok.text in names
        if tok.text.lower() in lowercase and not known:
            continue  # the document also writes it in lower case: a common word
        if names is not None and tok.is_sent_start and not known:
            continue  # capitalised only because it opens a sentence or an item
        inv.proper_nouns[tok.text] += 1
    return inv


@dataclass
class DocumentLock:
    """Everything the whole document already contains, so a rewrite may move an item between
    units without being charged for adding it."""

    citations: set[str]
    quotes: set[str]
    numbers: set[str]
    names: set[str]

    @classmethod
    def build(cls, text: str) -> DocumentLock:
        names = document_names(text)
        inv = inventory(text, names)
        return cls(
            citations=set(inv.citations),
            quotes=set(inv.quotes),
            numbers=set(inv.numbers),
            names=names | set(inv.proper_nouns),
        )

    def allows(self, cls_name: str, item: str) -> bool:
        if cls_name == "citations":
            return item in self.citations or is_primary_citation(item)
        if cls_name == "quotes":
            return item in self.quotes
        if cls_name == "numbers":
            return item in self.numbers
        return item in self.names


@dataclass
class LockViolation:
    cls: str
    missing: list[str]

    def __str__(self) -> str:
        return f"{self.cls}: missing " + ", ".join(self.missing)


@dataclass
class LockResult:
    violations: list[LockViolation]
    additions: list[str]  # new items the document does not establish; the semantic lock rules

    @property
    def ok(self) -> bool:
        return not self.violations


def check(before: str, after: str, document: DocumentLock) -> LockResult:
    """Lexical lock: every locked item in `before` must survive in `after`. Items `after`
    adds are fine when the document already has them or they cite the primary text; any
    other addition is reported for the semantic lock to judge, not rejected here."""
    a = inventory(before, document.names)
    b = inventory(after, document.names)
    after_lower, before_lower = after.lower(), before.lower()
    violations = []
    additions = []
    for cls in LOCK_CLASSES:
        before_items = a.by_class(cls)
        after_items = b.by_class(cls)
        after_keys = Counter(k.lower() for k in after_items.elements())
        before_keys = Counter(k.lower() for k in before_items.elements())
        if cls == "proper_nouns":
            missing = [t for t in before_items if t.lower() not in after_lower]
            added = [t for t in after_items if t.lower() not in before_lower]
        else:
            # Removals and additions are both case-insensitive: the words, not their case.
            missing = sorted(k for k, n in before_items.items() if after_keys[k.lower()] < n)
            added = sorted(set(k for k in after_items if before_keys[k.lower()] == 0))
        if missing:
            violations.append(LockViolation(cls, missing))
        for item in added:
            if not document.allows(cls, item):
                additions.append(f"{cls}: {item}")
    return LockResult(violations=violations, additions=additions)
