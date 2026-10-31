# ABOUTME: Content lock: a document-scoped allowlist of citations, quotes, numbers and names.
# ABOUTME: Removals from a unit are violations; additions pass when the document already has them.
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from deslop.metrics.anchors import CITATION_PATTERNS, NUMBER_RE
from deslop.nlp import load_nlp

LOCK_CLASSES = ("citations", "quotes", "numbers", "proper_nouns")
ANY_QUOTE_RE = re.compile(r"[\"“]([^\"”]{1,600})[\"”]")
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


def is_primary_citation(text: str) -> bool:
    item = text.strip()
    for pattern in PRIMARY_CITATION_PATTERNS:
        m = pattern.match(item + " ")  # trailing space satisfies end-of-token lookaheads
        if m and m.end() == len(item):
            return True
    return False


def document_names(text: str) -> set[str]:
    """Names the document establishes: entity tokens anywhere plus proper-noun tokens that
    are not sentence-initial. A capital that only ever opens a sentence is not a name."""
    doc = load_nlp(ner=True)(text)
    names: set[str] = set()
    for ent in doc.ents:
        if ent.label_ in NAME_LABELS:
            names.update(t.text for t in ent if t.text[:1].isupper())
    for tok in doc:
        if tok.pos_ == "PROPN" and tok.text[:1].isupper() and not tok.is_sent_start:
            names.add(tok.text)
    return names


def inventory(text: str, names: set[str] | None = None) -> Inventory:
    """Locked items in `text`. With `names` (from `document_names`), a sentence-initial
    capitalised token counts as a proper noun only when the document names it elsewhere."""
    inv = Inventory()
    seen: list[tuple[int, int]] = []
    for pattern in CITATION_PATTERNS:
        for m in pattern.finditer(text):
            if not any(s <= m.start() < e or s < m.end() <= e for s, e in seen):
                seen.append((m.start(), m.end()))
                inv.citations[m.group(0).strip()] += 1
    for m in ANY_QUOTE_RE.finditer(text):
        inv.quotes[_normalize_quote(m.group(1))] += 1
    quoted = [(m.start(), m.end()) for m in ANY_QUOTE_RE.finditer(text)]
    for m in NUMBER_RE.finditer(text):
        if any(s <= m.start() < e for s, e in seen):
            continue  # part of a citation
        if any(s <= m.start() < e for s, e in quoted):
            continue  # protected by the quote already
        inv.numbers[m.group(0)] += 1
    doc = load_nlp(ner=True)(text)
    entity_tokens = {t.i for ent in doc.ents if ent.label_ in NAME_LABELS for t in ent}
    for tok in doc:
        if not tok.text[:1].isupper():
            continue
        is_name = tok.pos_ == "PROPN" or tok.i in entity_tokens
        if not is_name:
            continue
        if tok.is_sent_start and names is not None and tok.text not in names:
            continue
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
    violations = []
    additions = []
    for cls in LOCK_CLASSES:
        before_items = a.by_class(cls)
        after_items = b.by_class(cls)
        if cls == "proper_nouns":
            missing = [t for t in before_items if t not in after_items and t not in after]
            added = [t for t in after_items if t not in before_items and t not in before]
        else:
            missing = sorted((before_items - after_items).elements())
            added = sorted(set((after_items - before_items).elements()))
        if missing:
            violations.append(LockViolation(cls, missing))
        for item in added:
            if not document.allows(cls, item):
                additions.append(f"{cls}: {item}")
    return LockResult(violations=violations, additions=additions)
