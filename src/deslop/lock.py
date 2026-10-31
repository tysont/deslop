# ABOUTME: Content lock: inventories citations, quotes, numbers, and proper nouns in a span
# ABOUTME: so a rewrite can be rejected if any of them changed.
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from deslop.metrics.anchors import CITATION_PATTERNS, NUMBER_RE
from deslop.nlp import load_nlp

LOCK_CLASSES = ("citations", "quotes", "numbers", "proper_nouns")
ANY_QUOTE_RE = re.compile(r"[\"“]([^\"”]{1,600})[\"”]")


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


def inventory(text: str) -> Inventory:
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
    doc = load_nlp()(text)
    for tok in doc:
        if tok.pos_ == "PROPN" and tok.text[:1].isupper():
            inv.proper_nouns[tok.text] += 1
    return inv


@dataclass
class LockViolation:
    cls: str
    missing: list[str]
    added: list[str]

    def __str__(self) -> str:
        parts = []
        if self.missing:
            parts.append("missing " + ", ".join(self.missing))
        if self.added:
            parts.append("added " + ", ".join(self.added))
        return f"{self.cls}: " + "; ".join(parts)


def check(
    before: str,
    after: str,
    classes: tuple[str, ...] = LOCK_CLASSES,
    *,
    allowed_names: set[str] | None = None,
) -> list[LockViolation]:
    """Return violations: locked items removed from or added to the span.

    Proper nouns listed in allowed_names (typically every name in the whole document) may be
    introduced, since naming a person already in the text adds no content.
    """
    a = inventory(before)
    b = inventory(after)
    allowed = allowed_names or set()
    violations = []
    for cls in classes:
        before_items = a.by_class(cls)
        after_items = b.by_class(cls)
        if cls == "proper_nouns":
            # Tagging varies with context; compare surface strings, not counts.
            missing = [t for t in before_items if t not in after and t not in after_items]
            added = [
                t
                for t in after_items
                if t not in before_items and t not in before and t not in allowed
            ]
        else:
            missing = sorted((before_items - after_items).elements())
            added = sorted((after_items - before_items).elements())
        if missing or added:
            violations.append(LockViolation(cls, missing, added))
    return violations
