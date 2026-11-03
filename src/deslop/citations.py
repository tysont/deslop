# ABOUTME: Citation and quotation policy for apply: which additions a primary text permits, and
# ABOUTME: the --citations keep|none checks that input citations survive in the same sentence.
from __future__ import annotations

import re
from typing import Literal

from deslop.numbers import words_to_digits
from deslop.reference import (
    Addition,
    PrimaryText,
    canonical_label,
    normalize,
    primary_citations,
    quoted_spans,
)
from deslop.result import CandidateLog
from deslop.semantic import AddedClaim

Citations = Literal["keep", "verified", "none"]

KEY_TOKEN_RE = re.compile(r"\b[A-Z][A-Za-z'-]+\b|\d+(?:[.:]\d+)*")


def permit_primary(
    cand: CandidateLog, texts: list[PrimaryText], allow_facts: bool, original: str = ""
) -> None:
    """The primary-text policy on the semantic lock's additions: a citation that resolves in a
    declared primary text, a sentence that merely names one, and a claim whose new names and
    numbers (those `original` lacks) are all inside a quotation verified verbatim are
    permitted and moved to `cleared` with the reason. With `allow_facts`, a claim whose
    sentence cites a resolved location and whose names and numbers all occur in that
    location is permitted too. Everything else stands."""
    if cand.semantic is None or not texts:
        return
    verified = [a for a in (cand.primary.additions if cand.primary else []) if a.verified]
    names = {w.lower() for t in texts for w in re.findall(r"[A-Za-z]+", t.name)}
    known = set(normalize(original).split())
    kept = []
    for item in cand.semantic.added:
        why = _primary_permission(item, verified, texts, names, allow_facts, known)
        if item.kind == "claim" and why:
            item.note = why
            cand.semantic.cleared.append(item)
        else:
            kept.append(item)
    cand.semantic.added = kept


STRUCTURE_WORDS = {"book", "books", "chapter", "chapters", "verse", "verses", "part"}

FUNCTION_WORDS = {"the", "a", "an", "it", "this", "that", "and", "but", "in", "of", "as", "its"}

ROMAN_RE = re.compile(r"^[IVXLC]+$")


def _primary_permission(
    item: AddedClaim,
    verified: list[Addition],
    texts: list[PrimaryText],
    names: set[str],
    allow_facts: bool,
    known: set[str] = frozenset(),  # type: ignore[assignment]
) -> str:
    sentence = item.sentence or ""
    tokens = [t for t in KEY_TOKEN_RE.findall(words_to_digits(item.text))]
    if item.text and item.text[:1].isupper() and tokens and item.text.startswith(tokens[0]):
        tokens = tokens[1:]  # a sentence-initial capital is not a name
    keys = [t.lower() for t in tokens if t.lower() not in FUNCTION_WORDS]
    cites = [a for a in verified if a.kind == "citation" and a.text in sentence and not a.coarse]
    cited = {canonical_label(a.text) for a in verified if a.kind == "citation"}
    cited |= {canonical_label(a.text.split()[-1]) for a in verified if a.kind == "citation"}
    if (
        cited
        and keys
        and all(
            canonical_label(k) in cited or k in STRUCTURE_WORDS or k in ("verse", "reference")
            for k in keys
        )
    ):
        return "a citation that resolves in a declared primary text"
    if (
        item.category == "entity"
        and keys
        and all(
            k in names or k in STRUCTURE_WORDS or k.isdigit() or ROMAN_RE.match(k.upper())
            for k in keys
        )
    ):
        return "names a declared primary text"
    for a in verified:
        if a.kind != "quote" or normalize(a.text) not in normalize(sentence):
            continue
        quoted = set(normalize(a.text).split())
        if keys and all(k in quoted or k in known for k in keys):
            return f"verbatim in {a.where}"
    if allow_facts and keys:
        for a in cites:
            for t in texts:
                if not t.answers_to(a.text):
                    continue
                span = words_to_digits(t.cited_text(a.text))
                if span and all(k in span.split() for k in keys):
                    return f"a fact verifiable at {a.where}"
    return ""


SENTENCE_SPLIT_RE = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][\"”’)]))\s+(?=[A-Z\"“(])")

CONTENT_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]{3,}")

SAME_POINT_OVERLAP = 0.3  # share of the input sentence's content words the output sentence keeps


def _content_words(sentence: str) -> set[str]:
    return {w.lower() for w in CONTENT_WORD_RE.findall(sentence)}


def _sentences_with(text: str, needle: str) -> list[str]:
    return [s for s in SENTENCE_SPLIT_RE.split(text) if needle in s]


def citation_policy_notes(
    section_text: str, candidate: str, texts: list[PrimaryText], policy: Citations
) -> list[str]:
    """`--citations keep` and `none`: every citation the input carries must appear in the
    output exactly as written and at least as often, each in a sentence that makes the same point as
    the input sentence carrying it (it keeps at least `SAME_POINT_OVERLAP` of that sentence's
    content words), and every input quotation byte-identically; under `none` the output may
    add no citation or quotation either. `verified` leaves this to the primary-text check."""
    if policy == "verified":
        return []
    notes = []
    before = primary_citations(section_text, texts)
    for c in sorted(set(before)):
        want, have = before.count(c), candidate.count(c)
        if have < want:
            notes.append(f"the citation {c} must appear exactly as written ({want}x, found {have})")
            continue
        for source in _sentences_with(section_text, c):
            words = _content_words(source)
            if not words:
                continue
            best = max(
                (
                    len(words & _content_words(out)) / len(words)
                    for out in _sentences_with(candidate, c)
                ),
                default=0.0,
            )
            if best < SAME_POINT_OVERLAP:
                notes.append(
                    f"the citation {c} appears, but not in the sentence that makes the same point "
                    f'("{source.strip()[:70]}")'
                )
                break
    for q in quoted_spans(section_text):
        if q not in candidate:
            notes.append(f'the quotation "{q[:60]}" must be reproduced exactly as written')
    if policy == "none":
        after = primary_citations(candidate, texts)
        for c in sorted(set(after) - set(before)):
            notes.append(f"the citation {c} is not in the original; add no citations")
        kept = {normalize(q) for q in quoted_spans(section_text)}
        for q in quoted_spans(candidate):
            if normalize(q) not in kept:
                notes.append(f'the quotation "{q[:60]}" is not in the original; add no quotations')
    return notes
