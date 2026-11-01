# ABOUTME: Recitation check: the runs of eight or more words a candidate shares with a reference
# ABOUTME: text (the human original) but not with its input, so a writer reciting a source is seen.
from __future__ import annotations

import re

from pydantic import BaseModel, Field

NGRAM = 8  # words; a shared run shorter than this is ordinary English, not recitation
DEFAULT_MAX_RECITATION = 2.0  # percent of candidate words covered by shared runs
FIXTURE_MAX_RECITATION = 1.0  # a text qualifies as a fixture only under this

TOKEN_RE = re.compile(r"[a-z0-9]+(?:['’][a-z0-9]+)*")


class RecitationResult(BaseModel):
    """What a candidate recites: how many of its `NGRAM`-word runs the reference contains and
    the input does not, how many of its words those runs cover, and the longest three runs."""

    ngrams: int = 0
    covered: int = 0
    words: int = 0
    share: float = 0.0  # percent of words covered
    spans: list[str] = Field(default_factory=list)  # the longest three runs
    runs: list[str] = Field(default_factory=list)  # every maximal run, in candidate order


def _tokens(text: str) -> list[tuple[str, int, int]]:
    """(word, start, end) for every word, lowercased with punctuation stripped."""
    return [
        (m.group(0).replace("'", "").replace("\u2019", ""), m.start(), m.end())
        for m in TOKEN_RE.finditer(text.lower())
    ]


def words_of(text: str) -> list[str]:
    """The words of `text` as the check sees them: lowercased, punctuation stripped."""
    return [w for w, _, _ in _tokens(text)]


def _ngrams(words: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def recitation(
    candidate: str, reference: str, input_text: str, *, n: int = NGRAM
) -> RecitationResult:
    """Count the `n`-word runs of `candidate` that `reference` contains and `input_text` does
    not. Case and punctuation are ignored. `covered` is the number of candidate words inside
    at least one such run, `share` that as a percent of the candidate's words, and `spans`
    the longest three maximal runs as they appear in the candidate (`runs` has them all)."""
    toks = _tokens(candidate)
    words = [w for w, _, _ in toks]
    result = RecitationResult(words=len(words))
    if len(words) < n:
        return result
    shared = _ngrams([w for w, _, _ in _tokens(reference)], n) - _ngrams(
        [w for w, _, _ in _tokens(input_text)], n
    )
    covered = [False] * len(words)
    count = 0
    for i in range(len(words) - n + 1):
        if tuple(words[i : i + n]) in shared:
            count += 1
            for k in range(i, i + n):
                covered[k] = True
    spans: list[tuple[int, int]] = []
    start = None
    for i, hit in enumerate(covered + [False]):
        if hit and start is None:
            start = i
        elif not hit and start is not None:
            spans.append((start, i - 1))
            start = None
    result.runs = [candidate[toks[a][1] : toks[b][2]] for a, b in spans]
    spans.sort(key=lambda s: (-(s[1] - s[0]), s[0]))
    result.ngrams = count
    result.covered = sum(covered)
    result.share = 100.0 * result.covered / len(words)
    result.spans = [candidate[toks[a][1] : toks[b][2]] for a, b in spans[:3]]
    return result


__all__ = [
    "DEFAULT_MAX_RECITATION",
    "FIXTURE_MAX_RECITATION",
    "NGRAM",
    "RecitationResult",
    "recitation",
    "words_of",
]
