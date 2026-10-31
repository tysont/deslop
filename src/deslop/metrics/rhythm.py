# ABOUTME: Rhythm metrics: sentence and paragraph length shape, openers, punctuation.
# ABOUTME: All are computed from the Analysis without model calls.
from __future__ import annotations

import re
from collections import Counter

from spacy.tokens import Span

from deslop.metrics.base import mean, metric, pct, stdev
from deslop.nlp import Analysis

COORDINATORS = {"and", "but", "or", "nor", "yet", "so"}


def sentence_lengths(a: Analysis) -> list[float]:
    return [float(s.word_count) for s in a.sentences]


@metric("sent_len_mean", group="rhythm", slop_side="none", description="Mean words per sentence")
def sent_len_mean(a: Analysis) -> float:
    return mean(sentence_lengths(a))


@metric(
    "sent_len_std", group="rhythm", slop_side="low", description="Std dev of words per sentence"
)
def sent_len_std(a: Analysis) -> float:
    return stdev(sentence_lengths(a))


@metric(
    "sent_len_cv",
    group="rhythm",
    slop_side="low",
    description="Coefficient of variation of sentence length",
)
def sent_len_cv(a: Analysis) -> float:
    lengths = sentence_lengths(a)
    m = mean(lengths)
    return stdev(lengths) / m if m else 0.0


@metric("pct_sent_lt8", group="rhythm", slop_side="none", description="% sentences under 8 words")
def pct_sent_lt8(a: Analysis) -> float:
    return pct(sentence_lengths(a), lambda n: n < 8)


@metric("pct_sent_gt30", group="rhythm", slop_side="none", description="% sentences over 30 words")
def pct_sent_gt30(a: Analysis) -> float:
    return pct(sentence_lengths(a), lambda n: n > 30)


@metric(
    "max_run_similar_len",
    group="rhythm",
    slop_side="high",
    description="Longest run of consecutive sentences within +/-5 words of each other",
)
def max_run_similar_len(a: Analysis) -> float:
    lengths = sentence_lengths(a)
    if not lengths:
        return 0.0
    best = run = 1
    for prev, cur in zip(lengths, lengths[1:], strict=False):
        if abs(cur - prev) <= 5:
            run += 1
            best = max(best, run)
        else:
            run = 1
    return float(best)


def coordinating_joins(sent: Span) -> int:
    return sum(1 for t in sent if t.dep_ == "cc" and t.lower_ in COORDINATORS)


@metric(
    "pct_compound_multi",
    group="rhythm",
    slop_side="none",
    description="% sentences with two or more coordinating joins",
)
def pct_compound_multi(a: Analysis) -> float:
    joins = [float(coordinating_joins(s.span)) for s in a.sentences]
    return pct(joins, lambda n: n >= 2)


@metric(
    "para_len_mean", group="rhythm", slop_side="none", description="Mean sentences per paragraph"
)
def para_len_mean(a: Analysis) -> float:
    return mean([float(len(p.sentences)) for p in a.paragraphs])


@metric(
    "para_len_std",
    group="rhythm",
    slop_side="low",
    description="Std dev of sentences per paragraph",
)
def para_len_std(a: Analysis) -> float:
    return stdev([float(len(p.sentences)) for p in a.paragraphs])


def opener_pos_counts(a: Analysis) -> Counter[str]:
    counts: Counter[str] = Counter()
    for s in a.sentences:
        for tok in s.span:
            if tok.is_alpha:
                counts[tok.pos_] += 1
                break
    return counts


def _opener_share(a: Analysis, pos_set: set[str]) -> float:
    counts = opener_pos_counts(a)
    total = sum(counts.values())
    return 100.0 * sum(counts[p] for p in pos_set) / total if total else 0.0


@metric(
    "opener_pct_det",
    group="rhythm",
    slop_side="none",
    description="% sentences opening with a determiner",
)
def opener_pct_det(a: Analysis) -> float:
    return _opener_share(a, {"DET"})


@metric(
    "opener_pct_pron",
    group="rhythm",
    slop_side="none",
    description="% sentences opening with a pronoun",
)
def opener_pct_pron(a: Analysis) -> float:
    return _opener_share(a, {"PRON"})


@metric(
    "opener_pct_adv_conj",
    group="rhythm",
    slop_side="high",
    description="% sentences opening with an adverb or conjunction",
)
def opener_pct_adv_conj(a: Analysis) -> float:
    return _opener_share(a, {"ADV", "CCONJ", "SCONJ"})


@metric(
    "opener_pct_noun",
    group="rhythm",
    slop_side="none",
    description="% sentences opening with a noun or proper noun",
)
def opener_pct_noun(a: Analysis) -> float:
    return _opener_share(a, {"NOUN", "PROPN"})


@metric(
    "opener_top_word_share",
    group="rhythm",
    slop_side="high",
    description="% of sentences that begin with the single most common first word",
)
def opener_top_word_share(a: Analysis) -> float:
    words: Counter[str] = Counter()
    for s in a.sentences:
        for tok in s.span:
            if tok.is_alpha:
                words[tok.lower_] += 1
                break
    total = sum(words.values())
    return 100.0 * words.most_common(1)[0][1] / total if total else 0.0


EM_DASH_RE = re.compile(r"—|--")


def _per_1k(a: Analysis, count: int) -> float:
    return count / a.words_per_1k


@metric(
    "semicolons_per_1k", group="rhythm", slop_side="none", description="Semicolons per 1k words"
)
def semicolons_per_1k(a: Analysis) -> float:
    return _per_1k(a, a.text.count(";"))


@metric("colons_per_1k", group="rhythm", slop_side="high", description="Colons per 1k words")
def colons_per_1k(a: Analysis) -> float:
    return _per_1k(a, a.text.count(":"))


@metric("em_dashes_per_1k", group="rhythm", slop_side="high", description="Em dashes per 1k words")
def em_dashes_per_1k(a: Analysis) -> float:
    return _per_1k(a, len(EM_DASH_RE.findall(a.text)))


@metric(
    "parens_per_1k", group="rhythm", slop_side="none", description="Open parentheses per 1k words"
)
def parens_per_1k(a: Analysis) -> float:
    return _per_1k(a, a.text.count("("))


@metric(
    "questions_per_1k", group="rhythm", slop_side="high", description="Question marks per 1k words"
)
def questions_per_1k(a: Analysis) -> float:
    return _per_1k(a, a.text.count("?"))
