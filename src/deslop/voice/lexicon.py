# ABOUTME: Vocabulary signature: log-odds of corpus words and bigrams against the generic
# ABOUTME: English baseline, plus per-pattern rates against the LLM reference.
from __future__ import annotations

import math
from collections import Counter

from deslop.baselines import ReferenceProfile
from deslop.patterns import Catalog
from deslop.stats import MetricsProfile, WordCounts
from deslop.voice.model import Lexicon, LexiconEntry, PatternRate

TOP_TERMS = 500
MIN_COUNT = 5
STOPLIKE_MAX_LEN = 2


def log_odds_with_prior(
    counts_a: Counter[str],
    total_a: int,
    counts_b: Counter[str],
    total_b: int,
    prior: Counter[str],
    prior_total: int,
    alpha_scale: float = 1000.0,
) -> dict[str, float]:
    """Monroe et al. (2008) weighted log-odds ratio with informative Dirichlet prior.

    Returns a z-score per term: positive means over-represented in A relative to B.
    """
    scores: dict[str, float] = {}
    alpha0 = alpha_scale
    for term in set(counts_a) | set(counts_b):
        a = counts_a.get(term, 0)
        b = counts_b.get(term, 0)
        if a + b < MIN_COUNT:
            continue
        alpha_w = alpha0 * (prior.get(term, 0) + 0.5) / (prior_total + 0.5 * len(prior) + 1)
        la = math.log((a + alpha_w) / (total_a + alpha0 - a - alpha_w))
        lb = math.log((b + alpha_w) / (total_b + alpha0 - b - alpha_w))
        delta = la - lb
        var = 1.0 / (a + alpha_w) + 1.0 / (b + alpha_w)
        scores[term] = delta / math.sqrt(var)
    return scores


def _entries(
    scores: dict[str, float],
    counts_a: Counter[str],
    total_a: int,
    counts_b: Counter[str],
    total_b: int,
    *,
    top: int,
    positive: bool,
) -> list[LexiconEntry]:
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=positive)
    out = []
    for term, z in ordered[:top]:
        if (positive and z <= 0) or (not positive and z >= 0):
            break
        if len(term.replace(" ", "")) <= STOPLIKE_MAX_LEN:
            continue
        out.append(
            LexiconEntry(
                term=term,
                log_odds=round(z, 3),
                corpus_per_million=round(1e6 * counts_a.get(term, 0) / max(total_a, 1), 2),
                baseline_per_million=round(1e6 * counts_b.get(term, 0) / max(total_b, 1), 2),
            )
        )
    return out


def build_lexicon(
    counts: WordCounts,
    profile: MetricsProfile,
    generic: ReferenceProfile,
    llm: ReferenceProfile | None,
    catalog: Catalog,
    *,
    ban_ratio: float = 10.0,
) -> Lexicon:
    gen_words = Counter(generic.word_freq)
    gen_bigrams = Counter(generic.bigram_freq)
    gen_total = generic.total_words
    word_scores = log_odds_with_prior(
        counts.unigrams, counts.total, gen_words, gen_total, gen_words, gen_total
    )
    bigram_scores = log_odds_with_prior(
        counts.bigrams, counts.total, gen_bigrams, gen_total, gen_bigrams, gen_total
    )
    lexicon = Lexicon(ban_ratio=ban_ratio)
    lexicon.encouraged_words = _entries(
        word_scores,
        counts.unigrams,
        counts.total,
        gen_words,
        gen_total,
        top=TOP_TERMS,
        positive=True,
    )
    lexicon.encouraged_bigrams = _entries(
        bigram_scores,
        counts.bigrams,
        counts.total,
        gen_bigrams,
        gen_total,
        top=TOP_TERMS,
        positive=True,
    )
    if llm is not None and llm.word_freq:
        llm_words = Counter(llm.word_freq)
        llm_scores = log_odds_with_prior(
            llm_words, llm.total_words, counts.unigrams, counts.total, gen_words, gen_total
        )
        lexicon.discouraged_words = _entries(
            llm_scores,
            llm_words,
            llm.total_words,
            counts.unigrams,
            counts.total,
            top=200,
            positive=True,
        )
    for pattern in catalog.patterns:
        corpus_rate = profile.tells.get(pattern.id)
        corpus_per_1k = corpus_rate.rate_per_1k if corpus_rate else 0.0
        llm_per_1k = llm.tells[pattern.id].rate_per_1k if llm and pattern.id in llm.tells else None
        ratio = None
        banned = False
        if llm_per_1k is not None:
            ratio = (
                llm_per_1k / corpus_per_1k
                if corpus_per_1k > 0
                else (math.inf if llm_per_1k > 0 else 1.0)
            )
            banned = ratio > ban_ratio
        lexicon.patterns.append(
            PatternRate(
                id=pattern.id,
                name=pattern.name,
                corpus_per_1k=round(corpus_per_1k, 3),
                llm_per_1k=None if llm_per_1k is None else round(llm_per_1k, 3),
                ratio=None if ratio is None else (999.0 if math.isinf(ratio) else round(ratio, 2)),
                banned=banned,
            )
        )
    return lexicon
