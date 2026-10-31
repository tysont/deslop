# ABOUTME: Statistical profiling shared by calibrate and the baseline builders.
# ABOUTME: Splits texts into chunks, computes metrics and tell rates, aggregates with weights.
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from deslop.metrics import compute_all, detect_tells, tell_rates
from deslop.nlp import WORD_RE, Analysis, Paragraph, analyze
from deslop.patterns import Catalog, load_catalog

DEFAULT_CHUNK_WORDS = 500


class MetricStat(BaseModel):
    mean: float
    std: float
    p10: float
    p50: float
    p90: float
    n: int


class TellStat(BaseModel):
    rate_per_1k: float
    hits: int
    words: int


class AuthorSamples(BaseModel):
    weight: float = 1.0
    words: int = 0
    chunks: int = 0
    metrics: dict[str, list[float]] = Field(default_factory=dict)
    tells: dict[str, list[float]] = Field(default_factory=dict)
    tell_hits: dict[str, int] = Field(default_factory=dict)


class MetricsProfile(BaseModel):
    """metrics.json: aggregate stats plus the per-author samples they came from."""

    chunk_words: int = DEFAULT_CHUNK_WORDS
    total_words: int = 0
    metrics: dict[str, MetricStat] = Field(default_factory=dict)
    tells: dict[str, TellStat] = Field(default_factory=dict)
    per_author: dict[str, AuthorSamples] = Field(default_factory=dict)


@dataclass
class WordCounts:
    unigrams: Counter[str] = field(default_factory=Counter)
    bigrams: Counter[str] = field(default_factory=Counter)
    total: int = 0

    def add(self, text: str) -> None:
        words = [w.lower() for w in WORD_RE.findall(text)]
        self.unigrams.update(words)
        self.bigrams.update(f"{a} {b}" for a, b in zip(words, words[1:], strict=False))
        self.total += len(words)


# ---------------------------------------------------------------- chunking


def subset_analysis(paragraphs: list[Paragraph]) -> Analysis:
    """Build an Analysis over a subset of paragraphs without re-parsing."""
    text_parts: list[str] = []
    new_paras: list[Paragraph] = []
    offset = 0
    sent_index = 0
    for i, p in enumerate(paragraphs):
        np_ = Paragraph(index=i, start=offset, end=offset + len(p.text), doc=p.doc)
        for s in p.sentences:
            np_.sentences.append(
                type(s)(
                    index=sent_index,
                    para_index=i,
                    start=offset + s.span.start_char,
                    end=offset + s.span.end_char,
                    span=s.span,
                )
            )
            sent_index += 1
        new_paras.append(np_)
        text_parts.append(p.text)
        offset += len(p.text) + 2
    return Analysis(text="\n\n".join(text_parts), paragraphs=new_paras)


def chunk_analysis(a: Analysis, target_words: int = DEFAULT_CHUNK_WORDS) -> list[Analysis]:
    chunks: list[Analysis] = []
    current: list[Paragraph] = []
    words = 0
    for p in a.paragraphs:
        current.append(p)
        words += p.word_count
        if words >= target_words:
            chunks.append(subset_analysis(current))
            current, words = [], 0
    if current:
        if chunks and words < target_words * 0.4:
            merged = chunks[-1].paragraphs + current
            chunks[-1] = subset_analysis(merged)
        else:
            chunks.append(subset_analysis(current))
    return chunks


# ---------------------------------------------------------------- sampling


def sample_text(
    text: str,
    *,
    chunk_words: int = DEFAULT_CHUNK_WORDS,
    catalog: Catalog | None = None,
) -> AuthorSamples:
    catalog = catalog or load_catalog()
    samples = AuthorSamples()
    analysis = analyze(text)
    for chunk in chunk_analysis(analysis, chunk_words):
        values = compute_all(chunk)
        hits = detect_tells(chunk, catalog)
        rates = tell_rates(chunk, hits, catalog)
        for k, v in values.items():
            samples.metrics.setdefault(k, []).append(v)
        for k, v in rates.items():
            samples.tells.setdefault(k, []).append(v)
        for h in hits:
            samples.tell_hits[h.pattern_id] = samples.tell_hits.get(h.pattern_id, 0) + 1
        samples.words += chunk.word_count
        samples.chunks += 1
    return samples


def merge_samples(a: AuthorSamples, b: AuthorSamples) -> AuthorSamples:
    out = AuthorSamples(weight=a.weight, words=a.words + b.words, chunks=a.chunks + b.chunks)
    for k in set(a.metrics) | set(b.metrics):
        out.metrics[k] = a.metrics.get(k, []) + b.metrics.get(k, [])
    for k in set(a.tells) | set(b.tells):
        out.tells[k] = a.tells.get(k, []) + b.tells.get(k, [])
    for k in set(a.tell_hits) | set(b.tell_hits):
        out.tell_hits[k] = a.tell_hits.get(k, 0) + b.tell_hits.get(k, 0)
    return out


# ---------------------------------------------------------------- aggregation


def weighted_quantile(values: list[float], weights: list[float], q: float) -> float:
    pairs = sorted(zip(values, weights, strict=True))
    total = sum(w for _, w in pairs)
    if total <= 0:
        return 0.0
    acc = 0.0
    for v, w in pairs:
        acc += w
        if acc / total >= q:
            return v
    return pairs[-1][0]


def weighted_stat(values: list[float], weights: list[float]) -> MetricStat:
    if not values:
        return MetricStat(mean=0.0, std=0.0, p10=0.0, p50=0.0, p90=0.0, n=0)
    total = sum(weights)
    mean = sum(v * w for v, w in zip(values, weights, strict=True)) / total
    var = sum(w * (v - mean) ** 2 for v, w in zip(values, weights, strict=True)) / total
    return MetricStat(
        mean=mean,
        std=math.sqrt(var),
        p10=weighted_quantile(values, weights, 0.10),
        p50=weighted_quantile(values, weights, 0.50),
        p90=weighted_quantile(values, weights, 0.90),
        n=len(values),
    )


def author_weights(per_author: dict[str, AuthorSamples], max_share: float) -> dict[str, float]:
    """Per-chunk weights so no author's weighted share exceeds max_share."""
    raw = {a: s.weight * s.words for a, s in per_author.items() if s.words > 0}
    if not raw:
        return {}
    shares = capped_shares(raw, max_share)
    return {a: shares[a] / per_author[a].chunks for a in raw}


def capped_shares(raw: dict[str, float], max_share: float) -> dict[str, float]:
    """Proportional shares with any share above max_share pinned to it exactly."""
    if len(raw) * max_share <= 1.0 + 1e-9 and max_share < 1.0 / len(raw):
        return {a: 1.0 / len(raw) for a in raw}
    capped: set[str] = set()
    while True:
        free_total = sum(v for a, v in raw.items() if a not in capped)
        remaining = 1.0 - len(capped) * max_share
        shares = {
            a: (max_share if a in capped else remaining * v / free_total) for a, v in raw.items()
        }
        over = [a for a, s in shares.items() if a not in capped and s > max_share + 1e-12]
        if not over:
            return shares
        capped.update(over)
        if len(capped) * max_share >= 1.0:
            return {a: 1.0 / len(raw) for a in raw}


def aggregate(
    per_author: dict[str, AuthorSamples], *, max_share: float = 0.25, chunk_words: int
) -> MetricsProfile:
    weights = author_weights(per_author, max_share)
    profile = MetricsProfile(chunk_words=chunk_words, per_author=per_author)
    profile.total_words = sum(s.words for s in per_author.values())
    metric_names = sorted({k for s in per_author.values() for k in s.metrics})
    for name in metric_names:
        values: list[float] = []
        ws: list[float] = []
        for a, s in per_author.items():
            vals = s.metrics.get(name, [])
            values.extend(vals)
            ws.extend([weights.get(a, 0.0)] * len(vals))
        profile.metrics[name] = weighted_stat(values, ws)
    tell_ids = sorted({k for s in per_author.values() for k in s.tells})
    for tid in tell_ids:
        values, ws = [], []
        hits = 0
        for a, s in per_author.items():
            vals = s.tells.get(tid, [])
            values.extend(vals)
            ws.extend([weights.get(a, 0.0)] * len(vals))
            hits += s.tell_hits.get(tid, 0)
        rate = weighted_stat(values, ws).mean
        profile.tells[tid] = TellStat(rate_per_1k=rate, hits=hits, words=profile.total_words)
    return profile
