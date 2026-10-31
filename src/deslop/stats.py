# ABOUTME: Statistical profiling shared by calibrate and the baseline builders.
# ABOUTME: Splits texts into chunks, computes metrics and tell rates, aggregates with weights.
from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from deslop.metrics import compute_all, detect_tells, tell_rates
from deslop.nlp import WORD_RE, Analysis, Paragraph, analyze
from deslop.patterns import Catalog, load_catalog

DEFAULT_CHUNK_WORDS = 500


class MetricStat(BaseModel):
    """Per-metric statistics over 500-word chunks. `within_std` is the pooled standard
    deviation of document means within one author, which is what the score bands use so that
    the spread between authors does not widen them; `ci_low`/`ci_high` bound the mean with a
    document-level bootstrap."""

    mean: float
    std: float
    p10: float
    p50: float
    p90: float
    n: int
    within_std: float = 0.0
    ci_low: float = 0.0
    ci_high: float = 0.0


class TellStat(BaseModel):
    rate_per_1k: float
    hits: int
    words: int


class DocSamples(BaseModel):
    """One document's chunk-averaged metric values and tell rates, kept for clustering,
    held-out threshold calibration and the bootstrap."""

    title: str
    author: str
    source: str = ""
    words: int = 0
    chunks: int = 0
    metrics: dict[str, float] = Field(default_factory=dict)
    tells: dict[str, float] = Field(default_factory=dict)


class AuthorSamples(BaseModel):
    weight: float = 1.0
    words: int = 0
    chunks: int = 0
    metrics: dict[str, list[float]] = Field(default_factory=dict)
    tells: dict[str, list[float]] = Field(default_factory=dict)
    tell_hits: dict[str, int] = Field(default_factory=dict)


class ClusterMetric(BaseModel):
    metric: str
    cluster_mean: float
    voice_mean: float
    z: float


class Cluster(BaseModel):
    """One register found by clustering documents on their metric vectors."""

    index: int
    label: str
    documents: int
    words: int
    median_doc_words: int
    top_authors: list[str] = Field(default_factory=list)
    titles: list[str] = Field(default_factory=list)
    distinguishing: list[ClusterMetric] = Field(default_factory=list)


class MetricsProfile(BaseModel):
    """metrics.json: aggregate stats plus the per-author samples they came from, and the
    registers (document clusters) found at calibration."""

    chunk_words: int = DEFAULT_CHUNK_WORDS
    total_words: int = 0
    metrics: dict[str, MetricStat] = Field(default_factory=dict)
    tells: dict[str, TellStat] = Field(default_factory=dict)
    per_author: dict[str, AuthorSamples] = Field(default_factory=dict)
    per_document: list[DocSamples] = Field(default_factory=list)
    clusters: list[Cluster] = Field(default_factory=list)


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


def doc_samples(
    samples: AuthorSamples, *, title: str, author: str, source: str = ""
) -> DocSamples:
    """Collapse one document's chunk samples to chunk-mean metrics and document tell rates."""
    metrics = {k: sum(v) / len(v) for k, v in samples.metrics.items() if v}
    tells = {k: sum(v) / len(v) for k, v in samples.tells.items() if v}
    return DocSamples(
        title=title,
        author=author,
        source=source,
        words=samples.words,
        chunks=samples.chunks,
        metrics=metrics,
        tells=tells,
    )


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


BOOTSTRAP_RESAMPLES = 200


def within_author_std(docs: list[DocSamples], name: str) -> float:
    """Pooled standard deviation of document means within authors. Authors with one
    document contribute nothing; with no author holding two documents, fall back to the
    spread across all documents."""
    by_author: dict[str, list[float]] = {}
    for d in docs:
        if name in d.metrics:
            by_author.setdefault(d.author, []).append(d.metrics[name])
    pooled_ss = 0.0
    pooled_n = 0
    for vals in by_author.values():
        if len(vals) < 2:
            continue
        m = sum(vals) / len(vals)
        pooled_ss += sum((v - m) ** 2 for v in vals)
        pooled_n += len(vals) - 1
    if pooled_n > 0:
        return math.sqrt(pooled_ss / pooled_n)
    vals = [d.metrics[name] for d in docs if name in d.metrics]
    if len(vals) < 2:
        return 0.0
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def bootstrap_ci(
    docs: list[DocSamples],
    name: str,
    doc_weights: dict[str, float],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 0,
) -> tuple[float, float]:
    """95% interval for the weighted mean of `name`, resampling documents with replacement."""
    pool = [(d.metrics[name], doc_weights.get(d.title, 0.0)) for d in docs if name in d.metrics]
    if len(pool) < 2:
        v = pool[0][0] if pool else 0.0
        return v, v
    rng = random.Random(seed)
    means = []
    for _ in range(resamples):
        draw = [pool[rng.randrange(len(pool))] for _ in pool]
        total = sum(w for _, w in draw)
        means.append(sum(v * w for v, w in draw) / total if total > 0 else 0.0)
    means.sort()
    lo = means[int(0.025 * (len(means) - 1))]
    hi = means[int(0.975 * (len(means) - 1))]
    return lo, hi


def aggregate(
    per_author: dict[str, AuthorSamples],
    *,
    max_share: float = 0.25,
    chunk_words: int,
    per_document: list[DocSamples] | None = None,
) -> MetricsProfile:
    weights = author_weights(per_author, max_share)
    docs = per_document or []
    profile = MetricsProfile(chunk_words=chunk_words, per_author=per_author, per_document=docs)
    # Each document's bootstrap weight is its words times its author's per-chunk weight, so
    # the resampled mean respects the same author cap as the point estimate.
    doc_weights = {d.title: d.words * weights.get(d.author, 0.0) for d in docs}
    profile.total_words = sum(s.words for s in per_author.values())
    metric_names = sorted({k for s in per_author.values() for k in s.metrics})
    for name in metric_names:
        values: list[float] = []
        ws: list[float] = []
        for a, s in per_author.items():
            vals = s.metrics.get(name, [])
            values.extend(vals)
            ws.extend([weights.get(a, 0.0)] * len(vals))
        stat = weighted_stat(values, ws)
        if docs:
            stat.within_std = within_author_std(docs, name)
            stat.ci_low, stat.ci_high = bootstrap_ci(docs, name, doc_weights)
        else:
            stat.within_std = stat.std
            stat.ci_low = stat.ci_high = stat.mean
        profile.metrics[name] = stat
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
