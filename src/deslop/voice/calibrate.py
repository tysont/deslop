# ABOUTME: Compiles a corpus into a Voice: metrics profile, lexicon, rules, profile, exemplars.
# ABOUTME: Deterministic except for the optional LLM-written profile.md.
from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

from deslop import __version__
from deslop.baselines import english_generic, has_baseline, llm_reference
from deslop.corpus import Corpus
from deslop.patterns import load_catalog
from deslop.providers.base import CachingProvider
from deslop.stats import (
    DEFAULT_CHUNK_WORDS,
    DEFAULT_MIN_DOCS,
    AuthorSamples,
    DocSamples,
    MetricsProfile,
    WordCounts,
    aggregate,
    doc_samples,
    merge_samples,
    paragraph_variance,
    sample_text,
    shrink_small_corpus,
)
from deslop.thresholds import HELDOUT_PASS_TARGET, calibrate_thresholds
from deslop.voice.clusters import DocumentVector, cluster_documents, with_voice_means
from deslop.voice.exemplars import build_exemplars
from deslop.voice.lexicon import build_lexicon
from deslop.voice.model import AuthorProvenance, DocumentProvenance, Voice, VoiceMeta
from deslop.voice.profile import llm_profile, template_profile
from deslop.voice.rules import build_rules


def calibrate(
    corpus: Corpus,
    *,
    name: str | None = None,
    max_share: float = 0.25,
    exemplar_words: int = 2000,
    use_llm: bool = True,
    llm_model: str | None = None,
    provider: CachingProvider | None = None,
    patterns: tuple[str, ...] = (),
    ban_ratio: float = 10.0,
    chunk_words: int = DEFAULT_CHUNK_WORDS,
    min_docs: int = DEFAULT_MIN_DOCS,
    progress: Callable[[str], None] | None = None,
) -> Voice:
    catalog = load_catalog(patterns)
    generic = english_generic()
    llm = llm_reference() if has_baseline("llm_reference") else None
    docs = corpus.documents()
    if not docs:
        raise ValueError("corpus has no documents")

    per_author: dict[str, AuthorSamples] = {}
    per_doc: list[DocSamples] = []
    counts = WordCounts()
    for doc in docs:
        if progress:
            progress(f"sampling {doc.entry.author}: {doc.entry.title}")
        samples = sample_text(doc.text, chunk_words=chunk_words, catalog=catalog)
        samples.weight = doc.entry.weight
        if samples.chunks > 0:
            per_doc.append(
                doc_samples(
                    samples,
                    title=doc.entry.title,
                    author=doc.entry.author,
                    source=doc.entry.source,
                )
            )
        author = doc.entry.author
        per_author[author] = (
            merge_samples(per_author[author], samples) if author in per_author else samples
        )
        per_author[author].weight = doc.entry.weight
        counts.add(doc.text)

    profile = aggregate(
        per_author, max_share=max_share, chunk_words=chunk_words, per_document=per_doc
    )
    if len(per_doc) < min_docs:
        if progress:
            progress(f"small corpus: {len(per_doc)} documents, shrinking toward generic English")
        mean_words = sum(d.words for d in per_doc) / max(len(per_doc), 1)
        profile.metrics, profile.small_corpus = shrink_small_corpus(
            profile.metrics,
            generic.metrics,
            paragraph_variance([d.text for d in docs]),
            documents=len(per_doc),
            min_docs=min_docs,
            mean_doc_words=mean_words,
        )
    if progress:
        progress("clustering documents into registers")
    metric_names = [n for n in sorted(profile.metrics) if profile.metrics[n].std > 0]
    vectors = [
        DocumentVector(
            title=d.title,
            author=d.author,
            words=d.words,
            vector=document_vector(d, profile, metric_names),
        )
        for d in per_doc
    ]
    clusters, assignment = cluster_documents(vectors, metric_names)
    profile.clusters = with_voice_means(clusters, profile)
    registers = {title: clusters[i].label for title, i in assignment.items()}
    lexicon = build_lexicon(counts, profile, generic, llm, catalog, ban_ratio=ban_ratio)
    if progress:
        progress("calibrating thresholds against held-out and LLM-reference documents")
    profile.thresholds = calibrate_thresholds(
        profile,
        llm.documents if llm else [],
        banned=set(lexicon.banned_ids()),
        heldout_target=1.0 if profile.small_corpus is not None else HELDOUT_PASS_TARGET,
    )
    rules = build_rules(profile, generic, lexicon)
    if progress:
        progress("selecting exemplars")
    exemplars = build_exemplars(
        [(d.entry.author, d.entry.title, d.entry.source, d.text) for d in docs],
        profile,
        target_words=exemplar_words,
        max_share=max_share,
        registers=registers,
    )
    voice_name = name or corpus.name
    profile_source = "template"
    if use_llm and provider is None:
        from deslop.providers.base import ProviderError, default_provider

        try:
            provider = default_provider()
        except ProviderError as exc:
            if progress:
                progress(f"no provider for the profile ({exc}); using the template")
            provider = None
    if use_llm and provider is not None:
        if progress:
            progress(f"writing profile with {provider.name}")
        profile_md = llm_profile(
            voice_name, profile, generic, exemplars, rules, provider, llm_model, llm=llm
        )
        profile_source = f"{provider.name}:{llm_model or provider.default_model}"
    else:
        profile_md = template_profile(voice_name, profile, generic, llm)

    meta = provenance(corpus, per_author, max_share=max_share, name=voice_name)
    meta.profile_source = profile_source
    return Voice(
        meta=meta,
        metrics=profile,
        lexicon=lexicon,
        rules=rules,
        profile=profile_md,
        exemplars=exemplars,
    )


def provenance(
    corpus: Corpus, per_author: dict[str, AuthorSamples], *, max_share: float, name: str
) -> VoiceMeta:
    """voice.yaml: counts, the author list with shares and sources, licenses and documents."""
    total = sum(s.words for s in per_author.values())
    docs = corpus.manifest.documents
    sources_by_author: dict[str, set[str]] = {}
    for d in docs:
        if d.source:
            sources_by_author.setdefault(d.author, set()).add(source_site(d.source))
    authors = [
        AuthorProvenance(
            name=a,
            words=s.words,
            share=round(s.words / total, 4) if total else 0.0,
            documents=sum(1 for d in docs if d.author == a),
            sources=sorted(sources_by_author.get(a, set())),
        )
        for a, s in sorted(per_author.items(), key=lambda kv: -kv[1].words)
    ]
    return VoiceMeta(
        name=name,
        deslop_version=__version__,
        corpus_name=corpus.name,
        corpus_hash=corpus.content_hash(),
        total_words=total,
        document_count=len(docs),
        authors=authors,
        author_shares={a.name: a.share for a in authors},
        documents=[
            DocumentProvenance(
                title=d.title,
                author=d.author,
                words=d.words or 0,
                source=d.source,
                license=d.license,
                year=d.year,
            )
            for d in docs
        ],
        licenses=sorted({d.license for d in docs if d.license}),
        max_share=max_share,
    )


def source_site(source: str) -> str:
    """The host of a URL, or the source note itself."""
    m = re.match(r"https?://([^/]+)", source)
    return m.group(1).removeprefix("www.") if m else source


def document_vector(doc: DocSamples, profile: MetricsProfile, names: list[str]) -> list[float]:
    """A document's chunk-mean metric values as z-scores against the voice."""
    out = []
    for n in names:
        stat = profile.metrics[n]
        mean = doc.metrics.get(n, stat.mean)
        out.append((mean - stat.mean) / stat.std if stat.std > 0 else 0.0)
    return out


def extend(
    base: Voice,
    corpus: Corpus,
    *,
    max_share: float | None = None,
    patterns: tuple[str, ...] = (),
    chunk_words: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> Voice:
    """Fold `corpus` (newly added texts) into `base`'s measurements: per-author and
    per-document samples are merged, the aggregate, registers and thresholds recomputed,
    provenance extended. profile.md, rules.md and exemplars.md are kept as they are."""
    catalog = load_catalog(patterns)
    share = base.meta.max_share if max_share is None else max_share
    chunk = chunk_words or base.metrics.chunk_words
    per_author = {a: s.model_copy(deep=True) for a, s in base.metrics.per_author.items()}
    per_doc = list(base.metrics.per_document)
    for doc in corpus.documents():
        if progress:
            progress(f"sampling {doc.entry.author}: {doc.entry.title}")
        samples = sample_text(doc.text, chunk_words=chunk, catalog=catalog)
        samples.weight = doc.entry.weight
        author = doc.entry.author
        per_author[author] = (
            merge_samples(per_author[author], samples) if author in per_author else samples
        )
        per_author[author].weight = doc.entry.weight
        if samples.chunks > 0:
            per_doc.append(
                doc_samples(
                    samples,
                    title=doc.entry.title,
                    author=author,
                    source=doc.entry.source,
                )
            )
    profile = aggregate(per_author, max_share=share, chunk_words=chunk, per_document=per_doc)
    metric_names = [n for n in sorted(profile.metrics) if profile.metrics[n].std > 0]
    vectors = [
        DocumentVector(
            title=d.title,
            author=d.author,
            words=d.words,
            vector=document_vector(d, profile, metric_names),
        )
        for d in per_doc
    ]
    clusters, _ = cluster_documents(vectors, metric_names)
    profile.clusters = with_voice_means(clusters, profile)
    llm = llm_reference() if has_baseline("llm_reference") else None
    profile.thresholds = calibrate_thresholds(
        profile, llm.documents if llm else [], banned=set(base.lexicon.banned_ids())
    )
    meta = provenance(corpus, per_author, max_share=share, name=base.meta.name)
    meta.profile_source = base.meta.profile_source
    meta.documents = base.meta.documents + meta.documents
    meta.document_count = len(meta.documents)
    meta.licenses = sorted(set(base.meta.licenses) | set(meta.licenses))
    meta.corpus_name = base.meta.corpus_name
    meta.corpus_hash = f"{base.meta.corpus_hash}+{corpus.content_hash()}"
    return Voice(
        meta=meta,
        metrics=profile,
        lexicon=base.lexicon,
        rules=base.rules,
        profile=base.profile,
        exemplars=base.exemplars,
    )


def recalibrate_metrics(voice: Voice, *, max_share: float | None = None) -> Voice:
    """Recompute aggregate stats from the retained per-author samples (no corpus needed)."""
    share = voice.meta.max_share if max_share is None else max_share
    profile = aggregate(
        voice.metrics.per_author,
        max_share=share,
        chunk_words=voice.metrics.chunk_words,
        per_document=voice.metrics.per_document,
    )
    profile.clusters = voice.metrics.clusters
    llm = llm_reference() if has_baseline("llm_reference") else None
    profile.thresholds = calibrate_thresholds(
        profile, llm.documents if llm else [], banned=set(voice.lexicon.banned_ids())
    )
    return voice.model_copy(update={"metrics": profile})


def load_voice(path: Path | str) -> Voice:
    return Voice.load(path)


def add_tells(
    voice: Voice,
    corpus: Corpus,
    *,
    patterns: tuple[str, ...] = (),
    ban_ratio: float | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[str]:
    """Measure on `corpus` every tell the catalog has and `voice` predates, and add only those:
    the voice's rate (author-capped like a build), the starting limit a build would give it,
    and the lexicon's rate against the LLM reference, banned by the same rule. Every other
    measurement, band, threshold and prose file is left as it is. Returns the ids added."""
    from deslop.thresholds import TELL_FLOORS, TELL_STEPS
    from deslop.voice.lexicon import MIN_LLM_RATE_TO_BAN
    from deslop.voice.model import PatternRate

    catalog = load_catalog(patterns)
    missing = [p for p in catalog.patterns if p.id not in voice.metrics.tells]
    if not missing:
        return []
    chunk = voice.metrics.chunk_words
    per_author: dict[str, AuthorSamples] = {}
    for doc in corpus.documents():
        if progress:
            progress(f"sampling {doc.entry.author}: {doc.entry.title}")
        samples = sample_text(doc.text, chunk_words=chunk, catalog=catalog)
        samples.weight = doc.entry.weight
        author = doc.entry.author
        per_author[author] = (
            merge_samples(per_author[author], samples) if author in per_author else samples
        )
        per_author[author].weight = doc.entry.weight
    profile = aggregate(per_author, max_share=voice.meta.max_share, chunk_words=chunk)
    llm = llm_reference() if has_baseline("llm_reference") else None
    ratio_limit = ban_ratio if ban_ratio is not None else voice.lexicon.ban_ratio
    added = []
    for pattern in missing:
        stat = profile.tells.get(pattern.id)
        if stat is None:
            continue
        voice.metrics.tells[pattern.id] = stat
        voice.metrics.thresholds.tells[pattern.id] = max(
            stat.rate_per_1k * TELL_STEPS[0], TELL_FLOORS[0]
        )
        llm_rate = llm.tells[pattern.id].rate_per_1k if llm and pattern.id in llm.tells else None
        ratio = None
        banned = False
        if llm_rate is not None:
            ratio = llm_rate / stat.rate_per_1k if stat.rate_per_1k > 0 else 999.0
            banned = ratio > ratio_limit and llm_rate >= MIN_LLM_RATE_TO_BAN
        voice.lexicon.patterns = [x for x in voice.lexicon.patterns if x.id != pattern.id]
        voice.lexicon.patterns.append(
            PatternRate(
                id=pattern.id,
                name=pattern.name,
                corpus_per_1k=round(stat.rate_per_1k, 3),
                llm_per_1k=None if llm_rate is None else round(llm_rate, 3),
                ratio=None if ratio is None else round(min(ratio, 999.0), 2),
                banned=banned,
            )
        )
        added.append(pattern.id)
    return added
