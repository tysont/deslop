# ABOUTME: Compiles a corpus into a Voice: metrics profile, lexicon, rules, profile, exemplars.
# ABOUTME: Deterministic except for the optional LLM-written profile.md.
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from deslop import __version__
from deslop.baselines import english_generic, has_baseline, llm_reference
from deslop.corpus import Corpus
from deslop.patterns import load_catalog
from deslop.providers.base import CachingProvider
from deslop.stats import (
    DEFAULT_CHUNK_WORDS,
    AuthorSamples,
    DocSamples,
    MetricsProfile,
    WordCounts,
    aggregate,
    doc_samples,
    merge_samples,
    sample_text,
)
from deslop.voice.clusters import DocumentVector, cluster_documents, with_voice_means
from deslop.voice.exemplars import build_exemplars
from deslop.voice.lexicon import build_lexicon
from deslop.voice.model import Voice, VoiceMeta
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
    rules = build_rules(profile, generic, lexicon)
    if progress:
        progress("selecting exemplars")
    exemplars = build_exemplars(
        [(d.entry.author, d.entry.title, d.text) for d in docs],
        profile,
        target_words=exemplar_words,
        max_share=max_share,
        registers=registers,
    )
    voice_name = name or corpus.name
    profile_source = "template"
    if use_llm and provider is None:
        from deslop.providers.base import first_available

        provider = first_available()
    if use_llm and provider is not None:
        if progress:
            progress(f"writing profile with {provider.name}")
        profile_md = llm_profile(
            voice_name, profile, generic, exemplars, rules, provider, llm_model, llm=llm
        )
        profile_source = f"{provider.name}:{llm_model or provider.default_model}"
    else:
        profile_md = template_profile(voice_name, profile, generic, llm)

    total = sum(s.words for s in per_author.values())
    meta = VoiceMeta(
        name=voice_name,
        deslop_version=__version__,
        corpus_name=corpus.name,
        corpus_hash=corpus.content_hash(),
        corpus_path=str(corpus.root.resolve()),
        total_words=total,
        author_shares={a: round(s.words / total, 4) for a, s in per_author.items()},
        max_share=max_share,
        profile_source=profile_source,
    )
    return Voice(
        meta=meta,
        metrics=profile,
        lexicon=lexicon,
        rules=rules,
        profile=profile_md,
        exemplars=exemplars,
    )


def document_vector(doc: DocSamples, profile: MetricsProfile, names: list[str]) -> list[float]:
    """A document's chunk-mean metric values as z-scores against the voice."""
    out = []
    for n in names:
        stat = profile.metrics[n]
        mean = doc.metrics.get(n, stat.mean)
        out.append((mean - stat.mean) / stat.std if stat.std > 0 else 0.0)
    return out


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
    return voice.model_copy(update={"metrics": profile})


def load_voice(path: Path | str) -> Voice:
    return Voice.load(path)
