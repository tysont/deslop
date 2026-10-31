# ABOUTME: score(text, voice): deterministic comparison of a text against a voice profile.
# ABOUTME: Produces a Report with z-scores, tell rates, paragraph deviations, and a verdict.
from __future__ import annotations

from deslop.baselines import has_baseline, llm_reference
from deslop.metrics import METRICS, compute_all, detect_tells
from deslop.nlp import Analysis, analyze
from deslop.patterns import load_catalog
from deslop.report import (
    MetricResult,
    ParagraphResult,
    Report,
    ScoreOptions,
    TellHitResult,
    TellResult,
)
from deslop.stats import chunk_analysis
from deslop.voice.model import Voice

Z_CAP = 3.0


def slop_z(z: float, side: str) -> float:
    """Signed deviation toward the slop side; zero when on the safe side."""
    if side == "high":
        return max(z, 0.0)
    if side == "low":
        return max(-z, 0.0)
    if side == "both":
        return abs(z)
    return 0.0


def score(
    text: str,
    voice: Voice,
    *,
    sigma: float = 1.5,
    tell_mult: float = 2.0,
    tell_floor: float = 1.0,
    patterns: tuple[str, ...] = (),
    analysis: Analysis | None = None,
) -> Report:
    catalog = load_catalog(patterns)
    a = analysis or analyze(text)
    options = ScoreOptions(sigma=sigma, tell_mult=tell_mult, tell_floor=tell_floor)
    llm = llm_reference() if has_baseline("llm_reference") else None
    banned = set(voice.lexicon.banned_ids())

    # ---- metrics: averaged over chunks of the same size the voice was measured on,
    # so length-sensitive metrics (max runs, extremes) stay comparable.
    chunks = chunk_analysis(a, voice.metrics.chunk_words) or [a]
    per_chunk = [compute_all(c) for c in chunks]
    values = {name: sum(v[name] for v in per_chunk) / len(per_chunk) for name in per_chunk[0]}
    metric_results: list[MetricResult] = []
    for name, value in values.items():
        spec = METRICS[name]
        stat = voice.metrics.metrics.get(name)
        if stat is None:
            continue
        z = (value - stat.mean) / stat.std if stat.std > 0 else 0.0
        flagged = slop_z(z, spec.slop_side) > sigma
        metric_results.append(
            MetricResult(
                name=name,
                group=spec.group,
                value=value,
                voice_mean=stat.mean,
                voice_std=stat.std,
                z=z,
                slop_side=spec.slop_side,
                flagged=flagged,
            )
        )

    # ---- tells
    hits = detect_tells(a, catalog)
    per_1k = a.words_per_1k
    tell_results: list[TellResult] = []
    for pattern in catalog.patterns:
        p_hits = [h for h in hits if h.pattern_id == pattern.id]
        count = len(p_hits)
        rate = count / per_1k
        voice_rate = (
            voice.metrics.tells[pattern.id].rate_per_1k
            if pattern.id in voice.metrics.tells
            else 0.0
        )
        llm_rate = llm.tells[pattern.id].rate_per_1k if llm and pattern.id in llm.tells else None
        threshold = max(voice_rate * tell_mult, tell_floor)
        if pattern.id in banned:
            threshold = min(threshold, tell_floor)
        tell_results.append(
            TellResult(
                id=pattern.id,
                name=pattern.name,
                count=count,
                rate_per_1k=rate,
                voice_rate=voice_rate,
                llm_rate=llm_rate,
                threshold=threshold,
                ratio_vs_voice=(rate / voice_rate) if voice_rate > 0 else None,
                banned=pattern.id in banned,
                flagged=rate > threshold,
                hits=[
                    TellHitResult(
                        pattern_id=h.pattern_id,
                        sentence_idx=h.sentence_idx,
                        para_idx=h.para_idx,
                        start=h.start,
                        end=h.end,
                        text=h.text,
                        note=h.note,
                    )
                    for h in p_hits
                ],
            )
        )

    # ---- paragraphs
    flagged_ids = {t.id for t in tell_results if t.flagged}
    para_results: list[ParagraphResult] = []
    for p in a.paragraphs:
        p_hits = [h for h in hits if h.para_idx == p.index]
        weighted = [h for h in p_hits if h.pattern_id in flagged_ids or h.pattern_id in banned]
        words = max(p.word_count, 1)
        para_results.append(
            ParagraphResult(
                index=p.index,
                start=p.start,
                end=p.end,
                words=p.word_count,
                deviation=100.0 * len(weighted) / words,
                flagged_sentences=sorted({h.sentence_idx for h in weighted}),
                tells=sorted({h.pattern_id for h in p_hits}),
            )
        )

    # ---- verdict, fixes, distance
    flagged_metrics = [m for m in metric_results if m.flagged]
    flagged_tells = [t for t in tell_results if t.flagged]
    verdict = "PASS" if not flagged_metrics and not flagged_tells else "FAIL"
    fixes: list[str] = []
    for t in sorted(flagged_tells, key=lambda t: -(t.rate_per_1k / max(t.threshold, 1e-9))):
        fixes.append(
            f"{t.id} {t.name}: {t.count} hits ({t.rate_per_1k:.1f}/1k; "
            f"voice {t.voice_rate:.1f}/1k, "
            f"limit {t.threshold:.1f})"
        )
    for m in sorted(flagged_metrics, key=lambda m: -slop_z(m.z, m.slop_side)):
        direction = "too high" if m.z > 0 else "too low"
        fixes.append(
            f"{m.name} {direction}: {m.value:.2f} vs voice {m.voice_mean:.2f} (z={m.z:+.1f})"
        )

    slop_metrics = [m for m in metric_results if m.slop_side != "none"]
    metric_component = (
        sum(min(slop_z(m.z, m.slop_side), Z_CAP) / Z_CAP for m in slop_metrics) / len(slop_metrics)
        if slop_metrics
        else 0.0
    )
    tell_component = (
        sum(min(t.rate_per_1k / max(t.threshold, 1e-9), 2.0) / 2.0 for t in tell_results)
        / len(tell_results)
        if tell_results
        else 0.0
    )
    distance = round(100.0 * (0.5 * metric_component + 0.5 * tell_component), 1)

    return Report(
        voice=voice.name,
        verdict=verdict,
        distance=distance,
        words=a.word_count,
        sentences=len(a.sentences),
        paragraphs_count=len(a.paragraphs),
        metrics=metric_results,
        tells=tell_results,
        paragraphs=para_results,
        fixes=fixes,
        options=options,
    )
