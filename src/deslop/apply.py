# ABOUTME: apply(text_or_brief, voice): generate-then-verify rewriting of a document as a whole.
# ABOUTME: n candidates must pass the locks, coherence checks and a four-item judge, else it stays.
from __future__ import annotations

import datetime as dt
import difflib
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from deslop.citations import Citations, citation_policy_notes, permit_primary
from deslop.coherence import check_coherence
from deslop.continuity import check_continuity, last_paragraphs
from deslop.draft import Brief, SectionLog, draft_sections
from deslop.home import deslop_home
from deslop.judge import (
    compare_document,
    document_win_rate,
)
from deslop.lock import DocumentLock, check
from deslop.metrics import detect_tells
from deslop.nlp import Analysis, analyze, split_paragraphs
from deslop.patterns import Catalog, load_catalog
from deslop.prompts import clean_response, document_prompt, edit_prompt, length_bounds
from deslop.providers.base import (
    DEFAULT_JUDGE,
    DEFAULT_WRITER,
    CachingProvider,
    ProviderError,
)
from deslop.providers.roles import Models, ProviderFactory, build_models
from deslop.recitation import DEFAULT_MAX_RECITATION, NGRAM, recitation
from deslop.reference import (
    CitationScheme,
    PrimaryText,
    PrimaryTextSpec,
    citation_guidance,
    load_primary_texts,
    primary_citations,
    quotable_passages,
    quoted_spans,
    verify_additions,
)
from deslop.report import Report
from deslop.result import (
    DEFAULT_MAX_GROWTH,
    CandidateLog,
    IterationLog,
    Result,
    UnitLog,
    cost_of,
    meaning_diff,
    metric_table,
    unverified_additions,
    verified_additions,
)
from deslop.score import score
from deslop.semantic import (
    Fidelity,
    PropositionCheck,
    SemanticResult,
    check_meaning,
    extract_propositions,
    extract_texture,
)
from deslop.stats import Band
from deslop.units import (
    DEFAULT_CHUNK_WORDS,
    Section,
    Unit,
    edit_units,
    metric_notes_for,
    sections_of,
)
from deslop.voice.exemplars import ExemplarIndex
from deslop.voice.model import Voice

Mode = Literal["rewrite", "edit"]


def pairs_dir() -> Path:
    """Where accepted and rejected rewrite pairs are appended: $DESLOP_HOME/pairs."""
    return deslop_home() / "pairs"


CANDIDATE_TEMPERATURE = 0.9

ROUNDS = 2  # the first round, then one retry per draft carrying what a reader noticed in it

ADVISORY_PARAGRAPHS = 10  # under this many input paragraphs the paragraph band only advises

DEFAULT_QUOTE_CONTEXT_WORDS = 1500  # cap on the verbatim passages offered to the writer


def max_words_for(words: int, max_growth: float) -> int:
    return int(math.floor(words * (1 + max_growth / 100.0)))


def growth_of(before: str, after: str) -> float:
    """Word-count growth of `after` over `before`, in percent."""
    a, b = len(before.split()), len(after.split())
    return round(100.0 * (b - a) / max(a, 1), 1)


def _splice(text: str, units: list[Unit], replacements: dict[int, str]) -> str:
    out = []
    pos = 0
    for u in sorted(units, key=lambda u: u.start):
        out.append(text[pos : u.start])
        out.append(replacements.get(u.index, u.text))
        pos = u.end
    out.append(text[pos:])
    return "".join(out)


def _tell_counts(text: str, catalog: Catalog) -> dict[str, int]:
    counts: dict[str, int] = {}
    for h in detect_tells(analyze(text), catalog):
        counts[h.pattern_id] = counts.get(h.pattern_id, 0) + 1
    return counts


def _new_tells(before_doc: str, after_doc: str, catalog: Catalog) -> list[str]:
    before = _tell_counts(before_doc, catalog)
    after = _tell_counts(after_doc, catalog)
    return sorted(pid for pid, n in after.items() if n > before.get(pid, 0))


def _log_pairs(voice: Voice, units: list[UnitLog], path: Path | None) -> None:
    target = path or (pairs_dir() / f"{voice.name}.jsonl")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as fh:
            for unit in units:
                for cand in unit.candidates:
                    record = {
                        "voice": voice.name,
                        "timestamp": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
                        "kind": unit.kind,
                        "before": unit.original,
                        "after": cand.text,
                        "provider": cand.provider,
                        "model": cand.model,
                        "accepted": cand.accepted,
                        "reason": cand.reason,
                    }
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass


def apply(
    text_or_brief: str | Brief,
    voice: Voice,
    *,
    mode: Mode = "rewrite",
    writer: str = DEFAULT_WRITER,
    judge: str = DEFAULT_JUDGE,
    provider: str | None = None,
    budget: float | None = None,
    n: int = 3,
    max_growth: float = DEFAULT_MAX_GROWTH,
    chunk_words: int = DEFAULT_CHUNK_WORDS,
    primary_texts: list[str | Path | PrimaryText | PrimaryTextSpec] | None = None,
    citation_schemes: Sequence[CitationScheme] = (),
    fidelity: Fidelity = "content",
    report_all: bool = False,
    reference: str | None = None,
    reference_name: str = "reference",
    max_recitation: float = DEFAULT_MAX_RECITATION,
    allow_primary_facts: bool = False,
    quote_context_words: int = DEFAULT_QUOTE_CONTEXT_WORDS,
    citations: Citations = "verified",
    preceding: str | None = None,
    following: str | None = None,
    max_iter: int = 4,
    preserve_length: float = 0.15,
    patterns: tuple[str, ...] = (),
    provider_factory: ProviderFactory | None = None,
    pairs_path: Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> Result:
    """Write `text_or_brief` in `voice` and return a [`Result`][deslop.Result].

    Rewrite mode (the default) writes the document as a whole. Every paragraph's
    propositions are extracted (cached), then `n` candidates are sampled from the `writer`
    model, which sees the profile, the rules, the exemplars nearest the piece, the full
    original and the propositions by paragraph. A candidate must pass:

    - the content lock (nothing locked is removed) and the length limit (`max_growth`
      percent);
    - the primary-text check, and the `citations` policy;
    - the semantic lock, paragraph by paragraph, run by the `judge` model: every proposition
      kept and nothing material added;
    - the coherence checks: the same tense, and every sentence-initial pronoun with an
      antecedent (the paragraph band is enforced only from `ADVISORY_PARAGRAPHS` input
      paragraphs, and the connective band only advises);
    - recitation of `reference`, when given;
    - the continuity checks against `preceding`, when given;
    - the `judge` model's comparison with the original on four items (voice, coherence,
      tense and person, order and emphasis), plus "same writer continuing" when `preceding`
      is given: a win or a tie on every one.

    Survivors are judged round-robin and the best win rate wins. If none passes, a second
    round redrafts each candidate with only the notes that draft earned; if that fails too,
    the result is the original with the report. A document longer than `chunk_words` is
    rewritten section by section.

    Edit mode rewrites only the flagged sentences, under the content lock and the
    primary-text check, with the meaning checked when the lock sees additions. A `Brief` is
    drafted section by section first, then rewritten.

    `primary_texts` (files, directories, `PrimaryTextSpec`s or loaded `PrimaryText`s) come
    with `citation_schemes` for those that declare none; without any, added quotations and
    citations are allowed and listed as unverified. `reference` is the human text the input
    was made from: it is never shown to a model, only compared against. `preceding` is the
    already-converted section before the input and `following` the unconverted one after it;
    both are context and neither is rewritten. `writer` and `judge` are model names or
    aliases (`opus`, `sonnet`, `haiku`) and must differ; `provider` is `anthropic` (the API)
    or `claude-cli`, by default from DESLOP_PROVIDER. `budget` is USD. `provider_factory`
    maps a provider name to a `CachingProvider` (tests replay a cache with it).
    """
    catalog = load_catalog(patterns)
    models = build_models(
        writer,
        judge,
        provider=provider,
        factory=provider_factory,
        calibration=voice.meta.judge_calibration,
    )
    draft_logs: list[SectionLog] | None = None
    draft_text: str | None = None
    if isinstance(text_or_brief, Brief):
        text, draft_logs = draft_sections(text_or_brief, voice, models.writer, progress=progress)
        draft_text = text
        mode = "rewrite"
    else:
        text = text_or_brief
    texts = _load_texts(list(primary_texts or []), citation_schemes)
    if progress:
        progress(models.describe())
        if texts:
            progress("primary texts: " + ", ".join(t.name for t in texts))
    report_before = score(text, voice, patterns=patterns)
    exemplars = ExemplarIndex(voice)
    document = DocumentLock.build(text)
    opts = Options(
        models=models,
        texts=texts,
        max_growth=max_growth,
        chunk_words=chunk_words,
        patterns=patterns,
        n=n,
        progress=progress,
        fidelity=fidelity,
        report_all=report_all,
        reference=reference,
        max_recitation=max_recitation,
        allow_primary_facts=allow_primary_facts,
        quote_context_words=quote_context_words,
        citations=citations,
        preceding=preceding.strip() if preceding else None,
        following=following.strip() if following else None,
    )

    if mode == "edit":
        final_text, units, iterations, stopped = _edit_loop(
            text,
            voice,
            report_before,
            exemplars,
            document,
            catalog,
            opts,
            budget=budget,
            max_iter=max_iter,
            preserve_length=preserve_length,
        )
    else:
        final_text, units, iterations, stopped = _rewrite_loop(
            text,
            voice,
            report_before,
            exemplars,
            document,
            catalog,
            opts,
            budget=budget,
        )
    report_after = score(final_text, voice, patterns=patterns)
    _log_pairs(voice, units, pairs_path)
    diff = "".join(
        difflib.unified_diff(
            text.splitlines(keepends=True),
            final_text.splitlines(keepends=True),
            fromfile="before",
            tofile="after",
        )
    )
    winner = next(
        (u.candidates[u.chosen] for u in units if not u.kept_original and u.chosen is not None),
        None,
    )
    vs = winner.vs_original if winner is not None else None
    return Result(
        mode=mode,
        text=final_text,
        report_before=report_before,
        report_after=report_after,
        diff=diff,
        meaning_diff=meaning_diff(units),
        units=units,
        iterations=iterations,
        cost=cost_of(models.usage),
        stopped=stopped,
        exemplars_used=exemplars.available,
        max_growth=max_growth,
        chunk_words=chunk_words,
        growth_pct=growth_of(text, final_text),
        coherence=winner.coherence if winner is not None else None,
        continuity=winner.continuity if winner is not None else None,
        preceding_given=preceding is not None,
        following_given=following is not None,
        judge_items=dict(vs.items) if vs is not None else {},
        metric_table=metric_table(report_before, report_after),
        unverified_additions=unverified_additions(units),
        verified_additions=verified_additions(units),
        primary_texts=[t.name for t in texts],
        judge_gate=models.describe(),
        fidelity=fidelity,
        report_all=report_all,
        draft=draft_logs,
        draft_text=draft_text,
        reference=reference_name if reference is not None else None,
        max_recitation=max_recitation,
        allow_primary_facts=allow_primary_facts,
        citations=citations,
        recitation=winner.recitation if winner is not None else None,
    )


@dataclass
class Options:
    """Everything the unit loops share: the models, the primary texts, the gates."""

    models: Models
    texts: list[PrimaryText]
    max_growth: float
    chunk_words: int
    patterns: tuple[str, ...]
    n: int
    progress: Callable[[str], None] | None
    fidelity: Fidelity = "content"
    report_all: bool = False
    reference: str | None = None  # never shown to a model; compared against only
    max_recitation: float = DEFAULT_MAX_RECITATION
    allow_primary_facts: bool = False  # facts verifiable in a primary text are not additions
    quote_context_words: int = DEFAULT_QUOTE_CONTEXT_WORDS
    citations: Citations = "verified"  # keep | verified | none
    preceding: str | None = None  # the converted section before; never rewritten
    following: str | None = None  # the unconverted section after; context only


def _load_texts(
    sources: list[str | Path | PrimaryText | PrimaryTextSpec],
    schemes: Sequence[CitationScheme] = (),
) -> list[PrimaryText]:
    loaded = [t for t in sources if isinstance(t, PrimaryText)]
    names = [t for t in sources if not isinstance(t, PrimaryText)]
    return loaded + load_primary_texts(names, schemes=schemes)


def _rewrite_loop(
    text: str,
    voice: Voice,
    report_before: Report,
    exemplars: ExemplarIndex,
    document: DocumentLock,
    catalog: Catalog,
    opts: Options,
    *,
    budget: float | None,
) -> tuple[str, list[UnitLog], list[IterationLog], str]:
    """Rewrite the document as a whole, or section by section when it is longer than
    `chunk_words`. Every section must find a passing candidate, else the whole original is
    returned: a partial rewrite is never produced."""
    progress = opts.progress
    a = analyze(text)
    sections = sections_of(a, opts.chunk_words)
    if not sections:
        return text, [], [], "nothing_to_rewrite"
    writer0 = opts.models.writer
    try:
        by_paragraph = [
            extract_propositions(p.text, writer0, fidelity=opts.fidelity) for p in a.paragraphs
        ]
        textures = [extract_texture(p.text, writer0) for p in a.paragraphs]
    except ProviderError as exc:
        kept = [
            UnitLog(
                index=sec.index,
                kind=sec.kind,
                start=sec.start,
                end=sec.end,
                original=text[sec.start : sec.end],
                final=text[sec.start : sec.end],
                kept_original=True,
                reason=f"propositions could not be extracted: {exc}"[:200],
            )
            for sec in sections
        ]
        return text, kept, [], "budget" if "budget" in str(exc) else "provider_error"
    if progress and len(sections) > 1:
        progress(f"{len(sections)} sections of up to {opts.chunk_words} words")
    logs: list[UnitLog] = []
    pieces: list[str] = []  # rewritten section texts, in order
    pos = 0
    for section in sections:
        if budget is not None and opts.models.usage.cost_usd > budget:
            return (
                _revert(text, a, sections, by_paragraph, logs, "budget exhausted"),
                logs,
                [],
                "budget",
            )
        section_text = text[section.start : section.end]
        props = by_paragraph[section.first : section.last + 1]
        previous = pieces[-1] if pieces else (opts.preceding or "")
        following = (
            text[sections[section.index + 1].start : sections[section.index + 1].end]
            if section.index + 1 < len(sections)
            else (opts.following or "")
        )
        log = _rewrite_section(
            section,
            text,
            a,
            section_text,
            props,
            voice,
            exemplars,
            document,
            catalog,
            opts,
            previous=previous,
            following=following,
            report_before=report_before,
            textures=textures[section.first : section.last + 1],
            seam=opts.preceding if section.index == 0 else None,
        )
        logs.append(log)
        if log.kept_original:
            return (
                _revert(text, a, sections, by_paragraph, logs, "no candidate passed"),
                logs,
                [],
                "no_candidate_passed",
            )
        pieces.append(text[pos : section.start] + log.final)
        pos = section.end
    final = "".join(pieces) + text[pos:]
    return final, logs, [], "done"


def _revert(
    text: str,
    a: Analysis,
    sections: list[Section],
    by_paragraph: list[list[str]],
    logs: list[UnitLog],
    reason: str,
) -> str:
    """Mark every section as kept (the ones already rewritten included) and return the
    original text."""
    done = {log.index for log in logs}
    for log in logs:
        if not log.kept_original:
            log.final = log.original
            log.kept_original = True
            log.reason = f"reverted: {reason} in a later section"
    for section in sections:
        if section.index in done:
            continue
        props = by_paragraph[section.first : section.last + 1]
        logs.append(
            UnitLog(
                index=section.index,
                kind=section.kind,
                start=section.start,
                end=section.end,
                original=text[section.start : section.end],
                final=text[section.start : section.end],
                propositions=[p for ps in props for p in ps],
                propositions_by_paragraph=props,
                kept_original=True,
                reason=f"not attempted: {reason}",
            )
        )
    return text


def _lock_notes(detail: str) -> list[str]:
    return [f"the {part.strip()} from the original" for part in detail.split(";") if part.strip()]


def _linter_notes(
    before_doc: str, spliced: str, report: Report, span: tuple[int, int], catalog: Catalog
) -> list[str]:
    """What a reader would say about the linter's findings inside `span` of `spliced`: each
    new tell with its span and the pattern's plain description, and each flagged metric
    phrased without its name."""
    notes = []
    before_counts = _tell_counts(before_doc, catalog)
    for t in report.tells:
        if not (t.flagged or t.banned) or t.count <= before_counts.get(t.id, 0):
            continue
        for h in t.hits:
            if span[0] <= h.start < span[1]:
                desc = catalog.by_id(h.pattern_id).description.rstrip(".")
                notes.append(f'"{h.text.strip()[:90]}": {desc[:1].lower()}{desc[1:]}')
                break
    notes.extend(metric_notes_for(report))
    return notes


def _verify_section(
    section_text: str,
    candidate: str,
    cand: CandidateLog,
    document: DocumentLock,
    voice: Voice,
    opts: Options,
) -> list[str]:
    """The checks that need no judge: shape, length, lexical lock, primary texts, coherence,
    the citation policy and recitation. Returns what a reader noticed (empty when the
    candidate may go on)."""
    notes: list[str] = []
    if not candidate.strip() or candidate.strip() == section_text.strip():
        cand.lexical_ok = False
        return ["the draft was empty or unchanged"]
    cand.growth_pct = growth_of(section_text, candidate)
    words, limit = len(candidate.split()), max_words_for(len(section_text.split()), opts.max_growth)
    if words > limit:
        notes.append(f"it runs {words} words against the original's {len(section_text.split())}")
    lock = check(section_text, candidate, document)
    if not lock.ok:
        cand.lexical_ok = False
        cand.lexical_detail = "; ".join(str(v) for v in lock.violations)
        notes.extend(_lock_notes(cand.lexical_detail))
    elif lock.additions:
        cand.lexical_detail = "new items for the semantic lock: " + "; ".join(lock.additions)
    cand.primary = verify_additions(section_text, candidate, opts.texts)
    if not cand.primary.ok:
        notes.extend(f"added {add.kind} {add}" for add in cand.primary.rejected[:3])
    band = voice.metrics.thresholds.metrics.get("connective_opener_pct")
    stat = voice.metrics.metrics.get("para_len_mean")
    para_band = Band(low=stat.p10, high=stat.p90) if stat is not None else None
    cand.coherence = check_coherence(section_text, candidate, band=band, para_band=para_band)
    paragraphs = cand.coherence.paragraphs
    if not paragraphs.ok and len(split_paragraphs(section_text)) < ADVISORY_PARAGRAPHS:
        paragraphs.advisory = True
        cand.advisories.append(paragraphs.note)
    connectives = cand.coherence.connectives
    if not connectives.ok:
        connectives.advisory = True  # the judge's coherence item decides whether it matters
        cand.connective_note = connectives.note
    notes.extend(cand.coherence.reader_notes())
    notes.extend(citation_policy_notes(section_text, candidate, opts.texts, opts.citations))
    if opts.reference is not None:
        cand.recitation = recitation(candidate, opts.reference, section_text)
        if cand.recitation.novel_share > opts.max_recitation:
            # The spans stay in the log for the report; the writer is told only that it recited.
            notes.append(
                f"recitation: {cand.recitation.novel_share:.1f}% of its words recited a "
                f"published text in runs of {NGRAM} words or more; it must be put in the "
                "writer's own words"
            )
    return notes


def _check_meaning_by_paragraph(
    section_text: str,
    candidate: str,
    props: list[list[str]],
    checker: CachingProvider,
    fidelity: Fidelity = "content",
) -> SemanticResult:
    """The semantic lock paragraph by paragraph when the candidate keeps the paragraphing
    (each check sees the original's neighbouring paragraphs, so what they state is not new),
    else once over the whole section; the checks are merged with a paragraph prefix."""
    before, after = analyze(section_text).paragraphs, analyze(candidate).paragraphs
    merged = SemanticResult()
    if len(before) == len(after) == len(props):
        for i, (p, q, ps) in enumerate(zip(before, after, props, strict=True), 1):
            context = (
                before[i - 2].text if i >= 2 else "",
                before[i].text if i < len(before) else "",
            )
            r = check_meaning(p.text, q.text, ps, checker, fidelity=fidelity, context=context)
            merged.checks.extend(
                PropositionCheck(proposition=f"¶{i} {c.proposition}", kept=c.kept, note=c.note)
                for c in r.checks
            )
            merged.added.extend(r.added)
            merged.cleared.extend(r.cleared)
            merged.provider, merged.model = r.provider, r.model
        return merged
    return check_meaning(
        section_text, candidate, [p for ps in props for p in ps], checker, fidelity=fidelity
    )


def _rewrite_section(
    section: Section,
    text: str,
    a: Analysis,
    section_text: str,
    props: list[list[str]],
    voice: Voice,
    exemplars: ExemplarIndex,
    document: DocumentLock,
    catalog: Catalog,
    opts: Options,
    *,
    previous: str,
    following: str,
    report_before: Report,
    textures: Sequence[str] = (),
    seam: str | None = None,
) -> UnitLog:
    models, n, progress = opts.models, opts.n, opts.progress
    writer, judge = models.writer, models.judge
    log = UnitLog(
        index=section.index,
        kind=section.kind,
        start=section.start,
        end=section.end,
        original=section_text,
        final=section_text,
        propositions=[p for ps in props for p in ps],
        propositions_by_paragraph=props,
        texture_by_paragraph=list(textures),
    )
    near = exemplars.closest(section_text, 2)
    passages = quotable_passages(section_text, opts.texts, limit_words=opts.quote_context_words)
    guidance = citation_guidance(opts.texts)
    reproduce = (
        primary_citations(section_text, opts.texts) + [f'"{q}"' for q in quoted_spans(section_text)]
        if opts.citations != "verified"
        else []
    )
    log.quotable = [{"text": p.text_name, "ref": p.ref, "words": p.words} for p in passages]
    if progress and passages:
        progress(
            f"  quotable passages: {len(passages)} ({sum(p.words for p in passages)} words) "
            + ", ".join(f"{p.text_name} {p.ref}" for p in passages[:6])
            + (" ..." if len(passages) > 6 else "")
        )
    noticed_by_draft: list[list[str]] = []  # round two's draft k hears only draft k's notes
    for round_index in range(ROUNDS):
        eligible: list[CandidateLog] = []
        for k in range(n):
            attempt = round_index * n + k + 1
            noticed = noticed_by_draft[k] if k < len(noticed_by_draft) else []
            system, user = document_prompt(
                section,
                text,
                section_text,
                props,
                voice,
                exemplars,
                previous=previous,
                following=following,
                attempt=attempt,
                n=ROUNDS * n,
                noticed=noticed,
                textures=textures,
                passages=passages,
                guidance=guidance,
                reproduce=reproduce,
                citations=opts.citations,
            )
            try:
                completion = writer.complete(system, user, temperature=CANDIDATE_TEMPERATURE)
            except ProviderError as exc:
                log.candidates.append(
                    CandidateLog(
                        attempt=attempt,
                        text="",
                        provider=writer.name,
                        model=writer.default_model,
                        lexical_ok=False,
                        reason=f"provider error: {exc}"[:200],
                    )
                )
                continue
            candidate = clean_response(completion.text)
            cand = CandidateLog(
                attempt=attempt,
                text=candidate,
                provider=completion.provider,
                model=completion.model,
                lexical_ok=True,
            )
            log.candidates.append(cand)
            notes = _verify_section(section_text, candidate, cand, document, voice, opts)
            if seam:
                spreads = {k: v.std for k, v in voice.metrics.metrics.items()}
                cand.continuity = check_continuity(seam, candidate, spreads=spreads)
                notes.extend(cand.continuity.reader_notes())
            if not notes or opts.report_all:
                try:
                    cand.semantic = _check_meaning_by_paragraph(
                        section_text, candidate, props, judge, opts.fidelity
                    )
                except ProviderError as exc:
                    cand.reason = f"meaning check unavailable: {exc}"[:200]
                    continue
                permit_primary(cand, opts.texts, opts.allow_primary_facts, section_text)
                if not cand.semantic.ok:
                    notes.extend(
                        f'the claim "{c.proposition}" was dropped'
                        for c in cand.semantic.checks
                        if not c.kept
                    )
                    notes.extend(
                        f'"{x.text}" is a claim the original does not make ({x.category})'
                        if x.category
                        else f'"{x.text}" is a claim the original does not make'
                        for x in cand.semantic.added
                        if x.kind == "claim"
                    )
            spliced = text[: section.start] + candidate + text[section.end :]
            report = score(spliced, voice, patterns=opts.patterns)
            cand.distance = report.distance
            cand.new_tells = _new_tells(text, spliced, catalog)
            linter = _linter_notes(
                text, spliced, report, (section.start, section.start + len(candidate)), catalog
            )
            failed = bool(notes)
            if failed:
                cand.reason = "; ".join(notes)[:300]
                cand.reader_notes = _retry_notes(cand, notes, linter)
                if not opts.report_all:
                    continue
            else:
                cand.reader_notes = _retry_notes(cand, [], linter)
            try:
                cand.verdicts.append(
                    compare_document(
                        candidate,
                        section_text,
                        section_text,
                        voice,
                        judge,
                        exemplars=near,
                        preceding=last_paragraphs(seam) if seam else None,
                    )
                )
            except ProviderError as exc:
                if not failed:
                    cand.reason = f"judge unavailable: {exc}"[:200]
                continue
            vs = cand.vs_original
            cand.judge_ok = vs is not None and vs.no_loss
            cand.reader_notes = _retry_notes(cand, notes, linter)
            if failed:
                continue  # judged for the record only
            if not cand.judge_ok:
                lost = [k for k, v in (vs.items if vs else {}).items() if v == "loss"]
                note = vs.note if vs else ""
                cand.reason = f"the judge preferred the original for {', '.join(lost)}: {note}"[
                    :300
                ]
                cand.reader_notes = [
                    f"a second reader preferred the original for {', '.join(lost)}: {note}",
                    *_retry_notes(cand, [], linter),
                ]
                continue
            eligible.append(cand)

        for i, x in enumerate(eligible):
            for y in eligible[i + 1 :]:
                try:
                    v = compare_document(
                        x.text,
                        y.text,
                        section_text,
                        voice,
                        judge,
                        exemplars=near,
                        against=f"candidate {y.attempt}",
                        preceding=last_paragraphs(seam) if seam else None,
                    )
                except ProviderError:
                    continue  # the pair stays uncompared
                x.verdicts.append(v)
                y.verdicts.append(v.flipped(f"candidate {x.attempt}"))
        for cand in eligible:
            cand.win_rate = document_win_rate(cand.verdicts)
            cand.accepted = True
            cand.reason = "passed the locks, the coherence checks and the judge"
        if eligible:
            winner = min(eligible, key=_rank_key)
            log.chosen = log.candidates.index(winner)
            log.final = winner.text
            rate = f"{100 * winner.win_rate:.0f}%" if winner.win_rate is not None else "n/a"
            log.reason = (
                f"candidate {log.chosen + 1} of {len(log.candidates)}: win rate {rate} over "
                f"{len(winner.verdicts)} comparison(s), distance {winner.distance:.1f}, "
                f"growth {winner.growth_pct:+.0f}%"
            )
            if progress:
                progress(f"  {section.kind}: {log.reason}")
            return log
        if round_index + 1 < ROUNDS:
            this_round = log.candidates[round_index * n : (round_index + 1) * n]
            noticed_by_draft = [list(dict.fromkeys(c.reader_notes))[-12:] for c in this_round]
    log.kept_original = True
    log.reason = "no candidate passed: " + "; ".join(
        f"{c.attempt}: {c.reason[:80]}" for c in log.candidates
    )
    if progress:
        progress(f"  {section.kind}: kept original (no candidate passed)")
    return log


def _retry_notes(cand: CandidateLog, hard: list[str], linter: list[str]) -> list[str]:
    """What the retry prompt is told about this candidate. The hard failures (lock, length,
    meaning, tense, antecedents, primary texts, citation policy, continuity, recitation)
    always; the linter's finds, the paragraph band and the connective band only when the
    judge found something to lose: a candidate the judge rated win or tie on every item
    hears only its hard failures, and the connective band only when the judge's coherence
    item was a loss. With no verdict yet, everything."""
    vs = cand.vs_original
    soft = linter + cand.advisories
    connective = [cand.connective_note] if cand.connective_note else []
    if vs is None:
        return hard + soft + connective
    if vs.no_loss:
        return list(hard)
    if vs.items.get("coherent") != "loss":
        connective = []
    return hard + soft + connective


def _rank_key(cand: CandidateLog) -> tuple[float, float]:
    """Comparative win rate first, then linter distance."""
    rate = cand.win_rate if cand.win_rate is not None else -1.0
    distance = cand.distance if cand.distance is not None else 1e9
    return (-rate, distance)


def _edit_loop(
    text: str,
    voice: Voice,
    report_before: Report,
    exemplars: ExemplarIndex,
    document: DocumentLock,
    catalog: Catalog,
    opts: Options,
    *,
    budget: float | None,
    max_iter: int,
    preserve_length: float,
) -> tuple[str, list[UnitLog], list[IterationLog], str]:
    patterns, progress = opts.patterns, opts.progress
    writer, judge = opts.models.writer, opts.models.judge
    iterations: list[IterationLog] = []
    logs: list[UnitLog] = []
    rejected_for: dict[str, list[tuple[str, str]]] = {}
    current = text
    report = report_before
    best_text, best_report = text, report_before
    stopped = "max_iter"
    if report_before.verdict == "PASS":
        return text, logs, iterations, "pass"
    for i in range(max_iter):
        a = analyze(current)
        units = edit_units(current, a, report)
        if not units:
            stopped = "nothing_flagged"
            break
        if progress:
            progress(f"iteration {i + 1}: {len(units)} edit unit(s) via {writer.key}")
        replacements: dict[int, str] = {}
        for unit in units:
            if budget is not None and opts.models.usage.cost_usd > budget:
                stopped = "budget"
                break
            para = a.paragraphs[unit.para_index]
            ctx_before = current[para.start : unit.start]
            ctx_after = current[unit.end : para.end]
            log = UnitLog(
                index=len(logs),
                kind=unit.kind,
                start=unit.start,
                end=unit.end,
                original=unit.text,
                final=unit.text,
            )
            for attempt in range(2):
                system, user = edit_prompt(
                    unit,
                    current,
                    a,
                    voice,
                    exemplars,
                    document,
                    catalog,
                    preserve_length,
                    rejected_for.get(unit.text, []),
                )
                try:
                    completion = writer.complete(system, user, temperature=0.7)
                except ProviderError as exc:
                    log.candidates.append(
                        CandidateLog(
                            attempt=attempt + 1,
                            text="",
                            provider=writer.name,
                            model=writer.default_model,
                            lexical_ok=False,
                            reason=f"provider error: {exc}"[:200],
                        )
                    )
                    continue
                candidate = clean_response(completion.text)
                cand = CandidateLog(
                    attempt=attempt + 1,
                    text=candidate,
                    provider=completion.provider,
                    model=completion.model,
                    lexical_ok=True,
                )
                reason = _edit_reject_reason(
                    unit, candidate, ctx_before, ctx_after, preserve_length, catalog
                )
                if reason is None:
                    lock = check(unit.text, candidate, document)
                    cand.growth_pct = growth_of(unit.text, candidate)
                    cand.primary = verify_additions(unit.text, candidate, opts.texts)
                    if not lock.ok:
                        cand.lexical_ok = False
                        reason = "content lock: " + "; ".join(str(v) for v in lock.violations)
                    elif not cand.primary.ok:
                        reason = "primary text: " + "; ".join(
                            str(x) for x in cand.primary.rejected[:3]
                        )
                    elif lock.additions:
                        props = extract_propositions(unit.text, writer)
                        log.propositions = props
                        cand.semantic = check_meaning(unit.text, candidate, props, judge)
                        if not cand.semantic.ok:
                            reason = "meaning: " + "; ".join(cand.semantic.failing[:3])
                cand.accepted = reason is None
                cand.reason = reason or "accepted"
                log.candidates.append(cand)
                if reason is None:
                    replacements[unit.index] = candidate
                    log.final = candidate
                    log.chosen = len(log.candidates) - 1
                    log.reason = f"accepted on attempt {attempt + 1}"
                    break
                rejected_for.setdefault(unit.text, []).append((candidate, reason))
            if log.chosen is None:
                log.kept_original = True
                log.reason = log.candidates[-1].reason if log.candidates else "no attempt"
                if progress:
                    progress(f"  unit {unit.index} kept original ({log.reason})")
            logs.append(log)
        if stopped == "budget":
            break
        current = _splice(current, units, replacements)
        report = score(current, voice, patterns=patterns)
        iterations.append(
            IterationLog(
                index=i + 1,
                provider=writer.name,
                model=writer.default_model,
                units=len(units),
                accepted=len(replacements),
                distance=report.distance,
                verdict=report.verdict,
            )
        )
        if progress:
            progress(
                f"iteration {i + 1}: accepted {len(replacements)}/{len(units)}, "
                f"{report.verdict} distance {report.distance:.0f}"
            )
        if report.distance < best_report.distance:
            best_text, best_report = current, report
        if report.verdict == "PASS":
            stopped = "pass"
            break
        if not replacements:
            stopped = "no_progress"
            break
    return best_text, logs, iterations, stopped


def _edit_reject_reason(
    unit: Unit,
    candidate: str,
    context_before: str,
    context_after: str,
    preserve_length: float,
    catalog: Catalog,
) -> str | None:
    if not candidate or candidate == unit.text:
        return "empty or unchanged"
    if "\n" in candidate:
        return "contains line breaks"
    for ctx in (context_before, context_after):
        tail = ctx.strip()
        if len(tail) > 40 and tail[-40:] in candidate:
            return "repeated the surrounding context"
    words_after = len(candidate.split())
    low, high = length_bounds(unit, preserve_length)
    if not low <= words_after <= high:
        return f"length {words_after} words outside {low}-{high}"
    worse = _new_tells(
        context_before + unit.text + context_after,
        context_before + candidate + context_after,
        catalog,
    )
    if worse:
        return "new tells: " + ", ".join(worse)
    return None
