# ABOUTME: What apply returns: the per-unit and per-candidate logs, the Result model, and the
# ABOUTME: text reports built from it (meaning diff, coherence report, metric table).
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from deslop.coherence import CoherenceResult
from deslop.continuity import ContinuityResult
from deslop.draft import SectionLog
from deslop.judge import (
    DocumentVerdict,
)
from deslop.providers.base import (
    Usage,
)
from deslop.recitation import DEFAULT_MAX_RECITATION, RecitationResult
from deslop.reference import (
    PrimaryCheck,
)
from deslop.report import MetricResult, Report
from deslop.semantic import (
    AddedClaim,
    SemanticResult,
)
from deslop.units import DEFAULT_CHUNK_WORDS

DEFAULT_MAX_GROWTH = 15.0  # percent by which a candidate may outgrow the original


class CandidateLog(BaseModel):
    """One sampled rewrite of a unit and everything the verifiers said about it."""

    attempt: int
    text: str
    provider: str
    model: str
    lexical_ok: bool
    lexical_detail: str = ""
    semantic: SemanticResult | None = None  # every paragraph's propositions, merged
    primary: PrimaryCheck | None = None  # added quotations and citations, verified or not
    coherence: CoherenceResult | None = None
    recitation: RecitationResult | None = None  # runs shared with the reference, when given
    verdicts: list[DocumentVerdict] = Field(default_factory=list)  # vs original, then others
    win_rate: float | None = None
    judge_ok: bool | None = None  # won or tied every item against the original
    distance: float | None = None
    growth_pct: float | None = None
    new_tells: list[str] = Field(default_factory=list)
    reader_notes: list[str] = Field(default_factory=list)  # what the retry prompt was told
    advisories: list[str] = Field(default_factory=list)  # noted to the writer, never a rejection
    connective_note: str = ""  # the connective band, told to the writer only on a coherence loss
    continuity: ContinuityResult | None = None  # the seam with --preceding, when given
    accepted: bool = False
    reason: str = ""

    @property
    def vs_original(self) -> DocumentVerdict | None:
        return next((v for v in self.verdicts if v.against == "original"), None)

    @property
    def confidence_vs_original(self) -> float:
        v = self.vs_original
        return v.confidence if v is not None else 0.0


class UnitLog(BaseModel):
    """What happened to one unit (the document, or one section of a long one): its
    propositions by paragraph, every candidate, which won and why."""

    index: int
    kind: str  # document | section k/m | sentence | passage
    start: int
    end: int
    original: str
    final: str
    propositions: list[str] = Field(default_factory=list)
    propositions_by_paragraph: list[list[str]] = Field(default_factory=list)
    texture_by_paragraph: list[str] = Field(default_factory=list)
    quotable: list[dict[str, Any]] = Field(
        default_factory=list
    )  # passages offered: ref, words  # hints to the writer only
    candidates: list[CandidateLog] = Field(default_factory=list)
    chosen: int | None = None
    kept_original: bool = False
    reason: str = ""


class IterationLog(BaseModel):
    index: int
    provider: str
    model: str
    units: int
    accepted: int
    distance: float
    verdict: str


class Cost(BaseModel):
    usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    cached: int = 0
    unpriced_calls: int = 0


class MetricRow(BaseModel):
    """One two-sided metric before and after, against the voice's band."""

    name: str
    before: float
    after: float
    band_low: float | None = None
    band_high: float | None = None
    status_before: str = "in"  # under | in | over
    status_after: str = "in"


class Result(BaseModel):
    """What [`apply`][deslop.apply.apply] produced.

    `text` is the output (the input where nothing was accepted); `report_before` and
    `report_after` score both; `diff` is a unified diff and `meaning_diff` lists every
    paragraph's propositions with whether the output kept them plus its added quotations and
    citations; `coherence` holds the winner's tense, antecedent and connective checks and
    `judge_items` its item-by-item verdict against the original; `metric_table` has every
    metric before and after against the voice's band; `unverified_additions` are the added
    quotations and citations no primary text confirmed; `growth_pct` is the document's
    word-count growth; `units` logs each unit's candidates, verdicts, win rates and the
    winner; `judge_gate` says which providers judged; `iterations` is filled in edit mode;
    `cost` totals tokens and USD; `stopped` says why the loop ended; `exemplars_used` is
    False when the voice was packed without exemplars; `draft` holds the section logs when
    a brief was applied; `recitation` is the winner's recitation check against the
    reference (`reference` names it), whose novel share must be under `max_recitation`
    percent, when one was given.
    """

    mode: str
    text: str
    report_before: Report
    report_after: Report
    diff: str
    meaning_diff: str
    units: list[UnitLog]
    iterations: list[IterationLog] = Field(default_factory=list)
    cost: Cost
    stopped: str
    exemplars_used: bool = True
    max_growth: float = DEFAULT_MAX_GROWTH
    chunk_words: int = DEFAULT_CHUNK_WORDS
    growth_pct: float = 0.0
    coherence: CoherenceResult | None = None
    judge_items: dict[str, str] = Field(default_factory=dict)
    metric_table: list[MetricRow] = Field(default_factory=list)
    fidelity: str = "content"
    report_all: bool = False
    unverified_additions: list[str] = Field(default_factory=list)
    verified_additions: list[str] = Field(default_factory=list)
    primary_texts: list[str] = Field(default_factory=list)
    judge_gate: str = ""
    draft: list[SectionLog] | None = None
    draft_text: str | None = None
    reference: str | None = None
    max_recitation: float = DEFAULT_MAX_RECITATION
    allow_primary_facts: bool = False
    citations: str = "verified"
    continuity: ContinuityResult | None = None  # the winner's seam with --preceding
    preceding_given: bool = False
    following_given: bool = False
    recitation: RecitationResult | None = None

    def to_json(self, indent: int | None = 1) -> str:
        return self.model_dump_json(indent=indent)


def meaning_diff(units: list[UnitLog]) -> str:
    lines = []
    offered = [(u, q) for u in units for q in u.quotable]
    if offered:
        lines.append("quotable passages offered to the writer:")
        lines.extend(f"  · {q['text']} {q['ref']} ({q['words']} words)" for _, q in offered)
    verified = verified_additions(units)
    if verified:
        lines.append("verified additions (resolved in a primary text; coarse ones say so):")
        lines.extend(f"  ✓ {u}" for u in verified)
    unverified = unverified_additions(units)
    if unverified:
        lines.append("unverified additions (no primary text confirmed them):")
        lines.extend(f"  ? {u}" for u in unverified)
        lines.append("")
    for u in units:
        head = f"unit {u.index + 1} ({u.kind})"
        if u.kept_original:
            lines.append(f"{head}: kept original ({u.reason})")
            for p in u.propositions:
                lines.append(f"  ✓ {p}")
            continue
        chosen = u.candidates[u.chosen] if u.chosen is not None else None
        if chosen is None or chosen.semantic is None:
            lines.append(f"{head}: {u.reason}")
            continue
        rate = f", win rate {100 * chosen.win_rate:.0f}%" if chosen.win_rate is not None else ""
        lines.append(f"{head}: candidate {u.chosen + 1} by {chosen.provider}{rate}")  # type: ignore[operator]
        for c in chosen.semantic.checks:
            mark = "✓" if c.kept else "✗"
            lines.append(f"  {mark} {c.proposition}" + (f" ({c.note})" if c.note else ""))
        for added in chosen.semantic.added:
            lines.append(f"  + [{added.kind}] {added.text}{_votes(added)}")
        for item in chosen.semantic.cleared:
            lines.append(
                f"  ~ cleared on review [{item.category}] {item.text}{_votes(item)}: {item.note}"
            )
        for add in chosen.primary.additions if chosen.primary else []:
            mark = "✓" if add.verified else "?"
            lines.append(f"  {mark} [{add.kind}] {add}")
        for adv in chosen.advisories:
            lines.append(f"  ~ advisory: {adv}")
        if chosen.connective_note:
            lines.append(f"  ~ advisory (connectives): {chosen.connective_note}")
    return "\n".join(lines)


def _votes(item: AddedClaim) -> str:
    return f" (vote {item.votes})" if item.votes else ""


def coherence_report(result: Result) -> str:
    """The winner's coherence checks and its item-by-item verdict against the original, as
    text for the CLI and the saved report."""
    lines = []
    c = result.coherence
    if c is None:
        lines.append("coherence: not checked (the original was kept)")
    else:
        lines.append(
            f"coherence: {'ok' if c.ok else 'FAILED'}; paragraphs {c.paragraphs_before} -> "
            f"{c.paragraphs_after} ({c.paragraphs.sentences_per_paragraph:.1f} sentences each"
            + (
                f"; the voice's run {c.paragraphs.band_low:.1f} to {c.paragraphs.band_high:.1f})"
                if c.paragraphs.band_high is not None and c.paragraphs.band_low is not None
                else ")"
            )
            + f"; tense drift in {len(c.tense)} paragraph(s); "
            f"{len(c.antecedents)} pronoun(s) without an antecedent; connective openers "
            f"{100 * c.connectives.before:.0f}% -> {100 * c.connectives.after:.0f}%"
            + (f" ({c.connectives.note})" if c.connectives.note else "")
        )
        for d in c.tense:
            lines.append(
                f'  tense: paragraph {d.paragraph} {d.before} -> {d.after}: "{d.example[:100]}"'
            )
        for f in c.antecedents:
            lines.append(f'  antecedent: "{f.text[:100]}" ({f.word})')
    k = result.continuity
    if k is not None:
        rhythm = (
            f"; rhythm distance to the section before {k.rhythm_distance:.2f}"
            if k.rhythm_distance is not None
            else ""
        )
        lines.append(
            f"continuity with the section before: {'ok' if k.ok else 'FAILED'}; tense "
            f"{k.tense_before or '-'} -> {k.tense_after or '-'}, person {k.person_before} -> "
            f"{k.person_after}; {len(k.terms)} term(s) rendered differently; "
            f"{len(k.reintroductions)} name(s) introduced again{rhythm} "
            f"(distance to the voice {result.report_after.distance:.1f})"
        )
        lines.extend(f"  {n}" for n in k.reader_notes())
    if result.judge_items:
        lines.append(
            "judge vs original: " + ", ".join(f"{k} {v}" for k, v in result.judge_items.items())
        )
    return "\n".join(lines)


def metric_table(before: Report, after: Report) -> list[MetricRow]:
    """Every metric the voice measures, before and after, with its band and whether each
    value sits under, in or over it."""

    def status(m: MetricResult) -> str:
        if m.band_high is not None and m.value > m.band_high:
            return "over"
        if m.band_low is not None and m.value < m.band_low:
            return "under"
        return "in"

    after_by = {m.name: m for m in after.metrics}
    rows = []
    for m in before.metrics:
        n = after_by.get(m.name)
        if n is None:
            continue
        rows.append(
            MetricRow(
                name=m.name,
                before=round(m.value, 3),
                after=round(n.value, 3),
                band_low=m.band_low,
                band_high=m.band_high,
                status_before=status(m),
                status_after=status(n),
            )
        )
    return rows


def cost_of(usage: Usage) -> Cost:
    return Cost(
        usd=round(usage.cost_usd, 4),
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        calls=usage.calls,
        cached=usage.cached,
        unpriced_calls=usage.unpriced,
    )


def verified_additions(units: list[UnitLog]) -> list[str]:
    out: list[str] = []
    for u in units:
        if u.kept_original or u.chosen is None or u.candidates[u.chosen].primary is None:
            continue
        for add in u.candidates[u.chosen].primary.additions:  # type: ignore[union-attr]
            if add.verified:
                out.append(f"unit {u.index + 1}: {add}")
    return out


def unverified_additions(units: list[UnitLog]) -> list[str]:
    out: list[str] = []
    for u in units:
        if u.kept_original or u.chosen is None:
            continue
        cand = u.candidates[u.chosen]
        if cand.primary is None:
            continue
        for add in cand.primary.unverified:
            out.append(f"unit {u.index + 1}: {add}")
    return out
