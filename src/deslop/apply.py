# ABOUTME: apply(text_or_brief, voice): generate-then-verify rewriting of a document as a whole.
# ABOUTME: n candidates must pass the locks, coherence checks and a four-item judge, else it stays.
from __future__ import annotations

import copy
import datetime as dt
import difflib
import json
import math
import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from deslop.coherence import CoherenceResult, check_coherence
from deslop.draft import Brief, SectionLog, draft_sections
from deslop.judge import (
    DocumentVerdict,
    JudgeCalibration,
    compare_document,
    document_win_rate,
)
from deslop.lock import DocumentLock, check, inventory
from deslop.metrics import compute_all, detect_tells
from deslop.nlp import WORD_RE, Analysis, Paragraph, Sentence, analyze, split_paragraphs
from deslop.patterns import Catalog, load_catalog
from deslop.providers.base import (
    DEFAULT_JUDGE,
    DEFAULT_WRITER,
    CachingProvider,
    ProviderError,
    Usage,
    get_provider,
    resolve_model,
)
from deslop.recitation import DEFAULT_MAX_RECITATION, NGRAM, RecitationResult, recitation
from deslop.reference import (
    Addition,
    CitationScheme,
    Passage,
    PrimaryCheck,
    PrimaryText,
    PrimaryTextSpec,
    canonical_label,
    citation_guidance,
    load_primary_texts,
    normalize,
    quotable_passages,
    verify_additions,
)
from deslop.report import MetricResult, Report, TellHitResult
from deslop.score import score
from deslop.semantic import (
    AddedClaim,
    Fidelity,
    PropositionCheck,
    SemanticResult,
    check_meaning,
    extract_propositions,
    extract_texture,
)
from deslop.stats import Band
from deslop.voice.exemplars import parse_exemplars
from deslop.voice.model import Voice

Mode = Literal["rewrite", "edit"]
DEFAULT_CHUNK_WORDS = 2500  # longer documents are rewritten section by section


def pairs_dir() -> Path:
    """Where accepted and rejected rewrite pairs are appended: $DESLOP_HOME/pairs."""
    return Path(os.environ.get("DESLOP_HOME", Path.home() / ".deslop")) / "pairs"


CANDIDATE_TEMPERATURE = 0.9
DEFAULT_MAX_GROWTH = 15.0  # percent by which a candidate may outgrow the original
ROUNDS = 2  # the first round, then one retry carrying what a reader noticed
ADVISORY_PARAGRAPHS = 10  # under this many input paragraphs the paragraph band only advises
DEFAULT_QUOTE_CONTEXT_WORDS = 1500  # cap on the verbatim passages offered to the writer
KEY_TOKEN_RE = re.compile(r"\b[A-Z][A-Za-z'-]+\b|\d+(?:[.:]\d+)*")


# ---------------------------------------------------------------- result models


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
    recitation: RecitationResult | None = None

    def to_json(self, indent: int | None = 1) -> str:
        return self.model_dump_json(indent=indent)

    @property
    def changed(self) -> bool:
        return self.text != self.units_source_text()

    def units_source_text(self) -> str:
        return self.draft_text if self.draft_text is not None else ""


# ---------------------------------------------------------------- units


@dataclass
class Unit:
    index: int
    kind: str
    start: int
    end: int
    text: str
    para_index: int
    hits: list[TellHitResult]
    metric_notes: list[str]

    @property
    def tell_ids(self) -> list[str]:
        return sorted({h.pattern_id for h in self.hits})


def flagged_hits(report: Report) -> list[TellHitResult]:
    return [h for t in report.tells if (t.flagged or t.banned) for h in t.hits]


def similar_runs(a: Analysis, min_run: int = 3, tolerance: int = 5) -> list[list[int]]:
    """Sentence indexes of every run of >= min_run consecutive sentences in one paragraph
    whose word counts all fall within `tolerance` of each other."""
    runs: list[list[int]] = []
    for para in a.paragraphs:
        run: list[Sentence] = []
        for sent in para.sentences:
            counts = [x.word_count for x in run] + [sent.word_count]
            if run and max(counts) - min(counts) <= tolerance:
                run.append(sent)
            else:
                if len(run) >= min_run:
                    runs.append([x.index for x in run])
                run = [sent]
        if len(run) >= min_run:
            runs.append([x.index for x in run])
    return runs


def rhythm_targets(a: Analysis) -> set[int]:
    runs = similar_runs(a, 3) or similar_runs(a, 2)
    return {i for run in runs for i in run}


READER_PHRASES = {
    "citation_per_100w": (
        "cites the source less often than this writer does",
        "cites the source more often than this writer does",
    ),
    "quote_per_para": (
        "quotes the source less than this writer does",
        "quotes the source more than this writer does",
    ),
    "concrete_per_100w": (
        "has fewer particulars (names, numbers, terms) than this writer gives",
        "has more particulars (names, numbers, terms) than this writer gives",
    ),
    "sent_len_cv": (
        "the sentences run to much the same length",
        "the sentence lengths swing more than this writer's do",
    ),
    "sent_len_std": (
        "the sentences run to much the same length",
        "the sentence lengths swing more than this writer's do",
    ),
    "max_run_similar_len": ("", "several sentences in a row run to the same length"),
    "opener_top_word_share": (
        "the sentence openings vary more than this writer's do",
        "many sentences begin with the same word",
    ),
    "opener_pct_adv_conj": ("", "many sentences open with an adverb or a conjunction"),
    "reader_address_per_1k": (
        "addresses the reader less often than this writer does",
        "addresses the reader more often than this writer does",
    ),
    "connective_opener_pct": (
        "fewer sentences open with and, but, so or the like than this writer's do",
        "more sentences open with and, but, so or the like than this writer's do",
    ),
    "first_person_per_1k": (
        "says 'I' less than this writer does",
        "says 'I' more than this writer does",
    ),
    "sent_len_mean": (
        "the sentences run shorter than this writer's",
        "the sentences run longer than this writer's",
    ),
    "colons_per_1k": ("", "leans on colons"),
    "em_dashes_per_1k": ("", "leans on dashes"),
    "questions_per_1k": ("", "asks more questions than this writer does"),
    "semicolons_per_1k": ("uses fewer semicolons than this writer", "leans on semicolons"),
}


def reader_phrase(m: MetricResult) -> str:
    """What a reader would say about a flagged metric, with no metric named."""
    low, high = READER_PHRASES.get(m.name, ("", ""))
    phrase = high if m.z > 0 else low
    if phrase:
        return phrase
    direction = "more" if m.z > 0 else "less"
    return f"{m.group} reads {direction} marked than this writer's"


def metric_notes_for(report: Report) -> list[str]:
    return sorted({reader_phrase(m) for m in report.flagged_metrics()})


def edit_units(text: str, a: Analysis, report: Report) -> list[Unit]:
    """Edit mode: flagged sentences, with adjacent flagged sentences in one paragraph merged
    into a passage the model may split or merge."""
    hits = flagged_hits(report)
    notes = metric_notes_for(report)
    by_sentence: dict[int, dict[tuple[str, int, int], TellHitResult]] = {}
    for h in hits:
        for s in a.sentences:
            if s.start < h.end and h.start < s.end:
                by_sentence.setdefault(s.index, {})[(h.pattern_id, h.start, h.end)] = h
    targets = set(by_sentence)
    if not targets and notes:
        targets = rhythm_targets(a)
    groups: list[list[Sentence]] = []
    for s in a.sentences:
        if s.index not in targets:
            continue
        last = groups[-1][-1] if groups else None
        if last and last.index == s.index - 1 and last.para_index == s.para_index:
            groups[-1].append(s)
        else:
            groups.append([s])
    units: list[Unit] = []
    for group in groups:
        merged: dict[tuple[str, int, int], TellHitResult] = {}
        for s in group:
            merged.update(by_sentence.get(s.index, {}))
        first, last = group[0], group[-1]
        units.append(
            Unit(
                index=len(units),
                kind="sentence" if len(group) == 1 else "passage",
                start=first.start,
                end=last.end,
                text=text[first.start : last.end],
                para_index=first.para_index,
                hits=list(merged.values()),
                metric_notes=notes,
            )
        )
    return units


# ---------------------------------------------------------------- prompts

SYSTEM_TEMPLATE = """You write prose in one specific writer's voice. You are given the voice's
description and rules, exemplars of it, and a passage to write afresh in that voice. You
return the new passage and nothing else: no preamble, no quotation marks around it, no notes.

# The voice
{profile}

{rules}
"""

DOCUMENT_TEMPLATE = """# Exemplars of the voice
{exemplars}

# The original, in full
{document}

# What it asserts, paragraph by paragraph (keep every one, add none)
{propositions}
{section_block}{quotable}
# Instruction
Write {what} as this writer would, from its first paragraph to its last. Keep every
proposition listed above and add none. Adopt this writer's grammatical person and mode of
address as shown in the exemplars and profile; the propositions' phrasing is not a constraint
on person or address. Where a paragraph's note says what the original paragraph also did
(its address, questions, images, exclamations, appeals to the reader), use those where this
writer would; they are hints, not requirements. Formatting (title, headings, lists, emphasis)
follows the exemplars; the input's formatting is not a constraint, and when the exemplars
have no headings the output has none. Paragraph length and count follow the exemplars, not
the input: merge or split the input's paragraphs freely. Keep the author's tense and order of
argument unless the voice's profile says otherwise. Keep it about the length of the original.
{quote_rule}Return {returned} and nothing else.{noticed}

Draft {attempt} of {n}."""

QUOTABLE_TEMPLATE = """
# Passages you may quote (verbatim, from the primary texts the original cites)
{passages}

# How these texts are cited
{guidance}
"""

QUOTE_RULE = (
    "Quote only from the passages listed above and only verbatim; do not quote from memory; "
    "if a passage you want is not there, paraphrase it without quotation marks. "
)

SECTION_BLOCK = """
# The section to write now (paragraphs {first} to {last} of the original)
{section}

# Already written in the voice (it precedes the section; continue from it)
{previous}

# What follows the section in the original (for continuity; do not write it)
{following}
"""

NOTICED_TEMPLATE = """

# What a reader noticed in earlier drafts (do not repeat these)
{notes}"""

EDIT_TEMPLATE = """# Exemplars of the voice
{exemplars}

# Context (DO NOT EDIT; shown only so the rewrite fits)
BEFORE: {before}
AFTER: {after}

# The {kind} to rewrite
{unit}

# What is wrong with it
{problems}

# Must survive the rewrite (content lock)
{lock}

# Constraints
- Return only the rewritten {kind}, in the voice, between {min_words} and {max_words} words.
- Do not repeat or alter the surrounding context.
- Keep every claim; add no new fact, name, number, quotation or citation.
{extra}{rejected}"""

REJECTED_TEMPLATE = """

# Earlier attempts and why they were rejected (do not repeat their mistakes)
{attempts}"""

FIX_HINTS = {
    "T1": "Make a person the subject (the writer, a character, the reader) or state the fact.",
    "T5": "End on a particular already in the text: a name, a quotation, a plain fact. No "
    "antithesis, no abstract-noun equation.",
    "T2": "Say what is the case directly; drop the 'not X but Y' frame.",
    "T9": "Either answer the question in the same sentence or turn it into a statement.",
    "T13": "Replace the colon reveal with an ordinary clause.",
    "T14": "Do not restate; add a new particular or cut the paragraph.",
    "T15": "Vary sentence length: at least one sentence under 8 words, one well over 20.",
}
SENTENCE_SLACK_WORDS = 4


def length_bounds(unit: Unit, preserve_length: float) -> tuple[int, int]:
    words = len(unit.text.split())
    slack = SENTENCE_SLACK_WORDS if unit.kind == "sentence" else 0
    low = max(1, int(words * (1 - preserve_length)) - slack)
    high = int(math.ceil(words * (1 + preserve_length))) + 1 + slack
    return low, high


def _problems(unit: Unit, catalog: Catalog) -> str:
    lines = []
    for pid in unit.tell_ids:
        p = catalog.by_id(pid)
        spans = sorted({h.text.strip()[:80] for h in unit.hits if h.pattern_id == pid})
        hint = FIX_HINTS.get(pid, "")
        lines.append(
            f"- {p.id} {p.name}: {p.description} Flagged span(s): "
            + " | ".join(spans)
            + (f" Fix: {hint}" if hint else "")
        )
    for note in unit.metric_notes:
        lines.append(f"- {note}")
    return "\n".join(lines) if lines else "- Does not read in the voice."


def _zvector(values: dict[str, float], voice: Voice, names: list[str]) -> list[float]:
    out = []
    for n in names:
        stat = voice.metrics.metrics[n]
        out.append((values[n] - stat.mean) / stat.std if stat.std > 0 else 0.0)
    return out


def _content(text: str) -> frozenset[str]:
    return frozenset(w.lower() for w in WORD_RE.findall(text) if len(w) > 4)


class ExemplarIndex:
    """Exemplar bodies with their register, metric vector and content words, for choosing
    the three nearest a unit: cluster-proportionally and by lexical overlap, not metrics alone."""

    def __init__(self, voice: Voice):
        self.names = [n for n, s in voice.metrics.metrics.items() if s.std > 0]
        self.voice = voice
        self.bodies = parse_exemplars(voice.exemplars)
        self.registers = _registers(voice.exemplars)
        self.vectors = [_zvector(compute_all(analyze(b)), voice, self.names) for b in self.bodies]
        self.contents = [_content(b) for b in self.bodies]

    @property
    def available(self) -> bool:
        return bool(self.bodies)

    def closest(self, paragraph_text: str, k: int = 3) -> list[str]:
        if not self.bodies:
            return []
        content = _content(paragraph_text)
        v = _zvector(compute_all(analyze(paragraph_text)), self.voice, self.names)
        scores = []
        for i in range(len(self.bodies)):
            overlap = len(content & self.contents[i]) / max(len(content | self.contents[i]), 1)
            metric = math.sqrt(
                sum((x - y) ** 2 for x, y in zip(v, self.vectors[i], strict=True))
            ) / max(len(self.names), 1)
            scores.append((-(overlap * 10.0) + metric, i))
        scores.sort()
        # One exemplar from each register first, in proportion to register size, then the
        # remaining nearest ones.
        chosen: list[int] = []
        seen_registers: set[str] = set()
        for _, i in scores:
            reg = self.registers[i] if i < len(self.registers) else ""
            if reg not in seen_registers:
                chosen.append(i)
                seen_registers.add(reg)
            if len(chosen) >= k:
                break
        for _, i in scores:
            if len(chosen) >= k:
                break
            if i not in chosen:
                chosen.append(i)
        return [self.bodies[i] for i in chosen[:k]]


def _registers(markdown: str) -> list[str]:
    out = []
    for line in markdown.splitlines():
        if line.startswith("## "):
            start, end = line.rfind("(register"), line.rfind(")")
            out.append(line[start + 1 : end] if start > 0 and end > start else "")
    return out


def _context(unit: Unit, text: str, a: Analysis) -> tuple[str, str]:
    para = a.paragraphs[unit.para_index]
    before = text[max(para.start, unit.start - 400) : unit.start].strip() or "(start of text)"
    after = text[unit.end : min(para.end, unit.end + 400)].strip() or "(end of paragraph)"
    return before, after


@dataclass
class Section:
    """A run of consecutive paragraphs rewritten as one unit: the whole document, or one
    chunk of a long one."""

    index: int
    count: int
    first: int  # first paragraph index, 0-based
    last: int  # last paragraph index, inclusive
    start: int  # character span in the original
    end: int

    @property
    def kind(self) -> str:
        return "document" if self.count == 1 else f"section {self.index + 1}/{self.count}"


def sections_of(a: Analysis, chunk_words: int) -> list[Section]:
    """One section for the whole document when it fits in `chunk_words`, else consecutive
    runs of paragraphs of at most `chunk_words` words (a single paragraph may exceed it)."""
    paragraphs = a.paragraphs
    if not paragraphs:
        return []
    runs: list[list[Paragraph]] = []
    current: list[Paragraph] = []
    words = 0
    for p in paragraphs:
        if current and words + p.word_count > chunk_words:
            runs.append(current)
            current, words = [], 0
        current.append(p)
        words += p.word_count
    runs.append(current)
    if len(runs) == 1:  # the whole document, headings and all
        return [Section(0, 1, paragraphs[0].index, paragraphs[-1].index, 0, len(a.text))]
    return [
        Section(
            index=i,
            count=len(runs),
            first=run[0].index,
            last=run[-1].index,
            start=run[0].start,
            end=run[-1].end,
        )
        for i, run in enumerate(runs)
    ]


TEXTURE_LEAD = "The original paragraph also did these things; use them where this writer would: "


def _numbered(propositions: list[list[str]], first: int, textures: Sequence[str] = ()) -> str:
    lines = []
    for i, props in enumerate(propositions, first + 1):
        lines.append(f"Paragraph {i}:")
        lines.extend(f"  {k}. {p}" for k, p in enumerate(props, 1))
        if not props:
            lines.append("  (none)")
        texture = textures[i - first - 1] if i - first - 1 < len(textures) else ""
        if texture:
            lines.append(f"  {TEXTURE_LEAD}{texture}")
    return "\n".join(lines)


def document_prompt(
    section: Section,
    original: str,
    section_text: str,
    propositions: list[list[str]],
    voice: Voice,
    exemplars: ExemplarIndex,
    *,
    previous: str,
    following: str,
    attempt: int,
    n: int,
    noticed: list[str],
    textures: Sequence[str] = (),
    passages: Sequence[Passage] = (),
    guidance: Sequence[str] = (),
) -> tuple[str, str]:
    """The writer's prompt: profile and rules (system), exemplars nearest the piece, the full
    original, its propositions by paragraph (each with its texture note, when there is one),
    the passages it may quote verbatim with how the texts are cited (when primary texts are
    declared), and the instruction. A long document adds the section to write with the
    already-rewritten section before it and the original section after it. `noticed` carries
    what a reader saw in earlier drafts, for the retry."""
    system = SYSTEM_TEMPLATE.format(profile=voice.profile.strip(), rules=voice.rules.strip())
    whole = section.count == 1
    block = (
        ""
        if whole
        else SECTION_BLOCK.format(
            first=section.first + 1,
            last=section.last + 1,
            section=section_text,
            previous=previous or "(this is the first section)",
            following=following or "(this is the last section)",
        )
    )
    user = DOCUMENT_TEMPLATE.format(
        exemplars="\n\n".join(exemplars.closest(section_text)) or "(no exemplars in this voice)",
        document=original,
        propositions=_numbered(propositions, 0 if whole else section.first, textures) or "(none)",
        section_block=block,
        quotable=QUOTABLE_TEMPLATE.format(
            passages="\n\n".join(f"[{p.text_name} {p.ref}] {p.text.strip()}" for p in passages)
            or "(the original cites nothing that resolves in the primary texts)",
            guidance="\n".join(f"- {g}" for g in guidance) or "- (none)",
        )
        if passages or guidance
        else "",
        quote_rule=QUOTE_RULE if passages or guidance else "",
        what="this piece" if whole else "this section",
        returned="the whole piece" if whole else "the section",
        noticed=NOTICED_TEMPLATE.format(notes="\n".join(f"- {x}" for x in noticed))
        if noticed
        else "",
        attempt=attempt,
        n=n,
    )
    return system, user


def edit_prompt(
    unit: Unit,
    text: str,
    a: Analysis,
    voice: Voice,
    exemplars: ExemplarIndex,
    document: DocumentLock,
    catalog: Catalog,
    preserve_length: float,
    rejected: list[tuple[str, str]],
) -> tuple[str, str]:
    system = SYSTEM_TEMPLATE.format(profile=voice.profile.strip(), rules=voice.rules.strip())
    before, after = _context(unit, text, a)
    low, high = length_bounds(unit, preserve_length)
    inv = inventory(unit.text, document.names)
    lock_lines = inv.describe() or ["(nothing locked in this unit)"]
    extra = "- You may split or merge sentences within this passage; keep it on one line."
    if unit.kind == "sentence":
        extra = "- Keep it on one line."
    if "T15" in unit.tell_ids:
        extra += "\n- Vary sentence length sharply: at least one sentence under 8 words."
    user = EDIT_TEMPLATE.format(
        exemplars="\n\n".join(
            exemplars.closest(
                text[a.paragraphs[unit.para_index].start : a.paragraphs[unit.para_index].end]
            )
        )
        or "(no exemplars in this voice)",
        before=before,
        after=after,
        kind=unit.kind,
        unit=unit.text,
        problems=_problems(unit, catalog),
        lock="\n".join(f"- {line}" for line in lock_lines),
        min_words=low,
        max_words=high,
        extra=extra,
        rejected=_rejected_section(rejected),
    )
    return system, user


def max_words_for(words: int, max_growth: float) -> int:
    return int(math.floor(words * (1 + max_growth / 100.0)))


def growth_of(before: str, after: str) -> float:
    """Word-count growth of `after` over `before`, in percent."""
    a, b = len(before.split()), len(after.split())
    return round(100.0 * (b - a) / max(a, 1), 1)


def _rejected_section(rejected: list[tuple[str, str]]) -> str:
    if not rejected:
        return ""
    lines = [f'- Rejected ({reason}): "{candidate[:300]}"' for candidate, reason in rejected]
    return REJECTED_TEMPLATE.format(attempts="\n".join(lines))


def clean_response(raw: str) -> str:
    out = raw.strip()
    for fence in ("```", "'''"):
        if out.startswith(fence) and out.endswith(fence):
            out = out[len(fence) : -len(fence)].strip()
    if len(out) > 2 and out[0] in '"“' and out[-1] in '"”' and out.count('"') == 2:
        out = out[1:-1].strip()
    return out


# ---------------------------------------------------------------- providers


ProviderFactory = Callable[[str], CachingProvider]


@dataclass
class Models:
    """The writer and the judge: two different Anthropic models served by one provider. The
    judge also runs the semantic lock. A model never judges its own output."""

    writer: CachingProvider
    judge: CachingProvider
    calibration: JudgeCalibration | None = None

    @property
    def usage(self) -> Usage:
        total = Usage()
        total.add(self.writer.usage)
        if self.judge is not self.writer:
            total.add(self.judge.usage)
        return total

    def describe(self) -> str:
        cal = self.calibration
        if cal is None:
            measured = "uncalibrated"
        else:
            measured = f"{100 * cal.accuracy:.0f}% on {cal.pairs} pairs"
            if not cal.usable:
                measured += "; below 80%, its verdicts are unreliable"
        return (
            f"writer {self.writer.key}; judge {self.judge.key} ({measured}); "
            f"provider {self.writer.name}"
        )


def build_models(
    writer: str = DEFAULT_WRITER,
    judge: str = DEFAULT_JUDGE,
    *,
    provider: str | None = None,
    factory: ProviderFactory | None = None,
    calibration: dict[str, Any] | None = None,
) -> Models:
    """Resolve the writer and judge model names (aliases allowed) on the configured provider.
    `factory` maps a provider name to a `CachingProvider` (tests replay a cache with it)."""
    from deslop.providers.base import default_provider_name

    writer_model, judge_model = resolve_model(writer), resolve_model(judge)
    if writer_model == judge_model:
        raise ValueError(
            f"the judge must be a different model from the writer ({writer_model}): "
            "a model never judges its own output"
        )
    name = provider or default_provider_name()
    make = factory or (lambda n: get_provider(n))
    base = make(name)
    writer_p, judge_p = copy.copy(base), copy.copy(base)
    writer_p.default_model, writer_p.usage = writer_model, Usage()
    judge_p.default_model, judge_p.usage = judge_model, Usage()
    cal = None
    for key, data in (calibration or {}).items():
        if key in (judge_model, judge_p.key):
            try:
                cal = JudgeCalibration.model_validate(data)
            except ValueError:
                cal = None
    return Models(writer=writer_p, judge=judge_p, calibration=cal)


# ---------------------------------------------------------------- the loops


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


def _meaning_diff(units: list[UnitLog]) -> str:
    lines = []
    offered = [(u, q) for u in units for q in u.quotable]
    if offered:
        lines.append("quotable passages offered to the writer:")
        lines.extend(f"  · {q['text']} {q['ref']} ({q['words']} words)" for _, q in offered)
    verified = _verified(units)
    if verified:
        lines.append("verified additions (resolved in a primary text):")
        lines.extend(f"  ✓ {u}" for u in verified)
    unverified = _unverified(units)
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


def _cost(usage: Usage) -> Cost:
    return Cost(
        usd=round(usage.cost_usd, 4),
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        calls=usage.calls,
        cached=usage.cached,
        unpriced_calls=usage.unpriced,
    )


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
    max_iter: int = 4,
    preserve_length: float = 0.15,
    patterns: tuple[str, ...] = (),
    provider_factory: ProviderFactory | None = None,
    pairs_path: Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> Result:
    """Write `text_or_brief` in `voice` and return a [`Result`][deslop.Result].

    Rewrite mode (default) writes the document as a whole: every paragraph's propositions are
    extracted (cached), then `n` candidates are sampled from the `writer` model with the
    profile, the rules, the exemplars nearest the piece, the full original and the propositions
    by paragraph. Each candidate must pass the content lock (nothing locked removed), the length
    limit (`max_growth` percent), the primary-text check, the semantic lock paragraph by
    paragraph (every proposition kept, none added, checked by the `judge` model) and the
    coherence checks (paragraphing in the voice's range, same tense, every sentence-initial
    pronoun with an antecedent, connective openers in the voice's band and near the original's
    share), then is judged against the original by the `judge` model (a different Anthropic
    model from the writer) on four items (voice, coherence, tense and person, order and
    emphasis) and must win or tie on every one. Survivors are judged round-robin; the best win
    rate wins. If no candidate passes, one retry carries what a reader noticed; if that fails
    too the result is the original with the report. A document longer than `chunk_words` is
    rewritten section by section, each with the already-rewritten section before it in context
    and the original section after it visible. Edit mode rewrites only flagged sentences under
    the same locks. A `Brief` is drafted section by section first, then rewritten.
    `primary_texts` (files, directories, `PrimaryTextSpec`s or loaded `PrimaryText`s) are
    supplied by the caller, with `citation_schemes` for items that declare none;
    without any, added quotations and citations are allowed and listed as unverified.
    `reference` is the human text the input was made from, when there is one: it is never
    shown to the writer or the judge, only compared against, and a candidate whose words are
    more than `max_recitation` percent of its words recited from the reference (words in runs
    of eight or more shared with the reference and not with the input, and not in the
    input's vocabulary at all) is rejected for recitation (rewrite mode).
    `provider_factory` maps a provider name to a `CachingProvider` (tests replay a cache with
    it); `budget` is USD. `writer` and `judge` are model names or aliases (`opus`, `sonnet`,
    `haiku`) and must differ; `provider` is `anthropic` (the API) or `claude-cli`, default from
    DESLOP_PROVIDER.
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
        meaning_diff=_meaning_diff(units),
        units=units,
        iterations=iterations,
        cost=_cost(models.usage),
        stopped=stopped,
        exemplars_used=exemplars.available,
        max_growth=max_growth,
        chunk_words=chunk_words,
        growth_pct=growth_of(text, final_text),
        coherence=winner.coherence if winner is not None else None,
        judge_items=dict(vs.items) if vs is not None else {},
        metric_table=metric_table(report_before, report_after),
        unverified_additions=_unverified(units),
        verified_additions=_verified(units),
        primary_texts=[t.name for t in texts],
        judge_gate=models.describe(),
        fidelity=fidelity,
        report_all=report_all,
        draft=draft_logs,
        draft_text=draft_text,
        reference=reference_name if reference is not None else None,
        max_recitation=max_recitation,
        allow_primary_facts=allow_primary_facts,
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


def _load_texts(
    sources: list[str | Path | PrimaryText | PrimaryTextSpec],
    schemes: Sequence[CitationScheme] = (),
) -> list[PrimaryText]:
    loaded = [t for t in sources if isinstance(t, PrimaryText)]
    names = [t for t in sources if not isinstance(t, PrimaryText)]
    return loaded + load_primary_texts(names, schemes=schemes)


def _verified(units: list[UnitLog]) -> list[str]:
    out: list[str] = []
    for u in units:
        if u.kept_original or u.chosen is None or u.candidates[u.chosen].primary is None:
            continue
        for add in u.candidates[u.chosen].primary.additions:  # type: ignore[union-attr]
            if add.verified:
                out.append(f"unit {u.index + 1}: {add}")
    return out


def permit_primary(
    cand: CandidateLog, texts: list[PrimaryText], allow_facts: bool, original: str = ""
) -> None:
    """The primary-text policy on the semantic lock's additions: a citation that resolves in a
    declared primary text, a sentence that merely names one, and a claim whose new names and
    numbers (those `original` lacks) are all inside a quotation verified verbatim are
    permitted and moved to `cleared` with the reason. With `allow_facts`, a claim whose
    sentence cites a resolved location and whose names and numbers all occur in that
    location is permitted too. Everything else stands."""
    if cand.semantic is None or not texts:
        return
    verified = [a for a in (cand.primary.additions if cand.primary else []) if a.verified]
    names = {w.lower() for t in texts for w in re.findall(r"[A-Za-z]+", t.name)}
    known = set(normalize(original).split())
    kept = []
    for item in cand.semantic.added:
        why = _primary_permission(item, verified, texts, names, allow_facts, known)
        if item.kind == "claim" and why:
            item.note = why
            cand.semantic.cleared.append(item)
        else:
            kept.append(item)
    cand.semantic.added = kept


STRUCTURE_WORDS = {"book", "books", "chapter", "chapters", "verse", "verses", "part"}
FUNCTION_WORDS = {"the", "a", "an", "it", "this", "that", "and", "but", "in", "of", "as", "its"}
ROMAN_RE = re.compile(r"^[IVXLC]+$")


def _primary_permission(
    item: AddedClaim,
    verified: list[Addition],
    texts: list[PrimaryText],
    names: set[str],
    allow_facts: bool,
    known: set[str] = frozenset(),  # type: ignore[assignment]
) -> str:
    sentence = item.sentence or ""
    tokens = [t for t in KEY_TOKEN_RE.findall(item.text)]
    if item.text and item.text[:1].isupper() and tokens and item.text.startswith(tokens[0]):
        tokens = tokens[1:]  # a sentence-initial capital is not a name
    keys = [t.lower() for t in tokens if t.lower() not in FUNCTION_WORDS]
    cites = [a for a in verified if a.kind == "citation" and a.text in sentence]
    cited = {canonical_label(a.text) for a in verified if a.kind == "citation"}
    cited |= {canonical_label(a.text.split()[-1]) for a in verified if a.kind == "citation"}
    if (
        cited
        and keys
        and all(
            canonical_label(k) in cited or k in STRUCTURE_WORDS or k in ("verse", "reference")
            for k in keys
        )
    ):
        return "a citation that resolves in a declared primary text"
    if (
        item.category == "entity"
        and keys
        and all(
            k in names or k in STRUCTURE_WORDS or k.isdigit() or ROMAN_RE.match(k.upper())
            for k in keys
        )
    ):
        return "names a declared primary text"
    for a in verified:
        if a.kind != "quote" or normalize(a.text) not in normalize(sentence):
            continue
        quoted = set(normalize(a.text).split())
        if keys and all(k in quoted or k in known for k in keys):
            return f"verbatim in {a.where}"
    if allow_facts and keys:
        for a in cites:
            for t in texts:
                if not a.where.startswith(t.name + " "):
                    continue
                span = t.segment_text(a.where[len(t.name) + 1 :])
                if span and all(k in span.split() for k in keys):
                    return f"a fact verifiable at {a.where}"
    return ""


def _unverified(units: list[UnitLog]) -> list[str]:
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
        previous = pieces[-1] if pieces else ""
        following = (
            text[sections[section.index + 1].start : sections[section.index + 1].end]
            if section.index + 1 < len(sections)
            else ""
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
    """The checks that need no judge: shape, length, lexical lock, primary texts, coherence.
    Returns what a reader noticed (empty when the candidate may go on)."""
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
    notes.extend(cand.coherence.reader_notes())
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
    log.quotable = [{"text": p.text_name, "ref": p.ref, "words": p.words} for p in passages]
    if progress and passages:
        progress(
            f"  quotable passages: {len(passages)} ({sum(p.words for p in passages)} words) "
            + ", ".join(f"{p.text_name} {p.ref}" for p in passages[:6])
            + (" ..." if len(passages) > 6 else "")
        )
    noticed: list[str] = []
    for round_index in range(ROUNDS):
        eligible: list[CandidateLog] = []
        for k in range(n):
            attempt = round_index * n + k + 1
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
                cand.reader_notes = notes + linter + cand.advisories
                if not opts.report_all:
                    continue
            else:
                cand.reader_notes = linter + cand.advisories
            try:
                cand.verdicts.append(
                    compare_document(
                        candidate, section_text, section_text, voice, judge, exemplars=near
                    )
                )
            except ProviderError as exc:
                if not failed:
                    cand.reason = f"judge unavailable: {exc}"[:200]
                    cand.reader_notes = linter
                continue
            vs = cand.vs_original
            cand.judge_ok = vs is not None and vs.no_loss
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
                    *linter,
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
            seen: list[str] = []
            for c in log.candidates:
                for note in c.reader_notes:
                    if note not in seen:
                        seen.append(note)
            noticed = seen[-12:]
    log.kept_original = True
    log.reason = "no candidate passed: " + "; ".join(
        f"{c.attempt}: {c.reason[:80]}" for c in log.candidates
    )
    if progress:
        progress(f"  {section.kind}: kept original (no candidate passed)")
    return log


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
        provider = writer
        if progress:
            progress(f"iteration {i + 1}: {len(units)} edit unit(s) via {provider.key}")
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
            attempt_provider = provider
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
                    completion = attempt_provider.complete(system, user, temperature=0.7)
                except ProviderError as exc:
                    log.candidates.append(
                        CandidateLog(
                            attempt=attempt + 1,
                            text="",
                            provider=attempt_provider.name,
                            model=attempt_provider.default_model,
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
                        props = extract_propositions(unit.text, attempt_provider)
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
                provider=provider.name,
                model=provider.default_model,
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
