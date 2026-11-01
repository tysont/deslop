# ABOUTME: apply(text_or_brief, voice): generate-then-verify rewriting. Each paragraph gets n
# ABOUTME: candidates checked by the content lock, the semantic lock, the linter and a judge.
from __future__ import annotations

import datetime as dt
import difflib
import json
import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from deslop.draft import Brief, SectionLog, draft_sections
from deslop.judge import (
    GATE_CONFIDENCE,
    JudgeCalibration,
    PairVerdict,
    beats_original,
    compare_with,
    judge_paragraph,
    win_rate,
)
from deslop.lock import DocumentLock, check, inventory
from deslop.metrics import compute_all, detect_tells
from deslop.nlp import WORD_RE, Analysis, Sentence, analyze
from deslop.patterns import Catalog, load_catalog
from deslop.providers.base import CachingProvider, ProviderError, Usage, get_provider
from deslop.reference import PrimaryCheck, PrimaryText, load_primary_texts, verify_additions
from deslop.report import JudgeScores, Report, TellHitResult
from deslop.score import score
from deslop.semantic import SemanticResult, check_meaning, extract_propositions
from deslop.voice.exemplars import parse_exemplars
from deslop.voice.model import Voice

Mode = Literal["rewrite", "edit"]
DEFAULT_MODELS = ("claude", "gpt")


def pairs_dir() -> Path:
    """Where accepted and rejected rewrite pairs are appended: $DESLOP_HOME/pairs."""
    return Path(os.environ.get("DESLOP_HOME", Path.home() / ".deslop")) / "pairs"


CANDIDATE_TEMPERATURE = 0.9
DEFAULT_MAX_GROWTH = 15.0  # percent by which a candidate may outgrow its paragraph
SECTION_TELLS = {"T14", "T15"}  # need cross-paragraph room
CLOSING_RULE_TELL = "T5"  # rule R3: do not end a paragraph on an abstract aphorism


# ---------------------------------------------------------------- result models


class CandidateLog(BaseModel):
    """One sampled rewrite of a unit and everything the verifiers said about it."""

    attempt: int
    text: str
    provider: str
    model: str
    lexical_ok: bool
    lexical_detail: str = ""
    semantic: SemanticResult | None = None
    primary: PrimaryCheck | None = None  # added quotations and citations, verified or not
    pairs: list[PairVerdict] = Field(default_factory=list)  # comparative judge verdicts
    win_rate: float | None = None
    beats_original: bool | None = None
    judge: JudgeScores | None = None  # absolute rubric, secondary (winner only)
    distance: float | None = None
    growth_pct: float | None = None
    new_tells: list[str] = Field(default_factory=list)
    accepted: bool = False
    reason: str = ""

    @property
    def confidence_vs_original(self) -> float:
        return max(
            (p.confidence for p in self.pairs if p.against == "original" and p.wins), default=0.0
        )


class UnitLog(BaseModel):
    """What happened to one unit: its propositions, every candidate, which won and why."""

    index: int
    kind: str  # paragraph | section | sentence | passage
    start: int
    end: int
    original: str
    final: str
    propositions: list[str] = Field(default_factory=list)
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


class Result(BaseModel):
    """What [`apply`][deslop.apply.apply] produced.

    `text` is the output (the input where nothing was accepted); `report_before` and
    `report_after` score both; `diff` is a unified diff and `meaning_diff` lists every unit's
    propositions with whether the output kept them plus its added quotations and citations;
    `unverified_additions` are the added quotations and citations no primary text confirmed;
    `growth_pct` is the document's word-count growth; `units` logs each unit's candidates,
    verdicts, win rates and the winner; `judge_gate` says which providers judged and whether
    the gate was on; `iterations` is filled in edit mode; `cost` totals tokens and USD;
    `stopped` says why the loop ended; `exemplars_used` is False when the voice was packed
    without exemplars; `draft` holds the section logs when a brief was applied.
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
    min_confidence: float = GATE_CONFIDENCE
    max_growth: float = DEFAULT_MAX_GROWTH
    growth_pct: float = 0.0
    unverified_additions: list[str] = Field(default_factory=list)
    primary_texts: list[str] = Field(default_factory=list)
    judge_gate: str = ""
    draft: list[SectionLog] | None = None
    draft_text: str | None = None

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


def metric_notes_for(report: Report) -> list[str]:
    notes = []
    for m in report.flagged_metrics():
        if m.name in ("sent_len_std", "sent_len_cv", "max_run_similar_len"):
            notes.append(
                "Sentence lengths are too uniform; vary them sharply: merge some into a "
                "single sentence of over 35 words and cut others to under 8."
            )
        elif m.name == "opener_top_word_share":
            notes.append("Too many sentences begin with the same word; open this one differently.")
        elif m.name == "opener_pct_adv_conj":
            notes.append(
                "Too many sentences open with an adverb or conjunction; start with the subject."
            )
        elif m.name in ("colons_per_1k", "em_dashes_per_1k", "questions_per_1k"):
            notes.append(f"{m.name.replace('_per_1k', '')} are overused; remove them here.")
        elif m.name in (
            "first_person_per_1k",
            "quote_per_para",
            "citation_per_100w",
            "concrete_per_100w",
        ):
            notes.append(f"{m.name} is too low for this voice; be more concrete and specific.")
        elif m.name == "sent_len_mean":
            direction = "longer" if m.z < 0 else "shorter"
            notes.append(f"Sentences should run {direction} on average in this voice.")
    return sorted(set(notes))


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


def rewrite_units(text: str, a: Analysis, report: Report) -> list[Unit]:
    """Rewrite mode: every prose paragraph is a unit. A paragraph flagged for a tell that
    needs cross-paragraph room (T14, T15) is merged with its predecessor into a section of
    two or three paragraphs."""
    hits = flagged_hits(report)
    by_para: dict[int, list[TellHitResult]] = {}
    for h in hits:
        by_para.setdefault(h.para_idx, []).append(h)
    notes = metric_notes_for(report)
    units: list[Unit] = []
    for p in a.paragraphs:
        p_hits = by_para.get(p.index, [])
        needs_room = any(h.pattern_id in SECTION_TELLS for h in p_hits)
        if needs_room and units and units[-1].para_index == p.index - 1:
            last = units[-1]
            span = sum(1 for u in [last] for _ in range(1)) + last.text.count("\n\n") + 1
            if span < 3:
                last.kind = "section"
                last.end = p.end
                last.text = text[last.start : p.end]
                last.hits = last.hits + p_hits
                last.para_index = p.index
                continue
        units.append(
            Unit(
                index=len(units),
                kind="paragraph",
                start=p.start,
                end=p.end,
                text=text[p.start : p.end],
                para_index=p.index,
                hits=p_hits,
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

REWRITE_TEMPLATE = """# Exemplars of the voice (nearest in subject)
{exemplars}

# Where this passage sits (DO NOT rewrite; shown for continuity)
BEFORE: {before}
AFTER: {after}

# The {kind} to write afresh
{unit}

# Its propositions (keep every one, add none)
{propositions}

# Names, numbers, quotations and citations already in the document (you may use these)
{allowlist}

# Instruction
Write this {kind} as this writer would. Keep every proposition above; add no new claim. You
may change every sentence, reorder, split and merge sentences, and drop connective filler.
Cite the primary text where this writer would, quoting it only in its exact words and citing
only locations that exist in it, and address the reader only where this writer would. Keep it
as {shape}, {min_words} to {max_words} words; it must not run longer than
{max_words}.{extra}{issues}{rejected}

Draft {attempt} of {n}."""

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
    if unit.kind in ("paragraph", "section"):
        first_para = next(p for p in a.paragraphs if p.start >= unit.start or p.end > unit.start)
        prev_p = a.paragraphs[first_para.index - 1] if first_para.index > 0 else None
        next_p = a.paragraphs[para.index + 1] if para.index + 1 < len(a.paragraphs) else None
        before = text[prev_p.start : prev_p.end][-400:].strip() if prev_p else "(start of text)"
        after = text[next_p.start : next_p.end][:400].strip() if next_p else "(end of text)"
        return before, after
    before = text[max(para.start, unit.start - 400) : unit.start].strip() or "(start of text)"
    after = text[unit.end : min(para.end, unit.end + 400)].strip() or "(end of paragraph)"
    return before, after


def rewrite_prompt(
    unit: Unit,
    text: str,
    a: Analysis,
    voice: Voice,
    exemplars: ExemplarIndex,
    document: DocumentLock,
    propositions: list[str],
    *,
    attempt: int,
    n: int,
    issues: list[str],
    rejected: list[tuple[str, str]],
    max_growth: float = DEFAULT_MAX_GROWTH,
) -> tuple[str, str]:
    system = SYSTEM_TEMPLATE.format(profile=voice.profile.strip(), rules=voice.rules.strip())
    before, after = _context(unit, text, a)
    words = len(unit.text.split())
    allow = []
    if document.names:
        allow.append("names: " + ", ".join(sorted(document.names)[:60]))
    if document.numbers:
        allow.append("numbers: " + ", ".join(sorted(document.numbers)[:40]))
    if document.citations:
        allow.append("citations: " + "; ".join(sorted(document.citations)[:40]))
    if document.quotes:
        allow.append("quotations: " + " | ".join(f'"{q}"' for q in sorted(document.quotes)[:20]))
    shape = "one paragraph" if unit.kind == "paragraph" else "the same number of paragraphs"
    user = REWRITE_TEMPLATE.format(
        exemplars="\n\n".join(exemplars.closest(unit.text)) or "(no exemplars in this voice)",
        before=before,
        after=after,
        kind=unit.kind,
        unit=unit.text,
        propositions="\n".join(f"{i}. {p}" for i, p in enumerate(propositions, 1)) or "(none)",
        allowlist="\n".join(f"- {line}" for line in allow) or "- (nothing yet)",
        shape=shape,
        min_words=min(max(40, int(words * 0.7)), words),
        max_words=max_words_for(words, max_growth),
        extra="",
        issues=(
            "\n\nAddress these faults in the current version:\n"
            + "\n".join(f"- {i}" for i in issues)
        )
        if issues
        else "",
        rejected=_rejected_section(rejected),
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


class ProviderPool:
    def __init__(self, names: tuple[str, ...], factory: ProviderFactory | None = None):
        self.names = names
        self.factory = factory or (lambda n: get_provider(n))
        self._cache: dict[str, CachingProvider] = {}
        self.unavailable: dict[str, str] = {}

    def get(self, name: str) -> CachingProvider | None:
        if name in self.unavailable:
            return None
        if name not in self._cache:
            try:
                self._cache[name] = self.factory(name)
            except ProviderError as exc:
                self.unavailable[name] = str(exc)
                return None
        return self._cache[name]

    def for_iteration(self, i: int) -> CachingProvider:
        order = [self.names[(i + k) % len(self.names)] for k in range(len(self.names))]
        for name in order:
            p = self.get(name)
            if p is not None:
                return p
        raise ProviderError("no provider available: " + "; ".join(self.unavailable.values()))

    def other_than(self, provider: CachingProvider) -> CachingProvider:
        for name in self.names:
            p = self.get(name)
            if p is not None and p.name != provider.name:
                return p
        return provider

    def usage(self) -> Usage:
        total = Usage()
        seen: set[int] = set()
        for p in self._cache.values():
            if id(p) not in seen:
                seen.add(id(p))
                total.add(p.usage)
        return total


class JudgePlan:
    """Who may judge whom. A provider judges only candidates it did not write, and only if
    its measured calibration (voice.yaml `judge_calibration`) reached 80% accuracy; a
    provider with no calibration is trusted. Writers are the providers some judge is clear
    of. When no judge qualifies the gate is off and candidates are ranked by the linter."""

    def __init__(self, pool: ProviderPool, calibration: dict[str, Any]):
        self.pool = pool
        self.available: list[tuple[str, CachingProvider]] = []
        for alias in pool.names:
            p = pool.get(alias)
            if p is not None and all(q.name != p.name for _, q in self.available):
                self.available.append((alias, p))
        self.calibration: dict[str, JudgeCalibration] = {}
        for name, data in calibration.items():
            try:
                self.calibration[name] = JudgeCalibration.model_validate(data)
            except ValueError:
                continue
        self.judges = [p for _, p in self.available if self._qualified(p)]
        self.writers = [
            alias for alias, p in self.available if any(j.name != p.name for j in self.judges)
        ]
        if not self.writers:
            self.writers = [alias for alias, _ in self.available]
            self.judges = []
        self.enabled = bool(self.judges)

    def _qualified(self, p: CachingProvider) -> bool:
        cal = self.calibration.get(p.name)
        return cal is None or cal.usable

    def judges_for(self, *writer_names: str) -> list[CachingProvider]:
        """Judges clear of every writer named; empty when none is (the pair is skipped)."""
        clear = [j for j in self.judges if j.name not in writer_names]
        return clear[:1]

    def writer(self, i: int) -> CachingProvider:
        for k in range(len(self.writers)):
            p = self.pool.get(self.writers[(i + k) % len(self.writers)])
            if p is not None:
                return p
        return self.pool.for_iteration(i)

    def describe(self) -> str:
        if not self.available:
            return "no provider available"
        parts = []
        for _, p in self.available:
            cal = self.calibration.get(p.name)
            role = "judge" if p in self.judges else "writer only"
            acc = f"{100 * cal.accuracy:.0f}% on {cal.pairs} pairs" if cal else "uncalibrated"
            parts.append(f"{p.name}: {role} ({acc})")
        state = "on" if self.enabled else "OFF (no provider reached 80% as judge)"
        return f"judge gate {state}; " + "; ".join(parts)


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
            lines.append(f"  + [{added.kind}] {added.text}")
        for add in chosen.primary.additions if chosen.primary else []:
            mark = "✓" if add.verified else "?"
            lines.append(f"  {mark} [{add.kind}] {add}")
    return "\n".join(lines)


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
    models: tuple[str, ...] = DEFAULT_MODELS,
    budget: float | None = None,
    n: int = 3,
    min_confidence: float = GATE_CONFIDENCE,
    max_growth: float = DEFAULT_MAX_GROWTH,
    primary_texts: list[str | Path | PrimaryText] | None = None,
    max_iter: int = 4,
    preserve_length: float = 0.15,
    patterns: tuple[str, ...] = (),
    provider_factory: ProviderFactory | None = None,
    pairs_path: Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> Result:
    """Write `text_or_brief` in `voice` and return a [`Result`][deslop.Result].

    Rewrite mode (default) treats every paragraph as a unit: its propositions are extracted,
    `n` candidates are sampled across the configured providers in rotation, and each is
    checked by the content lock, the length limit (`max_growth` percent over the original),
    the primary-text check, the semantic lock (a different provider) and the linter in
    document context; the survivors are judged pairwise against the original and each other
    by a provider that wrote neither, and the candidate with the best win rate that beats
    the original with at least `min_confidence` wins, with one retry naming the failures,
    else the original is kept. A document-level pass then rewrites units flagged for summary
    closes, monotone runs or aphoristic endings. Edit mode rewrites only flagged sentences
    and passages under the same locks. A `Brief` is drafted section by section first, then
    rewritten. `primary_texts` (bundled names, files, directories or loaded `PrimaryText`s)
    default to the brief's `primary_texts` plus the voice's; without any, added quotations
    and citations are allowed and listed as unverified. `provider_factory` maps a provider
    name to a `CachingProvider` (tests replay a cache with it); `budget` is USD.
    """
    catalog = load_catalog(patterns)
    pool = ProviderPool(models, provider_factory)
    plan = JudgePlan(pool, voice.meta.judge_calibration)
    draft_logs: list[SectionLog] | None = None
    draft_text: str | None = None
    sources: list[str | Path | PrimaryText] = list(primary_texts or [])
    if isinstance(text_or_brief, Brief):
        sources = sources or [*text_or_brief.primary_texts, *voice.meta.primary_texts]
        text, draft_logs = draft_sections(text_or_brief, voice, pool, progress=progress)
        draft_text = text
        mode = "rewrite"
    else:
        sources = sources or list(voice.meta.primary_texts)
        text = text_or_brief
    texts = _load_texts(sources)
    if progress:
        progress(plan.describe())
        if texts:
            progress("primary texts: " + ", ".join(t.name for t in texts))
    report_before = score(text, voice, patterns=patterns)
    exemplars = ExemplarIndex(voice)
    document = DocumentLock.build(text)
    opts = Options(
        plan=plan,
        texts=texts,
        min_confidence=min_confidence,
        max_growth=max_growth,
        patterns=patterns,
        n=n,
        progress=progress,
    )

    if mode == "edit":
        result = _edit_loop(
            text,
            voice,
            report_before,
            pool,
            exemplars,
            document,
            catalog,
            opts,
            budget=budget,
            max_iter=max_iter,
            preserve_length=preserve_length,
        )
    else:
        result = _rewrite_loop(
            text,
            voice,
            report_before,
            pool,
            exemplars,
            document,
            catalog,
            opts,
            budget=budget,
        )
    final_text, units, iterations, stopped = result
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
    return Result(
        mode=mode,
        text=final_text,
        report_before=report_before,
        report_after=report_after,
        diff=diff,
        meaning_diff=_meaning_diff(units),
        units=units,
        iterations=iterations,
        cost=_cost(pool.usage()),
        stopped=stopped,
        exemplars_used=exemplars.available,
        min_confidence=min_confidence,
        max_growth=max_growth,
        growth_pct=growth_of(text, final_text),
        unverified_additions=_unverified(units),
        primary_texts=[t.name for t in texts],
        judge_gate=plan.describe(),
        draft=draft_logs,
        draft_text=draft_text,
    )


@dataclass
class Options:
    """Everything the unit loops share: the judge plan, the primary texts, the gates."""

    plan: JudgePlan
    texts: list[PrimaryText]
    min_confidence: float
    max_growth: float
    patterns: tuple[str, ...]
    n: int
    progress: Callable[[str], None] | None


def _load_texts(sources: list[str | Path | PrimaryText]) -> list[PrimaryText]:
    loaded = [t for t in sources if isinstance(t, PrimaryText)]
    names = [t for t in sources if not isinstance(t, PrimaryText)]
    return loaded + load_primary_texts(names)


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
    pool: ProviderPool,
    exemplars: ExemplarIndex,
    document: DocumentLock,
    catalog: Catalog,
    opts: Options,
    *,
    budget: float | None,
) -> tuple[str, list[UnitLog], list[IterationLog], str]:
    patterns, progress = opts.patterns, opts.progress
    a = analyze(text)
    units = rewrite_units(text, a, report_before)
    logs: list[UnitLog] = []
    current = text
    stopped = "done"
    offset = 0  # how much earlier replacements shifted later units
    unit_spans: list[tuple[int, int]] = []
    for unit in units:
        if budget is not None and pool.usage().cost_usd > budget:
            stopped = "budget"
            logs.append(_kept(unit, [], "budget exhausted"))
            continue
        start, end = unit.start + offset, unit.end + offset
        log = _rewrite_unit(
            unit,
            current,
            start,
            end,
            voice,
            pool,
            exemplars,
            document,
            catalog,
            opts,
            issues=[],
        )
        logs.append(log)
        current = current[:start] + log.final + current[end:]
        unit_spans.append((start, start + len(log.final)))
        offset += len(log.final) - len(unit.text)

    # Document-level pass: summary closes, monotone runs across paragraphs, aphoristic ends.
    report = score(current, voice, patterns=patterns)
    doc_hits = [
        h
        for t in report.tells
        if t.id in SECTION_TELLS | {CLOSING_RULE_TELL} and (t.flagged or t.banned)
        for h in t.hits
    ]
    if doc_hits and stopped != "budget":
        a2 = analyze(current)
        touched: dict[int, list[TellHitResult]] = {}
        for h in doc_hits:
            touched.setdefault(h.para_idx, []).append(h)
        if progress:
            progress(f"document pass: {len(touched)} paragraph(s) with T14/T15/T5 after rewriting")
        offset2 = 0
        for p in a2.paragraphs:
            if p.index not in touched:
                continue
            unit = Unit(
                index=len(logs),
                kind="paragraph",
                start=p.start,
                end=p.end,
                text=current[p.start : p.end],
                para_index=p.index,
                hits=touched[p.index],
                metric_notes=[],
            )
            issues = [
                f"{h.pattern_id} {catalog.by_id(h.pattern_id).name}: {h.text.strip()[:80]}"
                + (f" Fix: {FIX_HINTS[h.pattern_id]}" if h.pattern_id in FIX_HINTS else "")
                for h in touched[p.index]
            ]
            start, end = p.start + offset2, p.end + offset2
            log = _rewrite_unit(
                unit,
                current,
                start,
                end,
                voice,
                pool,
                exemplars,
                document,
                catalog,
                opts,
                issues=issues,
            )
            log.kind = "paragraph (document pass)"
            logs.append(log)
            current = current[:start] + log.final + current[end:]
            offset2 += len(log.final) - len(unit.text)
    return current, logs, [], stopped


def _rank_key(cand: CandidateLog) -> tuple[float, float, float]:
    """Comparative win rate first, then confidence over the original, then linter distance."""
    rate = cand.win_rate if cand.win_rate is not None else -1.0
    distance = cand.distance if cand.distance is not None else 1e9
    return (-rate, -cand.confidence_vs_original, distance)


def _kept(unit: Unit, propositions: list[str], reason: str) -> UnitLog:
    return UnitLog(
        index=unit.index,
        kind=unit.kind,
        start=unit.start,
        end=unit.end,
        original=unit.text,
        final=unit.text,
        propositions=propositions,
        kept_original=True,
        reason=reason,
    )


def _verify_candidate(
    unit: Unit,
    candidate: str,
    cand: CandidateLog,
    document: DocumentLock,
    opts: Options,
) -> str | None:
    """The locks and limits that need no judge: shape, length, lexical lock, primary texts.
    Returns the rejection reason, or None when the candidate may go on to the semantic lock."""
    if not candidate or candidate == unit.text:
        cand.lexical_ok = False
        return "empty or unchanged"
    if unit.kind == "paragraph" and "\n\n" in candidate.strip():
        cand.lexical_ok = False
        return "split the paragraph"
    cand.growth_pct = growth_of(unit.text, candidate)
    limit = max_words_for(len(unit.text.split()), opts.max_growth)
    if len(candidate.split()) > limit:
        return (
            f"length: {len(candidate.split())} words, over the {limit}-word limit "
            f"(+{opts.max_growth:.0f}%)"
        )
    lock = check(unit.text, candidate, document)
    if not lock.ok:
        cand.lexical_ok = False
        cand.lexical_detail = "; ".join(str(v) for v in lock.violations)
        return "content lock: " + cand.lexical_detail
    if lock.additions:
        cand.lexical_detail = "new items for the semantic lock: " + "; ".join(lock.additions)
    cand.primary = verify_additions(unit.text, candidate, opts.texts)
    if not cand.primary.ok:
        return "primary text: " + "; ".join(str(a) for a in cand.primary.rejected[:3])
    return None


def _rewrite_unit(
    unit: Unit,
    current: str,
    start: int,
    end: int,
    voice: Voice,
    pool: ProviderPool,
    exemplars: ExemplarIndex,
    document: DocumentLock,
    catalog: Catalog,
    opts: Options,
    *,
    issues: list[str],
) -> UnitLog:
    plan, n, progress = opts.plan, opts.n, opts.progress
    a = analyze(current)
    writer0 = plan.writer(0)
    propositions = extract_propositions(unit.text, writer0)
    log = UnitLog(
        index=unit.index,
        kind=unit.kind,
        start=start,
        end=end,
        original=unit.text,
        final=unit.text,
        propositions=propositions,
    )
    near = exemplars.closest(unit.text, 2)
    rejected: list[tuple[str, str]] = []
    for round_index in range(2):
        eligible: list[CandidateLog] = []
        writers: dict[int, str] = {}
        for k in range(n):
            attempt = round_index * n + k + 1
            writer = plan.writer(round_index * n + k)
            system, user = rewrite_prompt(
                unit,
                current,
                a,
                voice,
                exemplars,
                document,
                propositions,
                attempt=attempt,
                n=2 * n,
                issues=issues,
                rejected=rejected if round_index else [],
                max_growth=opts.max_growth,
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
            reason = _verify_candidate(unit, candidate, cand, document, opts)
            if reason is not None:
                cand.reason = reason
                if reason != "empty or unchanged":
                    rejected.append((candidate, reason))
                continue
            checkers = plan.judges_for(writer.name) or [pool.other_than(writer)]
            cand.semantic = check_meaning(unit.text, candidate, propositions, checkers[0])
            if not cand.semantic.ok:
                cand.reason = "meaning: " + "; ".join(cand.semantic.failing[:3])
                rejected.append((candidate, cand.reason))
                continue
            spliced = current[:start] + candidate + current[end:]
            cand.distance = score(spliced, voice, patterns=opts.patterns).distance
            cand.new_tells = _new_tells(current, spliced, catalog)
            eligible.append(cand)
            writers[id(cand)] = writer.name

        # Comparative judging: each survivor against the original, then round-robin.
        for cand in eligible:
            judges = plan.judges_for(writers[id(cand)])
            if judges:
                cand.pairs.append(compare_with(cand.text, unit.text, voice, judges, exemplars=near))
        for i, x in enumerate(eligible):
            for y in eligible[i + 1 :]:
                judges = plan.judges_for(writers[id(x)], writers[id(y)])
                if not judges:
                    continue  # no provider is clear of both writers
                v = compare_with(
                    x.text, y.text, voice, judges, exemplars=near, against=f"candidate {y.attempt}"
                )
                x.pairs.append(v)
                y.pairs.append(
                    PairVerdict(
                        against=f"candidate {x.attempt}",
                        wins=not v.wins,
                        picked_as_model=not v.picked_as_model,
                        confidence=v.confidence,
                        shown_as="B" if v.shown_as == "A" else "A",
                        provider=v.provider,
                        model=v.model,
                        note=v.note,
                    )
                )
        for cand in eligible:
            cand.win_rate = win_rate(cand.pairs)
            if plan.enabled:
                cand.beats_original = beats_original(cand.pairs, opts.min_confidence)
                if cand.beats_original:
                    cand.accepted = True
                    cand.reason = (
                        f"passed locks; beat the original at {cand.confidence_vs_original:.2f}"
                    )
                else:
                    vs = next((p for p in cand.pairs if p.against == "original"), None)
                    if vs is None:
                        cand.reason = "no judge clear of the writer; not gated"
                    elif vs.wins:
                        cand.reason = (
                            f"beat the original only at {vs.confidence:.2f} < "
                            f"{opts.min_confidence:.2f}: {vs.note}"
                        )
                    else:
                        cand.reason = f"judged less like the writer than the original: {vs.note}"
                    rejected.append((cand.text, cand.reason))
            else:
                cand.accepted = True
                cand.reason = "passed locks (judge gate off)"
        passing = [c for c in log.candidates if c.accepted]
        if passing:
            winner = min(passing, key=_rank_key)
            log.chosen = log.candidates.index(winner)
            log.final = winner.text
            judges = plan.judges_for(writers.get(id(winner), winner.provider))
            if judges:
                winner.judge = judge_paragraph(
                    winner.text, voice, judges[0], exemplars=near, propositions=propositions
                )
            rate = f"{100 * winner.win_rate:.0f}%" if winner.win_rate is not None else "n/a"
            log.reason = (
                f"candidate {log.chosen + 1} of {len(log.candidates)}: win rate {rate} over "
                f"{len(winner.pairs)} pair(s), distance {winner.distance:.1f}, "
                f"growth {winner.growth_pct:+.0f}%"
            )
            if progress:
                progress(f"  unit {unit.index + 1}: {log.reason}")
            return log
        if round_index == 0:
            # Retry once with the specific failures appended.
            issues = issues + [c.reason for c in log.candidates if c.reason][-n:]
    log.kept_original = True
    log.reason = "no candidate passed: " + "; ".join(
        f"{c.attempt}: {c.reason[:80]}" for c in log.candidates
    )
    if progress:
        progress(f"  unit {unit.index + 1}: kept original (no candidate passed)")
    return log


def _edit_loop(
    text: str,
    voice: Voice,
    report_before: Report,
    pool: ProviderPool,
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
        provider = pool.for_iteration(i)
        if progress:
            progress(f"iteration {i + 1}: {len(units)} edit unit(s) via {provider.name}")
        replacements: dict[int, str] = {}
        for unit in units:
            if budget is not None and pool.usage().cost_usd > budget:
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
                    attempt_provider = pool.other_than(attempt_provider)
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
                        checker = pool.other_than(attempt_provider)
                        props = extract_propositions(unit.text, attempt_provider)
                        log.propositions = props
                        cand.semantic = check_meaning(unit.text, candidate, props, checker)
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
                if attempt == 0:
                    attempt_provider = pool.other_than(provider)
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
