# ABOUTME: convert(text, voice): iteratively rewrites only the flagged units of a text into a
# ABOUTME: voice, validating each rewrite (content lock, length, no new tells) before accepting.
from __future__ import annotations

import datetime as dt
import difflib
import json
import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from deslop.lock import LOCK_CLASSES, check, inventory
from deslop.metrics import compute_all, detect_tells
from deslop.nlp import Analysis, Sentence, analyze
from deslop.patterns import Catalog, load_catalog
from deslop.providers.base import CachingProvider, ProviderError, Usage, get_provider
from deslop.report import Report, TellHitResult
from deslop.score import score
from deslop.voice.exemplars import parse_exemplars
from deslop.voice.model import Voice

Mode = Literal["edit", "rewrite"]
DEFAULT_MODELS = ("claude", "gpt", "claude", "gpt")
PAIRS_DIR = Path(os.environ.get("DESLOP_HOME", Path.home() / ".deslop")) / "pairs"


# ---------------------------------------------------------------- result models


class UnitLog(BaseModel):
    iteration: int
    unit_index: int
    kind: str
    before: str
    after: str | None
    tells: list[str]
    provider: str
    model: str
    accepted: bool
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


class ConvertResult(BaseModel):
    text: str
    report_before: Report
    report_after: Report
    diff: str
    iterations: list[IterationLog]
    units: list[UnitLog]
    cost: Cost
    stopped: str  # pass | max_iter | budget | nothing_flagged

    def to_json(self, indent: int | None = 1) -> str:
        return self.model_dump_json(indent=indent)

    @property
    def changed(self) -> bool:
        return any(u.accepted for u in self.units)


# ---------------------------------------------------------------- units


@dataclass
class Unit:
    index: int
    kind: str  # sentence | passage (adjacent flagged sentences) | paragraph
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
    """Sentences to rewrite when only rhythm metrics are flagged: every monotone run of
    three, or failing that every pair of similar-length neighbours."""
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
        elif m.name in ("sent_len_mean",):
            direction = "longer" if m.z < 0 else "shorter"
            notes.append(f"Sentences should run {direction} on average in this voice.")
    return sorted(set(notes))


def select_units(text: str, a: Analysis, report: Report, mode: Mode) -> list[Unit]:
    hits = flagged_hits(report)
    notes = metric_notes_for(report)
    units: list[Unit] = []
    if mode == "edit":
        by_sentence: dict[int, dict[tuple[str, int, int], TellHitResult]] = {}
        for h in hits:
            for s in a.sentences:
                if s.start < h.end and h.start < s.end:
                    by_sentence.setdefault(s.index, {})[(h.pattern_id, h.start, h.end)] = h
        targets = set(by_sentence)
        if not targets and notes:
            targets = rhythm_targets(a)
        # Adjacent flagged sentences in one paragraph form a single unit so the model can
        # split or merge them; everything outside these units stays byte-identical.
        groups: list[list[Sentence]] = []
        for s in a.sentences:
            if s.index not in targets:
                continue
            last = groups[-1][-1] if groups else None
            if last and last.index == s.index - 1 and last.para_index == s.para_index:
                groups[-1].append(s)
            else:
                groups.append([s])
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
    else:
        by_para: dict[int, list[TellHitResult]] = {}
        for h in hits:
            by_para.setdefault(h.para_idx, []).append(h)
        targets = set(by_para)
        if not targets and notes:
            targets = {a.sentences[i].para_index for i in rhythm_targets(a)}
        for p in a.paragraphs:
            if p.index in targets:
                units.append(
                    Unit(
                        index=len(units),
                        kind="paragraph",
                        start=p.start,
                        end=p.end,
                        text=text[p.start : p.end],
                        para_index=p.index,
                        hits=by_para.get(p.index, []),
                        metric_notes=notes,
                    )
                )
    return units


# ---------------------------------------------------------------- prompts

SYSTEM_TEMPLATE = """You rewrite prose into a specific writer's voice. You change only the unit
you are given, keep every fact, citation, quotation, number and name exactly as they are, add
nothing and remove nothing of substance. You return the rewritten unit and nothing else: no
preamble, no quotation marks around it, no commentary.

# The voice
{profile}

{rules}
"""

UNIT_TEMPLATE = """# Exemplars of the voice
{exemplars}

# Context (DO NOT EDIT; shown only so the rewrite fits)
BEFORE: {before}
AFTER: {after}

# The {kind} to rewrite
{unit}

# What is wrong with it
{problems}

# Must survive the rewrite exactly (content lock)
{lock}

# Constraints
- Return only the rewritten {kind}, in the voice, between {min_words} and {max_words} words.
- Do not repeat or alter the surrounding context.
- Do not add a new fact, name, number, quotation or citation.
{extra}{rejected}"""

REJECTED_TEMPLATE = """

# Earlier attempts that were rejected (do not repeat them)
{attempts}"""


SENTENCE_SLACK_WORDS = 4  # a single sentence needs room to change rhythm


def length_bounds(unit: Unit, preserve_length: float) -> tuple[int, int]:
    """Allowed word count for a rewrite: +/-preserve_length, plus slack for sentences."""
    words = len(unit.text.split())
    slack = SENTENCE_SLACK_WORDS if unit.kind == "sentence" else 0
    low = max(1, int(words * (1 - preserve_length)) - slack)
    high = int(math.ceil(words * (1 + preserve_length))) + 1 + slack
    return low, high


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


class ExemplarIndex:
    """Exemplar bodies with metric vectors, for picking the closest to a unit."""

    def __init__(self, voice: Voice):
        self.names = [n for n, s in voice.metrics.metrics.items() if s.std > 0]
        self.voice = voice
        self.bodies = parse_exemplars(voice.exemplars)
        self.vectors = [_zvector(compute_all(analyze(b)), voice, self.names) for b in self.bodies]

    def closest(self, paragraph_text: str, k: int = 3) -> list[str]:
        if not self.bodies:
            return []
        v = _zvector(compute_all(analyze(paragraph_text)), self.voice, self.names)
        dists = [
            math.sqrt(sum((x - y) ** 2 for x, y in zip(v, w, strict=True))) for w in self.vectors
        ]
        order = sorted(range(len(self.bodies)), key=lambda i: dists[i])
        return [self.bodies[i] for i in order[:k]]


def build_prompt(
    unit: Unit,
    text: str,
    a: Analysis,
    voice: Voice,
    exemplars: ExemplarIndex,
    lock_classes: tuple[str, ...],
    preserve_length: float,
    catalog: Catalog,
    rejected: list[tuple[str, str]] | None = None,
) -> tuple[str, str]:
    """Build (system, user) prompts. `rejected` lists (candidate, reason) pairs from earlier
    attempts on this unit so the next attempt avoids the same mistake."""
    system = SYSTEM_TEMPLATE.format(profile=voice.profile.strip(), rules=voice.rules.strip())
    para = a.paragraphs[unit.para_index]
    before = text[max(para.start, unit.start - 400) : unit.start].strip() or "(start of text)"
    after = text[unit.end : min(para.end, unit.end + 400)].strip() or "(end of paragraph)"
    if unit.kind == "paragraph":
        prev_p = a.paragraphs[unit.para_index - 1] if unit.para_index > 0 else None
        next_p = (
            a.paragraphs[unit.para_index + 1] if unit.para_index + 1 < len(a.paragraphs) else None
        )
        before = text[prev_p.start : prev_p.end][-400:].strip() if prev_p else "(start of text)"
        after = text[next_p.start : next_p.end][:400].strip() if next_p else "(end of text)"
    low, high = length_bounds(unit, preserve_length)
    inv = inventory(unit.text)
    lock_lines = inv.describe(lock_classes) or ["(nothing locked in this unit)"]
    extra = ""
    if unit.kind == "paragraph":
        extra = "- Keep the paragraph as a single paragraph; do not split it."
    elif unit.kind == "passage":
        extra = "- You may split or merge sentences within this passage; keep it on one line."
    if "T15" in unit.tell_ids:
        extra += "\n- Vary sentence length sharply: at least one sentence under 8 words."
    user = UNIT_TEMPLATE.format(
        exemplars="\n\n".join(exemplars.closest(text[para.start : para.end])),
        before=before,
        after=after,
        kind=unit.kind,
        unit=unit.text,
        problems=_problems(unit, catalog),
        lock="\n".join(f"- {line}" for line in lock_lines),
        min_words=low,
        max_words=high,
        extra=extra,
        rejected=_rejected_section(rejected or [], catalog),
    )
    return system, user


def _rejected_section(rejected: list[tuple[str, str]], catalog: Catalog) -> str:
    if not rejected:
        return ""
    lines = []
    for candidate, reason in rejected:
        detail = reason
        if reason.startswith("new tells: "):
            names = []
            for pid in reason[len("new tells: ") :].split(", "):
                p = catalog.by_id(pid)
                names.append(f"{pid} {p.name} ({p.description.rstrip('.')})")
            detail = "introduced " + "; ".join(names)
        lines.append(f'- Rejected ({detail}): "{candidate[:300]}"')
    return REJECTED_TEMPLATE.format(attempts="\n".join(lines))


# ---------------------------------------------------------------- validation


def clean_response(raw: str) -> str:
    out = raw.strip()
    for fence in ("```", "'''"):
        if out.startswith(fence) and out.endswith(fence):
            out = out[len(fence) : -len(fence)].strip()
    if len(out) > 2 and out[0] in '"“' and out[-1] in '"”' and out.count('"') == 2:
        out = out[1:-1].strip()
    return out


def validate(
    unit: Unit,
    candidate: str,
    *,
    context_before: str,
    context_after: str,
    lock_classes: tuple[str, ...],
    preserve_length: float,
    catalog: Catalog,
    allowed_names: set[str] | None = None,
) -> str | None:
    """Return a rejection reason, or None when the candidate is acceptable."""
    if not candidate or candidate == unit.text:
        return "empty or unchanged"
    if unit.kind == "paragraph" and "\n\n" in candidate:
        return "split the paragraph"
    if unit.kind in ("sentence", "passage") and "\n" in candidate:
        return "contains line breaks"
    for ctx in (context_before, context_after):
        tail = ctx.strip()
        if len(tail) > 40 and tail[-40:] in candidate:
            return "repeated the surrounding context"
    words_after = len(candidate.split())
    low, high = length_bounds(unit, preserve_length)
    if not low <= words_after <= high:
        return f"length {words_after} words outside {low}-{high}"
    violations = check(unit.text, candidate, lock_classes, allowed_names=allowed_names)
    if violations:
        return "content lock: " + "; ".join(str(v) for v in violations)
    # Judge tells in paragraph context so cross-sentence patterns (unanswered questions,
    # paragraph closes) are detected the same way the final rescore will see them.
    before_hits = _tell_counts(context_before + unit.text + context_after, catalog)
    after_hits = _tell_counts(context_before + candidate + context_after, catalog)
    worse = [pid for pid, n in after_hits.items() if n > before_hits.get(pid, 0)]
    if worse:
        return "new tells: " + ", ".join(sorted(worse))
    return None


def _tell_counts(text: str, catalog: Catalog) -> dict[str, int]:
    counts: dict[str, int] = {}
    for h in detect_tells(analyze(text), catalog):
        counts[h.pattern_id] = counts.get(h.pattern_id, 0) + 1
    return counts


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


# ---------------------------------------------------------------- the loop


def _splice(text: str, units: list[Unit], replacements: dict[int, str]) -> str:
    out = []
    pos = 0
    for u in sorted(units, key=lambda u: u.start):
        out.append(text[pos : u.start])
        out.append(replacements.get(u.index, u.text))
        pos = u.end
    out.append(text[pos:])
    return "".join(out)


def _log_pairs(voice: Voice, logs: list[UnitLog], path: Path | None) -> None:
    target = path or (PAIRS_DIR / f"{voice.name}.jsonl")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as fh:
            for log in logs:
                record = log.model_dump()
                record["voice"] = voice.name
                record["timestamp"] = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass


def convert(
    text: str,
    voice: Voice,
    *,
    mode: Mode = "edit",
    max_iter: int = 4,
    models: tuple[str, ...] = DEFAULT_MODELS,
    budget: float | None = None,
    sigma: float = 1.5,
    tell_mult: float = 2.0,
    tell_floor: float = 1.0,
    lock: tuple[str, ...] = LOCK_CLASSES,
    preserve_length: float = 0.15,
    patterns: tuple[str, ...] = (),
    provider_factory: ProviderFactory | None = None,
    pairs_path: Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> ConvertResult:
    catalog = load_catalog(patterns)
    report_before = score(
        text, voice, sigma=sigma, tell_mult=tell_mult, tell_floor=tell_floor, patterns=patterns
    )
    pool = ProviderPool(models, provider_factory)
    exemplars = ExemplarIndex(voice)
    iterations: list[IterationLog] = []
    unit_logs: list[UnitLog] = []

    document_names = set(inventory(text).proper_nouns)
    rejected_for: dict[str, list[tuple[str, str]]] = {}
    current = text
    report = report_before
    best_text, best_report = text, report_before
    stopped = "max_iter"
    if report_before.verdict == "PASS":
        stopped = "pass"
        max_iter = 0

    for i in range(max_iter):
        a = analyze(current)
        units = select_units(current, a, report, mode)
        if not units:
            stopped = "nothing_flagged"
            break
        provider = pool.for_iteration(i)
        if progress:
            progress(f"iteration {i + 1}: {len(units)} {mode} unit(s) via {provider.name}")
        replacements: dict[int, str] = {}
        for unit in units:
            if budget is not None and pool.usage().cost_usd > budget:
                stopped = "budget"
                break
            para = a.paragraphs[unit.para_index]
            ctx_before = current[para.start : unit.start]
            ctx_after = current[unit.end : para.end]
            attempt_provider = provider
            accepted = False
            for attempt in range(2):
                system, user = build_prompt(
                    unit,
                    current,
                    a,
                    voice,
                    exemplars,
                    lock,
                    preserve_length,
                    catalog,
                    rejected=rejected_for.get(unit.text),
                )
                try:
                    completion = attempt_provider.complete(system, user, temperature=0.7)
                except ProviderError as exc:
                    unit_logs.append(
                        UnitLog(
                            iteration=i + 1,
                            unit_index=unit.index,
                            kind=unit.kind,
                            before=unit.text,
                            after=None,
                            tells=unit.tell_ids,
                            provider=attempt_provider.name,
                            model=attempt_provider.default_model,
                            accepted=False,
                            reason=f"provider error: {exc}"[:200],
                        )
                    )
                    attempt_provider = pool.other_than(attempt_provider)
                    continue
                candidate = clean_response(completion.text)
                reason = validate(
                    unit,
                    candidate,
                    context_before=ctx_before,
                    context_after=ctx_after,
                    lock_classes=lock,
                    preserve_length=preserve_length,
                    catalog=catalog,
                    allowed_names=document_names,
                )
                unit_logs.append(
                    UnitLog(
                        iteration=i + 1,
                        unit_index=unit.index,
                        kind=unit.kind,
                        before=unit.text,
                        after=candidate,
                        tells=unit.tell_ids,
                        provider=completion.provider,
                        model=completion.model,
                        accepted=reason is None,
                        reason=reason or "",
                    )
                )
                if reason is None:
                    replacements[unit.index] = candidate
                    accepted = True
                    break
                rejected_for.setdefault(unit.text, []).append((candidate, reason))
                if attempt == 0:
                    attempt_provider = pool.other_than(provider)
            if progress and not accepted:
                progress(f"  unit {unit.index} kept original ({unit_logs[-1].reason})")
        if stopped == "budget":
            break
        current = _splice(current, units, replacements)
        report = score(
            current,
            voice,
            sigma=sigma,
            tell_mult=tell_mult,
            tell_floor=tell_floor,
            patterns=patterns,
        )
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

    usage = pool.usage()
    _log_pairs(voice, unit_logs, pairs_path)
    diff = "".join(
        difflib.unified_diff(
            text.splitlines(keepends=True),
            best_text.splitlines(keepends=True),
            fromfile="before",
            tofile="after",
        )
    )
    return ConvertResult(
        text=best_text,
        report_before=report_before,
        report_after=best_report,
        diff=diff,
        iterations=iterations,
        units=unit_logs,
        cost=Cost(
            usd=round(usage.cost_usd, 4),
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            calls=usage.calls,
            cached=usage.cached,
            unpriced_calls=usage.unpriced,
        ),
        stopped=stopped,
    )
