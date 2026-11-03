# ABOUTME: The units apply works on: sections of a document for rewrite mode, flagged sentences
# ABOUTME: and passages for edit mode, and the reader's phrasing of a flagged metric.
from __future__ import annotations

from dataclasses import dataclass

from deslop.nlp import Analysis, Paragraph, Sentence
from deslop.report import MetricResult, Report, TellHitResult

DEFAULT_CHUNK_WORDS = 2500  # longer documents are rewritten section by section


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
