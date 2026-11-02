# ABOUTME: Deterministic coherence checks on a rewrite: paragraphing against the voice, tense per
# ABOUTME: paragraph, sentence-initial pronouns with no antecedent, connective-opened sentences.
from __future__ import annotations

from collections import Counter

from pydantic import BaseModel, Field

from deslop.metrics.rhythm import connective_opener_ratio
from deslop.nlp import Analysis, Paragraph, Sentence, analyze
from deslop.stats import Band

PRONOUN_OPENERS = {"this", "that", "it", "these", "those"}
# First and second person never need an antecedent: the writer and the reader are given.
PERSONAL = {
    "i",
    "we",
    "you",
    "me",
    "us",
    "my",
    "our",
    "your",
    "mine",
    "ours",
    "yours",
    "myself",
    "ourselves",
    "yourself",
    "yourselves",
}
DOMINANT_SHARE = 0.5  # a tense is dominant when it carries more than half the finite verbs
CONNECTIVE_TOLERANCE = 0.5  # the output's ratio may differ from the original's by this share


def _dominant_tense(paragraphs: list[Paragraph]) -> str:
    counts: Counter[str] = Counter()
    for para in paragraphs:
        for tok in para.doc:
            if tok.pos_ not in ("VERB", "AUX") or "Fin" not in tok.morph.get("VerbForm", []):
                continue
            for tense in tok.morph.get("Tense", []):
                counts[{"Past": "past", "Pres": "present"}.get(tense, tense.lower())] += 1
    if not counts:
        return "none"
    tense, n = counts.most_common(1)[0]
    return tense if n > DOMINANT_SHARE * sum(counts.values()) else "mixed"


def paragraph_tense(para: Paragraph) -> str:
    """The dominant tense of the paragraph's finite verbs: past or present when one carries
    more than half of them, mixed when none does, none when no finite verb carries a tense."""
    return _dominant_tense([para])


class TenseDrift(BaseModel):
    paragraph: int  # 1-based
    before: str
    after: str
    example: str  # a sentence of the output in the drifted tense


def _example_in_tense(para: Paragraph, tense: str) -> str:
    for s in para.sentences:
        for tok in s.span:
            if tok.pos_ in ("VERB", "AUX") and "Fin" in tok.morph.get("VerbForm", []):
                got = [
                    {"Past": "past", "Pres": "present"}.get(t, t)
                    for t in tok.morph.get("Tense", [])
                ]
                if tense in got:
                    return s.text.strip()
    return para.sentences[0].text.strip() if para.sentences else ""


def tense_drift(before: Analysis, after: Analysis) -> list[TenseDrift]:
    """Paragraphs whose dominant tense changed to a different dominant tense, paired by
    position when the paragraph counts agree. A re-paragraphed output has no positions to
    pair, so there a paragraph drifts only into a tense that no paragraph of the original is
    dominated by (the original's dominant tense as a whole is reported as `before`); a
    paragraph that is mixed or has no finite verb on either side is not drift."""
    out = []
    paired = len(before.paragraphs) == len(after.paragraphs)
    whole = _dominant_tense(before.paragraphs)
    present = {paragraph_tense(p) for p in before.paragraphs} - {"none", "mixed"}
    for i, q in enumerate(after.paragraphs, 1):
        b = paragraph_tense(q)
        if paired:
            a = paragraph_tense(before.paragraphs[i - 1])
        elif b in present or whole in ("none", "mixed"):
            continue
        else:
            a = whole
        if a != b and a not in ("none", "mixed") and b not in ("none", "mixed"):
            out.append(TenseDrift(paragraph=i, before=a, after=b, example=_example_in_tense(q, b)))
    return out


class AntecedentFlag(BaseModel):
    sentence: int  # 0-based index over the output's sentences
    paragraph: int  # 1-based
    word: str
    text: str


def _has_noun_phrase(text_sentences: list[Sentence]) -> bool:
    return any(any(t.pos_ in ("NOUN", "PROPN") for t in s.span) for s in text_sentences)


def antecedent_flags(a: Analysis) -> list[AntecedentFlag]:
    """Sentences opening with a third-person pronoun, or with this/that/it, when the previous
    two sentences hold no noun phrase for it to refer to. First and second person (I, we,
    you and their forms) are exempt."""
    sentences = a.sentences
    out = []
    for i, s in enumerate(sentences):
        first = next((t for t in s.span if t.is_alpha), None)
        if first is None:
            continue
        if first.lower_ in PERSONAL:
            continue  # first and second person need no antecedent
        if first.pos_ != "PRON" and first.lower_ not in PRONOUN_OPENERS:
            continue
        if not _has_noun_phrase(sentences[max(0, i - 2) : i]):
            out.append(
                AntecedentFlag(
                    sentence=i, paragraph=s.para_index + 1, word=first.text, text=s.text.strip()
                )
            )
    return out


class ParagraphCheck(BaseModel):
    """Paragraphing follows the voice, not the input: the output's mean sentences per
    paragraph must sit in the exemplars' range when the voice has one; without a range the
    output must keep the original's paragraph count."""

    before: int  # paragraphs in the original
    after: int  # paragraphs in the output
    sentences_per_paragraph: float  # the output's mean
    band_low: float | None = None  # the exemplars' range, sentences per paragraph
    band_high: float | None = None
    ok: bool = True
    note: str = ""
    advisory: bool = False  # the band is reported, not enforced (short inputs)


def paragraph_check(before: Analysis, after: Analysis, band: Band | None) -> ParagraphCheck:
    n_before, n_after = len(before.paragraphs), len(after.paragraphs)
    per = len(after.sentences) / max(n_after, 1)
    check = ParagraphCheck(
        before=n_before,
        after=n_after,
        sentences_per_paragraph=per,
        band_low=band.low if band else None,
        band_high=band.high if band else None,
    )
    if band is None:
        if n_before != n_after:
            check.ok = False
            check.note = f"it has {n_after} paragraphs where the original has {n_before}"
        return check
    low, high = band.low if band.low is not None else 0.0, band.high
    if per < low or (high is not None and per > high):
        check.ok = False
        check.note = (
            f"its {n_after} paragraphs average {per:.1f} sentences; this writer's paragraphs run "
            f"from {low:.1f} to {high if high is not None else per:.1f} sentences, so "
            + ("merge paragraphs" if per < low else "split paragraphs")
        )
    return check


class ConnectiveCheck(BaseModel):
    before: float  # share of the original's sentences opening with a connective
    after: float
    band_low: float | None = None  # the voice's band, in percent
    band_high: float | None = None
    ok: bool = True
    note: str = ""
    advisory: bool = False  # reported, and told to the writer only when the judge finds
    # the candidate incoherent; never a rejection on its own


def connective_check(before: Analysis, after: Analysis, band: Band | None) -> ConnectiveCheck:
    """The output's share of connective-opened sentences must sit in the voice's band and
    within half of the original's share either way, with one sentence of slack."""
    b, c = connective_opener_ratio(before), connective_opener_ratio(after)
    slack = 1.0 / max(len(after.sentences), 1)
    low, high = b * (1 - CONNECTIVE_TOLERANCE) - slack, b * (1 + CONNECTIVE_TOLERANCE) + slack
    notes = []
    if not low <= c <= high:
        notes.append(
            f"{100 * c:.0f}% of the sentences open with and, but, so or the like; "
            f"the original has {100 * b:.0f}%"
        )
    band_low = band.low if band else None
    band_high = band.high if band else None
    pct = 100 * c
    if (band_low is not None and pct < band_low) or (band_high is not None and pct > band_high):
        notes.append(
            f"{pct:.0f}% of the sentences open with a connective, outside this writer's band "
            f"of {band_low or 0:.0f}% to {band_high if band_high is not None else 100:.0f}%"
        )
    return ConnectiveCheck(
        before=b,
        after=c,
        band_low=band_low,
        band_high=band_high,
        ok=not notes,
        note="; ".join(notes),
    )


class CoherenceResult(BaseModel):
    """The deterministic coherence checks on one rewrite: paragraphing against the voice,
    tense drift per paragraph, pronouns without antecedents, and the connective-opener check.
    `ok` is True only when every check passes; `reader_notes()` phrases each failure for a
    retry prompt."""

    paragraphs_before: int
    paragraphs_after: int
    paragraphs: ParagraphCheck
    tense: list[TenseDrift] = Field(default_factory=list)
    antecedents: list[AntecedentFlag] = Field(default_factory=list)
    connectives: ConnectiveCheck

    @property
    def ok(self) -> bool:
        return (
            (self.paragraphs.ok or self.paragraphs.advisory)
            and not self.tense
            and not self.antecedents
            and (self.connectives.ok or self.connectives.advisory)
        )

    def reader_notes(self) -> list[str]:
        """What a reader would say about the failures; an advisory paragraph band is left
        out, since it is a note to the writer and not a rejection."""
        notes = []
        if not self.paragraphs.ok and not self.paragraphs.advisory:
            notes.append(self.paragraphs.note)
        for d in self.tense:
            notes.append(
                f"paragraph {d.paragraph} shifts from the {d.before} tense to the {d.after}: "
                f'"{d.example[:120]}"'
            )
        for f in self.antecedents:
            notes.append(
                f'"{f.text[:100]}" opens with "{f.word}" and nothing in the two sentences before '
                "it says what that refers to"
            )
        if not self.connectives.ok and not self.connectives.advisory:
            notes.append(self.connectives.note)
        return notes


def check_coherence(
    original: str,
    candidate: str,
    *,
    band: Band | None,
    para_band: Band | None = None,
    before: Analysis | None = None,
    after: Analysis | None = None,
) -> CoherenceResult:
    """Run every deterministic check on `candidate` against `original`. `band` is the voice's
    band for `connective_opener_pct` and `para_band` the exemplars' range of sentences per
    paragraph, when the voice has them."""
    before = before or analyze(original)
    after = after or analyze(candidate)
    return CoherenceResult(
        paragraphs_before=len(before.paragraphs),
        paragraphs_after=len(after.paragraphs),
        paragraphs=paragraph_check(before, after, para_band),
        tense=tense_drift(before, after),
        antecedents=antecedent_flags(after),
        connectives=connective_check(before, after, band),
    )
