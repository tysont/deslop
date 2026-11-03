# ABOUTME: Continuity with an already-converted preceding section: terms rendered the same way,
# ABOUTME: the same tense and person across the seam, no name re-introduced, and rhythm distance.
from __future__ import annotations

import difflib
import re
from collections import Counter

from pydantic import BaseModel, Field

from deslop.coherence import paragraph_tense
from deslop.metrics import compute_all
from deslop.metrics.base import METRICS
from deslop.nlp import Analysis, analyze, load_nlp

TOP_NOUNS = 20  # the preceding text's most frequent content nouns join its proper nouns
MIN_PERSON_PRONOUNS = 2  # a paragraph is first or second person with at least this many
FIRST = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"}
SECOND = {"you", "your", "yours", "yourself", "yourselves"}
EPITHET_DEPS = {"compound", "amod"}


class TermDrift(BaseModel):
    term: str  # as the preceding text writes it
    found: str  # as the candidate writes it


class Reintroduction(BaseModel):
    name: str
    preceding: str  # how the preceding text introduced it
    candidate: str  # how the candidate introduces it again


class ContinuityResult(BaseModel):
    """The seam between the preceding section and a candidate. `ok` when every term the
    preceding text uses is rendered the same way, the tense and person of its last paragraph
    carry into the candidate's first, and no name it introduced is introduced again."""

    terms: list[TermDrift] = Field(default_factory=list)
    tense_before: str = ""
    tense_after: str = ""
    person_before: str = ""
    person_after: str = ""
    reintroductions: list[Reintroduction] = Field(default_factory=list)
    rhythm_distance: float | None = None  # mean |difference| / voice spread, rhythm metrics

    @property
    def tense_ok(self) -> bool:
        skip = ("none", "mixed", "")
        if self.tense_before in skip or self.tense_after in skip:
            return True
        return self.tense_before == self.tense_after

    @property
    def person_ok(self) -> bool:
        return self.person_before == self.person_after

    @property
    def ok(self) -> bool:
        return not self.terms and self.tense_ok and self.person_ok and not self.reintroductions

    def reader_notes(self) -> list[str]:
        notes = [
            f'the section before writes "{t.term}"; this draft writes "{t.found}"'
            for t in self.terms
        ]
        if not self.tense_ok:
            notes.append(
                f"the section before ends in the {self.tense_before} tense; this draft opens in "
                f"the {self.tense_after}"
            )
        if not self.person_ok:
            notes.append(
                f"the section before ends in the {self.person_before} person; this draft opens "
                f"in the {self.person_after}"
            )
        notes.extend(
            f'{r.name} was introduced in the section before ("{r.preceding}"); this draft '
            f'introduces it again ("{r.candidate}")'
            for r in self.reintroductions
        )
        return notes


def paragraph_person(text: str) -> str:
    """first, second or third: the person of a paragraph's personal pronouns, third when
    neither first nor second reaches `MIN_PERSON_PRONOUNS`."""
    words = Counter(w.lower() for w in re.findall(r"[A-Za-z']+", text))
    first = sum(words[w] for w in FIRST)
    second = sum(words[w] for w in SECOND)
    if first >= MIN_PERSON_PRONOUNS and first >= second:
        return "first"
    if second >= MIN_PERSON_PRONOUNS:
        return "second"
    return "third"


def preceding_terms(text: str) -> dict[str, set[str]]:
    """Terms of the preceding text, keyed by a spelling-blind key (lower case, no spaces or
    hyphens), each with the surface forms the text uses: its proper nouns (named entities
    and proper-noun tokens) and its `TOP_NOUNS` most frequent content nouns."""
    doc = load_nlp(ner=True)(text)
    terms: dict[str, set[str]] = {}

    def add(form: str) -> None:
        form = form.strip()
        if len(form) < 3:
            return
        terms.setdefault(_key(form), set()).add(form)

    for ent in doc.ents:
        if ent.label_ in ("PERSON", "ORG", "GPE", "LOC", "NORP", "WORK_OF_ART", "EVENT", "FAC"):
            add(ent.text.removeprefix("the ").removeprefix("The "))
    for tok in doc:
        if tok.pos_ == "PROPN" and tok.is_title:
            add(tok.text)
    nouns = Counter(t.lemma_.lower() for t in doc if t.pos_ == "NOUN" and len(t.text) > 3)
    common = {lemma for lemma, _ in nouns.most_common(TOP_NOUNS)}
    for tok in doc:
        if tok.pos_ == "NOUN" and tok.lemma_.lower() in common:
            add(tok.text.lower())
    for m in re.finditer(r"\b[A-Za-z]+(?:-[A-Za-z]+)+\b", text):  # hyphenated terms
        add(m.group(0))
    return terms


def _key(form: str) -> str:
    return re.sub(r"[\s\-]", "", form.lower())


def term_drift(preceding: str, candidate: str) -> list[TermDrift]:
    """Every term the preceding text renders one way that the candidate renders another: a
    hyphen, space or join changed ("sea-water", "sea water", "seawater"), or a proper noun
    spelled differently (one or two letters apart in a name of six or more)."""
    terms = preceding_terms(preceding)
    out: list[TermDrift] = []
    seen: set[str] = set()
    for key, forms in terms.items():
        lower_forms = {f.lower() for f in forms}
        for phrase in _windows(candidate):
            if _key(phrase) != key or phrase.lower() in lower_forms or key in seen:
                continue
            out.append(TermDrift(term=sorted(forms)[0], found=phrase))
            seen.add(key)
    names = {f for forms in terms.values() for f in forms if f[:1].isupper() and " " not in f}
    cand_names = set(re.findall(r"\b[A-Z][a-z]{5,}\b", candidate))
    for name in sorted(names):
        if len(name) < 6 or name in cand_names:
            continue
        for other in sorted(cand_names):
            if other in names or _key(other) in seen:
                continue
            if other[0] == name[0] and _edits(name.lower(), other.lower()) <= 2:
                out.append(TermDrift(term=name, found=other))
                seen.add(_key(other))
    return out


def _windows(text: str) -> list[str]:
    """Every run of one to three words in `text`, with the spaces or hyphens between them."""
    words = list(re.finditer(r"[A-Za-z]+", text))
    out = []
    for i in range(len(words)):
        for j in range(i, min(i + 3, len(words))):
            out.append(text[words[i].start() : words[j].end()])
    return out


def _edits(a: str, b: str) -> int:
    matcher = difflib.SequenceMatcher(None, a, b)
    return sum(
        max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in matcher.get_opcodes() if op != "equal"
    )


def introductions(text: str) -> dict[str, str]:
    """Names the text introduces with an appositive ("Xenophanes, a poet,") or an epithet
    ("the philosopher Xenophanes"), each with the phrase that introduced it."""
    doc = load_nlp(ner=True)(text)
    out: dict[str, str] = {}
    for tok in doc:
        if tok.pos_ != "PROPN" or tok.dep_ == "compound":
            continue
        appos = [c for c in tok.children if c.dep_ == "appos"]
        epithet = [
            c
            for c in tok.children
            if c.dep_ in EPITHET_DEPS and c.pos_ == "NOUN" and c.i < tok.i and c.text.islower()
        ]
        if appos:
            span = doc[tok.i : appos[0].right_edge.i + 1]
        elif epithet and any(c.dep_ == "det" for c in tok.children):
            span = doc[tok.left_edge.i : tok.i + 1]
        else:
            continue
        name = " ".join(t.text for t in tok.subtree if t.pos_ == "PROPN" and t.i <= tok.i)
        out.setdefault(name or tok.text, span.text)
    # The parser often files "the patriarch Abraham" as a noun with a relative clause, so
    # the epithet is also read off the surface: the, a lower-case noun, then a name.
    for i in range(len(doc) - 2):
        det, noun, named = doc[i], doc[i + 1], doc[i + 2]
        if (
            det.lower_ in ("the", "a", "an")
            and noun.pos_ == "NOUN"
            and noun.text.islower()
            and named.pos_ == "PROPN"
            and named.is_title
        ):
            out.setdefault(named.text, doc[i : i + 3].text)
    return out


def rhythm_distance(preceding: str, candidate: str, spreads: dict[str, float]) -> float | None:
    """How far the candidate's rhythm sits from the preceding section's: the mean, over the
    rhythm metrics the voice gives a spread for, of |difference| / spread."""
    before = compute_all(analyze(preceding))
    after = compute_all(analyze(candidate))
    zs = [
        abs(after[name] - before[name]) / spread
        for name, spread in spreads.items()
        if spread > 0 and name in before and name in after and METRICS[name].group == "rhythm"
    ]
    return round(sum(zs) / len(zs), 3) if zs else None


def check_continuity(
    preceding: str, candidate: str, *, spreads: dict[str, float] | None = None
) -> ContinuityResult:
    """The continuity checks between the preceding section and a candidate (see
    `ContinuityResult`), plus the rhythm distance when the voice's spreads are given."""
    before: Analysis = analyze(preceding)
    after: Analysis = analyze(candidate)
    last = before.paragraphs[-1] if before.paragraphs else None
    first = after.paragraphs[0] if after.paragraphs else None
    result = ContinuityResult(
        terms=term_drift(preceding, candidate),
        tense_before=paragraph_tense(last) if last else "",
        tense_after=paragraph_tense(first) if first else "",
        person_before=paragraph_person(last.text if last else ""),
        person_after=paragraph_person(first.text if first else ""),
    )
    introduced = introductions(preceding)
    again = introductions(candidate)
    for name, phrase in introduced.items():
        if name in again:
            result.reintroductions.append(
                Reintroduction(name=name, preceding=phrase, candidate=again[name])
            )
    if spreads:
        result.rhythm_distance = rhythm_distance(preceding, candidate, spreads)
    return result


def last_paragraphs(text: str, count: int = 2) -> str:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    return "\n\n".join(paras[-count:])
