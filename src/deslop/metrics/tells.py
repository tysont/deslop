# ABOUTME: Tell detectors: each returns spans where a catalogued slop pattern fires.
# ABOUTME: Regex detectors are data-driven; dep-pattern and metric detectors live here by name.
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

from spacy.tokens import Span, Token

from deslop.metrics.anchors import CITATION_PATTERNS, NUMBER_RE, QUOTE_RE
from deslop.nlp import WORD_RE, Analysis, Sentence
from deslop.patterns import Catalog, Pattern, load_catalog


@dataclass(frozen=True)
class TellHit:
    pattern_id: str
    sentence_idx: int
    para_idx: int
    start: int
    end: int
    text: str
    note: str = ""


SentenceDetector = Callable[[Analysis, Sentence, Pattern, Catalog], list[TellHit]]
DocumentDetector = Callable[[Analysis, Pattern, Catalog], list[TellHit]]

SENTENCE_DETECTORS: dict[str, SentenceDetector] = {}
DOCUMENT_DETECTORS: dict[str, DocumentDetector] = {}


def sentence_detector(name: str):
    def register(fn: SentenceDetector) -> SentenceDetector:
        SENTENCE_DETECTORS[name] = fn
        return fn

    return register


def document_detector(name: str):
    def register(fn: DocumentDetector) -> DocumentDetector:
        DOCUMENT_DETECTORS[name] = fn
        return fn

    return register


# ---------------------------------------------------------------- helpers


def _hit(pattern: Pattern, sent: Sentence, start: int, end: int, text: str, note: str = ""):
    return TellHit(pattern.id, sent.index, sent.para_index, start, end, text, note)


def _token_hit(pattern: Pattern, sent: Sentence, first: Token, last: Token, note: str = ""):
    start = sent.start - sent.span.start_char + first.idx
    end = sent.start - sent.span.start_char + last.idx + len(last.text)
    text = sent.span.doc.text[first.idx : last.idx + len(last.text)]
    return _hit(pattern, sent, start, end, text, note)


def _span_hit(pattern: Pattern, sent: Sentence, span: Span, note: str = ""):
    return _token_hit(pattern, sent, span[0], span[-1], note)


def is_abstract(tok: Token, catalog: Catalog) -> bool:
    if tok.pos_ not in ("NOUN", "PROPN", "ADJ"):
        return False
    lemma = tok.lemma_.lower()
    if lemma in catalog.lexicon("abstract_nouns"):
        return True
    if tok.pos_ == "ADJ":
        return False
    return any(lemma.endswith(s) for s in catalog.lexicon("abstract_suffixes")) and len(lemma) > 6


def is_verse(sent: Sentence) -> bool:
    """Sentences spanning kept line breaks are quoted verse, not the writer's prose."""
    return "\n" in sent.text.strip()


def has_anchor(sent: Sentence) -> bool:
    text = sent.text
    if any(t.pos_ == "PROPN" for t in sent.span):
        return True
    if NUMBER_RE.search(text) or QUOTE_RE.search(text):
        return True
    return any(p.search(text) for p in CITATION_PATTERNS)


def subject_of(verb: Token) -> Token | None:
    for child in verb.children:
        if child.dep_ in ("nsubj", "nsubjpass"):
            return child
    return None


def noun_phrase_text(tok: Token) -> str:
    return " ".join(t.lower_ for t in tok.subtree if t.i <= tok.i)


@lru_cache(maxsize=256)
def _compile(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE | re.MULTILINE)


def content_lemmas(span) -> set[str]:
    return {
        t.lemma_.lower()
        for t in span
        if t.pos_ in ("NOUN", "VERB", "ADJ", "PROPN") and not t.is_stop and len(t.text) > 3
    }


# ---------------------------------------------------------------- regex


def regex_detector(a: Analysis, sent: Sentence, pattern: Pattern, catalog: Catalog):
    hits = []
    text = sent.text
    for raw in pattern.params.get("patterns", []):
        for m in _compile(raw).finditer(text):
            hits.append(
                _hit(pattern, sent, sent.start + m.start(), sent.start + m.end(), m.group(0))
            )
    return hits


# ---------------------------------------------------------------- dep-pattern


@sentence_detector("text_as_agent")
def text_as_agent(a: Analysis, sent: Sentence, pattern: Pattern, catalog: Catalog):
    nouns = catalog.lexicon("text_nouns")
    verbs = set(pattern.params.get("verbs", []))
    hits = []
    for tok in sent.span:
        if tok.pos_ not in ("VERB", "AUX") or tok.lemma_.lower() not in verbs:
            continue
        subj = subject_of(tok)
        if subj is None:
            continue
        is_text_noun = subj.pos_ == "NOUN" and subj.lemma_.lower() in nouns
        is_titled = subj.pos_ == "PROPN" and any(
            c.dep_ == "det" and c.lower_ == "the" for c in subj.children
        )
        if is_text_noun or is_titled:
            hits.append(_token_hit(pattern, sent, subj, tok, note=tok.lower_))
    return hits


@sentence_detector("triad")
def triad(a: Analysis, sent: Sentence, pattern: Pattern, catalog: Catalog):
    minimum = int(pattern.params.get("min_conjuncts", 3))
    hits = []
    for tok in sent.span:
        if tok.dep_ in CHAIN_DEPS:
            continue
        chain = conjunct_chain(tok)
        if len(chain) >= minimum and any(c.dep_ == "cc" for t in chain for c in t.children):
            hits.append(_token_hit(pattern, sent, tok, chain[-1], note=f"{len(chain)}"))
    if hits:
        return hits
    m = SURFACE_TRIAD_RE.search(sent.text)
    if m:
        hits.append(
            _hit(pattern, sent, sent.start + m.start(), sent.start + m.end(), m.group(0), "surface")
        )
    return hits


CHAIN_DEPS = ("conj", "parataxis")
SURFACE_TRIAD_RE = re.compile(
    r"(?<![,;])\b[^,;:.?!]{2,40},\s[^,;:.?!]{2,40},\s(?:and|or)\s[^,;:.?!]{2,40}"
)


def conjunct_chain(tok: Token) -> list[Token]:
    """Follow conj links (flat or chained) collecting same-POS conjuncts."""
    chain = [tok]
    frontier = [tok]
    while frontier:
        cur = frontier.pop()
        for c in cur.children:
            if c.dep_ in CHAIN_DEPS and c.pos_ == tok.pos_ and c not in chain:
                chain.append(c)
                frontier.append(c)
    return sorted(chain, key=lambda t: t.i)


ANTITHESIS_RE = re.compile(
    r"\bnot\b[^,;.?!]{1,50}\bbut\b|,\s*not\s+\w|\bless\b[^,;.?!]{1,40}\bthan\b|\bmore\b[^,;.?!]{1,40}\bthan\b"
    r"|^to \w+\b.{0,40}\bis to\b|^what\b.{0,40}\bis what\b"
    r"|\ba (?:kind|sort) of\b|\bis the \w+ of\b",
    re.IGNORECASE,
)


def is_chiastic(sent: Sentence) -> bool:
    """Two clauses whose content words mirror (A B ... B A)."""
    parts = re.split(r"[,;:]|\bbut\b|\band\b", sent.text)
    if len(parts) < 2:
        return False
    first = [w.lower() for w in WORD_RE.findall(parts[0]) if len(w) > 3]
    second = [w.lower() for w in WORD_RE.findall(" ".join(parts[1:])) if len(w) > 3]
    shared = [w for w in first if w in second]
    if len(shared) < 2:
        return False
    a, b = shared[0], shared[1]
    return second.index(b) < second.index(a)


@sentence_detector("aphoristic_close")
def aphoristic_close(a: Analysis, sent: Sentence, pattern: Pattern, catalog: Catalog):
    para = a.paragraphs[sent.para_index]
    if para.sentences[-1] is not sent or len(para.sentences) < 2:
        return []
    if sent.word_count > int(pattern.params.get("max_words", 30)) or has_anchor(sent):
        return []
    if is_verse(sent):
        return []
    nouns = [t for t in sent.span if t.pos_ in ("NOUN", "PROPN") or is_abstract(t, catalog)]
    abstract = [t for t in nouns if is_abstract(t, catalog)]
    density = len(abstract) / len(nouns) if nouns else 0.0
    dense = density >= float(pattern.params.get("abstract_density", 0.3)) and len(abstract) >= int(
        pattern.params.get("min_abstract", 2)
    )
    shaped = bool(ANTITHESIS_RE.search(sent.text)) or is_chiastic(sent)
    if dense or shaped:
        note = "antithesis" if shaped else f"abstract={density:.2f}"
        return [_span_hit(pattern, sent, sent.span, note)]
    return []


@sentence_detector("invented_label")
def invented_label(a: Analysis, sent: Sentence, pattern: Pattern, catalog: Catalog):
    hits = []
    for tok in sent.span:
        if tok.dep_ != "poss" or not is_abstract(tok, catalog):
            continue
        head = tok.head
        if not is_abstract(head, catalog):
            # Mis-parsed possessives: fall back to the token after the 's marker.
            after = [t for t in sent.span if t.i > tok.i and t.text != "'s"]
            head = after[0] if after else head
        if is_abstract(head, catalog) and head.i > tok.i:
            hits.append(_token_hit(pattern, sent, tok, head))
    return hits


@sentence_detector("meta_commentary")
def meta_commentary(a: Analysis, sent: Sentence, pattern: Pattern, catalog: Catalog):
    verbs = set(pattern.params.get("verbs", []))
    we_verbs = set(pattern.params.get("we_verbs", []))
    subjects = set(pattern.params.get("subjects", []))
    text_nouns = catalog.lexicon("text_nouns")
    hits = []
    for tok in sent.span:
        if tok.pos_ != "VERB":
            continue
        subj = subject_of(tok)
        if subj is None:
            continue
        verb = tok.lemma_.lower()
        lemma = subj.lemma_.lower()
        is_text = subj.pos_ == "NOUN" and lemma in text_nouns and verb in verbs
        is_deictic = subj.lower_ in subjects and verb in verbs
        is_we = (
            subj.lower_ == "we"
            and verb in we_verbs
            and not any(
                c.dep_ == "aux" and c.lower_ in ("can", "could", "may", "might", "shall", "will")
                for c in tok.children
            )
        )
        if is_text or is_deictic or is_we:
            hits.append(_token_hit(pattern, sent, subj, tok, note=tok.lower_))
    return hits


@sentence_detector("epithet_substitution")
def epithet_substitution(a: Analysis, sent: Sentence, pattern: Pattern, catalog: Catalog):
    roles = catalog.lexicon("role_nouns")
    introduced = any(t.pos_ == "PROPN" for s in a.sentences[: sent.index] for t in s.span)
    if not introduced:
        return []
    hits = []
    for tok in sent.span:
        if tok.pos_ != "NOUN" or tok.lemma_.lower() not in roles:
            continue
        children = list(tok.children)
        has_the = any(c.dep_ == "det" and c.lower_ == "the" for c in children)
        adjectives = [c for c in children if c.dep_ == "amod" and c.pos_ == "ADJ"]
        if has_the and adjectives:
            first = min(children, key=lambda t: t.i)
            hits.append(_token_hit(pattern, sent, first, tok))
    return hits


@sentence_detector("unanswered_question")
def unanswered_question(a: Analysis, sent: Sentence, pattern: Pattern, catalog: Catalog):
    if not sent.text.rstrip().endswith("?") or QUOTE_RE.search(sent.text) or is_verse(sent):
        return []
    following = a.sentences[sent.index + 1 : sent.index + 3]
    if following and following[0].text.rstrip().endswith("?"):
        return []  # a series of questions is judged by its last member
    openers = [o.lower() for o in pattern.params.get("answer_openers", [])]
    asked = content_lemmas(sent.span)
    for nxt in following:
        lowered = nxt.text.lower().lstrip('"“')
        if any(re.match(r"\b" + re.escape(o) + r"\b", lowered) for o in openers):
            return []
        if asked & content_lemmas(nxt.span):
            return []
    return [_span_hit(pattern, sent, sent.span)]


@sentence_detector("hedge_stack")
def hedge_stack(a: Analysis, sent: Sentence, pattern: Pattern, catalog: Catalog):
    minimum = int(pattern.params.get("min_hedges", 2))
    found = []
    lowered = sent.text.lower()
    for hedge in catalog.lexicon("hedges"):
        for m in re.finditer(r"\b" + re.escape(hedge) + r"\b", lowered):
            found.append(m.group(0))
    if len(found) >= minimum:
        return [_span_hit(pattern, sent, sent.span, note=",".join(sorted(set(found))))]
    return []


@sentence_detector("colon_reveal")
def colon_reveal(a: Analysis, sent: Sentence, pattern: Pattern, catalog: Catalog):
    max_words = int(pattern.params.get("max_words", 7))
    hits = []
    for tok in sent.span:
        if tok.text != ":":
            continue
        tail = sent.span.doc[tok.i + 1 : sent.span.end]
        words = [t for t in tail if t.is_alpha]
        if not words or len(words) > max_words:
            continue
        if any(t.pos_ in ("VERB", "AUX") for t in tail):
            continue
        if any(t.text in ('"', "“") for t in tail):
            continue
        nouns = [t for t in tail if t.pos_ == "NOUN"]
        if nouns and is_abstract(nouns[-1], catalog):
            hits.append(_token_hit(pattern, sent, tok, words[-1], note=nouns[-1].lower_))
    return hits


# ---------------------------------------------------------------- document-level


@document_detector("tidy_summary_close")
def tidy_summary_close(a: Analysis, pattern: Pattern, catalog: Catalog):
    if len(a.paragraphs) < int(pattern.params.get("min_paragraphs", 3)):
        return []
    last = a.paragraphs[-1]
    prior = set()
    for p in a.paragraphs[:-1]:
        prior |= content_lemmas(p.doc)
    own = content_lemmas(last.doc)
    if not own:
        return []
    overlap = len(own & prior) / len(own)
    prior_propn = {t.lower_ for p in a.paragraphs[:-1] for t in p.doc if t.pos_ == "PROPN"}
    new_anchor = (
        any(
            t.pos_ == "PROPN"
            and t.text[:1].isupper()
            and t.lower_ not in prior_propn
            and t.lemma_.lower() not in prior
            for t in last.doc
        )
        or bool(NUMBER_RE.search(last.text))
        or bool(QUOTE_RE.search(last.text))
    )
    if overlap >= float(pattern.params.get("min_overlap", 0.75)) and not new_anchor:
        first = last.sentences[0]
        return [
            TellHit(
                pattern.id,
                first.index,
                last.index,
                last.start,
                last.end,
                last.text,
                note=f"overlap={overlap:.2f}",
            )
        ]
    return []


# ---------------------------------------------------------------- driver


def detect_tells(a: Analysis, catalog: Catalog | None = None) -> list[TellHit]:
    catalog = catalog or load_catalog()
    hits: list[TellHit] = []
    for pattern in catalog.patterns:
        if pattern.detector == "regex":
            for sent in a.sentences:
                hits.extend(regex_detector(a, sent, pattern, catalog))
        elif pattern.detector == "dep-pattern":
            fn = SENTENCE_DETECTORS[pattern.function or pattern.name]
            for sent in a.sentences:
                hits.extend(fn(a, sent, pattern, catalog))
        elif pattern.detector == "metric":
            fn_doc = DOCUMENT_DETECTORS[pattern.function or pattern.name]
            hits.extend(fn_doc(a, pattern, catalog))
    hits.sort(key=lambda h: (h.start, h.pattern_id))
    return hits


def tell_rates(
    a: Analysis, hits: list[TellHit], catalog: Catalog | None = None
) -> dict[str, float]:
    """Hits per 1k words for every pattern in the catalog (zero when absent)."""
    catalog = catalog or load_catalog()
    counts = {p.id: 0 for p in catalog.patterns}
    for h in hits:
        counts[h.pattern_id] = counts.get(h.pattern_id, 0) + 1
    return {pid: n / a.words_per_1k for pid, n in counts.items()}
