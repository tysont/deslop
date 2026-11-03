# ABOUTME: Tell detectors: each returns spans where a catalogued slop pattern fires.
# ABOUTME: Regex detectors are data-driven; dep-pattern and metric detectors live here by name.
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

from spacy.tokens import Doc, Span, Token

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


def sentence_detector(name: str) -> Callable[[SentenceDetector], SentenceDetector]:
    def register(fn: SentenceDetector) -> SentenceDetector:
        SENTENCE_DETECTORS[name] = fn
        return fn

    return register


def document_detector(name: str) -> Callable[[DocumentDetector], DocumentDetector]:
    def register(fn: DocumentDetector) -> DocumentDetector:
        DOCUMENT_DETECTORS[name] = fn
        return fn

    return register


# ---------------------------------------------------------------- helpers


def _hit(
    pattern: Pattern, sent: Sentence, start: int, end: int, text: str, note: str = ""
) -> TellHit:
    return TellHit(pattern.id, sent.index, sent.para_index, start, end, text, note)


def _token_hit(
    pattern: Pattern, sent: Sentence, first: Token, last: Token, note: str = ""
) -> TellHit:
    start = sent.start - sent.span.start_char + first.idx
    end = sent.start - sent.span.start_char + last.idx + len(last.text)
    text = sent.span.doc.text[first.idx : last.idx + len(last.text)]
    return _hit(pattern, sent, start, end, text, note)


def _span_hit(pattern: Pattern, sent: Sentence, span: Span, note: str = "") -> TellHit:
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


@lru_cache(maxsize=256)
def _compile(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE | re.MULTILINE)


def content_lemmas(span: Span | Doc) -> set[str]:
    return {
        t.lemma_.lower()
        for t in span
        if t.pos_ in ("NOUN", "VERB", "ADJ", "PROPN") and not t.is_stop and len(t.text) > 3
    }


# ---------------------------------------------------------------- regex


def regex_detector(
    a: Analysis, sent: Sentence, pattern: Pattern, catalog: Catalog
) -> list[TellHit]:
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
def text_as_agent(a: Analysis, sent: Sentence, pattern: Pattern, catalog: Catalog) -> list[TellHit]:
    nouns = catalog.lexicon("text_nouns")
    verbs = set(pattern.params.get("verbs", []))
    hits = []
    for tok in sent.span:
        is_about = tok.lemma_.lower() == "be" and any(
            c.dep_ == "prep" and c.lower_ == "about" for c in tok.children
        )  # "the verse is about X": the text as the agent of a topic
        if tok.pos_ not in ("VERB", "AUX") or (tok.lemma_.lower() not in verbs and not is_about):
            continue
        subj = subject_of(tok)
        if subj is None:
            continue
        is_text_noun = subj.pos_ == "NOUN" and subj.lemma_.lower() in nouns
        is_titled = subj.pos_ == "PROPN" and any(
            c.dep_ == "det" and c.lower_ == "the" for c in subj.children
        )
        if is_text_noun or is_titled:
            note = f"{tok.lower_} about" if is_about else tok.lower_
            hits.append(_token_hit(pattern, sent, subj, tok, note=note))
    return hits


PSEUDO_CLEFT_RE = re.compile(r"^\W*What\b[^.?!;:]{2,90}?\b(?:is|was|are|were)\b", re.IGNORECASE)
IT_CLEFT_RE = re.compile(
    r"\bIt\s+(?:is|was)\s+(?P<focus>(?:(?!\bto\b)[^\s,;:.?!]+\s+){1,7}?)(?:that|who|whom)\b"
)
REVERSE_CLEFT_RE = re.compile(r"\b(?:is|was|are|were)\s+(?:what|why|how|where)\b", re.IGNORECASE)
OBJECT_DEPS = {"dobj", "obj", "attr", "oprd", "acomp", "dative"}
CLAUSE_OBJECT_DEPS = {"ccomp", "xcomp"}


@sentence_detector("cleft_inversion")
def cleft_inversion(
    a: Analysis, sent: Sentence, pattern: Pattern, catalog: Catalog
) -> list[TellHit]:
    """Clefts and inversions that hold the point back: a pseudo-cleft ("What X is Y"), an
    it-cleft ("It is X that Y"; not "It is clear that", which is extraposition), a reverse
    pseudo-cleft ("X is what/why Y"), and a fronted object ("Grief Plato will not allow"). A
    question is none of these."""
    text = sent.text.strip()
    if text.endswith("?"):
        return []
    hits: list[TellHit] = []
    m = PSEUDO_CLEFT_RE.search(text)
    if m:
        after_what = next((t for t in sent.span if t.lower_ == "what"), None)
        nxt = (
            sent.span.doc[after_what.i + 1]
            if after_what is not None and after_what.i + 1 < len(sent.span.doc)
            else None
        )
        if nxt is not None and (nxt.lower_ in ("a", "an") or nxt.pos_ in ("NOUN", "ADJ")):
            m = None  # "What a dreadful image is this", "What pictures are these": exclamations
    if m:
        start = sent.text.find(text)
        hits.append(
            _hit(
                pattern,
                sent,
                sent.start + start + m.start(),
                sent.start + start + m.end(),
                m.group(0).strip(),
                "pseudo-cleft",
            )
        )
    for m in IT_CLEFT_RE.finditer(sent.text):
        focus_char = sent.span.start_char + m.start("focus")
        first = next((t for t in sent.span if t.idx >= focus_char), None)
        if first is None or first.pos_ in ("ADJ", "ADV", "PART", "VERB") or first.lower_ == "not":
            continue  # "It is clear that", "It is not that": extraposition, not a cleft
        hits.append(
            _hit(
                pattern, sent, sent.start + m.start(), sent.start + m.end(), m.group(0), "it-cleft"
            )
        )
    for m in REVERSE_CLEFT_RE.finditer(sent.text):
        hits.append(
            _hit(
                pattern,
                sent,
                sent.start + m.start(),
                sent.start + m.end(),
                m.group(0),
                "reverse cleft",
            )
        )
    hits.extend(_fronted_object(a, sent, pattern, int(pattern.params.get("max_fronted", 5))))
    return hits


def _fronted_object(
    a: Analysis, sent: Sentence, pattern: Pattern, max_fronted: int
) -> list[TellHit]:
    """A noun phrase before the subject that is the verb's missing object: "Grief Plato
    will not allow". The parser often files the fronted noun as a compound of a named
    subject, so the test is on the surface: a short run of nominal tokens, then the
    subject's own core, then a transitive verb (not "be") with no object after it."""
    tokens = [t for t in sent.span if not t.is_punct]
    root = next((t for t in sent.span if t.dep_ == "ROOT"), None)
    if root is None or root.pos_ != "VERB" or root.lemma_.lower() == "be":
        return []
    subj = subject_of(root)
    if subj is None or subj.pos_ not in ("PROPN", "PRON"):
        return []
    for c in root.children:
        if c.i <= root.i:
            continue
        if c.dep_ in OBJECT_DEPS:
            return []
        between = sent.span.doc[root.i + 1 : c.left_edge.i]
        if c.dep_ in CLAUSE_OBJECT_DEPS and not any(t.text == "," for t in between):
            return []  # a clause after a comma is a mis-attached adverbial, not the object
    core = [subj] + [c for c in subj.children if c.dep_ in ("det", "poss") and c.i < subj.i]
    core_start = min(t.i for t in core)
    prefix = [t for t in tokens if t.i < core_start]
    if not 1 <= len(prefix) <= max_fronted:
        return []
    if any(t.pos_ not in ("NOUN", "PROPN", "ADJ", "DET") for t in prefix):
        return []
    head = prefix[-1]
    if head.pos_ not in ("NOUN", "PROPN"):
        return []
    if head.pos_ == "PROPN" and not re.search(rf"\b{re.escape(head.lower_)}\b", a.text):
        # The parser capitalises a sentence-initial common noun into a name ("Grief Plato");
        # it counts as a fronted common noun only when the document writes it in lower case.
        return []
    return [_token_hit(pattern, sent, prefix[0], root, note="fronted object")]


@sentence_detector("triad")
def triad(a: Analysis, sent: Sentence, pattern: Pattern, catalog: Catalog) -> list[TellHit]:
    minimum = int(pattern.params.get("min_conjuncts", 3))
    hits = []
    for tok in sent.span:
        if tok.dep_ in CHAIN_DEPS:
            continue
        chain = conjunct_chain(tok)
        if (
            len(chain) >= minimum
            and any(c.dep_ == "cc" for t in chain for c in t.children)
            and is_rhetorical_parallel(chain, catalog)
        ):
            hits.append(_token_hit(pattern, sent, tok, chain[-1], note=f"{len(chain)}"))
    if hits:
        return hits
    for m in SURFACE_TRIAD_RE.finditer(sent.text):
        if _parallel_items(sent, m, catalog):
            hits.append(
                _hit(
                    pattern,
                    sent,
                    sent.start + m.start(),
                    sent.start + m.end(),
                    m.group(0),
                    "surface",
                )
            )
    return hits


def _parallel_items(sent: Sentence, m: re.Match[str], catalog: Catalog) -> bool:
    """Surface triads count only when the three items end in the same part of speech and
    the items are a rhetorical parallel, not an enumeration."""
    ends = [m.end(1), m.end(2), m.end(3)]
    toks = []
    for end in ends:
        char = sent.span.start_char + end - 1
        tok = next((t for t in sent.span if t.idx <= char < t.idx + len(t.text)), None)
        if tok is None:
            return False
        toks.append(tok)
    return len({t.pos_ for t in toks}) == 1 and is_rhetorical_parallel(toks, catalog)


RHETORICAL_MODIFIER_SUFFIXES = ("ly", "ous", "ive", "al", "ic", "ful", "less", "able", "ible")


def is_rhetorical_parallel(items: list[Token], catalog: Catalog) -> bool:
    """A triad is three or more parallel clauses or predicates, or three abstract nouns,
    adjectives or adverbs in rhetorical parallel ("honestly, seriously, and
    philosophically"). A list of concrete nouns (materials, items, names, places: "gravel,
    chalk and clay") is an enumeration, not a triad."""
    pos = items[0].pos_
    if pos in ("VERB", "AUX"):
        return True
    if pos == "NOUN":
        return all(is_abstract(t, catalog) for t in items)
    if pos in ("ADJ", "ADV"):
        return all(
            is_abstract(t, catalog)
            or (len(t.lemma_) > 5 and t.lemma_.lower().endswith(RHETORICAL_MODIFIER_SUFFIXES))
            for t in items
        )
    return False


CHAIN_DEPS = ("conj", "parataxis")
SURFACE_TRIAD_RE = re.compile(
    r"(?<![,;])\b([^,;:.?!]{2,40}),\s([^,;:.?!]{2,40}),\s(?:and|or)\s([^,;:.?!]{2,40})"
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
def aphoristic_close(
    a: Analysis, sent: Sentence, pattern: Pattern, catalog: Catalog
) -> list[TellHit]:
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
def invented_label(
    a: Analysis, sent: Sentence, pattern: Pattern, catalog: Catalog
) -> list[TellHit]:
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
def meta_commentary(
    a: Analysis, sent: Sentence, pattern: Pattern, catalog: Catalog
) -> list[TellHit]:
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
def epithet_substitution(
    a: Analysis, sent: Sentence, pattern: Pattern, catalog: Catalog
) -> list[TellHit]:
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
def unanswered_question(
    a: Analysis, sent: Sentence, pattern: Pattern, catalog: Catalog
) -> list[TellHit]:
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
def hedge_stack(a: Analysis, sent: Sentence, pattern: Pattern, catalog: Catalog) -> list[TellHit]:
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
def colon_reveal(a: Analysis, sent: Sentence, pattern: Pattern, catalog: Catalog) -> list[TellHit]:
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
def tidy_summary_close(a: Analysis, pattern: Pattern, catalog: Catalog) -> list[TellHit]:
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


@document_detector("monotone_run")
def monotone_run(a: Analysis, pattern: Pattern, catalog: Catalog) -> list[TellHit]:
    """Runs of >= min_run consecutive prose sentences within one paragraph whose word
    counts all sit within `tolerance` of each other. The span is the whole run."""
    min_run = int(pattern.params.get("min_run", 3))
    tolerance = int(pattern.params.get("tolerance", 5))
    hits: list[TellHit] = []

    def flush(run: list[Sentence]) -> None:
        if len(run) >= min_run:
            first, last = run[0], run[-1]
            hits.append(
                TellHit(
                    pattern.id,
                    first.index,
                    first.para_index,
                    first.start,
                    last.end,
                    a.text[first.start : last.end],
                    note=f"{len(run)} sentences of {min(s.word_count for s in run)}-"
                    f"{max(s.word_count for s in run)} words",
                )
            )

    for para in a.paragraphs:
        run: list[Sentence] = []
        for sent in para.sentences:
            if is_verse(sent):
                flush(run)
                run = []
                continue
            counts = [s.word_count for s in run] + [sent.word_count]
            if run and max(counts) - min(counts) <= tolerance:
                run.append(sent)
            else:
                flush(run)
                run = [sent]
        flush(run)
    return hits


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
