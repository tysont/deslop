# ABOUTME: Shared text segmentation and spaCy pipeline access.
# ABOUTME: Turns raw text into paragraphs and sentences with parses for the metrics.
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

import spacy
from spacy.language import Language
from spacy.tokens import Doc, Span

WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")
PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")
SPACY_MODEL = "en_core_web_sm"
SPACY_MODEL_URL = (
    "https://github.com/explosion/spacy-models/releases/download/"
    "en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
)


@lru_cache(maxsize=1)
def load_nlp() -> Language:
    """Load the small English pipeline once per process."""
    try:
        return spacy.load(SPACY_MODEL, disable=["ner"])
    except OSError as exc:
        raise OSError(
            f"spaCy model {SPACY_MODEL} is not installed. Install it with "
            f"`pip install {SPACY_MODEL_URL}` (pipx: `pipx inject deslop {SPACY_MODEL_URL}`)."
        ) from exc


def split_paragraphs(text: str) -> list[tuple[int, int]]:
    """Return (start, end) character offsets of non-empty paragraphs."""
    spans: list[tuple[int, int]] = []
    pos = 0
    for chunk in PARAGRAPH_SPLIT_RE.split(text):
        start = text.index(chunk, pos) if chunk else pos
        end = start + len(chunk)
        pos = end
        stripped = chunk.strip()
        if stripped:
            lead = len(chunk) - len(chunk.lstrip())
            spans.append((start + lead, start + lead + len(stripped)))
    return spans


def count_words(text: str) -> int:
    return len(WORD_RE.findall(text))


@dataclass
class Sentence:
    index: int
    para_index: int
    start: int
    end: int
    span: Span

    @property
    def text(self) -> str:
        return self.span.text

    @property
    def word_count(self) -> int:
        return count_words(self.text)


@dataclass
class Paragraph:
    index: int
    start: int
    end: int
    doc: Doc
    sentences: list[Sentence] = field(default_factory=list)

    @property
    def text(self) -> str:
        return self.doc.text

    @property
    def word_count(self) -> int:
        return count_words(self.text)


@dataclass
class Analysis:
    """Parsed view of a text: paragraphs, sentences, offsets into the original."""

    text: str
    paragraphs: list[Paragraph]

    @property
    def sentences(self) -> list[Sentence]:
        return [s for p in self.paragraphs for s in p.sentences]

    @property
    def word_count(self) -> int:
        return count_words(self.text)

    @property
    def words_per_1k(self) -> float:
        return max(self.word_count, 1) / 1000.0


def looks_like_verse(lines: list[str]) -> bool:
    """Short lines that mostly begin with a capital are verse rather than wrapped prose."""
    lines = [ln for ln in lines if ln.strip()]
    if len(lines) < 2:
        return False
    mean_len = sum(len(ln) for ln in lines) / len(lines)
    capitalized = sum(1 for ln in lines if ln.strip()[:1].isupper() or ln.strip()[:1] in "\"'")
    return mean_len < 50 and capitalized / len(lines) >= 0.8


def is_prose_paragraph(text: str) -> bool:
    """Skip headings, list markers, verse blocks, and very short fragments."""
    stripped = text.strip()
    if stripped.startswith("#"):
        return False
    if looks_like_verse(stripped.split("\n")):
        return False
    return count_words(stripped) >= 3


def analyze(text: str) -> Analysis:
    nlp = load_nlp()
    offsets = split_paragraphs(text)
    chunks = [text[s:e] for s, e in offsets]
    paragraphs: list[Paragraph] = []
    sent_index = 0
    for (start, end), doc in zip(offsets, nlp.pipe(chunks, batch_size=32), strict=True):
        if not is_prose_paragraph(doc.text):
            continue
        para = Paragraph(index=len(paragraphs), start=start, end=end, doc=doc)
        for sent in doc.sents:
            if not WORD_RE.search(sent.text):
                continue
            para.sentences.append(
                Sentence(
                    index=sent_index,
                    para_index=para.index,
                    start=start + sent.start_char,
                    end=start + sent.end_char,
                    span=sent,
                )
            )
            sent_index += 1
        if para.sentences:
            paragraphs.append(para)
    return Analysis(text=text, paragraphs=paragraphs)
