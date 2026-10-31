# ABOUTME: Text normalization for ingested documents: UTF-8, straight quotes,
# ABOUTME: unwrapped lines, single blank line between paragraphs, Gutenberg boilerplate removed.
from __future__ import annotations

import re
import unicodedata

from deslop.nlp import looks_like_verse

QUOTE_MAP = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
        "«": '"',
        "»": '"',
        "′": "'",
        "″": '"',
        " ": " ",
    }
)

GUTENBERG_START_RE = re.compile(
    r"^\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG.*$", re.MULTILINE
)
GUTENBERG_END_RE = re.compile(r"^\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG.*$", re.MULTILINE)


def strip_gutenberg(text: str) -> str:
    start = GUTENBERG_START_RE.search(text)
    if start:
        text = text[start.end() :]
    end = GUTENBERG_END_RE.search(text)
    if end:
        text = text[: end.start()]
    return text


def straighten_quotes(text: str) -> str:
    return text.translate(QUOTE_MAP)


def unwrap_lines(text: str) -> str:
    """Join hard-wrapped lines inside a paragraph; keep blank lines as separators."""
    paragraphs = re.split(r"\n[ \t]*\n+", text)
    out = []
    for para in paragraphs:
        lines = [ln.strip() for ln in para.split("\n")]
        lines = [ln for ln in lines if ln]
        if not lines:
            continue
        if looks_like_verse(lines):
            out.append("\n".join(lines))
        else:
            out.append(" ".join(lines))
    return "\n\n".join(out)


def normalize(text: str, *, gutenberg: bool = True) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if gutenberg:
        text = strip_gutenberg(text)
    text = straighten_quotes(text)
    text = text.replace("\t", " ")
    text = re.sub(r"[ ]{2,}", " ", text)
    text = unwrap_lines(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"
