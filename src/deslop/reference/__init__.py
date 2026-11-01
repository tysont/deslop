# ABOUTME: Primary texts: plain-text sources with {{ref}} markers (verse or Stephanus), used to
# ABOUTME: verify that an added quotation is verbatim and that an added citation resolves.
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

from pydantic import BaseModel, Field

from deslop.lock import ANY_QUOTE_RE, PRIMARY_CITATION_PATTERNS

MARKER_RE = re.compile(r"\{\{([^{}]{1,40})\}\}")
HEADER_RE = re.compile(r"^#\s*([a-z_]+):\s*(.*)$")
VERSE_REF_RE = re.compile(
    r"^(?:(?P<book>[1-3]?\s?[A-Za-z]+)\.?\s+)?(?P<chapter>\d{1,3}):(?P<verse>\d{1,3})"
    r"(?:[-–](?P<end>\d{1,3}))?$"
)
STEPHANUS_REF_RE = re.compile(
    r"^(?P<page>\d{2,3})(?P<section>[a-e])?(?:[-–](?P<end>\d{2,3}[a-e]?|[a-e]))?$"
)
BUNDLED = {"kjv-genesis": "kjv_genesis.txt", "jowett-republic": "jowett_republic.txt"}
MIN_QUOTE_WORDS = 3  # shorter quoted spans are terms, not quotations


def normalize(text: str) -> str:
    """Whitespace- and punctuation-normalised form for verbatim comparison: lower case, curly
    quotes straightened, punctuation dropped, single spaces."""
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = text.replace("—", " ").replace("–", " ").replace("-", " ")
    text = re.sub(r"[^\w\s']", " ", text.lower())
    text = re.sub(r"'(?!\w)|(?<!\w)'", " ", text)
    return " ".join(text.split())


@dataclass
class Segment:
    ref: str
    start: int  # offset into the normalised text
    end: int


@dataclass
class PrimaryText:
    """One source: its normalised text, its markers as segments, and the names it answers to."""

    name: str
    path: str
    text: str  # normalised, markers removed
    segments: list[Segment] = field(default_factory=list)
    book: str = ""
    aliases: list[str] = field(default_factory=list)
    kinds: set[str] = field(default_factory=set)  # "verse" | "stephanus" present in markers
    _verses: dict[tuple[int, int], str] = field(default_factory=dict)
    _pages: dict[str, str] = field(default_factory=dict)

    @property
    def has_markers(self) -> bool:
        return bool(self.segments)

    @classmethod
    def load(cls, path: Path | str) -> PrimaryText:
        """Read a marked plain-text file. Header lines `# key: value` give `title`, `book` and
        `aliases`; `{{Genesis 6:5}}` and `{{369b}}` markers start a segment each. A file
        without markers still verifies quotations."""
        p = Path(path)
        raw = p.read_text(encoding="utf-8")
        meta: dict[str, str] = {}
        body_lines = []
        in_header = True
        for line in raw.splitlines():
            m = HEADER_RE.match(line) if in_header else None
            if m:
                meta[m.group(1)] = m.group(2).strip()
                continue
            in_header = False
            body_lines.append(line)
        body = "\n".join(body_lines)
        return cls.from_text(
            body,
            name=meta.get("title") or p.stem,
            path=str(p),
            book=meta.get("book", ""),
            aliases=[a.strip() for a in meta.get("aliases", "").split(",") if a.strip()],
        )

    @classmethod
    def from_text(
        cls,
        body: str,
        *,
        name: str,
        path: str = "",
        book: str = "",
        aliases: list[str] | None = None,
    ) -> PrimaryText:
        pieces: list[str] = []
        refs: list[tuple[str, int]] = []  # ref, offset into normalised text
        pos = 0
        length = 0
        for m in MARKER_RE.finditer(body):
            chunk = normalize(body[pos : m.start()])
            if chunk:
                pieces.append(chunk)
                length += len(chunk) + 1
            refs.append((m.group(1).strip(), length))
            pos = m.end()
        chunk = normalize(body[pos:])
        if chunk:
            pieces.append(chunk)
        text = " ".join(pieces)
        segments = []
        for i, (ref, start) in enumerate(refs):
            end = refs[i + 1][1] if i + 1 < len(refs) else len(text)
            segments.append(Segment(ref=ref, start=min(start, len(text)), end=end))
        pt = cls(
            name=name, path=path, text=text, segments=segments, book=book, aliases=aliases or []
        )
        pt._index()
        return pt

    def _index(self) -> None:
        for seg in self.segments:
            v = VERSE_REF_RE.match(seg.ref)
            if v and v.group("chapter"):
                self.kinds.add("verse")
                self._verses[(int(v.group("chapter")), int(v.group("verse")))] = seg.ref
                if not self.book and v.group("book"):
                    self.book = v.group("book")
                continue
            s = STEPHANUS_REF_RE.match(seg.ref.lower())
            if s:
                self.kinds.add("stephanus")
                self._pages[s.group("page") + (s.group("section") or "a")] = seg.ref

    # ------------------------------------------------------------ lookups
    def locate(self, offset: int) -> str:
        """The marker in force at an offset of the normalised text, or the text's name."""
        where = self.name
        for seg in self.segments:
            if seg.start <= offset:
                where = seg.ref
            else:
                break
        return where

    def find_quote(self, span: str) -> str | None:
        """Where `span` occurs verbatim (after normalisation), or None."""
        needle = normalize(span)
        if not needle:
            return None
        at = self.text.find(needle)
        if at < 0:
            return None
        return self.locate(at)

    def answers_to(self, book: str | None) -> bool:
        if not book:
            return True
        key = book.lower().replace(".", "").strip()
        key = re.sub(r"^([1-3])\s*", r"\1 ", key)
        names = [self.book.lower()] + [a.lower() for a in self.aliases]
        names = [re.sub(r"^([1-3])\s*", r"\1 ", n) for n in names if n]
        return any(n == key or (len(key) >= 3 and n.startswith(key)) for n in names)

    def resolve(self, citation: str) -> str | None:
        """The marker a citation names, or None. Verse references need the book to match
        this text (or to be absent); ranges resolve when both ends exist."""
        item = citation.strip().rstrip(".,;:)")
        item = re.sub(r"^(?:Republic|Rep\.?)\s*(?:[IVX]+|\d+)?,?\s*", "", item, flags=re.I)
        v = VERSE_REF_RE.match(item)
        if v and "verse" in self.kinds:
            if not self.answers_to(v.group("book")):
                return None
            chapter, verse = int(v.group("chapter")), int(v.group("verse"))
            start = self._verses.get((chapter, verse))
            if start is None:
                return None
            if v.group("end") and (chapter, int(v.group("end"))) not in self._verses:
                return None
            return start
        s = STEPHANUS_REF_RE.match(item.lower())
        if s and "stephanus" in self.kinds:
            start = self._pages.get(s.group("page") + (s.group("section") or "a"))
            if start is None:
                return None
            end = s.group("end")
            if end:
                end_key = end if end[0].isdigit() else s.group("page") + end
                if len(end_key) == 3:
                    end_key += "a"
                if end_key not in self._pages:
                    return None
            return start
        return None


def citation_kind(citation: str) -> str | None:
    item = citation.strip().rstrip(".,;:)")
    item = re.sub(r"^(?:Republic|Rep\.?)\s*(?:[IVX]+|\d+)?,?\s*", "", item, flags=re.I)
    if VERSE_REF_RE.match(item):
        return "verse"
    if STEPHANUS_REF_RE.match(item.lower()):
        return "stephanus"
    return None


def bundled_path(name: str) -> Path:
    """Path of a reference text shipped with deslop: `kjv-genesis` or `jowett-republic`."""
    if name not in BUNDLED:
        raise FileNotFoundError(f"no bundled reference text {name!r}; have {sorted(BUNDLED)}")
    with resources.as_file(resources.files("deslop.reference") / BUNDLED[name]) as p:
        return Path(p)


def load_primary_texts(
    items: Sequence[str | Path], *, relative_to: Path | None = None
) -> list[PrimaryText]:
    """Load each item: a bundled name (`kjv-genesis`), a file, or a directory of .txt/.md files.
    Relative paths resolve against `relative_to` when given."""
    out: list[PrimaryText] = []
    for item in items:
        name = str(item)
        if name in BUNDLED:
            out.append(PrimaryText.load(bundled_path(name)))
            continue
        p = Path(name)
        if not p.is_absolute() and relative_to is not None and not p.exists():
            p = relative_to / p
        if p.is_dir():
            for f in sorted(p.iterdir()):
                if f.suffix.lower() in (".txt", ".md") and f.is_file():
                    out.append(PrimaryText.load(f))
        elif p.is_file():
            out.append(PrimaryText.load(p))
        else:
            raise FileNotFoundError(f"primary text not found: {item}")
    return out


# ---------------------------------------------------------------- verification


class Addition(BaseModel):
    """A quotation or citation a rewrite added: where it was verified, or why it was not."""

    kind: str  # quote | citation
    text: str
    verified: bool = False
    where: str = ""  # marker and text name when verified
    note: str = ""  # "not in any primary text" / "no primary text given" / "unresolved"

    def __str__(self) -> str:
        label = f'"{self.text}"' if self.kind == "quote" else self.text
        if self.verified:
            return f"{label} ({self.where})"
        return f"{label}: {self.note}"


class PrimaryCheck(BaseModel):
    """Outcome of checking a candidate's added quotations and citations."""

    additions: list[Addition] = Field(default_factory=list)
    checked: bool = False  # True when primary texts were available

    @property
    def rejected(self) -> list[Addition]:
        return [
            a
            for a in self.additions
            if self.checked and not a.verified and a.note != "unverifiable"
        ]

    @property
    def unverified(self) -> list[Addition]:
        return [a for a in self.additions if not a.verified]

    @property
    def ok(self) -> bool:
        return not self.rejected


def quoted_spans(text: str) -> list[str]:
    return [
        m.group(1).strip()
        for m in ANY_QUOTE_RE.finditer(text)
        if len(m.group(1).split()) >= MIN_QUOTE_WORDS
    ]


def primary_citations(text: str) -> list[str]:
    seen: list[tuple[int, int]] = []
    out = []
    for pattern in PRIMARY_CITATION_PATTERNS:
        for m in pattern.finditer(text):
            if not any(s <= m.start() < e or s < m.end() <= e for s, e in seen):
                seen.append((m.start(), m.end()))
                out.append(m.group(0).strip())
    return out


def verify_additions(original: str, candidate: str, texts: list[PrimaryText]) -> PrimaryCheck:
    """Every quoted span (3+ words) and primary citation `candidate` has that `original` lacks.
    With `texts`, a quotation must occur verbatim in one of them and a citation must resolve
    in a text carrying markers of its kind; a kind no text carries markers for is
    `unverifiable` and merely listed. Without texts everything is listed as unverified."""
    check = PrimaryCheck(checked=bool(texts))
    before_quotes = {normalize(q) for q in quoted_spans(original)}
    before_cites = set(primary_citations(original))
    kinds = {k for t in texts for k in t.kinds}
    for q in quoted_spans(candidate):
        if normalize(q) in before_quotes:
            continue
        add = Addition(kind="quote", text=q, note="no primary text given")
        if texts:
            add.note = "not in any primary text"
            for t in texts:
                where = t.find_quote(q)
                if where:
                    add.verified, add.where, add.note = True, f"{t.name} {where}", ""
                    break
        check.additions.append(add)
    for c in primary_citations(candidate):
        if c in before_cites:
            continue
        add = Addition(kind="citation", text=c, note="no primary text given")
        if texts:
            kind = citation_kind(c)
            if kind is None or kind not in kinds:
                add.note = "unverifiable"
            else:
                add.note = "does not resolve in any primary text"
                for t in texts:
                    where = t.resolve(c)
                    if where:
                        add.verified, add.where, add.note = True, f"{t.name} {where}", ""
                        break
        check.additions.append(add)
    return check
