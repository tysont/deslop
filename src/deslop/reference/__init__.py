# ABOUTME: Primary texts: a plain text with an optional sidecar of labelled spans and per-text
# ABOUTME: citation schemes; verifies that an added quotation is verbatim and a citation resolves.
from __future__ import annotations

import difflib
import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, Field

from deslop.lock import ANY_QUOTE_RE, PRIMARY_CITATION_PATTERNS

MIN_QUOTE_WORDS = 3  # shorter quoted spans are terms, not quotations
NEAREST_MIN_RATIO = 0.5
SIDECAR_SUFFIX = ".locs.json"  # `<name>.locs.json` beside `<name>.md` or `<name>.txt`
RANGE_RE = re.compile(r"\s*[-–]\s*")


class CitationScheme(BaseModel):
    """How citations of one text look, so deslop can recognise them without knowing what they
    mean: `pattern` matches a citation that names the text or a location in it, and
    `bare_pattern`, when given, matches a bare location whose text is known from context."""

    name: str
    pattern: str
    bare_pattern: str | None = None
    coarse_pattern: str | None = None  # a coarser form (a whole chapter, a whole book)

    def matches(self, citation: str) -> bool:
        item = citation.strip()
        return any(
            pat and re.fullmatch(pat, item)
            for pat in (self.pattern, self.bare_pattern, self.coarse_pattern)
        )

    def is_coarse(self, citation: str) -> bool:
        """Matched by the coarse form and not by the fine ones."""
        item = citation.strip()
        if not self.coarse_pattern or not re.fullmatch(self.coarse_pattern, item):
            return False
        return not any(pat and re.fullmatch(pat, item) for pat in (self.pattern, self.bare_pattern))

    def finditer(self, text: str) -> Iterable[re.Match[str]]:
        for pat in (self.pattern, self.bare_pattern, self.coarse_pattern):
            if pat:
                yield from re.finditer(pat, text)


class PrimaryTextSpec(BaseModel):
    """One declared primary text: its path (relative to the voice file when declared in
    `voice.yaml`), an optional display title, and its citation schemes."""

    path: str
    title: str = ""
    schemes: list[CitationScheme] = Field(default_factory=list)


def normalize(text: str) -> str:
    """Whitespace- and punctuation-normalised form for verbatim comparison: lower case, curly
    quotes straightened, punctuation dropped, single spaces."""
    return _normalize_with_map(text)[0]


def _normalize_with_map(text: str) -> tuple[str, list[int]]:
    """`normalize(text)` and, for every raw character offset, the normalised offset its
    output starts at (plus one final entry for the end)."""
    out: list[str] = []
    offsets: list[int] = []
    length = 0
    pending_space = False
    n = len(text)
    for i, c in enumerate(text):
        offsets.append(length + (1 if pending_space and length else 0))
        if c in "’‘":
            c = "'"
        if c in "—–-":
            c = " "
        c = c.lower()
        if c == "'":
            before = i > 0 and (text[i - 1].isalnum() or text[i - 1] == "_")
            after = i + 1 < n and (text[i + 1].isalnum() or text[i + 1] == "_")
            if not (before and after):
                c = " "
        elif not (c.isalnum() or c == "_" or c.isspace()):
            c = " "
        if c.isspace():
            pending_space = True
            continue
        if pending_space and length:
            out.append(" ")
            length += 1
        pending_space = False
        out.append(c)
        length += 1
    offsets.append(length)
    return "".join(out), offsets


def canonical_label(citation: str) -> str:
    """A citation string as a sidecar key: lower case, periods and trailing punctuation
    dropped, single spaces."""
    item = citation.strip().rstrip(".,;:)").replace(".", "")
    return " ".join(item.lower().split())


@dataclass
class Segment:
    ref: str
    start: int  # offset into the normalised text
    end: int
    raw_start: int = 0  # offsets into the raw text, for quoting verbatim
    raw_end: int = 0


@dataclass
class Passage:
    """A verbatim span of a primary text offered to the writer, labelled with its ref."""

    text_name: str
    ref: str
    text: str

    @property
    def words(self) -> int:
        return len(self.text.split())


@dataclass
class PrimaryText:
    """One source: its normalised text, its labelled spans from the sidecar, and the
    citation schemes it answers to."""

    name: str
    path: str
    text: str  # normalised
    segments: list[Segment] = field(default_factory=list)  # in sidecar order
    schemes: list[CitationScheme] = field(default_factory=list)
    raw: str = ""
    _labels: dict[str, Segment] = field(default_factory=dict)

    @property
    def has_locations(self) -> bool:
        return bool(self.segments)

    @classmethod
    def load(
        cls,
        path: Path | str,
        *,
        schemes: Sequence[CitationScheme] = (),
        title: str = "",
    ) -> PrimaryText:
        """Read a plain-text or markdown file. A sidecar `<name>.locs.json` beside it, when
        present, lists `{"ref", "start", "end"}` entries with character offsets into the
        file; each becomes a labelled span. Without one the text still verifies quotations."""
        p = Path(path)
        raw = p.read_text(encoding="utf-8")
        sidecar = p.with_name(p.stem + SIDECAR_SUFFIX)
        locs: list[dict[str, object]] = []
        if sidecar.is_file():
            data = json.loads(sidecar.read_text(encoding="utf-8"))
            locs = list(data.get("locations", []) if isinstance(data, dict) else data)
        return cls.from_text(
            raw, name=title or p.stem, path=str(p), locations=locs, schemes=schemes
        )

    @classmethod
    def from_text(
        cls,
        raw: str,
        *,
        name: str,
        path: str = "",
        locations: Sequence[dict[str, object]] = (),
        schemes: Sequence[CitationScheme] = (),
    ) -> PrimaryText:
        text, offsets = _normalize_with_map(raw)
        segments = []
        for loc in locations:
            try:
                start, end = int(str(loc["start"])), int(str(loc["end"]))
                ref = str(loc["ref"]).strip()
            except (KeyError, TypeError, ValueError):
                continue
            if not ref or start < 0 or end < start:
                continue
            a = offsets[min(start, len(raw))]
            b = offsets[min(end, len(raw))]
            segments.append(
                Segment(
                    ref=ref, start=a, end=max(a, b), raw_start=start, raw_end=min(end, len(raw))
                )
            )
        pt = cls(name=name, path=path, text=text, segments=segments, schemes=list(schemes), raw=raw)
        for seg in segments:
            pt._labels.setdefault(canonical_label(seg.ref), seg)
        return pt

    # ------------------------------------------------------------ lookups
    def locate(self, offset: int) -> str:
        """The label of the first sidecar span containing an offset of the normalised text,
        or the text's name."""
        for seg in self.segments:
            if seg.start <= offset < seg.end:
                return seg.ref
        return self.name

    def segment_text(self, ref: str) -> str:
        """The normalised text of a labelled span, or empty."""
        seg = self._labels.get(canonical_label(ref))
        return self.text[seg.start : seg.end] if seg else ""

    def find_quote(self, span: str) -> str | None:
        """Where `span` occurs verbatim (after normalisation), or None."""
        needle = normalize(span)
        if not needle:
            return None
        at = self.text.find(needle)
        if at < 0:
            return None
        return self.locate(at)

    def nearest(self, span: str) -> tuple[str, str]:
        """The closest run of the text to `span`, as (location, run), for showing why a
        quotation was rejected; ("", "") when nothing comes near."""
        needle = normalize(span).split()
        words = self.text.split()
        if not needle or not words:
            return "", ""
        n = len(needle)
        wanted = set(needle)
        best, best_ratio, best_at = "", 0.0, -1
        offsets = []
        pos = 0
        for w in words:
            offsets.append(pos)
            pos += len(w) + 1
        for i in range(0, max(len(words) - n + 1, 1)):
            window = words[i : i + n]
            if len(wanted & set(window)) * 2 < len(wanted):
                continue
            matcher = difflib.SequenceMatcher(None, needle, window)
            ratio = matcher.ratio()
            if ratio > best_ratio:
                # show and locate the matched stretch, not the window's edges, which may sit
                # in the span before or after
                blocks = [b for b in matcher.get_matching_blocks() if b.size]
                lo = blocks[0].b if blocks else 0
                hi = blocks[-1].b + blocks[-1].size if blocks else n
                best, best_ratio, best_at = " ".join(window[lo:hi]), ratio, offsets[i + lo]
        if best_ratio < NEAREST_MIN_RATIO:
            return "", ""
        return self.locate(best_at), best

    # ------------------------------------------------------------ passages
    def spans(self) -> list[Segment]:
        """The distinct spans of the sidecar in order, one segment each (the first label)."""
        seen: set[tuple[int, int]] = set()
        out = []
        for seg in self.segments:
            key = (seg.raw_start, seg.raw_end)
            if key not in seen:
                seen.add(key)
                out.append(seg)
        return out

    def span_for(self, ref: str) -> Segment | None:
        """The distinct span a label names, under its first label."""
        seg = self._labels.get(canonical_label(ref))
        if seg is None:
            return None
        return next(
            (s for s in self.spans() if (s.raw_start, s.raw_end) == (seg.raw_start, seg.raw_end)),
            seg,
        )

    def has_bare_scheme(self) -> bool:
        return any(s.bare_pattern for s in self.schemes)

    def example_label(self, scheme: CitationScheme, *, coarse: bool = False) -> str:
        """A sidecar label the scheme's fine forms recognise (or, with `coarse`, its coarse
        form), or empty when none does."""
        for seg in self.segments:
            if scheme.is_coarse(seg.ref) == coarse and scheme.matches(seg.ref):
                return seg.ref
        return ""

    def is_coarse(self, citation: str) -> bool:
        return any(s.is_coarse(citation) for s in self.schemes)

    def passages_for(self, citation: str) -> list[Passage]:
        """The verbatim spans a citation of this text entitles the writer to quote: a coarse
        citation's span as it is; for a fine one, in a text with a bare (chapter:verse style)
        scheme the whole chapter, meaning every span whose label shares the text before the
        last colon; in a paragraph-labelled text the cited span and one on either side. Empty
        when the citation does not resolve."""
        ref = self.resolve(citation)
        if ref is None:
            return []
        target = self._labels[canonical_label(ref)]
        if self.is_coarse(citation):
            return [Passage(self.name, target.ref, self.raw[target.raw_start : target.raw_end])]
        spans = self.spans()
        index = next(
            (
                i
                for i, s in enumerate(spans)
                if (s.raw_start, s.raw_end) == (target.raw_start, target.raw_end)
            ),
            None,
        )
        if index is None:
            return [Passage(self.name, target.ref, self.raw[target.raw_start : target.raw_end])]
        first = spans[index]
        if self.has_bare_scheme() and ":" in first.ref:
            prefix = canonical_label(first.ref).rsplit(":", 1)[0] + ":"
            chapter = [s for s in spans if canonical_label(s.ref).startswith(prefix)]
            return [Passage(self.name, s.ref, self.raw[s.raw_start : s.raw_end]) for s in chapter]
        lo, hi = max(index - 1, 0), min(index + 1, len(spans) - 1)
        return [
            Passage(self.name, s.ref, self.raw[s.raw_start : s.raw_end]) for s in spans[lo : hi + 1]
        ]

    def answers_to(self, citation: str) -> bool:
        """Whether one of this text's schemes recognises the citation string."""
        return any(s.matches(citation) for s in self.schemes)

    def resolve(self, citation: str) -> str | None:
        """The sidecar label a citation names, or None: a scheme must match the string, and
        its canonical form (or, for a range, the part before the dash) must be a label."""
        if not self.answers_to(citation):
            return None
        key = canonical_label(citation)
        seg = self._labels.get(key)
        if seg is None and RANGE_RE.search(key):
            seg = self._labels.get(RANGE_RE.split(key, 1)[0].strip())
        return seg.ref if seg else None


def quotable_passages(
    source: str, texts: Sequence[PrimaryText], *, limit_words: int = 1500
) -> list[Passage]:
    """What the writer may quote verbatim, from every citation `source` contains that resolves
    in a text: for a chapter:verse text the whole chapter when it fits under `limit_words`,
    else the cited spans; for a paragraph-labelled text the cited paragraphs and their
    neighbours. The total is capped at `limit_words`, in citation order."""
    chosen: list[Passage] = []
    seen: set[tuple[str, str]] = set()
    total = 0
    for citation in primary_citations(source, texts):
        for t in texts:
            group = t.passages_for(citation)
            if not group:
                continue
            group_words = sum(p.words for p in group)
            if t.has_bare_scheme() and (
                group_words > limit_words or total + group_words > limit_words
            ):
                ref = t.resolve(citation)
                seg = t.span_for(ref) if ref else None
                group = (
                    [Passage(t.name, seg.ref, t.raw[seg.raw_start : seg.raw_end])] if seg else []
                )
            for passage in group:
                key = (passage.text_name, passage.ref)
                if key in seen:
                    continue
                if total + passage.words > limit_words:
                    if t.is_coarse(citation):
                        break  # a coarse span over the cap is left out, not truncated
                    return chosen
                seen.add(key)
                chosen.append(passage)
                total += passage.words
            break
    return chosen


def citation_guidance(texts: Sequence[PrimaryText]) -> list[str]:
    """One line per text for the writer: the scheme name, a resolvable example drawn from the
    sidecar, and the warning when a declared scheme has no labels in this edition."""
    lines = []
    for t in texts:
        for scheme in t.schemes:
            fine = t.example_label(scheme)
            coarse = t.example_label(scheme, coarse=True)
            labels = t.spans()
            if fine:
                extra = f', or coarsely as in "{coarse}"' if coarse else ""
                lines.append(f'{t.name}: cite by {scheme.name}, as in "{fine}"{extra}.')
            elif coarse:
                lines.append(
                    f'{t.name}: cite coarsely by {scheme.name}, as in "{coarse}"; finer '
                    f"{scheme.name} citations do not resolve in this edition and will be rejected."
                )
            elif labels:
                lines.append(
                    f'{t.name}: this edition locates passages as "{labels[0].ref}"; {scheme.name} '
                    "citations of this kind do not resolve in this edition and will be rejected."
                )
            else:
                lines.append(
                    f"{t.name}: {scheme.name} citations do not resolve in this edition and will "
                    "be rejected."
                )
        if not t.schemes and t.spans():
            lines.append(f'{t.name}: this edition locates passages as "{t.spans()[0].ref}".')
    return lines


def load_primary_texts(
    items: Sequence[str | Path | PrimaryTextSpec],
    *,
    relative_to: Path | None = None,
    schemes: Sequence[CitationScheme] = (),
) -> list[PrimaryText]:
    """Load each item: a file, a directory of .txt/.md files, or a `PrimaryTextSpec`. Relative
    paths resolve against `relative_to` when given. `schemes` apply to every item that
    carries none of its own."""
    out: list[PrimaryText] = []
    for item in items:
        spec = item if isinstance(item, PrimaryTextSpec) else PrimaryTextSpec(path=str(item))
        own = spec.schemes or list(schemes)
        p = Path(spec.path)
        if not p.is_absolute() and relative_to is not None and not p.exists():
            p = relative_to / p
        if p.is_dir():
            for f in sorted(p.iterdir()):
                if f.suffix.lower() in (".txt", ".md") and f.is_file():
                    out.append(PrimaryText.load(f, schemes=own))
        elif p.is_file():
            out.append(PrimaryText.load(p, schemes=own, title=spec.title))
        else:
            raise FileNotFoundError(f"primary text not found: {spec.path}")
    return out


# ---------------------------------------------------------------- verification


class Addition(BaseModel):
    """A quotation or citation a rewrite added: where it was verified, or why it was not."""

    kind: str  # quote | citation
    text: str
    verified: bool = False
    where: str = ""  # label and text name when verified
    note: str = ""  # "not in any primary text" / "no primary text given" / "unresolved"
    nearest: str = ""  # for a rejected quotation, the closest run of a primary text
    coarse: bool = False  # a citation matched by a scheme's coarse form (a chapter, a book)

    def __str__(self) -> str:
        label = f'"{self.text}"' if self.kind == "quote" else self.text
        if self.verified:
            return f"{label} ({self.where}{', coarse' if self.coarse else ''})"
        near = f'; nearest: "{self.nearest}"' if self.nearest else ""
        return f"{label}: {self.note}{near}"


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


def _place(text: PrimaryText, where: str) -> str:
    """`<text name> <label>`, without repeating a name the label already starts with."""
    return where if where.lower().startswith(text.name.lower()) else f"{text.name} {where}"


def quoted_spans(text: str) -> list[str]:
    return [
        m.group(1).strip()
        for m in ANY_QUOTE_RE.finditer(text)
        if len(m.group(1).split()) >= MIN_QUOTE_WORDS
    ]


def primary_citations(text: str, texts: Sequence[PrimaryText] = ()) -> list[str]:
    """Citation strings in `text`: those the texts' schemes recognise, then the generic
    citation shapes the lock knows (so an unrecognised scheme is still listed)."""
    seen: list[tuple[int, int]] = []
    out = []
    sources: list[Iterable[re.Match[str]]] = [s.finditer(text) for t in texts for s in t.schemes]
    sources += [p.finditer(text) for p in PRIMARY_CITATION_PATTERNS]
    for matches in sources:
        for m in matches:
            if not any(s <= m.start() < e or s < m.end() <= e for s, e in seen):
                seen.append((m.start(), m.end()))
                out.append(m.group(0).strip())
    return out


def verify_additions(original: str, candidate: str, texts: list[PrimaryText]) -> PrimaryCheck:
    """Every quoted span (3+ words) and citation `candidate` has that `original` lacks. With
    `texts`, a quotation must occur verbatim in one of them (the nearest run is shown when it
    does not) and a citation a scheme recognises must resolve to a labelled span of a text
    with that scheme; a citation no scheme recognises is `unverifiable` and merely listed.
    Without texts everything is listed as unverified."""
    check = PrimaryCheck(checked=bool(texts))
    before_quotes = {normalize(q) for q in quoted_spans(original)}
    before_cites = set(primary_citations(original, texts))
    for q in quoted_spans(candidate):
        if normalize(q) in before_quotes:
            continue
        add = Addition(kind="quote", text=q, note="no primary text given")
        if texts:
            add.note = "not in any primary text"
            for t in texts:
                where = t.find_quote(q)
                if where:
                    add.verified, add.where, add.note = True, _place(t, where), ""
                    break
            if not add.verified:
                for t in texts:
                    where, run = t.nearest(q)
                    if run:
                        add.nearest = f"{run} ({_place(t, where)})"
                        break
        check.additions.append(add)
    for c in primary_citations(candidate, texts):
        if c in before_cites:
            continue
        add = Addition(kind="citation", text=c, note="no primary text given")
        if texts:
            answering = [t for t in texts if t.answers_to(c)]
            if not answering:
                add.note = "unverifiable"
            else:
                add.note = "does not resolve in " + ", ".join(t.name for t in answering)
                for t in answering:
                    where = t.resolve(c)
                    if where:
                        add.verified, add.where, add.note = True, _place(t, where), ""
                        add.coarse = t.is_coarse(c)
                        break
        check.additions.append(add)
    return check


__all__ = [
    "Addition",
    "CitationScheme",
    "Passage",
    "citation_guidance",
    "quotable_passages",
    "PrimaryCheck",
    "PrimaryText",
    "PrimaryTextSpec",
    "canonical_label",
    "load_primary_texts",
    "normalize",
    "primary_citations",
    "quoted_spans",
    "verify_additions",
]
