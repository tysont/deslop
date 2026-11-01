# ABOUTME: Briefs and section drafting: writes a piece section by section from a brief, one
# ABOUTME: claim per section, stripping any citation or quotation the brief did not list.
from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from deslop.lock import ANY_QUOTE_RE
from deslop.metrics.anchors import CITATION_PATTERNS
from deslop.nlp import analyze
from deslop.providers.base import CachingProvider, ProviderError
from deslop.voice.exemplars import parse_exemplars
from deslop.voice.model import Voice

# ---------------------------------------------------------------- brief


class Section(BaseModel):
    """One section of a brief: a heading, the single claim it argues, and the references and
    quotations it must use verbatim."""

    heading: str
    claim: str
    refs: list[str] = Field(default_factory=list)
    quotes: list[str] = Field(default_factory=list)
    notes: str = ""


class Brief(BaseModel):
    """What to write: a title, an optional audience, a target length per section, the
    sections. Load one from yaml with
    [`load_brief`][deslop.load_brief]."""

    title: str
    audience: str = ""
    words_per_section: int = 180
    sections: list[Section]


def load_brief(path: Path | str) -> Brief:
    """Parse a brief yaml file; raises `ValueError` when a section lacks a heading or claim."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict) or not data.get("sections"):
        raise ValueError(f"{path}: a brief needs a title and a list of sections")
    for i, section in enumerate(data["sections"], 1):
        for key in ("heading", "claim"):
            if not isinstance(section, dict) or not str(section.get(key, "")).strip():
                raise ValueError(f"{path}: section {i} is missing its {key}")
        section["claim"] = " ".join(str(section["claim"]).split())
    return Brief.model_validate(data)


# ---------------------------------------------------------------- results


class SectionLog(BaseModel):
    heading: str
    text: str
    attempts: int
    complete: bool
    missing: list[str] = Field(default_factory=list)
    stripped: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------- A6 unlisted_citation


def _norm(text: str) -> str:
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    return " ".join(text.split())


def _listed(item: str, listed: list[str]) -> bool:
    item = _norm(item).strip(" .,;:")
    return any(item in _norm(x) or _norm(x) in item for x in listed if x.strip())


def strip_unlisted(text: str, *, refs: list[str], quotes: list[str]) -> tuple[str, list[str]]:
    """Remove every quotation and citation the brief did not list. A sentence that carries an
    unlisted quotation is dropped whole; an unlisted citation is cut out of its sentence."""
    stripped: list[str] = []
    bad_quote_spans: list[tuple[int, int]] = []
    for m in ANY_QUOTE_RE.finditer(text):
        if not _listed(m.group(1), quotes):
            stripped.append(_norm(m.group(1)))
            bad_quote_spans.append((m.start(), m.end()))
    for start, end in citation_spans(text):
        if not _listed(text[start:end], refs):
            stripped.append(text[start:end].strip())
    if not stripped:
        return text, []
    # drop sentences holding an unlisted quotation
    drop: list[tuple[int, int]] = []
    if bad_quote_spans:
        for sent in analyze(text).sentences:
            if any(sent.start <= s < sent.end for s, _ in bad_quote_spans):
                drop.append((sent.start, sent.end))
    out = _cut(text, drop)
    # cut remaining unlisted citations out of their sentences
    cuts = [(s, e) for s, e in citation_spans(out) if not _listed(out[s:e], refs)]
    out = _cut(out, cuts)
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r"\s+([.,;:])", r"\1", out)
    out = re.sub(r"\n[ \t]+", "\n", out)
    return out.strip(), sorted(set(stripped))


def citation_spans(text: str) -> list[tuple[int, int]]:
    """Non-overlapping citation matches; the first pattern to claim a span wins."""
    seen: list[tuple[int, int]] = []
    for pattern in CITATION_PATTERNS:
        for m in pattern.finditer(text):
            if not any(s <= m.start() < e or s < m.end() <= e for s, e in seen):
                seen.append((m.start(), m.end()))
    return sorted(seen)


def _cut(text: str, spans: list[tuple[int, int]]) -> str:
    if not spans:
        return text
    out, pos = [], 0
    for s, e in sorted(spans):
        if s < pos:
            continue
        out.append(text[pos:s])
        pos = e
    out.append(text[pos:])
    return "".join(out)


def missing_items(text: str, section: Section) -> list[str]:
    body = _norm(text)
    return [x for x in section.refs + section.quotes if _norm(x) not in body]


# ---------------------------------------------------------------- prompts

SYSTEM_TEMPLATE = """You write new prose in a specific writer's voice, from a brief. Each section
makes exactly one claim and uses exactly the references and quotations it is given, verbatim.
You return the section's prose and nothing else: no heading, no preamble, no commentary.

# The voice
{profile}

{rules}
"""

SECTION_TEMPLATE = """# Exemplars of the voice
{exemplars}

# Piece: {title}
{audience}
# Previous section (DO NOT repeat; shown only for continuity)
{previous}

# Section: {heading}
Claim (the one thing this section argues): {claim}
{notes}
Use each of these references verbatim: {refs}
Use each of these quotations verbatim, inside double quotes: {quotes}

# Constraints
- About {words} words of prose in one or two paragraphs. No heading, no list, no preamble.
- Make this claim and no other.
- Do not add any quotation, citation, number, or named source beyond those listed.
- Every listed reference and quotation must appear exactly as given.{feedback}"""


def build_section_prompt(
    brief: Brief,
    section: Section,
    voice: Voice,
    previous: str,
    missing: list[str],
) -> tuple[str, str]:
    system = SYSTEM_TEMPLATE.format(profile=voice.profile.strip(), rules=voice.rules.strip())
    exemplars = parse_exemplars(voice.exemplars)[:3]
    feedback = ""
    if missing:
        feedback = "\n- Your previous attempt omitted these; include them exactly: " + " | ".join(
            missing
        )
    user = SECTION_TEMPLATE.format(
        exemplars="\n\n".join(exemplars) or "(none)",
        title=brief.title,
        audience=f"Audience: {brief.audience}\n" if brief.audience else "",
        previous=previous[-600:].strip() or "(this is the first section)",
        heading=section.heading,
        claim=section.claim,
        notes=f"Notes: {section.notes}\n" if section.notes else "",
        refs="; ".join(section.refs) or "(none)",
        quotes="; ".join(f'"{q}"' for q in section.quotes) or "(none)",
        words=brief.words_per_section,
        feedback=feedback,
    )
    return system, user


def clean_response(raw: str) -> str:
    out = raw.strip()
    for fence in ("```", "'''"):
        if out.startswith(fence) and out.endswith(fence):
            out = out[len(fence) : -len(fence)].strip()
    if len(out) > 2 and out[0] in '"“' and out[-1] in '"”' and out.count('"') == 2:
        out = out[1:-1].strip()
    return out


def clean_section(raw: str) -> str:
    text = clean_response(raw)
    lines = [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


# ---------------------------------------------------------------- driver


def assemble(brief: Brief, sections: list[SectionLog]) -> str:
    parts = [f"# {brief.title}"]
    for s in sections:
        parts.append(f"## {s.heading}\n\n{s.text}")
    return "\n\n".join(parts).strip() + "\n"


def draft_sections(
    brief: Brief,
    voice: Voice,
    provider: CachingProvider,
    *,
    progress: Callable[[str], None] | None = None,
) -> tuple[str, list[SectionLog]]:
    """Write each section of `brief` in `voice` (two attempts when a listed reference or
    quotation is missing), strip unlisted quotations and citations, and assemble the piece.
    The result is what `apply` then rewrites as a whole."""
    logs: list[SectionLog] = []
    previous = ""
    for section in brief.sections:
        if progress:
            progress(f"drafting: {section.heading} via {provider.key}")
        best: SectionLog | None = None
        missing: list[str] = []
        attempt_provider = provider
        for attempt in range(1, 3):
            system, user = build_section_prompt(brief, section, voice, previous, missing)
            try:
                completion = attempt_provider.complete(system, user, temperature=0.7)
            except ProviderError as exc:
                if progress:
                    progress(f"  {attempt_provider.key} failed: {exc}")
                continue
            text, stripped = strip_unlisted(
                clean_section(completion.text), refs=section.refs, quotes=section.quotes
            )
            missing = missing_items(text, section)
            log = SectionLog(
                heading=section.heading,
                text=text,
                attempts=attempt,
                complete=not missing and bool(text),
                missing=missing,
                stripped=stripped,
            )
            if best is None or len(log.missing) < len(best.missing):
                best = log
            best.attempts = attempt
            if log.complete:
                break
        if best is None:
            raise ProviderError("the writer could not draft section " + section.heading)
        if progress and not best.complete:
            progress(f"  incomplete after {best.attempts} attempts; missing: {best.missing}")
        logs.append(best)
        previous = best.text
    return assemble(brief, logs), logs
