# ABOUTME: The writer's prompts: the whole-document rewrite prompt and the edit-mode prompt,
# ABOUTME: their templates, and the cleanup of a model's reply.
from __future__ import annotations

import math
from collections.abc import Sequence

from deslop.citations import Citations
from deslop.lock import DocumentLock, inventory
from deslop.nlp import Analysis
from deslop.patterns import Catalog
from deslop.reference import Passage
from deslop.units import Section, Unit
from deslop.voice.exemplars import ExemplarIndex
from deslop.voice.model import Voice

SYSTEM_TEMPLATE = """You write prose in one specific writer's voice. You are given the voice's
description and rules, exemplars of it, and a passage to write afresh in that voice. You
return the new passage and nothing else: no preamble, no quotation marks around it, no notes.

# The voice
{profile}

{rules}
"""


DOCUMENT_TEMPLATE = """# Exemplars of the voice
{exemplars}

# The original, in full
{document}

# What it asserts, paragraph by paragraph (keep every one, add none)
{propositions}
{section_block}{quotable}{reproduce}
# Instruction
Write {what} as this writer would, from its first paragraph to its last. Keep every
proposition listed above and add none. Adopt this writer's grammatical person and mode of
address as shown in the exemplars and profile; the propositions' phrasing is not a constraint
on person or address. Where a paragraph's note says what the original paragraph also did
(its address, questions, images, exclamations, appeals to the reader), use those where this
writer would; they are hints, not requirements. Formatting (title, headings, lists, emphasis)
follows the exemplars; the input's formatting is not a constraint, and when the exemplars
have no headings the output has none. Paragraph length and count follow the exemplars, not
the input: merge or split the input's paragraphs freely. Keep the author's tense and order of
argument unless the voice's profile says otherwise. Keep it about the length of the original.
{quote_rule}Return {returned} and nothing else.{noticed}

Draft {attempt} of {n}."""


QUOTABLE_TEMPLATE = """
# Passages you may quote (verbatim, from the primary texts the original cites)
{passages}

# How these texts are cited
{guidance}
"""


REPRODUCE_TEMPLATE = """
# Reproduce exactly (each once, in the sentence that makes the same point)
{items}
"""


REPRODUCE_RULE = {
    "keep": "Reproduce every citation and quotation listed above exactly as written, once each, "
    "in the sentence that makes the same point. ",
    "none": "Reproduce every citation and quotation listed above exactly as written, once each, "
    "in the sentence that makes the same point, and add no citation or quotation of your own. ",
}


QUOTE_RULE = (
    "Quote only from the passages listed above and only verbatim; do not quote from memory; "
    "if a passage you want is not there, paraphrase it without quotation marks. "
)


CONTINUITY_BLOCK = """
# The section that comes immediately before (already in the voice; continue it, do not rewrite it)
{preceding}

# What comes after (not yet converted; context only, do not rewrite it)
{following}
"""


SECTION_BLOCK = """
# The section to write now (paragraphs {first} to {last} of the original)
{section}

# Already written in the voice (it precedes the section; continue from it)
{previous}

# What follows the section in the original (for continuity; do not write it)
{following}
"""


NOTICED_TEMPLATE = """

# What a reader noticed in earlier drafts (do not repeat these)
{notes}"""


EDIT_TEMPLATE = """# Exemplars of the voice
{exemplars}

# Context (DO NOT EDIT; shown only so the rewrite fits)
BEFORE: {before}
AFTER: {after}

# The {kind} to rewrite
{unit}

# What is wrong with it
{problems}

# Must survive the rewrite (content lock)
{lock}

# Constraints
- Return only the rewritten {kind}, in the voice, between {min_words} and {max_words} words.
- Do not repeat or alter the surrounding context.
- Keep every claim; add no new fact, name, number, quotation or citation.
{extra}{rejected}"""


REJECTED_TEMPLATE = """

# Earlier attempts and why they were rejected (do not repeat their mistakes)
{attempts}"""


FIX_HINTS = {
    "T1": "Make a person the subject (the writer, a character, the reader) or state the fact.",
    "T5": "End on a particular already in the text: a name, a quotation, a plain fact. No "
    "antithesis, no abstract-noun equation.",
    "T2": "Say what is the case directly; drop the 'not X but Y' frame.",
    "T9": "Either answer the question in the same sentence or turn it into a statement.",
    "T13": "Replace the colon reveal with an ordinary clause.",
    "T14": "Do not restate; add a new particular or cut the paragraph.",
    "T15": "Vary sentence length: at least one sentence under 8 words, one well over 20.",
}


SENTENCE_SLACK_WORDS = 4


def length_bounds(unit: Unit, preserve_length: float) -> tuple[int, int]:
    words = len(unit.text.split())
    slack = SENTENCE_SLACK_WORDS if unit.kind == "sentence" else 0
    low = max(1, int(words * (1 - preserve_length)) - slack)
    high = int(math.ceil(words * (1 + preserve_length))) + 1 + slack
    return low, high


def _problems(unit: Unit, catalog: Catalog) -> str:
    lines = []
    for pid in unit.tell_ids:
        p = catalog.by_id(pid)
        spans = sorted({h.text.strip()[:80] for h in unit.hits if h.pattern_id == pid})
        hint = FIX_HINTS.get(pid, "")
        lines.append(
            f"- {p.id} {p.name}: {p.description} Flagged span(s): "
            + " | ".join(spans)
            + (f" Fix: {hint}" if hint else "")
        )
    for note in unit.metric_notes:
        lines.append(f"- {note}")
    return "\n".join(lines) if lines else "- Does not read in the voice."


def _context(unit: Unit, text: str, a: Analysis) -> tuple[str, str]:
    para = a.paragraphs[unit.para_index]
    before = text[max(para.start, unit.start - 400) : unit.start].strip() or "(start of text)"
    after = text[unit.end : min(para.end, unit.end + 400)].strip() or "(end of paragraph)"
    return before, after


TEXTURE_LEAD = "The original paragraph also did these things; use them where this writer would: "


def _numbered(propositions: list[list[str]], first: int, textures: Sequence[str] = ()) -> str:
    lines = []
    for i, props in enumerate(propositions, first + 1):
        lines.append(f"Paragraph {i}:")
        lines.extend(f"  {k}. {p}" for k, p in enumerate(props, 1))
        if not props:
            lines.append("  (none)")
        texture = textures[i - first - 1] if i - first - 1 < len(textures) else ""
        if texture:
            lines.append(f"  {TEXTURE_LEAD}{texture}")
    return "\n".join(lines)


def document_prompt(
    section: Section,
    original: str,
    section_text: str,
    propositions: list[list[str]],
    voice: Voice,
    exemplars: ExemplarIndex,
    *,
    previous: str,
    following: str,
    attempt: int,
    n: int,
    noticed: list[str],
    textures: Sequence[str] = (),
    passages: Sequence[Passage] = (),
    guidance: Sequence[str] = (),
    reproduce: Sequence[str] = (),
    citations: Citations = "verified",
) -> tuple[str, str]:
    """The writer's prompt: profile and rules (system), exemplars nearest the piece, the full
    original, its propositions by paragraph (each with its texture note, when there is one),
    the passages it may quote verbatim with how the texts are cited (when primary texts are
    declared), and the instruction. A long document adds the section to write with the
    already-rewritten section before it and the original section after it. `noticed` carries
    what a reader saw in this draft's first-round version, for the retry."""
    system = SYSTEM_TEMPLATE.format(profile=voice.profile.strip(), rules=voice.rules.strip())
    whole = section.count == 1
    block = (
        (
            CONTINUITY_BLOCK.format(
                preceding=previous or "(nothing comes before)",
                following=following or "(nothing comes after)",
            )
            if previous or following
            else ""
        )
        if whole
        else SECTION_BLOCK.format(
            first=section.first + 1,
            last=section.last + 1,
            section=section_text,
            previous=previous or "(this is the first section)",
            following=following or "(this is the last section)",
        )
    )
    user = DOCUMENT_TEMPLATE.format(
        exemplars="\n\n".join(exemplars.closest(section_text)) or "(no exemplars in this voice)",
        document=original,
        propositions=_numbered(propositions, 0 if whole else section.first, textures) or "(none)",
        section_block=block,
        quotable=QUOTABLE_TEMPLATE.format(
            passages="\n\n".join(f"[{p.text_name} {p.ref}] {p.text.strip()}" for p in passages)
            or "(the original cites nothing that resolves in the primary texts)",
            guidance="\n".join(f"- {g}" for g in guidance) or "- (none)",
        )
        if passages or guidance
        else "",
        quote_rule=(QUOTE_RULE if passages or guidance else "")
        + (REPRODUCE_RULE.get(citations, "") if reproduce or citations == "none" else ""),
        reproduce=REPRODUCE_TEMPLATE.format(items="\n".join(f"- {r}" for r in reproduce))
        if reproduce
        else "",
        what="this piece" if whole else "this section",
        returned="the whole piece" if whole else "the section",
        noticed=NOTICED_TEMPLATE.format(notes="\n".join(f"- {x}" for x in noticed))
        if noticed
        else "",
        attempt=attempt,
        n=n,
    )
    return system, user


def edit_prompt(
    unit: Unit,
    text: str,
    a: Analysis,
    voice: Voice,
    exemplars: ExemplarIndex,
    document: DocumentLock,
    catalog: Catalog,
    preserve_length: float,
    rejected: list[tuple[str, str]],
) -> tuple[str, str]:
    system = SYSTEM_TEMPLATE.format(profile=voice.profile.strip(), rules=voice.rules.strip())
    before, after = _context(unit, text, a)
    low, high = length_bounds(unit, preserve_length)
    inv = inventory(unit.text, document.names)
    lock_lines = inv.describe() or ["(nothing locked in this unit)"]
    extra = "- You may split or merge sentences within this passage; keep it on one line."
    if unit.kind == "sentence":
        extra = "- Keep it on one line."
    if "T15" in unit.tell_ids:
        extra += "\n- Vary sentence length sharply: at least one sentence under 8 words."
    user = EDIT_TEMPLATE.format(
        exemplars="\n\n".join(
            exemplars.closest(
                text[a.paragraphs[unit.para_index].start : a.paragraphs[unit.para_index].end]
            )
        )
        or "(no exemplars in this voice)",
        before=before,
        after=after,
        kind=unit.kind,
        unit=unit.text,
        problems=_problems(unit, catalog),
        lock="\n".join(f"- {line}" for line in lock_lines),
        min_words=low,
        max_words=high,
        extra=extra,
        rejected=_rejected_section(rejected),
    )
    return system, user


def _rejected_section(rejected: list[tuple[str, str]]) -> str:
    if not rejected:
        return ""
    lines = [f'- Rejected ({reason}): "{candidate[:300]}"' for candidate, reason in rejected]
    return REJECTED_TEMPLATE.format(attempts="\n".join(lines))


def clean_response(raw: str) -> str:
    out = raw.strip()
    for fence in ("```", "'''"):
        if out.startswith(fence) and out.endswith(fence):
            out = out[len(fence) : -len(fence)].strip()
    if len(out) > 2 and out[0] in '"“' and out[-1] in '"”' and out.count('"') == 2:
        out = out[1:-1].strip()
    return out
