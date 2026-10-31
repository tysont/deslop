# ABOUTME: Builds rules.md: the default rule set plus rules justified by the measured metrics.
# ABOUTME: Each derived rule cites the number that justifies it.
from __future__ import annotations

from importlib import resources

from deslop.baselines import ReferenceProfile
from deslop.stats import MetricsProfile
from deslop.voice.model import Lexicon

# (metric, low-threshold-ratio-vs-generic, high-threshold-ratio, low text, high text)
DERIVED_RULES = [
    (
        "first_person_per_1k",
        0.5,
        1.5,
        "First person singular appears {v:.1f}/1k words (generic English {g:.1f}): avoid it.",
        "First person singular appears {v:.1f}/1k words (generic English {g:.1f}): use it.",
    ),
    (
        "semicolons_per_1k",
        0.5,
        1.5,
        "Semicolons: {v:.1f}/1k words (generic {g:.1f}): avoid them.",
        "Semicolons: {v:.1f}/1k words (generic {g:.1f}): use them to join related clauses.",
    ),
    (
        "colons_per_1k",
        0.5,
        1.5,
        "Colons: {v:.1f}/1k words (generic {g:.1f}): avoid them.",
        "Colons: {v:.1f}/1k words (generic {g:.1f}): acceptable before lists and quotations.",
    ),
    (
        "em_dashes_per_1k",
        0.5,
        1.5,
        "Em dashes: {v:.1f}/1k words (generic {g:.1f}): avoid them.",
        "Em dashes: {v:.1f}/1k words (generic {g:.1f}): acceptable for asides.",
    ),
    (
        "questions_per_1k",
        0.5,
        1.5,
        "Questions: {v:.1f}/1k words (generic {g:.1f}): avoid rhetorical questions.",
        "Questions: {v:.1f}/1k words (generic {g:.1f}): questions are in voice when answered.",
    ),
    (
        "reader_address_per_1k",
        0.5,
        1.5,
        "Reader address (you): {v:.1f}/1k words (generic {g:.1f}): do not address the reader.",
        "Reader address (you): {v:.1f}/1k words (generic {g:.1f}): address the reader directly.",
    ),
    (
        "quote_per_para",
        0.5,
        1.5,
        "Quotations: {v:.2f} per paragraph (generic {g:.2f}): quote sparingly.",
        "Quotations: {v:.2f} per paragraph (generic {g:.2f}): quote the text often and at length.",
    ),
    (
        "citation_per_100w",
        0.5,
        1.5,
        "Citations: {v:.2f}/100 words (generic {g:.2f}): few formal references.",
        "Citations: {v:.2f}/100 words (generic {g:.2f}): cite by reference often.",
    ),
    (
        "sent_len_mean",
        0.75,
        1.25,
        "Sentences average {v:.0f} words (generic {g:.0f}): keep sentences short.",
        "Sentences average {v:.0f} words (generic {g:.0f}): long sentences are in voice.",
    ),
    (
        "para_len_mean",
        0.75,
        1.25,
        "Paragraphs average {v:.1f} sentences (generic {g:.1f}): keep paragraphs short.",
        "Paragraphs average {v:.1f} sentences (generic {g:.1f}): paragraphs run long.",
    ),
]


def default_rules() -> str:
    return (resources.files("deslop.baselines") / "default_rules.md").read_text(encoding="utf-8")


def derived_rules(
    profile: MetricsProfile, generic: ReferenceProfile, lexicon: Lexicon
) -> list[str]:
    rules: list[str] = []
    for name, low, high, low_text, high_text in DERIVED_RULES:
        if name not in profile.metrics or name not in generic.metrics:
            continue
        v = profile.metrics[name].mean
        g = generic.metrics[name].mean
        if g <= 0:
            continue
        ratio = v / g
        if ratio < low:
            rules.append(low_text.format(v=v, g=g))
        elif ratio > high:
            rules.append(high_text.format(v=v, g=g))
    cv = profile.metrics.get("sent_len_cv")
    if cv is not None:
        rules.append(
            f"Sentence length varies (coefficient of variation {cv.mean:.2f}, "
            f"p10 {cv.p10:.2f}): mix long and short sentences; "
            "never run several of the same length."
        )
    banned = [p for p in lexicon.patterns if p.banned]
    for p in banned:
        rules.append(
            f"Pattern {p.id} ({p.name}) is banned: {p.corpus_per_1k:.2f}/1k words in this voice "
            f"vs {p.llm_per_1k:.2f}/1k in LLM prose."
        )
    return rules


def build_rules(profile: MetricsProfile, generic: ReferenceProfile, lexicon: Lexicon) -> str:
    lines = [default_rules().rstrip(), "", "## Derived from this voice", ""]
    derived = derived_rules(profile, generic, lexicon)
    if derived:
        lines.extend(f"- {r}" for r in derived)
    else:
        lines.append("- (no metric differs enough from generic English to justify a rule)")
    return "\n".join(lines) + "\n"
