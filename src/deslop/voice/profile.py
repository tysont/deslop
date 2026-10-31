# ABOUTME: Writes profile.md, the positive description of a voice, from metrics and exemplars.
# ABOUTME: Uses an LLM when available and allowed, otherwise a templated description.
from __future__ import annotations

from deslop.baselines import ReferenceProfile
from deslop.providers.base import CachingProvider
from deslop.stats import MetricsProfile

PROFILE_SYSTEM = """You are a careful stylistician. You describe a writer's voice from
measurements and verbatim excerpts. You write 300-600 words of plain, concrete,
positive description: what this writer characteristically does, what they never do,
how sentences move, how paragraphs end, how the reader is addressed. No praise, no
hedging, no bullet points, no headings. Do not describe what the writing is "not" except
in a single short list of things the writer never does. Do not mention that you were given
metrics. Cite numbers sparingly and only when they sharpen a claim."""


def describe_metric_row(name: str, profile: MetricsProfile, generic: ReferenceProfile) -> str:
    v = profile.metrics[name].mean
    g = generic.metrics[name].mean if name in generic.metrics else None
    if g is None:
        return f"{name}: {v:.2f}"
    return f"{name}: {v:.2f} (generic English {g:.2f})"


def template_profile(name: str, profile: MetricsProfile, generic: ReferenceProfile) -> str:
    m = profile.metrics

    def mean(k: str) -> float:
        return m[k].mean if k in m else 0.0

    def gen(k: str) -> float:
        return generic.metrics[k].mean if k in generic.metrics else 0.0

    def compare(k: str, low: str, high: str, same: str) -> str:
        g = gen(k)
        v = mean(k)
        if g <= 0:
            return same
        if v < 0.6 * g:
            return low
        if v > 1.5 * g:
            return high
        return same

    paragraphs = [
        f"# Voice: {name}",
        "",
        f"Sentences in this voice average {mean('sent_len_mean'):.0f} words, with a coefficient "
        f"of variation of {mean('sent_len_cv'):.2f}; about {mean('pct_sent_lt8'):.0f}% are under "
        f"eight words and {mean('pct_sent_gt30'):.0f}% run past thirty. "
        + compare(
            "sent_len_mean",
            "The sentences are short by the standard of general English prose.",
            "The sentences are long by the standard of general English prose.",
            "Sentence length sits near the general English norm.",
        )
        + f" Paragraphs average {mean('para_len_mean'):.1f} sentences.",
        "",
        "Punctuation per thousand words: "
        f"semicolons {mean('semicolons_per_1k'):.1f}, colons {mean('colons_per_1k'):.1f}, "
        f"em dashes {mean('em_dashes_per_1k'):.1f}, parentheses {mean('parens_per_1k'):.1f}, "
        f"question marks {mean('questions_per_1k'):.1f}. "
        + compare(
            "semicolons_per_1k",
            "The writer rarely uses semicolons.",
            "The writer leans on the semicolon to chain related clauses.",
            "Semicolon use is ordinary.",
        ),
        "",
        f"The writing is anchored in particulars: {mean('concrete_per_100w'):.1f} concrete tokens "
        f"(names, numbers, foreign script, italic terms) per hundred words, "
        f"{mean('quote_per_para'):.2f} quotations of four or more words per paragraph, and "
        f"{mean('citation_per_100w'):.2f} formal citations per hundred words. "
        + compare(
            "quote_per_para",
            "Quotation is rare; claims are made in the writer's own words.",
            "The writer quotes the text under discussion often and at length.",
            "Quotation is moderate.",
        ),
        "",
        f"The first person singular appears {mean('first_person_per_1k'):.1f} times per thousand "
        f"words and the reader is addressed {mean('reader_address_per_1k'):.1f} times "
        "per thousand. "
        f"Committed evaluative claims (a reading is right or wrong) occur "
        f"{mean('committed_claim_per_para'):.2f} times per paragraph. "
        + compare(
            "first_person_per_1k",
            "The writer stays out of the frame.",
            "The writer speaks in their own person and takes responsibility for judgments.",
            "The writer uses the first person at an ordinary rate.",
        ),
        "",
        f"Sentence openers: {mean('opener_pct_det'):.0f}% begin with a determiner, "
        f"{mean('opener_pct_pron'):.0f}% with a pronoun, "
        f"{mean('opener_pct_noun'):.0f}% with a noun, "
        f"and {mean('opener_pct_adv_conj'):.0f}% with an adverb or conjunction. "
        f"The most common first word accounts for {mean('opener_top_word_share'):.0f}% "
        "of openings.",
        "",
        "This profile was generated from metrics alone. Edit it by hand, or re-run calibrate "
        "with an LLM provider configured to produce a fuller description.",
    ]
    return "\n".join(paragraphs) + "\n"


def llm_profile(
    name: str,
    profile: MetricsProfile,
    generic: ReferenceProfile,
    exemplars_md: str,
    rules_md: str,
    provider: CachingProvider,
    model: str | None = None,
) -> str:
    rows = "\n".join(describe_metric_row(k, profile, generic) for k in sorted(profile.metrics))
    user = (
        f"Voice name: {name}\n\nMeasurements (mean over 500-word chunks):\n{rows}\n\n"
        f"Rules derived from the measurements:\n{rules_md}\n\n"
        f"Verbatim exemplars:\n{exemplars_md}\n\n"
        "Write the 300-600 word voice description now."
    )
    completion = provider.complete(PROFILE_SYSTEM, user, model=model, temperature=0.4)
    return f"# Voice: {name}\n\n{completion.text.strip()}\n"
