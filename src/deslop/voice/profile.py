# ABOUTME: Writes profile.md, the positive description of a voice: a facts block rendered from
# ABOUTME: metrics.json, the registers found by clustering, and LLM or templated prose.
from __future__ import annotations

import math

from deslop.baselines import ReferenceProfile
from deslop.providers.base import CachingProvider
from deslop.stats import Cluster, MetricsProfile, MetricStat

PROFILE_SYSTEM = """You are a careful stylistician. You describe a writer's voice from a block
of measured facts, a summary of the registers (clusters of documents) the corpus contains, and
verbatim excerpts. Write 300-600 words of plain, concrete, positive description: what this
writer characteristically does, what they never do, how sentences move, how paragraphs end,
how the reader is addressed. No praise, no hedging, no bullet points, no headings.

Two hard constraints. First, every stylistic claim you make must be consistent with the facts
block: if the facts say a feature is low or rare, do not call it frequent; if variation is
high, do not call it steady; if the median is zero, say the feature is absent from most
documents rather than usual. Second, name the registers: say what each register is (who
writes it, how long it runs, what distinguishes it) and describe the voice as the thing they
share, noting where they differ. Do not mention that you were given metrics or a facts block.
Cite numbers sparingly and only when they sharpen a claim."""

# One-line meanings used in the facts block; the key is the metric name in metrics.json.
METRIC_DESCRIPTIONS: dict[str, str] = {
    "citation_per_100w": "formal citations per 100 words",
    "colons_per_1k": "colons per 1k words",
    "committed_claim_per_para": "committed evaluative claims per paragraph",
    "concrete_per_100w": "concrete tokens (names, numbers, foreign script) per 100 words",
    "em_dashes_per_1k": "em dashes per 1k words",
    "first_person_per_1k": "first person singular per 1k words",
    "max_run_similar_len": "longest run of similar-length sentences",
    "opener_pct_adv_conj": "share of sentences opening with an adverb or conjunction",
    "opener_pct_det": "share of sentences opening with a determiner",
    "opener_pct_noun": "share of sentences opening with a noun",
    "opener_pct_pron": "share of sentences opening with a pronoun",
    "opener_top_word_share": "share of openings taken by the single most common first word",
    "para_len_mean": "paragraph length in sentences",
    "para_len_std": "paragraph-length variation (sentences)",
    "parens_per_1k": "parentheses per 1k words",
    "pct_compound_multi": "share of sentences with three or more clauses",
    "pct_sent_gt30": "share of sentences over thirty words",
    "pct_sent_lt8": "share of sentences under eight words",
    "questions_per_1k": "question marks per 1k words",
    "quote_per_para": "quotations of four or more words per paragraph",
    "reader_address_per_1k": "reader address (you) per 1k words",
    "semicolons_per_1k": "semicolons per 1k words",
    "sent_len_cv": "sentence-length variation (coefficient of variation)",
    "sent_len_mean": "mean sentence length in words",
    "sent_len_std": "sentence-length standard deviation",
}


def percentile_of(value: float, stat: MetricStat) -> int:
    """Approximate percentile of `value` in a baseline's distribution (normal on mean/std)."""
    if stat.std <= 0:
        return 50 if value == stat.mean else (100 if value > stat.mean else 0)
    z = (value - stat.mean) / stat.std
    return int(round(100 * 0.5 * (1 + math.erf(z / math.sqrt(2)))))


def level_word(pct: int) -> str:
    if pct >= 90:
        return "very high"
    if pct >= 75:
        return "high"
    if pct <= 10:
        return "very low"
    if pct <= 25:
        return "low"
    return "typical"


def metric_fact(
    name: str, stat: MetricStat, generic: ReferenceProfile, llm: ReferenceProfile | None
) -> str:
    label = METRIC_DESCRIPTIONS.get(name, name)
    parts = [f"{label} is "]
    g = generic.metrics.get(name)
    if g is not None:
        pg = percentile_of(stat.mean, g)
        parts.append(f"{level_word(pg)}: mean {stat.mean:.2f} (median {stat.p50:.2f}), ")
        parts.append(f"{ordinal(pg)} percentile of generic English")
    else:
        parts.append(f"mean {stat.mean:.2f} (median {stat.p50:.2f})")
    if llm is not None and name in llm.metrics:
        parts.append(
            f", {ordinal(percentile_of(stat.mean, llm.metrics[name]))} percentile of LLM prose"
        )
    return "- " + "".join(parts) + f" [{name}]"


def ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def facts_block(
    profile: MetricsProfile, generic: ReferenceProfile, llm: ReferenceProfile | None
) -> str:
    """The verbatim facts the prose must agree with: every metric in words, with its value
    and percentile against the generic and LLM baselines, then tell rates against LLM prose."""
    lines = [
        "## Facts",
        "",
        "Measured over 500-word chunks of the corpus; percentiles place the",
        "mean against the generic English and LLM-prose baselines.",
        "",
    ]
    for name in sorted(profile.metrics):
        lines.append(metric_fact(name, profile.metrics[name], generic, llm))
    if profile.tells:
        lines.append("")
        for tid in sorted(profile.tells, key=lambda t: int(t[1:]) if t[1:].isdigit() else 0):
            rate = profile.tells[tid].rate_per_1k
            ref = llm.tells.get(tid) if llm is not None else None
            if ref is not None and ref.rate_per_1k > 0:
                lines.append(
                    f"- tell {tid}: {rate:.2f} per 1k words, "
                    f"{rate / ref.rate_per_1k:.1f}x the LLM-prose rate"
                )
            else:
                lines.append(f"- tell {tid}: {rate:.2f} per 1k words")
    return "\n".join(lines) + "\n"


def cluster_summary(clusters: list[Cluster], total_words: int) -> str:
    if not clusters:
        return ""
    lines = ["## Registers", ""]
    if len(clusters) == 1:
        lines.append("The documents form a single register.")
    for c in clusters:
        share = 100 * c.words / max(total_words, 1)
        marks = ", ".join(
            f"{METRIC_DESCRIPTIONS.get(m.metric, m.metric)} {'above' if m.z > 0 else 'below'} "
            f"the voice mean ({m.cluster_mean:.2f} vs {m.voice_mean:.2f})"
            for m in c.distinguishing
        )
        lines.append(
            f"- {c.label}: {c.documents} documents, {share:.0f}% of words, median document "
            f"{c.median_doc_words} words; top authors {', '.join(c.top_authors)}; "
            f"distinguished by {marks}."
        )
    return "\n".join(lines) + "\n"


def assemble(name: str, facts: str, registers: str, prose: str) -> str:
    parts = [f"# Voice: {name}", "", facts.rstrip(), ""]
    if registers:
        parts += [registers.rstrip(), ""]
    parts += ["## Description", "", prose.strip(), ""]
    return "\n".join(parts)


def template_profile(
    name: str,
    profile: MetricsProfile,
    generic: ReferenceProfile,
    llm: ReferenceProfile | None = None,
) -> str:
    """Profile without an LLM: the facts block and registers, plus a short templated note."""
    m = profile.metrics

    def mean(k: str) -> float:
        return m[k].mean if k in m else 0.0

    prose = (
        f"Sentences in this voice average {mean('sent_len_mean'):.0f} words with a "
        f"coefficient of variation of {mean('sent_len_cv'):.2f}; paragraphs average "
        f"{mean('para_len_mean'):.1f} sentences. Per thousand words: semicolons "
        f"{mean('semicolons_per_1k'):.1f}, colons {mean('colons_per_1k'):.1f}, em dashes "
        f"{mean('em_dashes_per_1k'):.1f}, parentheses {mean('parens_per_1k'):.1f}, questions "
        f"{mean('questions_per_1k'):.1f}. The first person singular appears "
        f"{mean('first_person_per_1k'):.1f} times per thousand words and the reader is "
        f"addressed {mean('reader_address_per_1k'):.1f} times. Quotations run "
        f"{mean('quote_per_para'):.2f} per paragraph and citations "
        f"{mean('citation_per_100w'):.2f} per hundred words.\n\n"
        "This description was generated from the facts above without a language model. "
        "Edit it by hand, or re-run calibrate with a provider configured for a fuller one."
    )
    return assemble(
        name,
        facts_block(profile, generic, llm),
        cluster_summary(profile.clusters, profile.total_words),
        prose,
    )


def llm_profile(
    name: str,
    profile: MetricsProfile,
    generic: ReferenceProfile,
    exemplars_md: str,
    rules_md: str,
    provider: CachingProvider,
    model: str | None = None,
    llm: ReferenceProfile | None = None,
) -> str:
    facts = facts_block(profile, generic, llm)
    registers = cluster_summary(profile.clusters, profile.total_words)
    user = (
        f"Voice name: {name}\n\n{facts}\n{registers or 'Registers: one register.'}\n\n"
        f"Rules derived from the measurements:\n{rules_md}\n\n"
        f"Verbatim exemplars (the heading of each names its register):\n{exemplars_md}\n\n"
        "Write the 300-600 word voice description now. Every claim must agree with the facts; "
        "name the registers."
    )
    completion = provider.complete(PROFILE_SYSTEM, user, model=model, temperature=0.4)
    return assemble(name, facts, registers, completion.text)
