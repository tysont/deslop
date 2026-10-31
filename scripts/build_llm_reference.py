# ABOUTME: Builds src/deslop/baselines/llm_reference.json by generating ~200k words of
# ABOUTME: LLM essay prose across providers and storing aggregate stats only (never the text).
from __future__ import annotations

import datetime as dt
import itertools
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from deslop.baselines import ReferenceProfile, baseline_path
from deslop.corpus.normalize import normalize
from deslop.providers.base import get_provider, list_providers
from deslop.stats import WordCounts, aggregate, merge_samples, sample_text

TARGET_WORDS = int(os.environ.get("DESLOP_LLM_REF_WORDS", "200000"))
ESSAY_WORDS = 700
WORKERS = int(os.environ.get("DESLOP_LLM_REF_WORKERS", "8"))
CACHE = Path(".cache/llm_reference")

TOPICS = [
    "the opening chapter of Genesis",
    "Hamlet's delay",
    "the ending of King Lear",
    "Plato's cave",
    "Aristotle on friendship",
    "the Book of Job",
    "Augustine's Confessions",
    "Dante's Inferno, canto 5",
    "Montaigne on cannibals",
    "Paradise Lost, book 9",
    "Pride and Prejudice's first sentence",
    "Moby-Dick's whiteness chapter",
    "Middlemarch's finale",
    "the Iliad's final book",
    "Odysseus and the Cyclops",
    "Virgil's Dido",
    "Ovid's Metamorphoses",
    "Beowulf's dragon",
    "Chaucer's Pardoner",
    "Macbeth's porter",
    "Othello's handkerchief",
    "The Tempest's epilogue",
    "Wordsworth's Tintern Abbey",
    "Keats's Grecian urn",
    "Dickinson's dashes",
    "Whitman's catalogues",
    "Emerson on self-reliance",
    "Thoreau at Walden",
    "Frederick Douglass's narrative",
    "Lincoln's second inaugural",
    "the Gettysburg Address",
    "Kafka's Metamorphosis",
    "Woolf's Mrs Dalloway",
    "Joyce's The Dead",
    "Eliot's The Waste Land",
    "Borges's Library of Babel",
    "Camus's Stranger",
    "Orwell's Politics and the English Language",
    "Baldwin's Notes of a Native Son",
    "Achebe's Things Fall Apart",
    "the parable of the prodigal son",
    "the Sermon on the Mount",
    "Ecclesiastes",
    "the Psalms of lament",
    "Exodus and the burning bush",
    "Ruth and Naomi",
    "the Gospel of Mark's ending",
    "Paul's letter to the Romans",
    "the Bhagavad Gita",
    "the Tao Te Ching",
    "Confucius on ritual",
    "the Epic of Gilgamesh",
    "Thucydides on the plague",
    "Herodotus on Croesus",
    "Plutarch's Caesar",
    "Tacitus on Nero",
    "Machiavelli's Prince",
    "Hobbes's state of nature",
    "Locke on property",
    "Rousseau's general will",
    "Kant on enlightenment",
    "Hume on miracles",
    "Mill on liberty",
    "Nietzsche's eternal return",
    "Darwin's tangled bank",
    "Freud on dreams",
    "Marx on commodity fetishism",
    "Weber's iron cage",
    "Arendt on the banality of evil",
    "Wittgenstein's beetle in the box",
    "the Turing test",
    "Kuhn's paradigms",
    "the trolley problem",
    "the prisoner's dilemma",
    "the Fermi paradox",
    "the printing press",
    "the French Revolution's terror",
    "the American Constitution's preamble",
    "the Magna Carta",
    "the fall of Constantinople",
    "the Black Death",
    "the Reformation",
    "the Scientific Revolution",
    "the Industrial Revolution",
    "the invention of photography",
    "jazz improvisation",
    "Bach's Goldberg Variations",
    "Beethoven's late quartets",
    "Vermeer's light",
    "Caravaggio's shadows",
    "Hokusai's wave",
    "Gothic cathedrals",
    "the Parthenon",
    "Japanese tea ceremony",
    "chess openings",
    "the rules of cricket",
    "why we dream",
    "the history of the alphabet",
    "the invention of zero",
]

STYLES = [
    "a critical essay for a literary review",
    "a chapter of a popular nonfiction book",
    "a reflective personal essay",
    "an analytical blog post for educated general readers",
    "a book review",
    "an introduction to a scholarly edition",
]

SYSTEM = (
    "You are a thoughtful essayist. Write in polished, publishable English prose. "
    "Do not use headings or bullet points."
)


def prompt_for(topic: str, style: str) -> str:
    return f"Write {style} of about {ESSAY_WORDS} words on {topic}. Prose only."


# provider:model pairs; override with
# DESLOP_LLM_REF_SOURCES="anthropic:claude-opus-5-5,openai:gpt-4.1"
DEFAULT_SOURCES = [
    ("anthropic", "claude-opus-5-5"),
    ("anthropic", "claude-sonnet-5-5"),
    ("openai", "gpt-4.1"),
]


def working_sources() -> list[tuple[str, str, object]]:
    """Return (provider name, model, provider) for every source that answers a ping."""
    configured = list_providers()
    raw = os.environ.get("DESLOP_LLM_REF_SOURCES")
    pairs = [tuple(x.split(":", 1)) for x in raw.split(",")] if raw else DEFAULT_SOURCES
    out = []
    for name, model in pairs:
        if not configured.get(name):
            print(f"skip {name}:{model}: not configured", file=sys.stderr)
            continue
        provider = get_provider(name, cache_dir=CACHE / "responses")
        try:
            provider.complete(
                "Reply with the single word OK.", "ping", model=model, temperature=0.0
            )
        except Exception as exc:  # noqa: BLE001 - report and move on
            print(f"skip {name}:{model}: {str(exc)[:120]}", file=sys.stderr)
            continue
        out.append((name, model, provider))
    return out


def main() -> None:
    sources = working_sources()
    if not sources:
        print("no working providers (set ANTHROPIC_API_KEY and/or OPENAI_API_KEY)", file=sys.stderr)
        sys.exit(1)
    print("sources: " + ", ".join(f"{n}:{m}" for n, m, _ in sources), flush=True)
    counts = WordCounts()
    per_source = {}
    words = 0
    jobs = list(itertools.product(TOPICS, STYLES))
    # ~ESSAY_WORDS per job; request a little more than needed and stop at the target.
    needed = int(TARGET_WORDS / ESSAY_WORDS * 1.15) + 1
    assigned = [(job, sources[i % len(sources)]) for i, job in enumerate(jobs[:needed])]

    def run(item):
        (topic, style), (name, model, provider) = item
        completion = provider.complete(
            SYSTEM, prompt_for(topic, style), model=model, temperature=0.8
        )
        return name, completion.model, normalize(completion.text, gutenberg=False)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for i, (name, model, text) in enumerate(pool.map(run, assigned)):
            n = len(text.split())
            if n < 200 or words >= TARGET_WORDS:
                continue
            words += n
            counts.add(text)
            key = f"{name}:{model}"
            samples = sample_text(text)
            if key in per_source:
                per_source[key] = merge_samples(per_source[key], samples)
            else:
                per_source[key] = samples
            if i % 10 == 0:
                spent = sum(p.usage.cost_usd for _, _, p in sources)
                print(f"{i:4d} {words:7d} words  ${spent:.2f} {key}", flush=True)
    profile = aggregate(per_source, max_share=1.0, chunk_words=500)
    ref = ReferenceProfile(
        name="llm_reference",
        created=dt.date.today().isoformat(),
        source=(
            f"{words} words of generated essays on {len(TOPICS)} topics in {len(STYLES)} styles; "
            f"sources: {', '.join(per_source)}"
        ),
        total_words=counts.total,
        metrics=profile.metrics,
        tells=profile.tells,
        word_freq=dict(counts.unigrams.most_common(30000)),
        bigram_freq=dict(counts.bigrams.most_common(30000)),
    )
    ref.save(baseline_path("llm_reference"))
    print(f"wrote {baseline_path('llm_reference')} ({counts.total} words)")
    print(json.dumps({k: round(v.rate_per_1k, 2) for k, v in profile.tells.items()}, indent=1))


if __name__ == "__main__":
    main()
