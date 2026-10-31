# ABOUTME: Builds src/deslop/baselines/llm_reference.json by generating ~200k words of
# ABOUTME: LLM essay prose across providers and storing aggregate stats only (never the text).
from __future__ import annotations

import datetime as dt
import itertools
import json
import os
import sys
from pathlib import Path

from deslop.baselines import ReferenceProfile, baseline_path
from deslop.corpus.normalize import normalize
from deslop.providers.base import get_provider, list_providers
from deslop.stats import WordCounts, aggregate, sample_text

TARGET_WORDS = int(os.environ.get("DESLOP_LLM_REF_WORDS", "200000"))
ESSAY_WORDS = 700
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


def main() -> None:
    available = [n for n, ok in list_providers().items() if ok and n != "compatible"]
    if not available:
        print(
            "no providers configured (set ANTHROPIC_API_KEY and/or OPENAI_API_KEY)", file=sys.stderr
        )
        sys.exit(1)
    providers = {n: get_provider(n, cache_dir=CACHE / "responses") for n in available}
    models = {
        "anthropic": os.environ.get("DESLOP_LLM_REF_ANTHROPIC_MODEL"),
        "openai": os.environ.get("DESLOP_LLM_REF_OPENAI_MODEL"),
    }
    counts = WordCounts()
    per_source = {}
    words = 0
    jobs = list(itertools.product(TOPICS, STYLES))
    cycle = itertools.cycle(available)
    for i, (topic, style) in enumerate(jobs):
        if words >= TARGET_WORDS:
            break
        name = next(cycle)
        provider = providers[name]
        completion = provider.complete(
            SYSTEM, prompt_for(topic, style), model=models.get(name), temperature=0.8
        )
        text = normalize(completion.text, gutenberg=False)
        n = len(text.split())
        if n < 200:
            continue
        words += n
        counts.add(text)
        key = f"{name}:{completion.model}"
        samples = sample_text(text)
        if key in per_source:
            from deslop.stats import merge_samples

            per_source[key] = merge_samples(per_source[key], samples)
        else:
            per_source[key] = samples
        if i % 10 == 0:
            print(f"{i:4d} {words:7d} words  ${provider.usage.cost_usd:.2f} {key}", flush=True)
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
