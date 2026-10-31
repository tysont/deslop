# ABOUTME: Builds src/deslop/baselines/english_generic.json from public-domain books.
# ABOUTME: Downloads Gutenberg texts into a scratch directory, stores only aggregate stats.
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import httpx

from deslop.baselines import ReferenceProfile, baseline_path
from deslop.corpus.normalize import normalize
from deslop.stats import WordCounts, aggregate, sample_text

# Mixed registers: fiction, essays, science, history, philosophy.
BOOKS = {
    1342: "Austen, Pride and Prejudice",
    158: "Austen, Emma",
    1400: "Dickens, Great Expectations",
    98: "Dickens, A Tale of Two Cities",
    2009: "Darwin, Origin of Species",
    34901: "Mill, On Liberty",
    76: "Twain, Huckleberry Finn",
    2701: "Melville, Moby Dick",
    120: "Stevenson, Treasure Island",
    174: "Wilde, Dorian Gray",
    1260: "Bronte, Jane Eyre",
    345: "Stoker, Dracula",
    5500: "Bacon, Advancement of Learning",
    1232: "Machiavelli, The Prince",
    11: "Carroll, Alice in Wonderland",
    16209: "Hazlitt, Lectures on the English Poets",
    4679: "Johnson, Lives of the Poets vol 1",
}
TOP_N = 30000
MAX_WORDS_PER_BOOK = 120_000


def fetch(gid: int, cache: Path) -> str:
    target = cache / f"{gid}.txt"
    if not target.exists():
        url = f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt"
        resp = httpx.get(url, follow_redirects=True, timeout=120.0)
        resp.raise_for_status()
        target.write_text(resp.text, encoding="utf-8")
    return target.read_text(encoding="utf-8")


def truncate_words(text: str, limit: int) -> str:
    paras = text.split("\n\n")
    out, n = [], 0
    for p in paras:
        out.append(p)
        n += len(p.split())
        if n >= limit:
            break
    return "\n\n".join(out)


def main(cache_dir: str) -> None:
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    counts = WordCounts()
    per_author = {}
    for gid, label in BOOKS.items():
        text = truncate_words(normalize(fetch(gid, cache)), MAX_WORDS_PER_BOOK)
        print(f"{label:45s} {len(text.split()):7d} words", flush=True)
        counts.add(text)
        per_author[label] = sample_text(text)
    profile_stats = aggregate(per_author, max_share=1.0, chunk_words=500)
    ref = ReferenceProfile(
        name="english_generic",
        created=dt.date.today().isoformat(),
        source=f"{len(BOOKS)} Project Gutenberg books, up to {MAX_WORDS_PER_BOOK} words each",
        total_words=counts.total,
        metrics=profile_stats.metrics,
        tells=profile_stats.tells,
        word_freq=dict(counts.unigrams.most_common(TOP_N)),
        bigram_freq=dict(counts.bigrams.most_common(TOP_N)),
    )
    ref.save(baseline_path("english_generic"))
    print(f"wrote {baseline_path('english_generic')} ({counts.total} words)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".cache/gutenberg")
