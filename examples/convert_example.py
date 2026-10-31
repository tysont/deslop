# ABOUTME: Worked example: build a voice from the fixture corpus, convert examples/slop_sample.md
# ABOUTME: in edit mode, and print the before/after report. Replays from tests/cache without keys.
from __future__ import annotations

import os
import sys
from pathlib import Path

from rich.console import Console

from deslop import Corpus, convert
from deslop.providers.base import CachingProvider, get_provider
from deslop.voice.calibrate import calibrate

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "examples" / "build"
CACHE = Path(os.environ.get("DESLOP_CACHE_DIR", ROOT / "tests" / "cache"))
HAS_KEY = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY"))
REPLAY_ONLY = not HAS_KEY or os.environ.get("DESLOP_CACHE_ONLY", "") == "1"


def provider(name: str) -> CachingProvider:
    """Real providers, answering from the repository cache. Without credentials a prompt that
    is not cached raises CacheMissError instead of silently skipping the call."""
    return get_provider(name, cache_dir=CACHE, cache_only=REPLAY_ONLY)


def main() -> int:
    console = Console()
    BUILD.mkdir(parents=True, exist_ok=True)

    # 1. A voice from the four shipped essays on Shakespeare (Johnson, Hazlitt, Bradley, Arnold).
    # The name matches the test suite's voice so the prompts, and so the cache keys, are the same.
    corpus = Corpus.load(ROOT / "tests" / "fixtures" / "corpus_small")
    voice = calibrate(corpus, name="fixture", use_llm=False)
    voice.save(BUILD / "fixture.voice")
    console.print(f"voice [bold]{voice.name}[/]: {voice.meta.total_words} words of corpus")

    # 2. Convert the machine-written sample, rewriting only the flagged sentences.
    text = (ROOT / "examples" / "slop_sample.md").read_text(encoding="utf-8")
    result = convert(
        text,
        voice,
        mode="edit",
        max_iter=4,
        provider_factory=provider,
        pairs_path=BUILD / "pairs.jsonl",
        progress=lambda msg: console.print(f"[dim]{msg}[/]"),
    )
    (BUILD / "slop_sample.converted.md").write_text(result.text, encoding="utf-8")

    # 3. Before and after.
    console.rule("before")
    for item in result.report_before.renderables(max_hits=12):
        console.print(item)
    console.rule("after")
    for item in result.report_after.renderables(max_hits=12):
        console.print(item)
    console.rule("diff")
    console.print(result.diff, markup=False, highlight=False)
    accepted = sum(1 for u in result.units if u.accepted)
    console.print(
        f"\n{result.report_before.verdict} -> {result.report_after.verdict} in "
        f"{len(result.iterations)} iteration(s); {accepted}/{len(result.units)} rewrites "
        f"accepted; ${result.cost.usd:.2f} ({result.cost.cached} cached, "
        f"{result.cost.calls} live calls); output in {BUILD / 'slop_sample.converted.md'}"
    )
    return 0 if result.report_after.verdict == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
