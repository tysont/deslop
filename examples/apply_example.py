# ABOUTME: Worked example: build a voice from the fixture essays, apply it to the slop sample
# ABOUTME: in edit mode, and print the before/after report. Replays from tests/cache without keys.
from __future__ import annotations

import os
import sys
from pathlib import Path

from rich.console import Console

from deslop import Voice, apply
from deslop.providers.base import CachingProvider, get_provider

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "examples" / "build"
CACHE = Path(os.environ.get("DESLOP_CACHE_DIR", ROOT / "tests" / "cache"))
HAS_KEY = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY"))
REPLAY_ONLY = not HAS_KEY or os.environ.get("DESLOP_CACHE_ONLY", "") == "1"
AUTHORS = {
    "johnson_*": "Samuel Johnson",
    "hazlitt_*": "William Hazlitt",
    "arnold_*": "Matthew Arnold",
    "bradley_*": "A. C. Bradley",
}


def provider(name: str) -> CachingProvider:
    """Real providers, answering from the repository cache. Without credentials a prompt that
    is not cached raises CacheMissError instead of silently skipping the call."""
    return get_provider(name, cache_dir=CACHE, cache_only=REPLAY_ONLY)


def main() -> int:
    console = Console()
    BUILD.mkdir(parents=True, exist_ok=True)

    # 1. A voice from the four shipped essays on Shakespeare (Johnson, Hazlitt, Bradley, Arnold).
    # The name matches the test suite's voice so the prompts, and so the cache keys, are the same.
    voice = Voice.from_texts(
        [ROOT / "tests" / "fixtures" / "corpus_small"],
        name="fixture",
        author_map=AUTHORS,
        use_llm=False,
    )
    voice.save(BUILD / "fixture.voice")
    console.print(f"voice [bold]{voice.name}[/]: {voice.meta.total_words} words of corpus")

    # 2. Apply the voice to the machine-written sample, rewriting only the flagged sentences.
    text = (ROOT / "examples" / "slop_sample.md").read_text(encoding="utf-8")
    result = apply(
        text,
        voice,
        mode="edit",
        provider_factory=provider,
        pairs_path=BUILD / "pairs.jsonl",
        progress=lambda msg: console.print(f"[dim]{msg}[/]"),
    )
    (BUILD / "slop_sample.applied.md").write_text(result.text, encoding="utf-8")

    # 3. Before and after.
    console.rule("before")
    for item in result.report_before.renderables(show_hits=False):
        console.print(item)
    console.rule("after")
    for item in result.report_after.renderables(show_hits=False):
        console.print(item)
    console.rule("diff")
    console.print(result.diff, markup=False, highlight=False)
    before, after = result.report_before, result.report_after
    rewritten = sum(1 for u in result.units if not u.kept_original)
    console.print(
        f"{before.verdict} -> {after.verdict} in {len(result.iterations)} iteration(s); "
        f"{rewritten}/{len(result.units)} units rewritten; ${result.cost.usd:.2f} "
        f"({result.cost.cached} cached, {result.cost.calls} live calls); "
        f"output in {BUILD / 'slop_sample.applied.md'}"
    )
    return 0 if after.verdict == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
