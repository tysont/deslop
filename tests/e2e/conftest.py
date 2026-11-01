# ABOUTME: End-to-end fixtures: real providers whose responses are cached under tests/cache.
# ABOUTME: Misses go live and are recorded only with DESLOP_RECORD=1 and a key; else they skip.
from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from deslop.providers.base import CacheMissError, CachingProvider, cache_only_default, get_provider

load_dotenv(override=True)
CACHE = Path(__file__).parent.parent / "cache"
HAS_KEY = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY"))
RECORD = os.environ.get("DESLOP_RECORD", "").strip().lower() in ("1", "true", "yes")
REPLAY_ONLY = cache_only_default() or not HAS_KEY or not RECORD


@pytest.fixture(autouse=True)
def live_calls_are_logged(fresh_spend_guard, monkeypatch):
    """A recording run spends real money, so its calls go to the real spend log, not the
    per-test one the unit-test fixture sets up."""
    if not REPLAY_ONLY:
        from deslop import spend

        monkeypatch.setattr(spend, "spend_path", lambda: Path.home() / ".deslop" / "spend.jsonl")
        spend.set_context(command="pytest tests/e2e (recording)")


@pytest.fixture(scope="session")
def provider_factory():
    def factory(name: str) -> CachingProvider:
        return get_provider(name, cache_dir=CACHE, cache_only=REPLAY_ONLY)

    return factory


def pytest_runtest_call(item: pytest.Item) -> None:
    """In replay mode a prompt missing from tests/cache means the cache was not recorded for
    this version of the prompts; skip loudly rather than fail. With DESLOP_RECORD=1 and a
    key the miss goes live and is recorded, and any failure is real."""
    try:
        item.runtest()
    except CacheMissError as exc:
        if REPLAY_ONLY:
            pytest.skip(f"tests/cache not recorded for this prompt; record live: {exc}")
        raise
