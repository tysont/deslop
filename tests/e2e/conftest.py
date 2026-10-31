# ABOUTME: End-to-end fixtures: real providers whose responses are cached under tests/cache.
# ABOUTME: With credentials, misses go live and are recorded; without, a miss fails the test.
from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from deslop.providers.base import CachingProvider, cache_only_default, get_provider

load_dotenv(override=True)
CACHE = Path(__file__).parent.parent / "cache"
HAS_KEY = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY"))
REPLAY_ONLY = cache_only_default() or not HAS_KEY


@pytest.fixture(scope="session")
def provider_factory():
    def factory(name: str) -> CachingProvider:
        return get_provider(name, cache_dir=CACHE, cache_only=REPLAY_ONLY)

    return factory
