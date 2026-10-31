# ABOUTME: Provider protocol, pricing table, response cache, and provider registry.
# ABOUTME: complete(system, user, model, temperature) -> Completion is the only entry point.
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from dotenv import load_dotenv

load_dotenv()

CACHE_DIR = Path(os.environ.get("DESLOP_HOME", Path.home() / ".deslop")) / "cache"

# USD per million tokens (input, output). Unknown models fall back to zero cost.
PRICING: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-opus-4-1": (15.0, 75.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.6),
    "gpt-4.1": (2.0, 8.0),
    "gpt-4.1-mini": (0.4, 1.6),
    "gpt-5": (1.25, 10.0),
    "gpt-5-mini": (0.25, 2.0),
}

DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-5",
    "openai": "gpt-4.1",
    "compatible": os.environ.get("DESLOP_COMPATIBLE_MODEL", "default"),
}


class ProviderError(Exception):
    pass


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 0
    cached: int = 0

    def add(self, other: Usage) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cost_usd += other.cost_usd
        self.calls += other.calls
        self.cached += other.cached


@dataclass
class Completion:
    text: str
    model: str
    provider: str
    usage: Usage = field(default_factory=Usage)
    from_cache: bool = False


def price(model: str, input_tokens: int, output_tokens: int) -> float:
    for prefix, (pin, pout) in PRICING.items():
        if model.startswith(prefix):
            return (input_tokens * pin + output_tokens * pout) / 1e6
    return 0.0


class Provider(Protocol):
    name: str
    default_model: str

    def complete(
        self, system: str, user: str, *, model: str | None = None, temperature: float = 0.7
    ) -> Completion: ...


class CachingProvider:
    """Wraps a raw provider call with a disk cache keyed on the full prompt."""

    name: str = "base"
    default_model: str = ""

    def __init__(
        self, *, cache_dir: Path | None = None, use_cache: bool = True, max_retries: int = 4
    ):
        self.cache_dir = cache_dir or CACHE_DIR
        self.use_cache = use_cache
        self.max_retries = max_retries
        self.usage = Usage()

    def _raw_complete(self, system: str, user: str, model: str, temperature: float) -> Completion:
        raise NotImplementedError

    def _cache_key(self, system: str, user: str, model: str, temperature: float) -> Path:
        payload = json.dumps(
            {"p": self.name, "m": model, "t": temperature, "s": system, "u": user}, sort_keys=True
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return self.cache_dir / digest[:2] / f"{digest}.json"

    def complete(
        self, system: str, user: str, *, model: str | None = None, temperature: float = 0.7
    ) -> Completion:
        model = model or self.default_model
        key = self._cache_key(system, user, model, temperature)
        if self.use_cache and key.exists():
            data = json.loads(key.read_text(encoding="utf-8"))
            usage = Usage(cached=1)
            self.usage.add(usage)
            return Completion(
                text=data["text"], model=model, provider=self.name, usage=usage, from_cache=True
            )
        delay = 1.0
        last: Exception | None = None
        for _ in range(self.max_retries):
            try:
                result = self._raw_complete(system, user, model, temperature)
                break
            except Exception as exc:  # noqa: BLE001 - retry any transport/API error
                last = exc
                time.sleep(delay)
                delay *= 2
        else:
            raise ProviderError(f"{self.name} failed after {self.max_retries} attempts: {last}")
        self.usage.add(result.usage)
        if self.use_cache:
            key.parent.mkdir(parents=True, exist_ok=True)
            key.write_text(json.dumps({"text": result.text, "model": model}), encoding="utf-8")
        return result


def list_providers() -> dict[str, bool]:
    """Provider name -> whether credentials are configured."""
    return {
        "anthropic": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "openai": bool(os.environ.get("OPENAI_API_KEY")),
        "compatible": bool(os.environ.get("DESLOP_COMPATIBLE_BASE_URL")),
    }


def get_provider(name: str, **kwargs) -> CachingProvider:
    from deslop.providers.anthropic import AnthropicProvider
    from deslop.providers.openai import CompatibleProvider, OpenAIProvider

    aliases = {"claude": "anthropic", "gpt": "openai"}
    name = aliases.get(name, name)
    if name == "anthropic":
        return AnthropicProvider(**kwargs)
    if name == "openai":
        return OpenAIProvider(**kwargs)
    if name == "compatible":
        return CompatibleProvider(**kwargs)
    raise ProviderError(f"unknown provider: {name}")


def first_available() -> CachingProvider | None:
    for name, ok in list_providers().items():
        if ok:
            return get_provider(name)
    return None
