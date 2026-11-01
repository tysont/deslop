# ABOUTME: Provider protocol, response cache, retry policy, and provider registry.
# ABOUTME: complete(system, user, model, temperature) -> Completion is the only entry point.
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from dotenv import load_dotenv

load_dotenv(override=True)  # a project .env beats stale shell exports


def default_cache_dir() -> Path:
    """Where responses are cached: $DESLOP_CACHE_DIR, else $DESLOP_HOME/cache."""
    explicit = os.environ.get("DESLOP_CACHE_DIR")
    if explicit:
        return Path(explicit)
    return Path(os.environ.get("DESLOP_HOME", Path.home() / ".deslop")) / "cache"


DEFAULT_MODELS = {
    "anthropic": "claude-opus-5-5",
    "claude-cli": "claude-opus-5-5",
    "openai": "gpt-5.5",
    "compatible": os.environ.get("DESLOP_COMPATIBLE_MODEL", "default"),
}
DEFAULT_WRITER = "claude-opus-5-5"
DEFAULT_JUDGE = "claude-sonnet-5-5"
MODEL_ALIASES = {
    "opus": "claude-opus-5-5",
    "sonnet": "claude-sonnet-5-5",
    "haiku": "claude-haiku-5-5",
}
ANTHROPIC_PROVIDERS = ("anthropic", "claude-cli")  # the default set; OpenAI only when named


def resolve_model(name: str) -> str:
    """`opus`, `sonnet` and `haiku` resolve to the current model of that family, the same
    for every provider; any other name is taken as a model id."""
    return MODEL_ALIASES.get(name.strip().lower(), name.strip())


def default_provider_name() -> str:
    """Which Anthropic provider serves calls when none is named: DESLOP_PROVIDER, else the
    API (`anthropic`). The config file's `provider:` key is applied by the CLI."""
    name = os.environ.get("DESLOP_PROVIDER", "").strip().lower()
    return name if name in ANTHROPIC_PROVIDERS else "anthropic"


class ProviderError(Exception):
    pass


class NonRetryableError(ProviderError):
    """Auth, bad-request, and programming errors: fail fast instead of backing off."""


class BudgetExceeded(ProviderError):
    """The invocation's spend budget is used up; no further live call is made. A
    ProviderError so every loop that tolerates a failed call stops cleanly."""


class CacheMissError(Exception):
    """A cache-only provider was asked for a prompt that is not in its cache. Deliberately
    not a ProviderError so retry loops do not swallow it."""


def cache_only_default() -> bool:
    return os.environ.get("DESLOP_CACHE_ONLY", "").strip().lower() in ("1", "true", "yes")


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 0
    cached: int = 0
    unpriced: int = 0  # calls to models missing from pricing.yaml; their cost is not counted

    def add(self, other: Usage) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cost_usd += other.cost_usd
        self.calls += other.calls
        self.cached += other.cached
        self.unpriced += other.unpriced


@dataclass
class Completion:
    text: str
    model: str
    provider: str
    usage: Usage = field(default_factory=Usage)
    from_cache: bool = False


class Provider(Protocol):
    name: str
    default_model: str

    def complete(
        self, system: str, user: str, *, model: str | None = None, temperature: float = 0.7
    ) -> Completion: ...


class CachingProvider:
    """Wraps a raw provider call with a disk cache keyed on the full prompt. `cache_namespace`
    is the vendor the key is filed under, so two providers serving the same models (the
    Anthropic API and the Claude Code CLI) replay each other's responses; `billing` says
    whether a live call costs dollars (`api`) or a subscription call (`subscription`)."""

    name: str = "base"
    default_model: str = ""
    cache_namespace: str = ""
    billing: str = "api"

    def __init__(
        self,
        *,
        cache_dir: Path | None = None,
        use_cache: bool = True,
        max_retries: int = 4,
        cache_only: bool | None = None,
    ):
        """cache_only providers answer from the cache alone: they need no credentials and
        raise CacheMissError for any prompt not already cached (default: $DESLOP_CACHE_ONLY)."""
        self.cache_dir = cache_dir or default_cache_dir()
        self.use_cache = use_cache
        self.max_retries = max_retries
        self.cache_only = cache_only_default() if cache_only is None else cache_only
        self.usage = Usage()

    def _raw_complete(self, system: str, user: str, model: str, temperature: float) -> Completion:
        raise NotImplementedError

    @property
    def key(self) -> str:
        """Vendor and model, the identity a same-vendor judge is told apart by."""
        return f"{self.name}/{self.default_model}"

    def _cache_key(self, system: str, user: str, model: str, temperature: float) -> Path:
        payload = json.dumps(
            {
                "p": self.cache_namespace or self.name,
                "m": model,
                "t": temperature,
                "s": system,
                "u": user,
            },
            sort_keys=True,
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return self.cache_dir / digest[:2] / f"{digest}.json"

    def complete(
        self, system: str, user: str, *, model: str | None = None, temperature: float = 0.7
    ) -> Completion:
        model = resolve_model(model or self.default_model)
        key = self._cache_key(system, user, model, temperature)
        if self.use_cache and key.exists():
            data = json.loads(key.read_text(encoding="utf-8"))
            usage = Usage(cached=1)
            self.usage.add(usage)
            return Completion(
                text=data["text"], model=model, provider=self.name, usage=usage, from_cache=True
            )
        if self.cache_only:
            raise CacheMissError(
                f"{self.name}/{model}: prompt not in cache {self.cache_dir} and "
                "cache-only mode is on (unset DESLOP_CACHE_ONLY and provide credentials)"
            )
        from deslop import spend

        over = spend.over_budget()
        if over is not None:
            raise BudgetExceeded(over)
        delay = 1.0
        last: Exception | None = None
        for _ in range(self.max_retries):
            try:
                result = self._raw_complete(system, user, model, temperature)
                break
            except NonRetryableError:
                raise
            except Exception as exc:  # noqa: BLE001 - retry any transport/API error
                last = exc
                time.sleep(delay)
                delay *= 2
        else:
            raise ProviderError(f"{self.name} failed after {self.max_retries} attempts: {last}")
        self.usage.add(result.usage)
        spend.record(provider=self.name, model=model, usage=result.usage, billing=self.billing)
        if self.use_cache:
            key.parent.mkdir(parents=True, exist_ok=True)
            key.write_text(json.dumps({"text": result.text, "model": model}), encoding="utf-8")
        return result


def list_providers() -> dict[str, bool]:
    """Provider name -> whether it is configured (a key, or the CLI on PATH). OpenAI and the
    compatible endpoint are listed but never selected unless named."""
    import shutil

    return {
        "anthropic": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "claude-cli": shutil.which("claude") is not None,
        "openai": bool(os.environ.get("OPENAI_API_KEY")),
        "compatible": bool(os.environ.get("DESLOP_COMPATIBLE_BASE_URL")),
    }


def get_provider(name: str, **kwargs: Any) -> CachingProvider:
    from deslop.providers.anthropic import AnthropicProvider
    from deslop.providers.claude_cli import ClaudeCliProvider
    from deslop.providers.openai import CompatibleProvider, OpenAIProvider

    aliases = {"claude": "anthropic", "gpt": "openai", "cli": "claude-cli"}
    name = aliases.get(name, name)
    if name == "anthropic":
        return AnthropicProvider(**kwargs)
    if name == "claude-cli":
        return ClaudeCliProvider(**kwargs)
    if name == "openai":
        return OpenAIProvider(**kwargs)
    if name == "compatible":
        return CompatibleProvider(**kwargs)
    raise ProviderError(f"unknown provider: {name}")


def default_provider(**kwargs: Any) -> CachingProvider:
    """The configured Anthropic provider (see `default_provider_name`). Never OpenAI."""
    return get_provider(default_provider_name(), **kwargs)
