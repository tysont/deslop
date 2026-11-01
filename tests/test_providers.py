# ABOUTME: Tests for the provider cache layer: hits replay without credentials, misses in
# ABOUTME: cache-only mode raise CacheMissError, and the cache directory is configurable.
import pytest

from deslop.providers import base
from deslop.providers.anthropic import AnthropicProvider
from deslop.providers.openai import CompatibleProvider, OpenAIProvider


class Echo(base.CachingProvider):
    name = "echo"
    default_model = "echo-1"

    def _raw_complete(self, system, user, model, temperature):
        return base.Completion(
            text=f"{user}!", model=model, provider=self.name, usage=base.Usage(calls=1)
        )


def test_cache_hit_replays_and_cache_only_miss_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("DESLOP_CACHE_ONLY", raising=False)
    live = Echo(cache_dir=tmp_path)
    assert live.complete("s", "hi").text == "hi!"
    replay = Echo(cache_dir=tmp_path, cache_only=True)
    got = replay.complete("s", "hi")
    assert got.from_cache and got.text == "hi!" and replay.usage.cached == 1
    with pytest.raises(base.CacheMissError) as exc:
        replay.complete("s", "something new")
    assert "echo" in str(exc.value) and str(tmp_path) in str(exc.value)
    assert not isinstance(exc.value, base.ProviderError)


def test_cache_only_providers_need_no_credentials(tmp_path, monkeypatch):
    for var in (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "DESLOP_COMPATIBLE_BASE_URL",
        "DESLOP_CACHE_ONLY",
    ):
        monkeypatch.delenv(var, raising=False)
    for cls in (AnthropicProvider, OpenAIProvider, CompatibleProvider):
        with pytest.raises(base.ProviderError):
            cls(cache_dir=tmp_path)
        provider = cls(cache_dir=tmp_path, cache_only=True)
        with pytest.raises(base.CacheMissError):
            provider.complete("s", "u")
    monkeypatch.setenv("DESLOP_CACHE_ONLY", "1")
    assert base.get_provider("claude", cache_dir=tmp_path).cache_only


def test_cache_dir_resolution(tmp_path, monkeypatch):
    monkeypatch.setenv("DESLOP_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("DESLOP_CACHE_DIR", raising=False)
    assert base.default_cache_dir() == tmp_path / "home" / "cache"
    monkeypatch.setenv("DESLOP_CACHE_DIR", str(tmp_path / "elsewhere"))
    assert base.default_cache_dir() == tmp_path / "elsewhere"
    assert Echo().cache_dir == tmp_path / "elsewhere"
