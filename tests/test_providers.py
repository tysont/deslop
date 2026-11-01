# ABOUTME: Tests for the provider cache layer: hits replay without credentials, misses in
# ABOUTME: cache-only mode raise CacheMissError, and the cache directory is configurable.
import os

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


FAKE_CLI = """#!/usr/bin/env python3
import json, os, sys
args = sys.argv[1:]
if args[:2] == ["auth", "status"]:
    print(json.dumps({"loggedIn": os.environ.get("FAKE_CLAUDE_LOGGED_IN", "1") == "1"}))
    sys.exit(0)
if args == ["--version"]:
    print("9.9.9 (fake)")
    sys.exit(0)
if os.environ.get("FAKE_CLAUDE_FAIL"):
    sys.stderr.write("fake failure: not logged in\\n")
    sys.exit(1)
if os.environ.get("FAKE_CLAUDE_HANG"):
    import time
    time.sleep(5)
prompt = sys.stdin.read()
model = args[args.index("--model") + 1]
system = args[args.index("--system-prompt") + 1]
assert "ANTHROPIC_API_KEY" not in os.environ, "the CLI must use the local login"
print(json.dumps({
    "type": "result", "subtype": "success", "is_error": False,
    "result": f"{prompt.strip()}! ({model}; system {len(system)} chars)",
    "usage": {"input_tokens": 12, "output_tokens": 7}, "total_cost_usd": 0.0,
}))
"""


@pytest.fixture
def fake_claude(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "claude"
    script.write_text(FAKE_CLI)
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ.get('PATH', '')}")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")  # must be stripped for the CLI
    monkeypatch.delenv("DESLOP_CACHE_ONLY", raising=False)
    return script


def test_model_aliases_resolve_the_same_for_every_provider():
    assert base.resolve_model("opus") == "claude-opus-5-5"
    assert base.resolve_model("sonnet") == "claude-sonnet-5-5"
    assert base.resolve_model("haiku") == "claude-haiku-5-5"
    assert base.resolve_model("claude-sonnet-5") == "claude-sonnet-5"
    assert base.DEFAULT_WRITER == "claude-opus-5-5" and base.DEFAULT_JUDGE == "claude-sonnet-5-5"


def test_claude_cli_provider_runs_the_binary_and_shares_the_api_cache(fake_claude, tmp_path):
    from deslop.providers.claude_cli import ClaudeCliProvider

    p = ClaudeCliProvider(cache_dir=tmp_path / "cache")
    got = p.complete("be brief", "hello", model="sonnet")
    assert got.text.startswith("hello! (claude-sonnet-5-5;") and got.provider == "claude-cli"
    assert got.usage.input_tokens == 12 and got.usage.output_tokens == 7 and got.usage.calls == 1
    assert got.usage.cost_usd == 0.0 and p.billing == "subscription"
    # The cache key is the same one the API provider would use, so replays cross providers.
    api = AnthropicProvider(cache_dir=tmp_path / "cache", cache_only=True)
    assert api.complete("be brief", "hello", model="claude-sonnet-5-5").from_cache
    assert p._cache_key("s", "u", "m", 0.5) == api._cache_key("s", "u", "m", 0.5)


def test_claude_cli_failures_carry_stderr_and_retry(fake_claude, tmp_path, monkeypatch):
    from deslop.providers.claude_cli import ClaudeCliProvider

    monkeypatch.setenv("FAKE_CLAUDE_FAIL", "1")
    p = ClaudeCliProvider(cache_dir=tmp_path / "cache", max_retries=2)
    monkeypatch.setattr(base.time, "sleep", lambda s: None)
    with pytest.raises(base.ProviderError) as exc:
        p.complete("s", "u")
    assert "fake failure: not logged in" in str(exc.value) and "2 attempts" in str(exc.value)


def test_claude_cli_timeout_is_a_provider_error(fake_claude, tmp_path, monkeypatch):
    from deslop.providers import claude_cli

    monkeypatch.setenv("FAKE_CLAUDE_HANG", "1")
    monkeypatch.setattr(claude_cli, "TIMEOUT_SECONDS", 0.5)
    monkeypatch.setattr(base.time, "sleep", lambda s: None)
    p = claude_cli.ClaudeCliProvider(cache_dir=tmp_path / "cache", max_retries=1)
    with pytest.raises(base.ProviderError) as exc:
        p.complete("s", "u")
    assert "timed out" in str(exc.value)


def test_claude_cli_needs_the_binary_unless_replaying(tmp_path, monkeypatch):
    from deslop.providers.claude_cli import ClaudeCliProvider, cli_status

    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.delenv("DESLOP_CACHE_ONLY", raising=False)
    with pytest.raises(base.ProviderError):
        ClaudeCliProvider(cache_dir=tmp_path)
    ClaudeCliProvider(cache_dir=tmp_path, cache_only=True)
    assert cli_status().installed is False


def test_cli_status_reports_login(fake_claude, monkeypatch):
    from deslop.providers.claude_cli import cli_status

    st = cli_status()
    assert st.installed and st.logged_in and st.version == "9.9.9 (fake)"
    monkeypatch.setenv("FAKE_CLAUDE_LOGGED_IN", "0")
    assert cli_status().logged_in is False


def test_default_provider_never_falls_back_to_openai(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("DESLOP_PROVIDER", raising=False)
    monkeypatch.delenv("DESLOP_CACHE_ONLY", raising=False)
    assert base.default_provider_name() == "anthropic"
    with pytest.raises(base.ProviderError):
        base.default_provider(cache_dir=tmp_path)
    monkeypatch.setenv("DESLOP_PROVIDER", "claude-cli")
    assert base.default_provider_name() == "claude-cli"
    assert (
        base.get_provider(
            "opus-writer" if False else "claude", cache_dir=tmp_path, cache_only=True
        ).name
        == "anthropic"
    )
