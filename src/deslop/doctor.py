# ABOUTME: `deslop doctor` checks: spaCy model, provider reachability, pricing table age,
# ABOUTME: cache directory writability, the default voice, and live spend from spend.jsonl.
from __future__ import annotations

import datetime as dt
from collections.abc import Callable

from pydantic import BaseModel

from deslop.cli.config import Config, ConfigError, resolve_voice
from deslop.pricing import home, load_pricing

PRICING_MAX_AGE_DAYS = 90


class Check(BaseModel):
    name: str
    status: str  # ok | warn | fail
    detail: str = ""


def check_spacy() -> Check:
    from deslop.nlp import SPACY_MODEL, load_nlp

    try:
        nlp = load_nlp()
    except OSError as exc:
        return Check(name="spacy", status="fail", detail=" ".join(str(exc).split()))
    return Check(name="spacy", status="ok", detail=f"{SPACY_MODEL} {nlp.meta.get('version', '')}")


def _ping(name: str) -> str:
    from deslop.providers.base import get_provider

    provider = get_provider(name, use_cache=False)
    return provider.complete("Reply with the single word OK.", "ping", temperature=0.0).model


def check_providers(
    ping: Callable[[str], str] = _ping, *, default: str | None = None
) -> list[Check]:
    """The Anthropic API (pinged when a key is set), the Claude Code CLI (installed and
    logged in, no model call), and the configured default (`default`, else DESLOP_PROVIDER,
    else the API); OpenAI is only noted."""
    from deslop.providers.base import default_provider_name, list_providers
    from deslop.providers.claude_cli import cli_status

    configured = list_providers()
    checks = []
    if configured.get("anthropic"):
        try:
            model = ping("anthropic")
        except Exception as exc:  # noqa: BLE001 - any failure is the finding
            checks.append(Check(name="providers", status="fail", detail=f"anthropic: {exc}"[:200]))
        else:
            checks.append(Check(name="providers", status="ok", detail=f"anthropic: {model}"))
    else:
        checks.append(
            Check(name="providers", status="warn", detail="anthropic: ANTHROPIC_API_KEY not set")
        )
    st = cli_status()
    if not st.installed:
        checks.append(Check(name="providers", status="warn", detail=f"claude-cli: {st.detail}"))
    elif st.logged_in:
        checks.append(
            Check(name="providers", status="ok", detail=f"claude-cli: {st.version}, logged in")
        )
    else:
        checks.append(
            Check(
                name="providers",
                status="warn",
                detail=f"claude-cli: {st.version}, not logged in "
                f"({st.detail or 'run claude auth login'})",
            )
        )
    default = default or default_provider_name()
    ok_default = (default == "anthropic" and configured.get("anthropic")) or (
        default == "claude-cli" and st.installed and st.logged_in
    )
    checks.append(
        Check(
            name="providers",
            status="ok" if ok_default else "warn",
            detail=f"default provider: {default}"
            + ("" if ok_default else " (not usable; apply, draft and calibrate need it)"),
        )
    )
    if configured.get("openai"):
        checks.append(
            Check(name="providers", status="ok", detail="openai: key set; unused unless named")
        )
    return checks


def check_pricing(today: dt.date | None = None) -> Check:
    table = load_pricing()
    today = today or dt.date.today()
    try:
        fetched = dt.date.fromisoformat(table.fetched)
    except ValueError:
        return Check(name="pricing", status="warn", detail="pricing.yaml has no fetched date")
    age = (today - fetched).days
    override = home() / "pricing.yaml"
    extra = f"; override {override} in use" if override.is_file() else ""
    if age > PRICING_MAX_AGE_DAYS:
        return Check(
            name="pricing",
            status="warn",
            detail=f"prices fetched {table.fetched}, {age} days ago; "
            f"check {', '.join(table.sources.values())}{extra}",
        )
    return Check(
        name="pricing", status="ok", detail=f"fetched {table.fetched}, {age} days old{extra}"
    )


def check_cache() -> Check:
    from deslop.providers.base import default_cache_dir

    target = default_cache_dir()
    try:
        target.mkdir(parents=True, exist_ok=True)
        probe = target / ".doctor"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        return Check(name="cache", status="fail", detail=f"{target} is not writable: {exc}")
    entries = sum(1 for p in target.rglob("*.json"))
    return Check(name="cache", status="ok", detail=f"{target} ({entries} cached responses)")


def check_voice(cfg: Config) -> Check:
    from deslop.voice.validate import validate

    try:
        voice_dir = resolve_voice(cfg, "doctor", None)
    except ConfigError as exc:
        return Check(name="voice", status="warn", detail=str(exc))
    if not voice_dir.exists():
        return Check(name="voice", status="fail", detail=f"{voice_dir} does not exist")
    result = validate(voice_dir)
    failed = [f"{c.name}: {c.detail}" for c in result.checks if not c.ok]
    if failed:
        return Check(name="voice", status="fail", detail=f"{voice_dir}: " + "; ".join(failed))
    return Check(name="voice", status="ok", detail=f"{voice_dir} validates")


def check_spend() -> Check:
    """Live spend today and over the last seven days, from spend.jsonl."""
    from deslop import spend

    return Check(name="spend", status="ok", detail=spend.summary().describe())


def run(cfg: Config | None = None, *, ping: Callable[[str], str] = _ping) -> list[Check]:
    cfg = cfg or Config()
    default = cfg.get("doctor", "provider", None)
    checks = [
        check_spacy(),
        *check_providers(ping, default=str(default) if default else None),
        check_pricing(),
        check_cache(),
    ]
    checks.append(check_voice(cfg))
    return checks


__all__ = [
    "Check",
    "check_cache",
    "check_pricing",
    "check_providers",
    "check_spacy",
    "check_spend",
    "check_voice",
    "run",
]
