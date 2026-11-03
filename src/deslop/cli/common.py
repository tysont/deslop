# ABOUTME: Helpers every deslop command shares: the consoles, error exit, per-command settings,
# ABOUTME: the spend guard, provider choice, and reading input and voices.
from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, NoReturn

import typer
from rich.console import Console

from deslop.cli.config import Config, ConfigError, resolve_voice

console = Console()

err = Console(stderr=True)


def arm_budget(
    s: Settings,
    budget: float | None,
    *,
    command: str,
    voice: str = "",
    input_file: str = "",
    max_calls: int | None = None,
) -> float:
    """Start the spend guard for this invocation: the context every live call is logged with,
    the USD limit (--budget, config, else $5) and the live-call limit (--max-calls, 200)."""
    from deslop import spend

    limit = float(s.get("budget", budget, spend.DEFAULT_BUDGET))
    spend.set_context(command=command, voice=voice, input_file=input_file)
    spend.set_budget(limit)
    spend.set_max_calls(int(s.get("max_calls", max_calls, spend.DEFAULT_MAX_CALLS)))
    return limit


def provider_name(s: Settings, explicit: str | None) -> str:
    """Which Anthropic provider serves this invocation: --provider, else the config file's
    `provider:` (anthropic | claude-cli), else DESLOP_PROVIDER, else the API."""
    from deslop.providers.base import ANTHROPIC_PROVIDERS, default_provider_name

    name = str(s.get("provider", explicit, default_provider_name())).strip().lower()
    if name not in (*ANTHROPIC_PROVIDERS, "openai", "compatible"):
        fail(f"unknown provider {name!r}; use anthropic or claude-cli")
    return name


def report_budget() -> None:
    from deslop import spend

    if spend.budget_hit():
        err.print(f"[yellow]{spend.over_budget()}[/]")


def print_json(payload: str) -> None:
    """Emit JSON untouched by rich wrapping or markup."""
    sys.stdout.write(payload if payload.endswith("\n") else payload + "\n")


def fail(message: str, code: int = 1) -> NoReturn:
    err.print(f"[red]error:[/] {message}")
    raise typer.Exit(code)


class Settings:
    """Per-invocation option resolver bound to one command name."""

    def __init__(self, ctx: typer.Context, command: str):
        self.cfg: Config = ctx.obj if isinstance(ctx.obj, Config) else Config()
        self.command = command

    def get(self, key: str, value: Any, default: Any = None) -> Any:
        return self.cfg.get(self.command, key, value, default=default)

    def path(self, key: str, value: Path | None, default: Path | None = None) -> Path | None:
        got = self.get(key, value, default)
        return None if got is None else Path(got)

    def paths(self, key: str, value: list[Path] | None) -> tuple[str, ...]:
        got = self.get(key, value or None, [])
        return tuple(str(p) for p in as_list(got))

    def voice(self, value: Path | None) -> Path:
        try:
            return resolve_voice(self.cfg, self.command, value)
        except ConfigError as exc:
            fail(str(exc))

    def progress(self, json_out: bool) -> Callable[[str], None] | None:
        return None if json_out else (lambda msg: err.print(f"[dim]{msg}[/]"))


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]
    return list(value)


def load_voice(path: Path) -> Any:
    from deslop.voice import Voice
    from deslop.voice.pack import PackError

    try:
        return Voice.load(path)
    except FileNotFoundError as exc:
        fail(str(exc))
    except PackError as exc:
        fail(str(exc))


def read_input(path: Path | None) -> str:
    if path is None or str(path) == "-":
        return sys.stdin.read()
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        fail(f"cannot read {path}: {exc.strerror or exc}")


def split_frontmatter(text: str) -> tuple[str, str]:
    """YAML frontmatter and the body: the frontmatter is stripped on ingest and put back on
    output, never rewritten or treated as prose."""
    from deslop.voice.collect import FRONTMATTER_RE

    m = FRONTMATTER_RE.match(text)
    if not m:
        return "", text
    return text[: m.end()], text[m.end() :]
