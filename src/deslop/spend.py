# ABOUTME: The spend guard: appends every live provider call to $DESLOP_HOME/spend.jsonl with
# ABOUTME: its command context, enforces a per-invocation budget, and sums today and the week.
from __future__ import annotations

import datetime as dt
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deslop.providers.base import Usage

DEFAULT_BUDGET = 5.0  # USD per invocation of apply or any lab command
DEFAULT_MAX_CALLS = 200  # live calls per invocation, so subscription runs have a cap too


def spend_path() -> Path:
    return Path(os.environ.get("DESLOP_HOME", Path.home() / ".deslop")) / "spend.jsonl"


@dataclass
class _State:
    command: str = ""
    voice: str = ""
    input_file: str = ""
    budget: float | None = None
    max_calls: int | None = None
    spent: float = 0.0
    calls: int = 0
    hit: bool = False
    extra: dict[str, str] = field(default_factory=dict)


_state = _State()


def reset() -> None:
    """Forget the context, the budget and the running total (a new invocation)."""
    global _state
    _state = _State()


def set_context(command: str = "", voice: str = "", input_file: str = "") -> None:
    """Name the command, voice and input file that later live calls are charged to."""
    _state.command = command
    _state.voice = voice
    _state.input_file = input_file


def set_budget(limit: float | None) -> None:
    """USD this invocation may spend on live calls; None removes the limit."""
    _state.budget = limit
    _state.spent = 0.0
    _state.hit = False


def set_max_calls(limit: int | None) -> None:
    """Live calls this invocation may make (API or subscription); None removes the limit."""
    _state.max_calls = limit
    _state.calls = 0


def budget() -> float | None:
    return _state.budget


def calls() -> int:
    """Live calls made since the limits were set."""
    return _state.calls


def spent() -> float:
    """USD spent on live calls since the budget was set (or the context reset)."""
    return _state.spent


def budget_hit() -> bool:
    return _state.hit


def over_budget() -> str | None:
    """The message to abort with when the budget is spent, else None."""
    if _state.budget is not None and _state.spent >= _state.budget:
        _state.hit = True
        return (
            f"budget of ${_state.budget:.2f} spent (${_state.spent:.2f} on live calls this "
            "invocation); stopping with the best result so far. Raise --budget to continue."
        )
    if _state.max_calls is not None and _state.calls >= _state.max_calls:
        _state.hit = True
        return (
            f"--max-calls {_state.max_calls} reached ({_state.calls} live calls this "
            "invocation); stopping with the best result so far. Raise --max-calls to continue."
        )
    return None


def record(
    *,
    provider: str,
    model: str,
    usage: Usage,
    when: dt.datetime | None = None,
    billing: str = "api",
) -> dict[str, object]:
    """Append one live call to spend.jsonl and add it to the running totals. A
    `subscription` call (the Claude Code CLI) is logged with cost null."""
    when = when or dt.datetime.now(dt.UTC)
    subscription = billing == "subscription"
    row: dict[str, object] = {
        "timestamp": when.isoformat(timespec="seconds"),
        "command": _state.command,
        "provider": provider,
        "model": model,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cost": None if subscription else round(usage.cost_usd, 6),
        "billing": billing,
        "voice": _state.voice,
        "input": _state.input_file,
    }
    _state.spent += 0.0 if subscription else usage.cost_usd
    _state.calls += max(usage.calls, 1)
    path = spend_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
    except OSError:
        pass  # the guard still counts; only the log line is lost
    return row


@dataclass
class Summary:
    """API dollars and subscription calls, today and over the last seven days."""

    today: float = 0.0
    week: float = 0.0
    calls_today: int = 0  # API calls
    calls_week: int = 0
    subscription_calls_today: int = 0
    subscription_calls_week: int = 0
    path: Path = field(default_factory=spend_path)
    exists: bool = False

    def describe(self) -> str:
        if not self.exists:
            return f"no live calls logged ({self.path})"
        return (
            f"today ${self.today:.2f} over {self.calls_today} API call(s) and "
            f"{self.subscription_calls_today} subscription call(s); last 7 days "
            f"${self.week:.2f} over {self.calls_week} API call(s) and "
            f"{self.subscription_calls_week} subscription call(s) ({self.path})"
        )


def summary(now: dt.datetime | None = None) -> Summary:
    """Spend today and over the last seven days, read from spend.jsonl."""
    now = now or dt.datetime.now(dt.UTC)
    path = spend_path()
    out = Summary(path=path, exists=path.is_file())
    if not out.exists:
        return out
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_start = now - dt.timedelta(days=7)
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
            when = dt.datetime.fromisoformat(str(row["timestamp"]))
            subscription = row.get("billing") == "subscription" or row.get("cost") is None
            cost = 0.0 if subscription else float(row.get("cost", 0.0))
        except (ValueError, KeyError, TypeError):
            continue
        if when.tzinfo is None:
            when = when.replace(tzinfo=dt.UTC)
        if when >= week_start:
            out.week += cost
            if subscription:
                out.subscription_calls_week += 1
            else:
                out.calls_week += 1
        if when >= day_start:
            out.today += cost
            if subscription:
                out.subscription_calls_today += 1
            else:
                out.calls_today += 1
    return out
