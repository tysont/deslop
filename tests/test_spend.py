# ABOUTME: Tests for the spend guard: every live call is appended to spend.jsonl with its
# ABOUTME: context, a per-invocation budget stops further live calls, and doctor sums the log.
from __future__ import annotations

import datetime as dt
import json

import pytest

from deslop import spend
from deslop.providers.base import BudgetExceeded, CachingProvider, Completion, Usage


class Live(CachingProvider):
    """A provider whose every call is live and costs `cost`."""

    name = "anthropic"
    default_model = "claude-test"

    def __init__(self, cost: float, **kwargs):
        super().__init__(use_cache=False, cache_only=False, **kwargs)
        self.cost = cost

    def _raw_complete(self, system, user, model, temperature):
        usage = Usage(calls=1, input_tokens=100, output_tokens=20, cost_usd=self.cost)
        return Completion(text="ok", model=model, provider=self.name, usage=usage)


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("DESLOP_HOME", str(tmp_path / "home"))
    spend.reset()
    yield tmp_path / "home"
    spend.reset()


def test_every_live_call_is_appended_with_its_context(home):
    spend.set_context(command="apply", voice="essay", input_file="draft.md")
    Live(0.25).complete("sys", "user")
    Live(0.25).complete("sys", "user2")
    rows = [json.loads(line) for line in (home / "spend.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    row = rows[0]
    assert row["command"] == "apply" and row["voice"] == "essay" and row["input"] == "draft.md"
    assert row["provider"] == "anthropic" and row["model"] == "claude-test"
    assert row["input_tokens"] == 100 and row["output_tokens"] == 20 and row["cost"] == 0.25
    assert dt.datetime.fromisoformat(row["timestamp"]).tzinfo is not None


def test_cached_calls_are_not_spend(home, tmp_path):
    p = Live(0.5, cache_dir=tmp_path / "cache")
    p.use_cache = True
    p.complete("sys", "user")
    p.complete("sys", "user")  # replayed
    rows = (home / "spend.jsonl").read_text().splitlines()
    assert len(rows) == 1


def test_budget_stops_live_calls_with_a_clear_message(home):
    spend.set_budget(1.0)
    p = Live(0.6)
    p.complete("sys", "a")
    p.complete("sys", "b")  # 1.2 spent: over budget after this call
    with pytest.raises(BudgetExceeded) as exc:
        p.complete("sys", "c")
    assert "budget" in str(exc.value) and "$1.00" in str(exc.value)
    assert spend.spent() == pytest.approx(1.2) and spend.budget_hit()
    assert len((home / "spend.jsonl").read_text().splitlines()) == 2


def test_totals_for_today_and_the_last_seven_days(home):
    now = dt.datetime.now(dt.UTC)
    spend.record(provider="x", model="m", usage=Usage(cost_usd=1.0), when=now)
    spend.record(
        provider="x", model="m", usage=Usage(cost_usd=2.0), when=now - dt.timedelta(days=3)
    )
    spend.record(
        provider="x", model="m", usage=Usage(cost_usd=4.0), when=now - dt.timedelta(days=9)
    )
    summary = spend.summary(now=now)
    assert summary.today == pytest.approx(1.0) and summary.week == pytest.approx(3.0)
    assert summary.calls_today == 1 and summary.calls_week == 2
    assert "today $1.00" in summary.describe() and "7 days $3.00" in summary.describe()


def test_summary_without_a_log_is_zero(home):
    s = spend.summary()
    assert s.today == 0.0 and s.week == 0.0 and "no live calls" in s.describe()


def test_subscription_calls_log_without_dollars_and_count_against_max_calls(home):
    spend.set_context(command="apply")
    spend.set_max_calls(2)
    row = spend.record(
        provider="claude-cli", model="m", usage=Usage(calls=1), billing="subscription"
    )
    assert row["cost"] is None and row["billing"] == "subscription"
    spend.record(provider="anthropic", model="m", usage=Usage(calls=1, cost_usd=0.2))
    assert spend.calls() == 2 and spend.over_budget() is not None
    assert "max-calls" in (spend.over_budget() or "")
    s = spend.summary()
    assert s.today == pytest.approx(0.2) and s.subscription_calls_today == 1
    assert "1 subscription call" in s.describe() and "$0.20" in s.describe()
