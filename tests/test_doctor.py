# ABOUTME: Tests for `deslop doctor`: each check reports ok/warn/fail with a useful detail,
# ABOUTME: and the command exits nonzero only when a check fails.
import datetime as dt
import json

from typer.testing import CliRunner

from deslop import doctor
from deslop.cli.app import app
from deslop.cli.config import Config

runner = CliRunner()


def test_spacy_and_cache_checks(tmp_path, monkeypatch):
    assert doctor.check_spacy().status == "ok"
    monkeypatch.setenv("DESLOP_HOME", str(tmp_path / "home"))
    c = doctor.check_cache()
    assert c.status == "ok" and "home" in c.detail
    blocker = tmp_path / "file"
    blocker.write_text("x")
    monkeypatch.setenv("DESLOP_HOME", str(blocker))
    assert doctor.check_cache().status == "fail"


def test_pricing_age_warns_after_ninety_days(monkeypatch):
    fetched = dt.date.fromisoformat(doctor.load_pricing().fetched)
    assert doctor.check_pricing(today=fetched + dt.timedelta(days=10)).status == "ok"
    old = doctor.check_pricing(today=fetched + dt.timedelta(days=91))
    assert old.status == "warn" and "91 days" in old.detail


def test_provider_checks_use_the_ping_and_the_cli_status(monkeypatch):
    from deslop.providers import claude_cli

    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DESLOP_PROVIDER", raising=False)
    monkeypatch.setattr(doctor, "_ping", lambda name: "claude-x")
    monkeypatch.setattr(claude_cli, "cli_status", lambda: claude_cli.CliStatus(True, True, "9.9.9"))
    checks = doctor.check_providers(ping=lambda name: "claude-x")
    details = [c.detail for c in checks]
    assert [c.status for c in checks] == ["ok", "ok", "ok"]
    assert "anthropic: claude-x" in details[0] and "claude-cli: 9.9.9, logged in" in details[1]
    assert "default provider: anthropic" in details[2]

    def boom(name):
        raise RuntimeError("401 bad key")

    checks = doctor.check_providers(ping=boom)
    assert checks[0].status == "fail" and "401" in checks[0].detail
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    monkeypatch.setattr(
        claude_cli, "cli_status", lambda: claude_cli.CliStatus(True, False, "9.9.9", "no login")
    )
    checks = doctor.check_providers(ping=lambda name: "x")
    assert [c.status for c in checks] == ["warn", "warn", "warn"]
    assert "not logged in" in checks[1].detail and "not usable" in checks[2].detail
    monkeypatch.setenv("DESLOP_PROVIDER", "claude-cli")
    monkeypatch.setattr(claude_cli, "cli_status", lambda: claude_cli.CliStatus(True, True, "9.9.9"))
    checks = doctor.check_providers(ping=lambda name: "x")
    assert checks[-1].status == "ok" and "default provider: claude-cli" in checks[-1].detail
    monkeypatch.setenv("OPENAI_API_KEY", "sk")
    assert "unused unless named" in doctor.check_providers(ping=lambda n: "x")[-1].detail


def test_voice_check_resolves_and_validates(fixture_voice, tmp_path, monkeypatch):
    monkeypatch.delenv("DESLOP_VOICE", raising=False)
    missing = doctor.check_voice(Config())
    assert missing.status == "warn" and "DESLOP_VOICE" in missing.detail
    voice_dir = tmp_path / "v.voice"
    fixture_voice.save(voice_dir)
    ok = doctor.check_voice(Config({"default_voice": str(voice_dir)}))
    assert ok.status == "ok", ok.detail
    monkeypatch.setenv("DESLOP_VOICE", str(tmp_path / "nope.voice"))
    assert doctor.check_voice(Config()).status == "fail"
    work = tmp_path / "work"
    work.mkdir()
    fixture_voice.save(work)
    (work / "metrics.json").write_text("{}")
    monkeypatch.setenv("DESLOP_VOICE", str(work))
    bad = doctor.check_voice(Config())
    assert bad.status == "fail" and "metrics" in bad.detail


def test_doctor_command(fixture_voice, tmp_path, monkeypatch):
    monkeypatch.setenv("DESLOP_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DESLOP_COMPATIBLE_BASE_URL", raising=False)
    voice_dir = tmp_path / "v.voice"
    fixture_voice.save(voice_dir)
    monkeypatch.setenv("DESLOP_VOICE", str(voice_dir))
    r = runner.invoke(app, ["doctor"])
    assert r.exit_code == 0, r.output
    assert "spacy" in r.output and "voice" in r.output
    r = runner.invoke(app, ["doctor", "--json"])
    data = json.loads(r.stdout)
    assert {c["name"] for c in data["checks"]} >= {
        "spacy",
        "providers",
        "pricing",
        "cache",
        "voice",
    }
    assert data["ok"] is True
    monkeypatch.setenv("DESLOP_VOICE", str(tmp_path / "missing.voice"))
    r = runner.invoke(app, ["doctor"])
    assert r.exit_code == 1


def test_spend_check_sums_the_log(tmp_path, monkeypatch):
    from deslop import spend
    from deslop.providers.base import Usage

    monkeypatch.setenv("DESLOP_HOME", str(tmp_path / "home"))
    assert "no live calls" in doctor.check_spend().detail
    spend.record(provider="anthropic", model="m", usage=Usage(cost_usd=0.5))
    c = doctor.check_spend()
    assert c.status == "ok" and "today $0.50" in c.detail and "7 days $0.50" in c.detail
