# ABOUTME: Tests for CLI configuration: deslop.yaml discovery, per-command sections, and
# ABOUTME: the precedence CLI flag > command section > top-level key > built-in default.
import json
import pathlib

import pytest
from typer.testing import CliRunner

from deslop.cli import config
from deslop.cli.app import app

runner = CliRunner()


def test_precedence_flag_over_section_over_top_level_over_default(tmp_path):
    cfg = config.Config({"sigma": 1.0, "score": {"sigma": 2.0}, "tell_mult": 9.0})
    assert cfg.get("score", "sigma", 3.0, default=0.5) == 3.0
    assert cfg.get("score", "sigma", None, default=0.5) == 2.0
    assert cfg.get("convert", "sigma", None, default=0.5) == 1.0
    assert cfg.get("convert", "tell_mult", None, default=0.5) == 9.0
    assert cfg.get("convert", "max_iter", None, default=4) == 4


def test_keys_accept_hyphens_and_lists():
    cfg = config.Config({"convert": {"max-iter": 2, "models": ["claude", "gpt"]}})
    assert cfg.get("convert", "max_iter", None, default=4) == 2
    assert cfg.get("convert", "models", None, default="x") == ["claude", "gpt"]


def test_discovery_order(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("DESLOP_HOME", str(home))
    (home / "config.yaml").write_text("sigma: 7\n")
    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.chdir(project)
    assert config.load_config(None).get("score", "sigma", None, default=1) == 7
    (project / "deslop.yaml").write_text("sigma: 8\n")
    assert config.load_config(None).get("score", "sigma", None, default=1) == 8
    explicit = tmp_path / "x.yaml"
    explicit.write_text("sigma: 9\n")
    assert config.load_config(explicit).get("score", "sigma", None, default=1) == 9
    assert config.load_config(None).path == project / "deslop.yaml"


def test_cli_reads_voice_and_options_from_config(
    fixture_voice, fixtures_dir, tmp_path, monkeypatch
):
    voice_dir = tmp_path / "v.voice"
    fixture_voice.save(voice_dir)
    monkeypatch.chdir(tmp_path)
    heldout = str(fixtures_dir / "heldout_bradley.md")
    (tmp_path / "deslop.yaml").write_text(f"default_voice: {voice_dir}\nscore:\n  quiet: true\n")
    r = runner.invoke(app, ["score", heldout, "--json"])
    assert r.exit_code == 0, r.output  # the voice came from the config file
    assert json.loads(r.stdout)["voice"] == "fixture"
    r = runner.invoke(app, ["score", heldout, "--voice", str(tmp_path / "nope.voice")])
    assert r.exit_code == 1, r.output  # the flag overrides the config, and is checked
    r = runner.invoke(app, ["--config", str(tmp_path / "missing.yaml"), "score", heldout])
    assert r.exit_code == 1 and "missing.yaml" in "".join(r.output.split())


def test_cli_requires_voice_when_absent_everywhere(fixtures_dir, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DESLOP_HOME", str(tmp_path))
    monkeypatch.delenv("DESLOP_VOICE", raising=False)
    r = runner.invoke(app, ["score", str(fixtures_dir / "heldout_bradley.md")])
    assert r.exit_code == 1
    for marker in ("--voice", "DESLOP_VOICE", "default_voice"):
        assert marker in r.output


def test_voice_resolution_order(tmp_path, monkeypatch):
    monkeypatch.delenv("DESLOP_VOICE", raising=False)
    cfg = config.Config({"default_voice": "cfg.voice", "blind": {"default_voice": "blind.voice"}})
    assert config.resolve_voice(cfg, "score", tmp_path / "flag.voice") == tmp_path / "flag.voice"
    assert config.resolve_voice(cfg, "score", None) == pathlib.Path("cfg.voice")
    assert config.resolve_voice(cfg, "blind", None) == pathlib.Path("blind.voice")
    monkeypatch.setenv("DESLOP_VOICE", "env.voice")
    assert config.resolve_voice(cfg, "score", None) == pathlib.Path("env.voice")
    assert config.resolve_voice(cfg, "score", tmp_path / "flag.voice") == tmp_path / "flag.voice"
    monkeypatch.setenv("DESLOP_VOICE", "")
    with pytest.raises(config.ConfigError) as exc:
        config.resolve_voice(config.Config(), "score", None)
    assert "DESLOP_VOICE" in str(exc.value) and "default_voice" in str(exc.value)
