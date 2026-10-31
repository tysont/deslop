# ABOUTME: CLI smoke tests via typer's runner: corpus, calibrate, score, voice show.
# ABOUTME: Checks exit codes and --json output shapes.
import json

from typer.testing import CliRunner

from deslop.cli.app import app

runner = CliRunner()


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0 and "deslop" in result.stdout


def test_corpus_init_add_stats(tmp_path):
    src = tmp_path / "a.txt"
    src.write_text(("Plain prose sentence number one here. " * 40).strip())
    c = tmp_path / "corpus"
    assert runner.invoke(app, ["corpus", "init", str(c), "--name", "t"]).exit_code == 0
    r = runner.invoke(
        app, ["corpus", "add", str(c), str(src), "--author", "Ann Author", "--title", "T", "--json"]
    )
    assert r.exit_code == 0, r.output
    assert json.loads(r.stdout)[0]["author"] == "Ann Author"
    r = runner.invoke(app, ["corpus", "stats", str(c), "--json"])
    assert json.loads(r.stdout)["authors"][0]["share"] == 1.0
    r = runner.invoke(app, ["corpus", "add", str(c), str(src), "--author", "Ann Author"])
    assert r.exit_code == 1 and "duplicate" in r.output


def test_calibrate_score_show(fixture_voice, fixtures_dir, slop_text, tmp_path):
    voice_dir = tmp_path / "v.voice"
    fixture_voice.save(voice_dir)
    r = runner.invoke(
        app, ["score", str(fixtures_dir / "slop_sample.md"), "--voice", str(voice_dir), "--json"]
    )
    assert r.exit_code == 2
    data = json.loads(r.stdout)
    assert data["verdict"] == "FAIL"
    out = tmp_path / "annotated.md"
    r = runner.invoke(
        app,
        [
            "score",
            str(fixtures_dir / "heldout_bradley.md"),
            "--voice",
            str(voice_dir),
            "--annotate",
            str(out),
            "-q",
        ],
    )
    assert r.exit_code == 0 and "PASS" in r.stdout
    assert out.exists()
    r = runner.invoke(app, ["voice", "show", str(voice_dir)])
    assert r.exit_code == 0 and "sent_len_mean" in r.stdout
    r = runner.invoke(app, ["voice", "show", str(voice_dir), "--json"])
    assert json.loads(r.stdout)["meta"]["name"] == "fixture"


def test_calibrate_command_small(tmp_path, fixtures_dir):
    # Calibrate a one-document corpus to exercise the command end to end quickly.
    c = tmp_path / "c"
    assert runner.invoke(app, ["corpus", "init", str(c)]).exit_code == 0
    r = runner.invoke(
        app,
        [
            "corpus",
            "add",
            str(c),
            str(fixtures_dir / "heldout_bradley.md"),
            "--author",
            "A. C. Bradley",
        ],
    )
    assert r.exit_code == 0, r.output
    r = runner.invoke(
        app, ["calibrate", str(c), "--out", str(tmp_path / "mini.voice"), "--no-llm", "--json"]
    )
    assert r.exit_code == 0, r.output
    assert json.loads(r.stdout)["name"] == "mini"
    assert (tmp_path / "mini.voice" / "exemplars.md").exists()


def test_providers_test_runs_without_keys(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DESLOP_COMPATIBLE_BASE_URL", raising=False)
    r = runner.invoke(app, ["providers", "test", "--json"])
    assert r.exit_code == 0
    assert all(not v["configured"] for v in json.loads(r.stdout).values())
