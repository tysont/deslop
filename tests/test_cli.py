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


def test_convert_command_with_fake_provider(fixture_voice, tmp_path, monkeypatch):
    import deslop.converter as convert_module
    from tests.test_convert import FakeProvider

    text = (
        "Abraham climbs the hill in Genesis 22 with the wood on Isaac's back and the knife in his "
        "own hand. The chapter asks what he owes. He builds the altar there."
    )
    script = {"The chapter asks what he owes.": "What he owes is never said."}
    fake = FakeProvider("claude", script)
    monkeypatch.setattr(convert_module, "get_provider", lambda name, **kw: fake)
    monkeypatch.setattr(convert_module, "PAIRS_DIR", tmp_path / "pairs")
    voice_dir = tmp_path / "v.voice"
    fixture_voice.save(voice_dir)
    src = tmp_path / "in.md"
    src.write_text(text)
    out = tmp_path / "out.md"
    r = runner.invoke(
        app,
        [
            "convert",
            str(src),
            "--voice",
            str(voice_dir),
            "--models",
            "claude",
            "--out",
            str(out),
            "--json",
        ],
    )
    assert r.exit_code in (0, 2), r.output
    data = json.loads(r.stdout)
    assert data["text"] == out.read_text()
    assert "What he owes is never said." in data["text"]
    assert data["cost"]["calls"] >= 1
    assert (tmp_path / "pairs" / "fixture.jsonl").exists()
    r = runner.invoke(app, ["convert", str(src), "--voice", str(voice_dir), "--lock", "bogus"])
    assert r.exit_code == 1 and "unknown lock class" in r.output


def _workspace(fixture_voice, tmp_path):
    import shutil
    from pathlib import Path

    fixtures = Path(__file__).parent / "fixtures"
    corpus_dir = tmp_path / "corpus"
    shutil.copytree(fixtures / "corpus_small", corpus_dir)
    voice_dir = tmp_path / "v.voice"
    v = fixture_voice.model_copy(deep=True)
    v.meta.corpus_path = str(corpus_dir)
    v.save(voice_dir)
    accepted = tmp_path / "accepted.md"
    accepted.write_text((fixtures / "heldout_hazlitt.md").read_text(encoding="utf-8"))
    return voice_dir, corpus_dir, accepted, fixtures


def test_voice_update_diff_validate_commands(fixture_voice, tmp_path, monkeypatch):
    voice_dir, corpus_dir, accepted, _ = _workspace(fixture_voice, tmp_path)
    monkeypatch.setenv("DESLOP_HOME", str(tmp_path / "home"))
    r = runner.invoke(app, ["voice", "validate", str(voice_dir)])
    assert r.exit_code == 0, r.output
    r = runner.invoke(app, ["voice", "update", str(voice_dir), "--accept", str(accepted), "--json"])
    assert r.exit_code == 0, r.output
    data = json.loads(r.stdout)
    assert data["added"] == ["accepted.md"]
    r = runner.invoke(app, ["voice", "diff", f"{voice_dir}@prev", str(voice_dir)])
    assert r.exit_code == 0 and "sent_len_mean" in r.stdout and "(accepted)" in r.stdout
    r = runner.invoke(app, ["voice", "diff", f"{voice_dir}@prev", str(voice_dir), "--json"])
    assert json.loads(r.stdout)["words_after"] > json.loads(r.stdout)["words_before"]
    r = runner.invoke(app, ["voice", "diff", f"{voice_dir}@nope", str(voice_dir)])
    assert r.exit_code == 1
    r = runner.invoke(app, ["voice", "validate", str(voice_dir), "--json"])
    assert r.exit_code == 0 and json.loads(r.stdout)["ok"] is True
    (corpus_dir / "corpus.yaml").write_text("name: broken\ndocuments: []\n")
    r = runner.invoke(app, ["voice", "validate", str(voice_dir)])
    assert r.exit_code == 1 and "FAIL" in r.stdout


def test_voice_export_command(fixture_voice, tmp_path, monkeypatch):
    import deslop.voice.export as export_module
    from tests.test_export_blind import Summarizer, write_pairs

    voice_dir, _, _, _ = _workspace(fixture_voice, tmp_path)
    home = tmp_path / "home"
    (home / "pairs").mkdir(parents=True)
    write_pairs(home / "pairs" / "fixture.jsonl", "fixture")
    monkeypatch.setattr(export_module, "PAIRS_DIR", home / "pairs")
    monkeypatch.setattr(export_module, "first_available", lambda: Summarizer(), raising=False)
    monkeypatch.setattr("deslop.providers.base.first_available", lambda: Summarizer())
    out = tmp_path / "train.jsonl"
    r = runner.invoke(
        app,
        ["voice", "export", str(voice_dir), "--out", str(out), "--max-chunks", "2", "--json"],
    )
    assert r.exit_code == 0, r.output
    data = json.loads(r.stdout)
    assert data["pairs"] == 2 and data["chunks"] == 2 and data["estimated_training_cost"] > 0
    assert len(out.read_text().splitlines()) == 4
    r = runner.invoke(
        app,
        ["voice", "export", str(voice_dir), "--out", str(out), "--no-chunks", "--format", "jsonl"],
    )
    assert r.exit_code == 0 and "No job was" in r.stdout
    r = runner.invoke(
        app, ["voice", "export", str(voice_dir), "--out", str(out), "--format", "csv"]
    )
    assert r.exit_code == 1


def test_blind_command_logs_and_summarizes(fixture_voice, tmp_path, monkeypatch):
    import deslop.blind as blind_module

    voice_dir, _, _, fixtures = _workspace(fixture_voice, tmp_path)
    monkeypatch.setattr(blind_module, "BLIND_DIR", tmp_path / "blind")
    cands = tmp_path / "cands"
    cands.mkdir()
    (cands / "draft.md").write_text((fixtures / "slop_sample.md").read_text(encoding="utf-8"))
    r = runner.invoke(
        app,
        ["blind", "--voice", str(voice_dir), "--candidates", str(cands), "--n", "2", "--seed", "3"],
        input="A\ntoo tidy\nB\n\n",
    )
    assert r.exit_code == 0, r.output
    assert "this session: " in r.stdout and "cumulative: 2" in r.stdout
    log = (tmp_path / "blind" / "fixture.jsonl").read_text().splitlines()
    assert len(log) == 2 and json.loads(log[0])["reason"] == "too tidy"
    r = runner.invoke(
        app,
        ["blind", "--voice", str(voice_dir), "--candidates", str(cands), "--n", "1", "--json"],
        input="B\nhmm\n",
    )
    assert r.exit_code == 0 and json.loads(r.stdout[r.stdout.index("{") :])["total"] == 3
    r = runner.invoke(
        app, ["blind", "--voice", str(voice_dir), "--candidates", str(tmp_path / "x")]
    )
    assert r.exit_code == 1


def test_draft_command_with_fake_provider(fixture_voice, tmp_path, monkeypatch):
    import deslop.converter as convert_module
    from deslop.draft import load_brief
    from tests.test_draft import FIXTURES, SectionProvider

    brief = load_brief(FIXTURES / "brief_sample.yaml")
    script = {
        s.heading: f'Johnson says "{s.quotes[0]}" (Johnson 1765), and he is right.'
        for s in brief.sections
    }
    monkeypatch.setattr(convert_module, "get_provider", lambda name, **kw: SectionProvider(script))
    monkeypatch.setattr(convert_module, "PAIRS_DIR", tmp_path / "pairs")
    voice_dir = tmp_path / "v.voice"
    fixture_voice.save(voice_dir)
    out = tmp_path / "draft.md"
    r = runner.invoke(
        app,
        [
            "draft",
            str(FIXTURES / "brief_sample.yaml"),
            "--voice",
            str(voice_dir),
            "--out",
            str(out),
            "--max-iter",
            "0",
            "--json",
        ],
    )
    assert r.exit_code in (0, 2), r.output
    data = json.loads(r.stdout)
    assert data["text"] == out.read_text() and out.read_text().startswith("# What Johnson")
    assert len(data["sections"]) == 3 and data["convert"]["iterations"] == []
    r = runner.invoke(app, ["draft", str(tmp_path / "nope.yaml"), "--voice", str(voice_dir)])
    assert r.exit_code == 1
