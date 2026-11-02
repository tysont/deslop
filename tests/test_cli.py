# ABOUTME: CLI tests via typer's runner: voice build/show/diff/validate/pack, score, apply with
# ABOUTME: scripted providers, lab commands, and clean errors. Checks exit codes and --json shapes.
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml
from typer.testing import CliRunner

from deslop.cli.app import app
from tests.test_apply import Scripted

runner = CliRunner()
FIXTURES = Path(__file__).parent / "fixtures"


def flat(r) -> str:
    """stdout with rich's line wrapping undone, so substrings survive narrow terminals."""
    return " ".join(r.stdout.split())


def scripted_factory(monkeypatch, providers=None, **kw):
    providers = providers or {"anthropic": Scripted("anthropic", **kw)}
    # `deslop.apply` names the function in the package namespace; patch the module itself.
    for module in ("deslop.apply", "deslop.providers.base"):
        monkeypatch.setattr(
            sys.modules[module],
            "get_provider",
            lambda name, **_: providers.get(name, providers["anthropic"]),
        )
    return providers


def test_version():
    r = runner.invoke(app, ["--version"])
    assert r.exit_code == 0 and "deslop 2." in flat(r)


def test_voice_build_show_validate(tmp_path):
    amap = tmp_path / "authors.yaml"
    amap.write_text(
        yaml.safe_dump(
            {
                "johnson_*": "Samuel Johnson",
                "hazlitt_*": "William Hazlitt",
                "arnold_*": "Matthew Arnold",
                "bradley_*": "A. C. Bradley",
            }
        )
    )
    out = tmp_path / "critics.voice"
    r = runner.invoke(
        app,
        [
            "voice",
            str(FIXTURES / "corpus_small"),
            "-o",
            str(out),
            "--author-map",
            str(amap),
            "--no-llm",
            "--exemplar-words",
            "1500",
        ],
    )
    assert r.exit_code == 0, r.output
    assert out.is_file() and "4 documents" in flat(r) and "thresholds" in flat(r)
    r = runner.invoke(app, ["voice", "show", str(out), "--clusters"])
    assert r.exit_code == 0, r.output
    assert "Registers" in flat(r) and "95% CI" in flat(r) and "Samuel Johnson" in flat(r)
    r = runner.invoke(app, ["voice", "show", str(out), "--json"])
    data = json.loads(r.stdout)
    assert data["meta"]["document_count"] == 4 and data["metrics"]["thresholds"]["stats"]
    r = runner.invoke(app, ["voice", "validate", str(out)])
    assert r.exit_code == 0, r.output
    assert "checksum" in flat(r) and "thresholds" in flat(r)
    r = runner.invoke(app, ["voice", "--help"])
    assert r.exit_code == 0 and "show" in flat(r) and "pack" in flat(r)


def test_voice_from_add_and_diff(fixture_voice, fixtures_dir, tmp_path):
    base = fixture_voice.save(tmp_path / "base.voice")
    out = tmp_path / "more.voice"
    r = runner.invoke(
        app,
        [
            "voice",
            "--from",
            str(base),
            "--add",
            str(fixtures_dir / "heldout_bradley.md"),
            "-o",
            str(out),
        ],
    )
    assert r.exit_code == 0, r.output
    assert "5 documents" in flat(r)
    r = runner.invoke(app, ["voice", "diff", str(base), str(out)])
    assert r.exit_code == 0 and "Metric means" in flat(r)
    r = runner.invoke(app, ["voice", "--from", str(base), "-o", str(out)])
    assert r.exit_code == 1 and "--add" in r.output


def test_voice_pack_unpack_strip(fixture_voice, tmp_path):
    zipped = fixture_voice.save(tmp_path / "f.voice")
    r = runner.invoke(app, ["voice", "unpack", str(zipped), "-o", str(tmp_path / "work")])
    assert r.exit_code == 0 and (tmp_path / "work" / "profile.md").exists()
    r = runner.invoke(
        app,
        [
            "voice",
            "pack",
            str(tmp_path / "work"),
            "-o",
            str(tmp_path / "shared.voice"),
            "--strip-exemplars",
        ],
    )
    assert r.exit_code == 0 and "stripped" in flat(r)
    r = runner.invoke(app, ["voice", "validate", str(tmp_path / "shared.voice")])
    assert r.exit_code == 0 and "exemplars stripped" in flat(r)


def test_score_exit_codes_and_json(fixture_voice, fixtures_dir, slop_text, tmp_path):
    voice = fixture_voice.save(tmp_path / "f.voice")
    sample = tmp_path / "sample.md"
    sample.write_text(slop_text)
    r = runner.invoke(app, ["score", str(sample), "--voice", str(voice), "--json"])
    assert r.exit_code == 2
    data = json.loads(r.stdout)
    assert data["verdict"] == "FAIL" and data["separation"]["llm_median"] == 70.0
    assert any(m["band_high"] is not None or m["band_low"] is not None for m in data["metrics"])
    r = runner.invoke(
        app,
        [
            "score",
            str(fixtures_dir / "heldout_bradley.md"),
            "--voice",
            str(voice),
            "--annotate",
            str(tmp_path / "a.md"),
            "-q",
        ],
    )
    assert r.exit_code == 0, r.output
    assert (tmp_path / "a.md").exists()


def test_score_with_judge(fixture_voice, fixtures_dir, tmp_path, monkeypatch):
    voice = fixture_voice.save(tmp_path / "f.voice")
    scripted_factory(monkeypatch)
    r = runner.invoke(
        app,
        [
            "score",
            str(fixtures_dir / "heldout_bradley.md"),
            "--voice",
            str(voice),
            "--judge",
            "--json",
        ],
    )
    assert r.exit_code == 0, r.output
    assert json.loads(r.stdout)["judge"]["provider"] == "anthropic"


def test_apply_rewrite_and_edit_with_scripted_providers(
    fixture_voice, slop_text, tmp_path, monkeypatch
):
    voice = fixture_voice.save(tmp_path / "f.voice")
    sample = tmp_path / "sample.md"
    sample.write_text(slop_text)
    scripted_factory(monkeypatch)
    out = tmp_path / "out.md"
    r = runner.invoke(
        app,
        ["apply", str(sample), "--voice", str(voice), "-o", str(out), "--n", "2", "--diff"],
    )
    assert r.exit_code in (0, 2), r.output
    assert out.exists() and "Plainly," in out.read_text()
    assert "Meaning" in flat(r) and "unit 1" in flat(r) and "candidate(s)" in flat(r)
    assert "writer anthropic/claude-opus-5-5" in flat(r) and "growth" in flat(r)
    assert "coherence: ok" in flat(r) and "judge vs original" in flat(r)
    assert "against the voice's band" in flat(r)
    marked = tmp_path / "gen.txt"
    marked.write_text(
        "# title: Genesis\n# book: Genesis\n\n{{Genesis 22:2}} Take now thy son.\n",
        encoding="utf-8",
    )
    r = runner.invoke(
        app,
        [
            "apply",
            str(sample),
            "--voice",
            str(voice),
            "--n",
            "1",
            "--primary-text",
            str(marked),
            "--max-growth",
            "40",
            "--chunk-words",
            "150",
            "--json",
        ],
    )
    assert r.exit_code in (0, 2), r.output
    data = json.loads(r.stdout)
    assert data["primary_texts"] == ["Genesis"] and data["max_growth"] == 40.0
    assert data["chunk_words"] == 150 and "growth_pct" in data
    assert len(data["units"]) > 1 and data["units"][0]["kind"].startswith("section 1/")
    assert "coherence" in data and "metric_table" in data and "judge_items" in data
    r = runner.invoke(
        app, ["apply", str(sample), "--voice", str(voice), "--primary-text", "nope.txt"]
    )
    assert r.exit_code == 1 and "primary text not found" in " ".join(r.output.split())
    r = runner.invoke(
        app,
        [
            "apply",
            str(sample),
            "--voice",
            str(voice),
            "--mode",
            "edit",
            "--max-iter",
            "1",
            "--json",
        ],
    )
    assert r.exit_code in (0, 2), r.output
    data = json.loads(r.stdout)
    assert data["mode"] == "edit" and data["iterations"] and data["units"]
    r = runner.invoke(app, ["apply", str(sample), "--voice", str(voice), "--mode", "sideways"])
    assert r.exit_code == 1


def test_apply_brief(fixture_voice, fixtures_dir, tmp_path, monkeypatch):
    from deslop.providers.base import Completion, Usage

    class Drafting(Scripted):
        def _raw_complete(self, system, user, model, temperature):
            if "from a brief" in system:
                return Completion(
                    text="Abraham climbs the mountain in Gen 22:1-3. The chapter withholds its answer.",
                    model=model,
                    provider=self.name,
                    usage=Usage(calls=1),
                )
            return super()._raw_complete(system, user, model, temperature)

    scripted_factory(monkeypatch, {"anthropic": Drafting("anthropic")})
    voice = fixture_voice.save(tmp_path / "f.voice")
    r = runner.invoke(
        app,
        [
            "apply",
            "--brief",
            str(fixtures_dir / "brief_sample.yaml"),
            "--voice",
            str(voice),
            "--n",
            "1",
            "--json",
        ],
    )
    assert r.exit_code in (0, 2), r.output
    data = json.loads(r.stdout)
    assert data["draft"] and data["draft_text"]


def test_voice_build_resolves_the_profile_provider_like_every_live_command(tmp_path):
    r = runner.invoke(
        app,
        [
            "voice",
            str(FIXTURES / "corpus_small"),
            "-o",
            str(tmp_path / "x.voice"),
            "--provider",
            "bogus",
        ],
    )
    assert r.exit_code == 1 and "unknown provider 'bogus'" in r.output
    assert not (tmp_path / "x.voice").exists()


def test_lab_calibrate_thresholds_and_export(fixture_voice, tmp_path, monkeypatch):
    monkeypatch.setenv("DESLOP_HOME", str(tmp_path / "home"))
    voice = fixture_voice.save(tmp_path / "f.voice")
    r = runner.invoke(
        app, ["lab", "calibrate-thresholds", str(voice), "-o", str(tmp_path / "re.voice")]
    )
    assert r.exit_code == 0, r.output
    assert "LLM-reference documents fail" in flat(r) and (tmp_path / "re.voice").exists()
    r = runner.invoke(
        app,
        ["lab", "export", str(voice), "--out", str(tmp_path / "ft.jsonl"), "--no-chunks"],
    )
    assert r.exit_code == 0, r.output
    assert "No job" in flat(r) and "0 pairs" in flat(r)


def test_lab_blind_with_voice_exemplars(fixture_voice, tmp_path, monkeypatch):
    monkeypatch.setenv("DESLOP_HOME", str(tmp_path / "home"))
    voice = fixture_voice.save(tmp_path / "f.voice")
    cands = tmp_path / "cands"
    cands.mkdir()
    (cands / "d.md").write_text("The play opens on a battlement at night. " * 20)
    r = runner.invoke(
        app,
        [
            "lab",
            "blind",
            "--voice",
            str(voice),
            "--candidates",
            str(cands),
            "--n",
            "1",
            "--min-overlap",
            "0",
            "--seed",
            "1",
        ],
        input="A\nhmm\n",
    )
    assert r.exit_code == 0, r.output
    assert "this session: " in flat(r)
    r = runner.invoke(
        app,
        [
            "lab",
            "blind",
            "--voice",
            str(voice),
            "--candidates",
            str(cands),
            "--n",
            "1",
            "--min-overlap",
            "0",
            "--export",
            str(tmp_path / "blind"),
        ],
    )
    assert r.exit_code == 0, r.output
    assert (tmp_path / "blind" / "pairs.md").exists() and (tmp_path / "blind" / "key.json").exists()


def test_lab_calibrate_judge_stores_accuracy_in_the_voice(
    fixture_voice, fixtures_dir, tmp_path, monkeypatch
):
    from deslop.providers.base import Completion, Usage

    class Judge:
        name = "anthropic"
        default_model = "m"

        def __init__(self):
            self.usage = Usage()

        @property
        def key(self):
            return f"{self.name}/{self.default_model}"

        def complete(self, system, user, *, model=None, temperature=0.7):
            self.usage.calls += 1
            a = user.split("PARAGRAPH A:\n", 1)[1].split("\n\nPARAGRAPH B:", 1)[0]
            pick = "B" if a.startswith("Model") else "A"
            return Completion(
                text=json.dumps(
                    {
                        "more_like_writer": pick,
                        "more_like_model": "A" if pick == "B" else "B",
                        "confidence": 0.9,
                    }
                ),
                model=self.default_model,
                provider="anthropic",
            )

    monkeypatch.setattr("deslop.providers.base.get_provider", lambda n, **_: Judge())
    voice = fixture_voice.save(tmp_path / "f.voice")
    llm = tmp_path / "llm"
    llm.mkdir()
    (llm / "e.md").write_text("\n\n".join(["Model paragraph " + "word " * 50] * 3))
    r = runner.invoke(
        app,
        [
            "lab",
            "calibrate-judge",
            str(voice),
            "--corpus",
            str(fixtures_dir / "corpus_small"),
            "--llm-texts",
            str(llm),
            "--pairs",
            "3",
            "--judges",
            "sonnet",
        ],
    )
    assert r.exit_code == 0, r.output
    assert "100% picked the corpus paragraph" in flat(r) and "usable" in flat(r)
    from deslop import Voice

    cal = Voice.load(voice).meta.judge_calibration["claude-sonnet-5-5"]
    assert cal["accuracy"] == 1.0 and cal["pairs"] == 3 and cal["model"] == "claude-sonnet-5-5"


def test_missing_input_file_is_an_error_not_a_traceback(fixture_voice, tmp_path):
    voice = fixture_voice.save(tmp_path / "f.voice")
    r = runner.invoke(app, ["score", str(tmp_path / "nope.md"), "--voice", str(voice)])
    assert r.exit_code == 1 and "cannot read" in r.output and "Traceback" not in r.output
    r = runner.invoke(app, ["voice", str(tmp_path / "nothing"), "-o", str(tmp_path / "x.voice")])
    assert r.exit_code == 1 and "Traceback" not in r.output


def test_apply_budget_aborts_with_best_so_far_and_logs_spend(
    fixture_voice, slop_text, tmp_path, monkeypatch
):
    from deslop import spend

    monkeypatch.setenv("DESLOP_HOME", str(tmp_path / "home"))
    spend.reset()
    voice = fixture_voice.save(tmp_path / "f.voice")
    sample = tmp_path / "sample.md"
    sample.write_text(slop_text)
    scripted_factory(monkeypatch)  # every scripted call costs $0.01 and counts as live
    r = runner.invoke(
        app, ["apply", str(sample), "--voice", str(voice), "--n", "1", "--budget", "0.03"]
    )
    assert r.exit_code in (0, 2), r.output
    assert "budget of $0.03 spent" in flat(r) and "best result so far" in flat(r)
    rows = (tmp_path / "home" / "spend.jsonl").read_text().splitlines()
    assert rows and json.loads(rows[0])["command"] == "apply"
    assert json.loads(rows[0])["voice"] == "fixture" and "sample.md" in json.loads(rows[0])["input"]
    spend.reset()
    r = runner.invoke(app, ["lab", "calibrate-thresholds", str(voice), "--budget", "1"])
    assert r.exit_code == 0, r.output
    import inspect

    from deslop.cli.app import lab_blind, lab_calibrate_judge, lab_export

    for command in (lab_blind, lab_calibrate_judge, lab_export):
        assert "budget" in inspect.signature(command).parameters


def test_apply_writer_and_judge_flags(fixture_voice, slop_text, tmp_path, monkeypatch):
    monkeypatch.setenv("DESLOP_HOME", str(tmp_path / "home"))  # no real config file
    monkeypatch.delenv("DESLOP_PROVIDER", raising=False)
    voice = fixture_voice.save(tmp_path / "f.voice")
    sample = tmp_path / "sample.md"
    sample.write_text(slop_text)
    seen = []
    monkeypatch.setattr(
        sys.modules["deslop.apply"],
        "get_provider",
        lambda name, **_: seen.append(name) or Scripted("anthropic"),
    )
    base = ["apply", str(sample), "--voice", str(voice), "--n", "1"]
    r = runner.invoke(app, [*base, "--writer", "sonnet", "--judge", "opus"])
    assert r.exit_code in (0, 2), r.output
    assert "writer anthropic/claude-sonnet-5-5; judge anthropic/claude-opus-5-5" in flat(r)
    assert seen == ["anthropic"], "the default provider is the API, once"
    r = runner.invoke(app, [*base, "--writer", "opus", "--judge", "claude-opus-5-5"])
    assert r.exit_code == 1 and "never judges its own output" in " ".join(r.output.split())
    r = runner.invoke(app, [*base, "--provider", "bogus"])
    assert r.exit_code == 1 and "unknown provider" in " ".join(r.output.split())
    r = runner.invoke(app, [*base, "--provider", "claude-cli", "--max-calls", "3"])
    assert r.exit_code in (0, 2), r.output
    assert seen[-1] == "claude-cli"
    (tmp_path / "deslop.yaml").write_text("provider: claude-cli\n")
    monkeypatch.chdir(tmp_path)
    r = runner.invoke(app, base)
    assert r.exit_code in (0, 2), r.output
    assert seen[-1] == "claude-cli", "the config file's provider is honoured"


def test_apply_strips_frontmatter_on_ingest_and_keeps_it_on_output(
    fixture_voice, slop_text, tmp_path, monkeypatch
):
    voice = fixture_voice.save(tmp_path / "f.voice")
    sample = tmp_path / "sample.md"
    front = "---\ntitle: slop\nnote: Generated by a script.\n---\n\n"
    sample.write_text(front + slop_text)
    providers = scripted_factory(monkeypatch)
    out = tmp_path / "out.md"
    r = runner.invoke(
        app, ["apply", str(sample), "--voice", str(voice), "--n", "1", "-o", str(out), "--json"]
    )
    assert r.exit_code in (0, 2), r.output
    data = json.loads(r.stdout)
    assert not data["text"].startswith("---") and "Generated by a script" not in data["text"]
    assert out.read_text().startswith(front), "frontmatter is kept on output"
    assert "Plainly," in out.read_text()
    for _, user in providers["anthropic"].calls:
        if "list the propositions" in user or "# The original, in full" in user:
            assert "Generated by a script" not in user, "propositions come from the body only"


def test_apply_compare_page_and_report_html(fixture_voice, slop_text, tmp_path, monkeypatch):
    voice = fixture_voice.save(tmp_path / "f.voice")
    sample = tmp_path / "sample.md"
    sample.write_text(slop_text)
    ref = tmp_path / "ref.md"
    ref.write_text("---\ntitle: ref\n---\n\nThe human reference.\n")
    scripted_factory(monkeypatch)
    page = tmp_path / "cmp.html"
    r = runner.invoke(
        app,
        [
            "apply",
            str(sample),
            "--voice",
            str(voice),
            "--n",
            "1",
            "--report-all",
            "--reference",
            str(ref),
            "--compare",
            str(page),
            "--json",
        ],
    )
    assert r.exit_code in (0, 2), r.output
    assert page.exists() and "The human reference." in page.read_text()
    assert "title: ref" not in page.read_text()
    assert "recitation vs ref.md" in page.read_text()
    result = tmp_path / "result.json"
    result.write_text(r.stdout)
    data = json.loads(r.stdout)
    assert data["reference"] == "ref.md" and data["max_recitation"] == 2.0
    assert data["units"][0]["candidates"][0]["recitation"]["ngrams"] == 0
    out = tmp_path / "again.html"
    r = runner.invoke(
        app, ["report", "--html", str(result), "-o", str(out), "--reference", str(ref)]
    )
    assert r.exit_code == 0, r.output
    assert out.exists() and "<!doctype html>" in out.read_text()


def test_lab_recitation_and_fixture_check(fixture_voice, slop_text, tmp_path, monkeypatch):
    from tests.test_apply import RECITED

    voice = fixture_voice.save(tmp_path / "f.voice")
    ref = tmp_path / "ref.md"
    ref.write_text("---\ntitle: ref\n---\n\nA human wrote this.\n\n" + RECITED + "\n")
    sample = tmp_path / "sample.md"
    sample.write_text(slop_text)
    scripted_factory(monkeypatch, recite=RECITED)
    r = runner.invoke(
        app, ["apply", str(sample), "--voice", str(voice), "--n", "1", "--report-all", "--json"]
    )
    assert r.exit_code in (0, 2), r.output
    result = tmp_path / "result.json"
    result.write_text(r.stdout)
    r = runner.invoke(app, ["lab", "recitation", str(result), "--reference", str(ref)])
    assert r.exit_code == 0, r.output
    assert "shared run(s)" in r.output and "resister of force" in r.output
    assert "shared run(s)" in r.output and "2.6%" in r.output
    r = runner.invoke(
        app, ["lab", "fixture-check", str(ref), "--voice", str(voice), "--max-calls", "10"]
    )
    assert r.exit_code == 2, r.output
    assert "does not qualify" in r.output and "resister of force" in r.output
    assert "outside the propositions' vocabulary" in r.output
    scripted_factory(monkeypatch)
    r = runner.invoke(app, ["lab", "fixture-check", str(ref), "--voice", str(voice), "--json"])
    assert r.exit_code == 0, r.output
    assert json.loads(r.stdout)["recitation"]["ngrams"] == 0
