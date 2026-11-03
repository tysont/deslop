# ABOUTME: Tests for building voices from files: author resolution, manifests, provenance in
# ABOUTME: voice.yaml, zip-canonical save/load, stripped exemplars, and extending a voice.
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
import yaml

from deslop.voice import Voice
from deslop.voice.collect import author_for, gather_files, title_for
from deslop.voice.exemplars import parse_exemplars
from deslop.voice.pack import PackError, pack_voice, unpack_voice

FIXTURES = Path(__file__).parent / "fixtures" / "corpus_small"


def test_author_resolution_order(tmp_path):
    fm = tmp_path / "essay.md"
    fm.write_text("---\nauthor: Ann Author\ntitle: On Foxes\nurl: https://x.org/a\n---\n\nText.\n")
    plain = tmp_path / "hazlitt_on_actors.txt"
    plain.write_text("Text.")
    assert author_for(fm, {}) == "Ann Author" and title_for(fm) == "On Foxes"
    assert author_for(plain, {}) == "Hazlitt" and title_for(plain) == "on actors"
    assert author_for(plain, {"hazlitt_*": "William Hazlitt"}) == "William Hazlitt"
    assert author_for(fm, {"default": "Someone"}) == "Ann Author"
    assert author_for(tmp_path / "x.txt", {"default": "Someone"}) == "Someone"


def test_gather_files_walks_directories_and_spots_manifests(tmp_path):
    (tmp_path / "a.txt").write_text("a")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.md").write_text("b")
    (sub / "notes.json").write_text("{}")
    crawl = tmp_path / "crawl"
    crawl.mkdir()
    (crawl / "manifest.jsonl").write_text("")
    (crawl / "c.txt").write_text("c")
    files = gather_files([tmp_path])
    assert tmp_path / "a.txt" in files and sub / "b.md" in files
    assert crawl in files and crawl / "c.txt" not in files
    assert sub / "notes.json" not in files


def test_from_texts_builds_zip_with_provenance(tmp_path):
    voice = Voice.from_texts(
        [FIXTURES],
        name="critics",
        author_map={
            "johnson_*": "Samuel Johnson",
            "hazlitt_*": "William Hazlitt",
            "arnold_*": "Matthew Arnold",
            "bradley_*": "A. C. Bradley",
        },
        use_llm=False,
    )
    assert voice.meta.document_count == 4 and voice.meta.total_words > 50000
    assert {a.name for a in voice.meta.authors} == {
        "Samuel Johnson",
        "William Hazlitt",
        "Matthew Arnold",
        "A. C. Bradley",
    }
    assert voice.meta.corpus_path is None and voice.meta.schema_version == 2
    assert voice.metrics.thresholds.stats is not None
    out = voice.save(tmp_path / "critics.voice")
    assert zipfile.is_zipfile(out)
    with zipfile.ZipFile(out) as zf:
        meta = yaml.safe_load(zf.read("voice.yaml"))
    assert meta["checksum"] and meta["document_count"] == 4
    again = Voice.load(out)
    assert again.read_only and again.name == "critics" and again.meta.checksum
    with pytest.raises(PackError):
        again.save(out)
    # The directory form is a working copy.
    work = tmp_path / "work"
    work.mkdir()
    again.save(work)
    assert (work / "voice.yaml").exists() and Voice.load(work).read_only is False


def test_from_texts_reads_a_crawler_manifest(tmp_path):
    crawl = tmp_path / "crawl"
    crawl.mkdir()
    rows = []
    for i, src in enumerate(sorted(FIXTURES.glob("*.txt"))):
        name = f"doc{i}.txt"
        (crawl / name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        rows.append(
            {
                "file": name,
                "author": f"Writer {i}",
                "title": f"Piece {i}",
                "url": f"https://site{i % 2}.org/{name}",
                "license": "CC BY 4.0",
                "date": "2001",
            }
        )
    (crawl / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    voice = Voice.from_texts([crawl], name="crawl", use_llm=False)
    assert voice.meta.document_count == 4
    assert voice.meta.licenses == ["CC BY 4.0"]
    assert sorted(voice.meta.authors[0].sources)[0].startswith("site")
    assert any(
        "<https://site" in line for line in voice.exemplars.splitlines() if line.startswith("## ")
    )


def test_strip_exemplars_pack(fixture_voice, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    fixture_voice.save(work)
    stripped = pack_voice(work, tmp_path / "shared.voice", strip_exemplars=True)
    shared = Voice.load(stripped)
    assert shared.meta.has_exemplars is False and parse_exemplars(shared.exemplars) == []
    unpack_voice(stripped, tmp_path / "u")  # checksum covers the stripped file
    assert "Stripped for sharing" in (tmp_path / "u" / "exemplars.md").read_text()


def test_extend_folds_new_texts_and_keeps_prose(fixture_voice, fixtures_dir):
    extended = Voice.from_texts(
        [fixtures_dir / "heldout_bradley.md"],
        name="fixture",
        author_map={"default": "A. C. Bradley"},
        base=fixture_voice,
        use_llm=False,
    )
    assert extended.meta.document_count == fixture_voice.meta.document_count + 1
    assert extended.meta.total_words > fixture_voice.meta.total_words
    assert (
        extended.profile == fixture_voice.profile and extended.exemplars == fixture_voice.exemplars
    )
    assert extended.metrics.thresholds.stats is not None
    assert len(extended.metrics.per_document) == len(fixture_voice.metrics.per_document) + 1


def test_small_corpus_voice_passes_its_own_documents_and_says_so(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from deslop.cli.app import app

    voice = Voice.from_texts([FIXTURES], name="small", use_llm=False, min_docs=10)
    info = voice.metrics.small_corpus
    assert info is not None and info.documents == 4 and info.min_docs == 10
    assert info.weight == pytest.approx(0.4) and info.words_needed > 0
    assert voice.metrics.thresholds.stats is not None
    assert voice.metrics.thresholds.stats.heldout_pass_rate == 1.0
    assert all(b.low is None or b.low >= 0 for b in voice.metrics.thresholds.metrics.values())
    path = voice.save(tmp_path / "small.voice")
    r = CliRunner().invoke(app, ["voice", "show", str(path)])
    flat = " ".join(r.output.split())
    assert r.exit_code == 0 and "small corpus" in flat and "4 documents" in flat
    assert "more words" in flat
    big = Voice.from_texts([FIXTURES], name="big", use_llm=False, min_docs=2)
    assert big.metrics.small_corpus is None


def test_a_voice_records_its_corpus_relative_to_itself_or_not_at_all(tmp_path):
    import re
    import shutil

    corpus = tmp_path / "corpus"
    shutil.copytree(FIXTURES, corpus)
    voice = Voice.from_texts([corpus], name="critics", use_llm=False)
    assert voice.meta.corpus_path is None, "nothing is recorded until the voice has a home"
    voice.save(tmp_path / "critics.voice")
    assert voice.meta.corpus_path == "corpus"
    nested = tmp_path / "voices"
    nested.mkdir()
    voice.save(nested / "critics.voice")
    assert voice.meta.corpus_path is None, "the corpus is outside the voice's tree"
    elsewhere = Voice.from_texts([FIXTURES], name="critics", use_llm=False)
    out = elsewhere.save(tmp_path / "elsewhere.voice")
    assert elsewhere.meta.corpus_path is None
    with zipfile.ZipFile(out) as zf:
        text = zf.read("voice.yaml").decode("utf-8")
    assert not re.search(r":\s+['\"]?[/~]", text), "no absolute or home path in voice.yaml"
    # An older voice that recorded the machine's absolute path loads with it dropped.
    work = tmp_path / "old"
    work.mkdir()
    voice.save(work)
    meta = yaml.safe_load((work / "voice.yaml").read_text())
    meta["corpus_path"] = "/home/someone/corpus"
    (work / "voice.yaml").write_text(yaml.safe_dump(meta, sort_keys=False, allow_unicode=True))
    assert Voice.load(work).meta.corpus_path is None


def test_add_tells_measures_only_the_tells_a_voice_predates(tmp_path, fixture_voice):
    from deslop.voice.calibrate import add_tells
    from deslop.voice.collect import collect

    v = fixture_voice.model_copy(deep=True)
    before = {k: x.rate_per_1k for k, x in v.metrics.tells.items()}
    limits = dict(v.metrics.thresholds.tells)
    v.metrics.tells.pop("T16", None)
    v.metrics.thresholds.tells.pop("T16", None)
    v.lexicon.patterns = [x for x in v.lexicon.patterns if x.id != "T16"]
    corpus, _ = collect([FIXTURES], name="critics")
    assert add_tells(v, corpus) == ["T16"]
    assert v.metrics.tells["T16"].rate_per_1k >= 0
    assert v.metrics.thresholds.tells["T16"] >= 2.0, "the starting floor a build would give it"
    assert any(x.id == "T16" for x in v.lexicon.patterns)
    for k, rate in before.items():
        if k != "T16":
            assert (
                v.metrics.tells[k].rate_per_1k == rate
                and v.metrics.thresholds.tells[k] == limits[k]
            )
    assert add_tells(v, corpus) == [], "nothing left to add"
