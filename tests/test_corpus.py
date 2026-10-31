# ABOUTME: Tests for corpus init/add/stats and simhash dedupe.
# ABOUTME: Uses temporary directories; never touches the shipped fixture corpus.
import json

import pytest
import yaml

from deslop.corpus import Corpus, DuplicateError
from deslop.corpus.dedupe import fingerprint, is_duplicate

PROSE = ("The quick brown fox jumps over the lazy dog near the riverbank. " * 30).strip()


def test_init_and_add_txt(tmp_path):
    src = tmp_path / "doc.txt"
    src.write_text(PROSE, encoding="utf-8")
    corpus = Corpus.init(tmp_path / "c", name="demo")
    entry = corpus.add(str(src), author="Jane Doe", title="Foxes", year=1900, weight=0.5)
    assert entry.path == "doe_foxes.txt"
    assert entry.words == 360
    manifest = yaml.safe_load((tmp_path / "c" / "corpus.yaml").read_text())
    assert manifest["name"] == "demo"
    assert manifest["documents"][0]["weight"] == 0.5
    reloaded = Corpus.load(tmp_path / "c")
    assert reloaded.documents()[0].text.startswith("The quick brown fox")


def test_add_markdown_strips_markup(tmp_path):
    src = tmp_path / "doc.md"
    src.write_text(
        "# Heading\n\n" + "Some **bold** text with a [link](http://x). " * 5, encoding="utf-8"
    )
    corpus = Corpus.init(tmp_path / "c")
    corpus.add(str(src), author="A B")
    text = corpus.documents()[0].text
    assert "**" not in text and "#" not in text and "http" not in text
    assert "bold text with a link" in text


def test_duplicate_rejected(tmp_path, heldout_texts):
    src = tmp_path / "doc.txt"
    src.write_text(heldout_texts[0], encoding="utf-8")
    near = tmp_path / "near.txt"
    near.write_text(heldout_texts[0].replace("poetry", "verse", 3), encoding="utf-8")
    corpus = Corpus.init(tmp_path / "c")
    corpus.add(str(src), author="A")
    with pytest.raises(DuplicateError):
        corpus.add(str(near), author="A")


def test_simhash_distance(heldout_texts):
    a = fingerprint(heldout_texts[0])
    b = fingerprint(heldout_texts[0].replace("the", "a", 5))
    c = fingerprint(heldout_texts[1])
    assert is_duplicate(a, [b])
    assert not is_duplicate(a, [c])


def test_stats_share_cap(corpus_small):
    stats = corpus_small.stats(max_share=0.25)
    assert stats.total_words > 55000
    assert len(stats.authors) == 4
    assert all(0.2 < a.share < 0.3 for a in stats.authors)
    assert stats.over_cap  # four equal authors: at least one is just over 25%
    assert not corpus_small.stats(max_share=0.3).over_cap


def test_unsupported_type(tmp_path):
    from deslop.corpus.ingest import IngestError, ingest

    bad = tmp_path / "x.xyz"
    bad.write_text("x")
    with pytest.raises(IngestError):
        ingest(str(bad))


def test_add_manifest_adds_skips_duplicates_and_reports_failures(tmp_path, heldout_texts):
    src = tmp_path / "crawl"
    src.mkdir()
    (src / "a.txt").write_text(heldout_texts[0], encoding="utf-8")
    (src / "b.txt").write_text(heldout_texts[0].replace("poetry", "verse", 3), encoding="utf-8")
    (src / "c.txt").write_text(PROSE, encoding="utf-8")
    rows = [
        {
            "file": "a.txt",
            "author": "Ann Author",
            "title": "First",
            "date": "2001-05-04",
            "license": "CC BY 4.0",
            "url": "https://example.org/a",
        },
        {"file": "b.txt", "author": "Bob Byline", "title": "Near duplicate", "year": 2002},
        {"file": "c.txt", "author": "Cy Writer", "title": "Foxes"},
        {"file": "missing.txt", "author": "Dee", "title": "Gone"},
    ]
    (src / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    corpus = Corpus.init(tmp_path / "c")
    report = corpus.add_manifest(src)
    assert [e.title for e in report.added] == ["First", "Foxes"]
    assert report.added[0].year == 2001 and report.added[0].license == "CC BY 4.0"
    assert report.duplicates == ["Near duplicate"]
    assert [title for title, _reason in report.failed] == ["Gone"]
    assert len(Corpus.load(tmp_path / "c").manifest.documents) == 2
