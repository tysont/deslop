# ABOUTME: End-to-end tests: calibrate the fixture corpus, save/load the voice, score texts.
# ABOUTME: Encodes the Phase 1 definition of done.
from collections import Counter

from deslop import Voice, score
from deslop.voice.calibrate import recalibrate_metrics
from deslop.voice.exemplars import parse_exemplars


def test_voice_round_trip(fixture_voice, tmp_path):
    path = fixture_voice.save(tmp_path / "rt.voice")
    assert {p.name for p in path.iterdir()} == {
        "voice.yaml",
        "metrics.json",
        "lexicon.json",
        "rules.md",
        "profile.md",
        "exemplars.md",
    }
    loaded = Voice.load(path)
    assert loaded.meta == fixture_voice.meta
    assert loaded.metrics == fixture_voice.metrics
    assert loaded.lexicon == fixture_voice.lexicon
    assert loaded.rules == fixture_voice.rules


def test_voice_meta(fixture_voice):
    assert fixture_voice.meta.total_words > 50000
    assert set(fixture_voice.meta.author_shares) == {
        "Samuel Johnson",
        "William Hazlitt",
        "Matthew Arnold",
        "A. C. Bradley",
    }
    assert fixture_voice.meta.profile_source == "template"
    assert len(fixture_voice.metrics.per_author) == 4


def test_lexicon_content(fixture_voice):
    lex = fixture_voice.lexicon
    assert 100 <= len(lex.encouraged_words) <= 500
    terms = {e.term for e in lex.encouraged_words}
    assert "shakespeare" in terms or "poet" in terms or "criticism" in terms
    assert all(e.log_odds > 0 for e in lex.encouraged_words)
    assert len(lex.patterns) == 14


def test_rules_and_profile_and_exemplars(fixture_voice):
    assert (
        "## Default" in fixture_voice.rules and "## Derived from this voice" in fixture_voice.rules
    )
    assert fixture_voice.profile.startswith("# Voice: fixture")
    bodies = parse_exemplars(fixture_voice.exemplars)
    assert bodies
    total = sum(len(b.split()) for b in bodies)
    assert 1500 <= total <= 3000
    authors = Counter(
        line for line in fixture_voice.exemplars.splitlines() if line.startswith("## ")
    )
    assert len(authors) >= 2


def test_exemplars_self_score_pass(fixture_voice):
    bodies = parse_exemplars(fixture_voice.exemplars)
    report = score("\n\n".join(bodies), fixture_voice)
    assert report.verdict == "PASS", report.fixes


def test_slop_sample_fails_with_t1_and_t5(fixture_voice, slop_text):
    report = score(slop_text, fixture_voice)
    assert report.verdict == "FAIL"
    t1 = next(t for t in report.tells if t.id == "T1")
    assert t1.count >= 8 and t1.flagged
    t5 = next(t for t in report.tells if t.id == "T5")
    assert {h.para_idx for h in t5.hits} == set(range(report.paragraphs_count))
    assert report.fixes and report.fixes[0].startswith("T1")
    assert report.distance > 20


def test_heldout_passes(fixture_voice, heldout_texts):
    for text in heldout_texts:
        report = score(text, fixture_voice)
        assert report.verdict == "PASS", report.fixes
        assert report.distance < 20


def test_report_outputs(fixture_voice, slop_text):
    report = score(slop_text, fixture_voice)
    data = report.model_dump()
    assert data["verdict"] == "FAIL" and "metrics" in data and "tells" in data
    pretty = report.pretty()
    assert "FAIL" in pretty and "Metrics" in pretty and "Fix first" in pretty
    annotated = report.annotate(slop_text)
    assert "[T1:asks]" in annotated
    assert annotated.replace("[", "").count("chapter asks") == slop_text.count("chapter asks")
    import re

    assert re.sub(r"\[T\d+[^\]]*\]", "", annotated) == slop_text


def test_recalibrate_without_corpus(fixture_voice):
    v2 = recalibrate_metrics(fixture_voice, max_share=1.0)
    assert set(v2.metrics.metrics) == set(fixture_voice.metrics.metrics)
    assert v2.metrics.total_words == fixture_voice.metrics.total_words


def test_score_options_change_verdict(fixture_voice, slop_text):
    lenient = score(slop_text, fixture_voice, sigma=50, tell_mult=100, tell_floor=1000)
    assert lenient.verdict == "PASS"
