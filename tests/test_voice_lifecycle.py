# ABOUTME: Tests for voice update (snapshot + recalibration from accepted files), @prev/@stamp
# ABOUTME: references, voice diff, and voice validate.
import shutil
from pathlib import Path

import pytest

from deslop.corpus import Corpus
from deslop.voice import Voice
from deslop.voice.diff import diff_voices
from deslop.voice.update import resolve_ref, snapshot, update
from deslop.voice.validate import validate

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def workspace(fixture_voice, tmp_path):
    corpus_dir = tmp_path / "corpus"
    shutil.copytree(FIXTURES / "corpus_small", corpus_dir)
    voice_dir = tmp_path / "fixture.voice"
    v = fixture_voice.model_copy(deep=True)
    v.meta.corpus_path = str(corpus_dir)
    v.save(voice_dir)
    accepted = tmp_path / "accepted.md"
    accepted.write_text((FIXTURES / "heldout_hazlitt.md").read_text(encoding="utf-8"))
    return voice_dir, corpus_dir, accepted


def test_snapshot_copies_everything_but_history(workspace):
    voice_dir, _, _ = workspace
    first = snapshot(voice_dir)
    assert first.parent == voice_dir / "history"
    assert (first / "metrics.json").exists() and not (first / "history").exists()
    second = snapshot(voice_dir)
    assert second != first and not (second / "history").exists()


def test_update_adds_accepted_docs_and_keeps_prose_files(workspace):
    voice_dir, corpus_dir, accepted = workspace
    before = Voice.load(voice_dir)
    res = update(voice_dir, [accepted], weight=0.5)
    after = Voice.load(voice_dir)
    assert res.added == [accepted.name] and res.skipped == []
    corpus = Corpus.load(corpus_dir)
    added = [d for d in corpus.manifest.documents if d.weight == 0.5]
    assert len(added) == 1 and added[0].author.endswith("(accepted)")
    assert after.meta.corpus_hash == corpus.content_hash() != before.meta.corpus_hash
    assert after.meta.total_words > before.meta.total_words
    assert after.profile == before.profile
    assert after.rules == before.rules
    assert after.exemplars == before.exemplars
    assert (
        after.metrics.metrics["sent_len_mean"].mean != before.metrics.metrics["sent_len_mean"].mean
    )
    assert res.snapshot.exists()
    assert Voice.load(res.snapshot).meta.corpus_hash == before.meta.corpus_hash
    # re-running with the same file skips the duplicate
    again = update(voice_dir, [accepted])
    assert again.skipped == [accepted.name] and again.added == []


def test_update_regenerates_prose_with_template_when_asked(workspace):
    voice_dir, _, accepted = workspace
    before = Voice.load(voice_dir)
    update(voice_dir, [accepted], regenerate=True, use_llm=False)
    after = Voice.load(voice_dir)
    assert after.rules != before.rules or after.exemplars != before.exemplars


def test_resolve_ref_prev_and_timestamp(workspace):
    voice_dir, _, accepted = workspace
    assert resolve_ref(str(voice_dir)) == voice_dir
    with pytest.raises(FileNotFoundError):
        resolve_ref(f"{voice_dir}@prev")
    res = update(voice_dir, [accepted])
    assert resolve_ref(f"{voice_dir}@prev") == res.snapshot
    stamp = res.snapshot.name
    assert resolve_ref(f"{voice_dir}@{stamp}") == res.snapshot
    assert resolve_ref(f"{voice_dir}@{stamp[:8]}") == res.snapshot
    with pytest.raises(FileNotFoundError):
        resolve_ref(f"{voice_dir}@1999")


def test_diff_reports_metric_rule_lexicon_and_share_changes(workspace):
    voice_dir, _, accepted = workspace
    res = update(voice_dir, [accepted])
    a, b = Voice.load(res.snapshot), Voice.load(voice_dir)
    d = diff_voices(a, b)
    names = {m.name for m in d.metrics}
    assert "sent_len_mean" in names
    moved = [m for m in d.metrics if m.name == "sent_len_mean"][0]
    assert moved.before != moved.after and moved.direction in (
        "toward_slop",
        "away_from_slop",
        "neutral",
    )
    assert any(author.endswith("(accepted)") for author in d.author_shares)
    assert d.words_before < d.words_after
    assert d.rules_diff == ""  # rules.md untouched by update
    assert isinstance(d.encouraged_added, list) and isinstance(d.banned_added, list)
    assert d.to_json().startswith("{")
    same = diff_voices(a, a)
    assert all(m.delta == 0 for m in same.metrics) and not same.changed


def test_validate_passes_on_fresh_voice_and_fails_on_damage(workspace):
    voice_dir, corpus_dir, _ = workspace
    result = validate(voice_dir)
    assert result.ok, [c for c in result.checks if not c.ok]
    assert {c.name for c in result.checks} == {"schema", "corpus_hash", "exemplars_self_score"}
    (corpus_dir / "extra.txt").write_text("More words in the corpus now.")
    corpus = Corpus.load(corpus_dir)
    corpus.add(str(corpus_dir / "extra.txt"), author="X")
    result = validate(voice_dir)
    assert not result.ok and not result.by_name("corpus_hash").ok
    (voice_dir / "metrics.json").write_text("{")
    result = validate(voice_dir)
    assert not result.by_name("schema").ok and not result.ok


def test_validate_skips_hash_when_corpus_unknown(workspace):
    voice_dir, _, _ = workspace
    v = Voice.load(voice_dir)
    v.meta.corpus_path = None
    v.save(voice_dir)
    result = validate(voice_dir)
    assert result.ok and "skipped" in result.by_name("corpus_hash").detail
    result = validate(voice_dir, corpus=Path("/nonexistent"))
    assert not result.by_name("corpus_hash").ok
