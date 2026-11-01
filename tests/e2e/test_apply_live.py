# ABOUTME: End-to-end tests: apply the fixture voice with real providers, replayed from
# ABOUTME: tests/cache without credentials. Encodes the v2 definition of done for apply.
from deslop import apply, load_brief
from deslop.lock import DocumentLock, check
from deslop.nlp import analyze


def check_locks(res, source):
    document = DocumentLock.build(source)
    for unit in res.units:
        if unit.kept_original:
            continue
        result = check(unit.original, unit.final, document)
        assert result.ok, (unit.index, result.violations)
    assert res.cost.calls + res.cost.cached > 0
    assert "--- before" in res.diff


def test_slop_sample_rewrite_mode_passes_locks_judge_and_linter(
    fixture_voice, slop_text, tmp_path, provider_factory
):
    res = apply(
        slop_text,
        fixture_voice,
        mode="rewrite",
        n=3,
        pairs_path=tmp_path / "pairs.jsonl",
        provider_factory=provider_factory,
    )
    assert res.report_before.verdict == "FAIL"
    rewritten = [u for u in res.units if not u.kept_original]
    assert rewritten, "at least one paragraph was rewritten"
    for u in rewritten:
        chosen = u.candidates[u.chosen]
        assert chosen.semantic is not None and chosen.semantic.ok
        assert chosen.beats_original and chosen.confidence_vs_original >= res.min_confidence
        assert all(p.provider != chosen.provider for p in chosen.pairs)
    assert res.report_after.distance < res.report_before.distance
    assert "✓" in res.meaning_diff
    check_locks(res, slop_text)


def test_slop_sample_edit_mode_keeps_unflagged_sentences(
    fixture_voice, slop_text, tmp_path, provider_factory
):
    res = apply(
        slop_text,
        fixture_voice,
        mode="edit",
        max_iter=4,
        pairs_path=tmp_path / "pairs.jsonl",
        provider_factory=provider_factory,
    )
    assert res.report_after.distance <= res.report_before.distance
    accepted = [u.original for u in res.units if not u.kept_original]
    for sent in analyze(slop_text).sentences:
        if not any(sent.text in before for before in accepted):
            assert sent.text in res.text, sent.text
    check_locks(res, slop_text)


def test_sample_brief_is_drafted_then_rewritten(
    fixture_voice, fixtures_dir, tmp_path, provider_factory
):
    brief = load_brief(fixtures_dir / "brief_sample.yaml")
    res = apply(
        brief,
        fixture_voice,
        n=2,
        pairs_path=tmp_path / "pairs.jsonl",
        provider_factory=provider_factory,
    )
    assert res.draft is not None and len(res.draft) == len(brief.sections)
    assert res.draft_text and res.text
    for section in brief.sections:
        for ref in section.refs:
            assert ref in res.text, ref
    assert res.report_after.distance <= 100
