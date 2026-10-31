# ABOUTME: Live end-to-end test: converts the slop sample with real providers (cached on disk).
# ABOUTME: Skipped unless an Anthropic or OpenAI key is configured. Encodes the Phase 2 definition of done.
import os

import pytest
from dotenv import load_dotenv

from deslop import convert
from deslop.lock import inventory

load_dotenv(override=True)
HAS_KEY = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY"))

pytestmark = pytest.mark.skipif(not HAS_KEY, reason="no provider credentials")


def check_locks_and_untouched(res, source):
    before, after = inventory(source), inventory(res.text)
    assert before.citations == after.citations
    assert before.quotes == after.quotes
    assert before.numbers == after.numbers
    assert set(before.proper_nouns) <= set(after.proper_nouns) | set(res.text.split())
    touched = {u.before for u in res.units if u.accepted}
    for u in res.units:
        if u.before not in touched:
            assert u.before in res.text
    assert res.cost.calls + res.cost.cached > 0
    assert "--- before" in res.diff


def test_slop_sample_converts_to_pass_in_rewrite_mode(fixture_voice, slop_text, tmp_path):
    res = convert(
        slop_text, fixture_voice, mode="rewrite", max_iter=4, pairs_path=tmp_path / "pairs.jsonl"
    )
    assert res.report_before.verdict == "FAIL"
    assert res.report_after.verdict == "PASS", res.report_after.fixes
    assert len(res.iterations) <= 4
    check_locks_and_untouched(res, slop_text)


def test_edit_mode_clears_every_tell(fixture_voice, slop_text, tmp_path):
    # Sentence-level edits cannot restore sentence-length variance, so edit mode is judged
    # on tells: none may remain flagged, and every unflagged sentence stays byte-identical.
    res = convert(
        slop_text, fixture_voice, mode="edit", max_iter=4, pairs_path=tmp_path / "pairs.jsonl"
    )
    assert res.report_after.flagged_tells() == [], res.report_after.fixes
    assert res.report_after.distance < 15
    check_locks_and_untouched(res, slop_text)
