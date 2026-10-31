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


def test_slop_sample_converts_to_pass(fixture_voice, slop_text, tmp_path):
    res = convert(
        slop_text, fixture_voice, mode="edit", max_iter=4, pairs_path=tmp_path / "pairs.jsonl"
    )
    assert res.report_before.verdict == "FAIL"
    assert res.report_after.verdict == "PASS", res.report_after.fixes
    assert len(res.iterations) <= 4
    # Lock inventory identical before/after for the whole text.
    before, after = inventory(slop_text), inventory(res.text)
    assert before.citations == after.citations
    assert before.quotes == after.quotes
    assert before.numbers == after.numbers
    assert set(before.proper_nouns) <= set(after.proper_nouns) | set(res.text.split())
    # Unflagged sentences byte-identical: every rejected or untouched unit still present.
    touched = {u.before for u in res.units if u.accepted}
    for u in res.units:
        if u.before not in touched:
            assert u.before in res.text
    assert res.cost.calls > 0
    assert "--- before" in res.diff
