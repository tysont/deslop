# ABOUTME: Live end-to-end test: converts the slop sample with real providers (cached on disk).
# ABOUTME: Skipped unless an Anthropic or OpenAI key is configured. Encodes the Phase 2 definition of done.
import os

import pytest
from dotenv import load_dotenv

from deslop import convert
from deslop.lock import inventory
from deslop.nlp import analyze

load_dotenv(override=True)
HAS_KEY = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY"))

pytestmark = pytest.mark.skipif(not HAS_KEY, reason="no provider credentials")


def check_locks_and_untouched(res, source):
    before, after = inventory(source), inventory(res.text)
    assert before.citations == after.citations
    assert before.quotes == after.quotes
    assert before.numbers == after.numbers
    assert set(before.proper_nouns) <= set(after.proper_nouns) | set(res.text.split())
    # Every source sentence outside an accepted unit survives byte-identical.
    accepted = [u.before for u in res.units if u.accepted]
    for sent in analyze(source).sentences:
        if not any(sent.text in before for before in accepted):
            assert sent.text in res.text, sent.text
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


def test_slop_sample_converts_to_pass_in_edit_mode(fixture_voice, slop_text, tmp_path):
    # Adjacent flagged sentences (including monotone runs) are rewritten as one passage, so
    # edit mode can restore rhythm while every sentence outside a unit stays byte-identical.
    res = convert(
        slop_text, fixture_voice, mode="edit", max_iter=4, pairs_path=tmp_path / "pairs.jsonl"
    )
    assert res.report_after.verdict == "PASS", res.report_after.fixes
    check_locks_and_untouched(res, slop_text)
