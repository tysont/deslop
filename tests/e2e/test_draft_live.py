# ABOUTME: Live end-to-end test: drafts the sample brief with real providers (cached on disk).
# ABOUTME: Skipped unless a provider key is configured. Encodes the Phase 3 draft definition of done.
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from deslop.draft import draft, load_brief

load_dotenv(override=True)
HAS_KEY = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY"))
pytestmark = pytest.mark.skipif(not HAS_KEY, reason="no provider credentials")
FIXTURES = Path(__file__).parent.parent / "fixtures"


def test_sample_brief_drafts_to_pass_within_two_iterations(fixture_voice, tmp_path):
    brief = load_brief(FIXTURES / "brief_sample.yaml")
    res = draft(brief, fixture_voice, convert_max_iter=2, pairs_path=tmp_path / "pairs.jsonl")
    assert all(s.complete for s in res.sections), [s.missing for s in res.sections]
    for section in brief.sections:
        for quote in section.quotes:
            assert quote in res.text, quote
        for ref in section.refs:
            assert ref in res.text, ref
    assert res.report.verdict == "PASS", res.report.fixes
    assert len(res.convert.iterations) <= 2
    assert res.cost.calls + res.cost.cached > 0
