# ABOUTME: End-to-end test: drafts the sample brief with real providers, replayed from
# ABOUTME: tests/cache when no credentials are present. Encodes the Phase 3 draft definition of done.
from pathlib import Path

from deslop.draft import draft, load_brief

FIXTURES = Path(__file__).parent.parent / "fixtures"


def test_sample_brief_drafts_to_pass_within_two_iterations(
    fixture_voice, tmp_path, provider_factory
):
    brief = load_brief(FIXTURES / "brief_sample.yaml")
    res = draft(
        brief,
        fixture_voice,
        convert_max_iter=2,
        pairs_path=tmp_path / "pairs.jsonl",
        provider_factory=provider_factory,
    )
    assert all(s.complete for s in res.sections), [s.missing for s in res.sections]
    for section in brief.sections:
        for quote in section.quotes:
            assert quote in res.text, quote
        for ref in section.refs:
            assert ref in res.text, ref
    assert res.report.verdict == "PASS", res.report.fixes
    assert len(res.convert.iterations) <= 2
    assert res.cost.calls + res.cost.cached > 0
