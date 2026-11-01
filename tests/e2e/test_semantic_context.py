# ABOUTME: Replays the semantic lock on the paragraph that cost candidate 2 of the v2.3.2
# ABOUTME: explainer run: with the neighbouring paragraphs shown, the judge no longer flags it.
from __future__ import annotations

import json
from pathlib import Path

from deslop.providers.base import DEFAULT_JUDGE
from deslop.semantic import check_meaning

FIXTURE = Path(__file__).parent.parent / "fixtures" / "explainer_candidate2_context.json"


def test_the_stearin_paragraph_passes_once_the_previous_paragraph_is_in_view(provider_factory):
    fx = json.loads(FIXTURE.read_text(encoding="utf-8"))
    judge = provider_factory("anthropic")
    judge.default_model = DEFAULT_JUDGE
    alone = check_meaning(fx["original"], fx["candidate"], fx["propositions"], judge)
    assert not alone.ok, "the defect: paragraph-local entailment flags tallow and stearin"
    assert any("tallow" in a.text.lower() or "stearin" in a.text.lower() for a in alone.added)
    seen = check_meaning(
        fx["original"],
        fx["candidate"],
        fx["propositions"],
        judge,
        context=(fx["original_before"], fx["original_after"]),
    )
    assert seen.ok, seen.failing
    assert all(c.kept for c in seen.checks)
