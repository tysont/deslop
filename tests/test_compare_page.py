# ABOUTME: Tests for the HTML comparison page: sentence alignment (same, changed, moved, new),
# ABOUTME: notes pinned to paragraphs, the candidate selector, and a self-contained document.
from __future__ import annotations

import json

from deslop.compare_page import align_sentences, build_page, page_data, word_diff


def test_sentences_align_by_content_and_mark_moves_and_additions():
    source = "The miller closed his gate at six. His daughter watched the window. A bird sang."
    candidate = (
        "His daughter watched the window. The miller shut his gate at six. "
        "Nobody mentioned the storm. A bird sang."
    )
    aligned = align_sentences(source, candidate)
    assert [s.status for s in aligned] == ["same", "moved", "new", "same"]
    assert aligned[1].source == 0 and aligned[0].source == 1
    assert ("delete", "closed") in aligned[1].words and ("insert", "shut") in aligned[1].words
    assert word_diff("a b c", "a x c") == [
        ("equal", "a "),
        ("delete", "b"),
        ("insert", "x"),
        ("equal", " c"),
    ]


def test_page_is_self_contained_and_pins_notes(fixture_voice, slop_text):
    from tests.test_apply import Scripted, apply, factory_for

    a = Scripted("anthropic")
    result = apply(slop_text, fixture_voice, n=2, report_all=True, provider_factory=factory_for(a))
    data = page_data(
        json.loads(result.to_json()), reference="Reference paragraph one.\n\nAnd paragraph two."
    )
    assert data["reference"] == ["Reference paragraph one.", "And paragraph two."]
    assert len(data["candidates"]) == 2 and data["header"]["selected"] == 1
    cand = data["candidates"][0]
    assert cand["sentences"] and any(s["status"] == "changed" for s in cand["sentences"])
    assert any(n["kind"] == "judge" for n in cand["notes"])
    page = build_page(json.loads(result.to_json()), reference="Reference paragraph one.")
    assert page.startswith("<!doctype html>") and "http" not in page.split("<body>")[0]
    assert "<script src" not in page and "<link" not in page
    for needle in (
        "Human reference",
        "Input",
        "Candidate",
        'id="select"',
        'id="clean"',
        "read clean",
    ):
        assert needle in page
    assert "Reference paragraph one." in page


def test_kept_original_without_report_all_shows_no_candidate(fixture_voice, slop_text):
    from tests.test_apply import Scripted, apply, factory_for

    result = apply(
        slop_text,
        fixture_voice,
        n=1,
        provider_factory=factory_for(Scripted("anthropic", lexical_break=True)),
    )
    data = page_data(json.loads(result.to_json()), reference=None)
    assert data["candidates"] == [] and data["header"]["kept_original"]
    assert "no reference" in build_page(json.loads(result.to_json()))
