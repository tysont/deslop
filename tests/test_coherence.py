# ABOUTME: Tests for the deterministic coherence checks: tense per paragraph, sentence-initial
# ABOUTME: pronouns without an antecedent, and the share of connective-opened sentences.
from __future__ import annotations

import pytest

from deslop.coherence import (
    antecedent_flags,
    check_coherence,
    connective_check,
    paragraph_tense,
    tense_drift,
)
from deslop.nlp import analyze
from deslop.stats import Band

PAST = (
    "The road ran north through the valley. Nobody walked it after dark. "
    "The miller closed his gate at six and lit the lamp.\n\n"
    "His daughter watched the window. She counted the carts that passed."
)
PRESENT = (
    "The road runs north through the valley. Nobody walks it after dark. "
    "The miller closes his gate at six and lights the lamp.\n\n"
    "His daughter watched the window. She counted the carts that passed."
)


def test_paragraph_tense_is_the_dominant_finite_tense():
    a = analyze(PAST)
    assert paragraph_tense(a.paragraphs[0]) == "past"
    assert paragraph_tense(analyze(PRESENT).paragraphs[0]) == "present"
    assert paragraph_tense(analyze("Nothing finite here, only gerunds.").paragraphs[0]) in (
        "present",
        "none",
    )


def test_tense_drift_names_the_paragraph_and_an_example():
    drift = tense_drift(analyze(PAST), analyze(PRESENT))
    assert len(drift) == 1
    d = drift[0]
    assert d.paragraph == 1 and d.before == "past" and d.after == "present"
    assert "runs" in d.example or "walks" in d.example
    assert tense_drift(analyze(PAST), analyze(PAST)) == []


def test_antecedent_flags_sentence_initial_pronouns_without_a_noun_phrase_behind_them():
    ok = analyze("The miller closed the gate. It was old and heavy.")
    assert antecedent_flags(ok) == []
    first = analyze("It was old and heavy. The miller closed the gate.")
    flags = antecedent_flags(first)
    assert len(flags) == 1 and flags[0].word == "It" and flags[0].sentence == 0
    far = analyze(
        "The miller closed the gate. Then, slowly, quietly. Yes, slowly. This surprised everyone."
    )
    assert [f.word for f in antecedent_flags(far)] == ["This"]


def test_connective_check_matches_the_original_and_the_band():
    before = analyze("He left. And he did not return. The house stood empty. Nobody came.")
    same = analyze("He went. But nobody saw him go. The house stayed empty. Nobody came.")
    more = analyze("He left. And he did not return. But the house stood empty. So nobody came.")
    c = connective_check(before, same, Band(low=0.0, high=60.0))
    assert c.ok and c.before == pytest.approx(0.25) and c.after == pytest.approx(0.25)
    c = connective_check(before, more, Band(low=0.0, high=60.0))
    assert not c.ok and "original" in c.note
    c = connective_check(before, same, Band(low=30.0, high=60.0))
    assert not c.ok and "band" in c.note
    none = analyze("He left. He did not return. The house stood empty. Nobody came.")
    one = analyze("He left. And he did not return. The house stood empty. Nobody came.")
    assert connective_check(none, one, None).ok  # one sentence of slack when the original has none


def test_check_coherence_collects_every_failure_as_a_reader_note():
    result = check_coherence(PAST, PRESENT, band=None)
    assert not result.ok
    assert result.paragraphs_before == 2 and result.paragraphs_after == 2
    notes = result.reader_notes()
    assert any("paragraph 1" in n and "past" in n for n in notes)
    good = check_coherence(PAST, PAST, band=None)
    assert good.ok and good.reader_notes() == []
    short = check_coherence(PAST, PAST.split("\n\n")[0], band=None)
    assert not short.ok and any("paragraphs" in n for n in short.reader_notes())
