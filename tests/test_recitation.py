# ABOUTME: Tests for the recitation check: runs of eight or more words a candidate shares with
# ABOUTME: a reference text but not with its input, their count, coverage and longest spans.
from __future__ import annotations

from deslop.recitation import NGRAM, RecitationResult, recitation

REFERENCE = (
    "The candle-wood is a hard, strong, excellent wood, evidently fitted for good work as a "
    "resister of force, and yet withal burning so well that where it is found the people make "
    "splinters of it, and torches. Here is a candle which Colonel Pasley is said to have taken "
    "out of the Royal George."
)
INPUT = (
    "Candle-wood is a hard, strong, excellent wood, well suited to resisting force, and it also "
    "burns well. Colonel Pasley reportedly took a candle from the wreck of the Royal George."
)


def test_no_shared_runs_means_zero():
    r = recitation("A plain sentence about candles that recites nothing at all.", REFERENCE, INPUT)
    assert r == RecitationResult(words=10)
    assert r.share == 0.0 and r.spans == []


def test_a_recited_run_is_counted_covered_and_reported_as_a_span():
    candidate = (
        "The wood is evidently fitted for good work as a resister of force, and yet withal "
        "burning so well that the people make torches of it."
    )
    r = recitation(candidate, REFERENCE, INPUT)
    # "evidently fitted ... so well that" is 17 words: ten 8-grams, all covered
    assert r.ngrams == 10
    assert r.covered == 17
    assert r.words == 26
    assert r.share == 100 * 17 / 26
    assert r.spans == [
        "evidently fitted for good work as a resister of force, and yet withal burning so well that"
    ]


def test_runs_also_present_in_the_input_do_not_count():
    # eight words shared with the reference that the input also contains
    reference = "one two three four five six seven eight nine ten"
    candidate = "say one two three four five six seven eight nine ten now"
    assert recitation(candidate, reference, reference).ngrams == 0
    assert recitation(candidate, reference, "nothing shared").ngrams == 3


def test_matching_ignores_case_and_punctuation():
    reference = "Keep, the AUTHOR'S tense; person and 'argumentative' order unless told otherwise."
    candidate = "keep the authors tense person and argumentative order unless told otherwise"
    assert recitation(candidate, reference, "").ngrams == len(candidate.split()) - NGRAM + 1


def test_spans_are_the_longest_three_in_candidate_text():
    runs = [
        " ".join(f"a{i}" for i in range(12)),
        " ".join(f"b{i}" for i in range(9)),
        " ".join(f"c{i}" for i in range(10)),
        " ".join(f"d{i}" for i in range(8)),
    ]
    reference = ". ".join(runs)
    candidate = " x y z ".join(runs)
    r = recitation(candidate, reference, "")
    assert [len(s.split()) for s in r.spans] == [12, 10, 9]
    assert [len(s.split()) for s in r.runs] == [12, 9, 10, 8]


def test_result_serialises_for_the_run_log():
    r = recitation(
        "eight words shared with the reference text here",
        "eight words shared with the reference text here",
        "",
    )
    assert r.model_dump() == {
        "ngrams": 1,
        "covered": 8,
        "words": 8,
        "share": 100.0,
        "spans": ["eight words shared with the reference text here"],
        "runs": ["eight words shared with the reference text here"],
    }
