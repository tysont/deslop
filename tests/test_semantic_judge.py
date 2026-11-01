# ABOUTME: Tests for the semantic lock parsing and the judges: the pairwise comparison and
# ABOUTME: its calibration, plus the absolute rubric kept as a secondary score.
from __future__ import annotations

import json

from deslop.judge import (
    JudgeCalibration,
    beats_original,
    calibrate_judge,
    compare,
    compare_with,
    judge_paragraph,
    win_rate,
)
from deslop.providers.base import CachingProvider, Completion, Usage
from deslop.semantic import check_meaning, extract_propositions, parse_numbered


class Canned(CachingProvider):
    def __init__(self, name: str, answer: str):
        super().__init__(use_cache=False, cache_only=False)
        self.name = name
        self.default_model = f"{name}-model"
        self.answer = answer
        self.prompts: list[str] = []

    def _raw_complete(self, system: str, user: str, model: str, temperature: float) -> Completion:
        self.prompts.append(user)
        return Completion(text=self.answer, model=model, provider=self.name, usage=Usage(calls=1))


class PicksMarked(CachingProvider):
    """A judge that picks whichever paragraph starts with 'Plainly,' as the writer's."""

    def __init__(self, name: str = "openai", confidence: float = 0.9):
        super().__init__(use_cache=False, cache_only=False)
        self.name = name
        self.default_model = f"{name}-model"
        self.confidence = confidence

    def _raw_complete(self, system: str, user: str, model: str, temperature: float) -> Completion:
        a = user.split("PARAGRAPH A:\n", 1)[1].split("\n\nPARAGRAPH B:", 1)[0]
        pick = "A" if a.startswith("Plainly,") else "B"
        other = "B" if pick == "A" else "A"
        text = json.dumps(
            {
                "more_like_writer": pick,
                "more_like_model": other,
                "confidence": self.confidence,
                "note": "the marked one",
            }
        )
        return Completion(text=text, model=model, provider=self.name, usage=Usage(calls=1))


def test_parse_numbered_accepts_dots_parens_and_bullets():
    text = "1. First claim.\n2) Second claim.\n- Third claim.\n\nnot a line"
    assert parse_numbered(text) == ["First claim.", "Second claim.", "Third claim."]


def test_extract_propositions_falls_back_to_lines():
    p = Canned("anthropic", "The chapter withholds its answer.\nAbraham obeys.")
    assert extract_propositions("text", p) == [
        "The chapter withholds its answer.",
        "Abraham obeys.",
    ]


def test_check_meaning_reports_dropped_and_added_claims():
    answer = json.dumps(
        {
            "original": [{"n": 1, "kept": True}, {"n": 2, "kept": False, "note": "gone"}],
            "added": [
                {"text": "Zeus fought his father", "kind": "claim"},
                {"text": "Consider this", "kind": "reader_address"},
                {"text": "Gen 6:5", "kind": "claim"},
            ],
        }
    )
    p = Canned("openai", answer)
    result = check_meaning("orig", "cand", ["A", "B"], p)
    assert not result.ok
    assert result.failing == ["dropped: B", "added: Zeus fought his father"]
    assert [a.kind for a in result.added] == ["claim", "reader_address", "citation"]


def test_check_meaning_without_json_fails_every_proposition():
    p = Canned("openai", "I cannot answer")
    result = check_meaning("orig", "cand", ["A"], p)
    assert not result.ok and result.checks[0].note == "judge gave no JSON"


def test_compare_reports_from_the_candidates_side_whatever_the_order(fixture_voice):
    judge = PicksMarked()
    v = compare("Plainly, the ark.", "The ark.", fixture_voice, judge, exemplars=["x", "y"])
    assert v.wins and not v.picked_as_model and v.confidence == 0.9
    assert v.provider == "openai" and v.against == "original"
    v2 = compare("The ark.", "Plainly, the ark.", fixture_voice, judge, exemplars=["x"])
    assert not v2.wins and v2.picked_as_model
    assert {v.shown_as, v2.shown_as} <= {"A", "B"}


def test_compare_without_json_is_a_loss_at_coin_toss(fixture_voice):
    v = compare("a", "b", fixture_voice, Canned("openai", "no"), exemplars=[])
    assert not v.wins and v.confidence == 0.5 and "no JSON" in v.note


def test_win_rate_and_gate():
    pairs = [
        compare("Plainly, x.", "x.", _voice(), PicksMarked(confidence=0.8), exemplars=[]),
        compare(
            "Plainly, x.",
            "Plainly, y.",
            _voice(),
            PicksMarked(),
            exemplars=[],
            against="candidate 2",
        ),
    ]
    assert win_rate(pairs) in (0.5, 1.0)
    assert beats_original(pairs, 0.7)
    assert not beats_original(pairs, 0.85)
    assert win_rate([]) is None and not beats_original([])


def test_compare_with_several_judges_needs_agreement(fixture_voice):
    agree = compare_with(
        "Plainly, x.", "x.", fixture_voice, [PicksMarked("a"), PicksMarked("b", 0.7)], exemplars=[]
    )
    assert agree.wins and agree.confidence == 0.8 and agree.provider == "a+b"
    disagree = compare_with(
        "Plainly, x.", "x.", fixture_voice, [PicksMarked("a"), Canned("b", "{}")], exemplars=[]
    )
    assert not disagree.wins


def test_calibrate_judge_measures_accuracy(fixture_voice):
    corpus = [f"Plainly, corpus paragraph {i} about the ark." for i in range(6)]
    llm = [f"Model paragraph {i} about the ark." for i in range(6)]
    cal = calibrate_judge(
        fixture_voice, PicksMarked(), llm, corpus, exemplars=["e1", "e2"], pairs=4
    )
    assert cal.pairs == 4 and cal.accuracy == 1.0 and cal.model_prose_accuracy == 1.0
    assert cal.usable and cal.provider == "openai"
    assert JudgeCalibration.model_validate(cal.model_dump()).accuracy == 1.0
    bad = calibrate_judge(fixture_voice, Canned("openai", "{}"), llm, corpus, exemplars=[], pairs=4)
    assert bad.accuracy == 0.0 and not bad.usable


def test_judge_paragraph_clamps_and_computes_overall(fixture_voice):
    p = Canned("openai", '{"voice": 9, "model_prose": 0, "fidelity": 4, "note": "fine"}')
    scores = judge_paragraph("text", fixture_voice, p, exemplars=["a", "b"], propositions=["x"])
    assert (scores.voice, scores.model_prose, scores.fidelity) == (5, 1, 4)
    assert scores.overall == round((5 + 5 + 4) / 3, 2)
    assert "PROPOSITIONS" in p.prompts[0] and "SAMPLE 2" in p.prompts[0]


def test_judge_paragraph_without_json_is_the_worst_score(fixture_voice):
    scores = judge_paragraph("text", fixture_voice, Canned("openai", "nope"), exemplars=[])
    assert scores.overall == 1.0 and "no JSON" in scores.note


def _voice():
    from deslop.stats import MetricsProfile
    from deslop.voice.model import Lexicon, Voice, VoiceMeta

    return Voice(
        meta=VoiceMeta(
            name="t", deslop_version="0", corpus_name="t", corpus_hash="", total_words=0
        ),
        metrics=MetricsProfile(),
        lexicon=Lexicon(),
        rules="",
        profile="A voice.",
        exemplars="",
    )
