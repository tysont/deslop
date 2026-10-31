# ABOUTME: Tests for the semantic lock and the rubric judge using a scripted provider, so the
# ABOUTME: parsing, exemptions and gating logic are exercised without a network.
from __future__ import annotations

from deslop.judge import judge_paragraph
from deslop.providers.base import CachingProvider, Completion, Usage
from deslop.semantic import check_meaning, extract_propositions, parse_numbered


class Scripted(CachingProvider):
    name = "scripted"
    default_model = "fake-1"

    def __init__(self, answers: dict[str, str]):
        super().__init__(use_cache=False, cache_only=False)
        self.answers = answers
        self.calls: list[str] = []

    def _raw_complete(self, system: str, user: str, model: str, temperature: float) -> Completion:
        self.calls.append(system[:40])
        for key, answer in self.answers.items():
            if key in system:
                return Completion(
                    text=answer, model=model, provider=self.name, usage=Usage(calls=1)
                )
        raise AssertionError("no scripted answer")


def test_parse_numbered_accepts_common_list_shapes():
    assert parse_numbered("1. Noah builds.\n2) God grieves.\n- The ark is wood.") == [
        "Noah builds.",
        "God grieves.",
        "The ark is wood.",
    ]


def test_extract_and_check_meaning_with_exemptions():
    p = Scripted(
        {
            "list the propositions": "1. Noah builds an ark.\n2. God grieves over creation.",
            "compare an original passage": (
                'Here: {"original": [{"n": 1, "kept": true, "note": ""}, '
                '{"n": 2, "kept": false, "note": "grief is gone"}], '
                '"added": [{"text": "Consider the timber.", "kind": "reader_address"}, '
                '{"text": "Gen 6:14", "kind": "claim"}, '
                '{"text": "The ark had three decks.", "kind": "claim"}]}'
            ),
        }
    )
    props = extract_propositions("Noah builds an ark because God grieves.", p)
    assert props == ["Noah builds an ark.", "God grieves over creation."]
    result = check_meaning("orig", "cand", props, p)
    assert not result.ok
    assert result.failing == [
        "dropped: God grieves over creation.",
        "added: The ark had three decks.",
    ]
    kinds = {a.text: a.kind for a in result.added}
    assert kinds["Gen 6:14"] == "citation" and kinds["Consider the timber."] == "reader_address"


def test_check_meaning_without_json_fails_closed():
    p = Scripted({"compare an original passage": "I cannot say."})
    result = check_meaning("o", "c", ["A claim."], p)
    assert not result.ok and result.checks[0].note == "judge gave no JSON"


def test_judge_scores_and_overall(fixture_voice):
    p = Scripted(
        {"demanding editor": '{"voice": 4, "model_prose": 2, "fidelity": 5, "note": "fine"}'}
    )
    scores = judge_paragraph(
        "Some paragraph.", fixture_voice, p, exemplars=["s1", "s2"], propositions=["x"]
    )
    assert (scores.voice, scores.model_prose, scores.fidelity) == (4, 2, 5)
    assert scores.overall == 4.33 and scores.provider == "scripted"
    bad = Scripted({"demanding editor": "no json here"})
    worst = judge_paragraph("p", fixture_voice, bad, exemplars=[])
    assert worst.overall == 1.0 and "no JSON" in worst.note
