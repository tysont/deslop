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
        self.systems: list[str] = []

    def _raw_complete(self, system: str, user: str, model: str, temperature: float) -> Completion:
        self.prompts.append(user)
        self.systems.append(system)
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


def test_extract_texture_is_a_short_note_or_nothing():
    from deslop.semantic import TEXTURE_SYSTEM, extract_texture

    p = Canned("anthropic", 'Second person ("you"); a rhetorical\nquestion; an exclamation.')
    assert (
        extract_texture("text", p)
        == 'Second person ("you"); a rhetorical question; an exclamation.'
    )
    assert "never" not in TEXTURE_SYSTEM and "quoted briefly" in TEXTURE_SYSTEM
    assert extract_texture("text", Canned("anthropic", "None.")) == ""
    assert extract_texture("text", Canned("anthropic", "  none ")) == ""


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
    result = check_meaning("orig", "cand", ["A", "B"], p, fidelity="strict")
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


def test_compare_document_reports_each_item_from_the_candidate_side(fixture_voice):
    from deslop.judge import DocumentVerdict, compare_document, document_win_rate

    answer = json.dumps(
        {
            "voice": "A",
            "coherent": "tie",
            "tense_person": "B",
            "order": "tie",
            "confidence": 0.8,
            "note": "A keeps the writer's cadence; B stays in one tense",
        }
    )
    judge = Canned("openai", answer)
    v = compare_document(
        "CANDIDATE text.", "ORIGINAL text.", "ORIGINAL text.", fixture_voice, judge, exemplars=["x"]
    )
    assert isinstance(v, DocumentVerdict) and v.against == "original"
    assert set(v.items) == {"voice", "coherent", "tense_person", "order"}
    assert set(v.items.values()) <= {"win", "tie", "loss"}
    assert sorted(v.items.values()).count("tie") == 2
    assert v.confidence == 0.8 and v.provider == "openai"
    assert v.losses == 1 and v.wins == 1 and not v.no_loss
    assert "THE ORIGINAL" in judge.prompts[0] and "TEXT A" in judge.prompts[0]
    # win or tie on every item passes; a loss anywhere fails
    clean = DocumentVerdict(against="original", items={k: "tie" for k in v.items}, confidence=0.6)
    assert clean.no_loss and clean.net == 0
    assert document_win_rate([]) is None
    assert document_win_rate([v, clean]) == 0.5  # one net loss (0), one tie (0.5)


def test_compare_document_without_json_is_a_loss_everywhere(fixture_voice):
    from deslop.judge import compare_document

    v = compare_document("a", "b", "b", fixture_voice, Canned("openai", "no idea"), exemplars=[])
    assert v.losses == 4 and v.note == "judge gave no JSON"


def test_content_fidelity_votes_on_each_flag_and_reports_cleared_items():
    from deslop.semantic import (
        MATERIALITY,
        REVIEW_QUESTION,
        REVIEW_SAMPLES,
        REVIEW_TEMPERATURE,
        check_meaning,
    )

    class TwoPass(CachingProvider):
        """The first reader flags a date, a restatement, a citation, the cattle sentence
        and a distant shore; the reviewers vote on each flag: the cattle sentence is cleared
        two to one, the shore and the date stand three to nothing."""

        def __init__(self):
            super().__init__(use_cache=False, cache_only=False)
            self.name, self.default_model, self.systems, self.prompts = "anthropic", "m", [], []
            self.temperatures: list[float] = []

        def _raw_complete(self, system, user, model, temperature):
            self.systems.append(system)
            self.prompts.append(user)
            self.temperatures.append(temperature)
            if "first reader flagged" in system:
                if "cattle" in user:
                    nth = sum("cattle" in p for p in self.prompts)
                    new, why = nth == 2, "the fields are sea, so nothing grows there"
                elif "shore" in user:
                    new, why = True, "no shore is named"
                else:
                    new = "1834" in user
                    why = "a date the original lacks" if new else "restates the sinking"
                text = json.dumps({"new": new, "why": why})
            else:
                text = json.dumps(
                    {
                        "original": [{"n": 1, "kept": True}],
                        "added": [
                            {
                                "sentence": "The lamp was lit in 1834.",
                                "text": "1834",
                                "category": "number",
                            },
                            {
                                "sentence": "It had gone down with the ship.",
                                "text": "went down with the ship",
                                "category": "event",
                            },
                            {"sentence": "See Gen 6:5.", "text": "Gen 6:5", "category": "entity"},
                            {
                                "sentence": "The cattle found nothing there to graze.",
                                "text": "cattle finding nothing to graze",
                                "category": "scene",
                            },
                            {
                                "sentence": "Across the water lay a distant shore.",
                                "text": "a distant shore",
                                "category": "scene",
                            },
                        ],
                    }
                )
            return Completion(text=text, model=model, provider=self.name, usage=Usage(calls=1))

    p = TwoPass()
    result = check_meaning(
        "The fields are now sea at high tide.",
        "cand",
        ["The fields are sea."],
        p,
        fidelity="content",
    )
    assert MATERIALITY in p.systems[0] and "sentence by sentence" in p.systems[0]
    rule = " ".join(MATERIALITY.split())
    assert "cattle have nothing to graze where the fields are now sea" in rule
    assert "a distant shore, when the original names no shore" in rule
    assert "is an inference, not an addition, and is permitted" in rule
    assert REVIEW_SAMPLES == 3 and REVIEW_TEMPERATURE == 0.3
    assert len(p.systems) == 1 + 4 * REVIEW_SAMPLES, (
        "pass one, then a sampled vote per flagged claim (the citation is exempt)"
    )
    assert p.temperatures[0] == 0.0 and set(p.temperatures[1:]) == {REVIEW_TEMPERATURE}
    assert REVIEW_QUESTION in p.systems[1] and "ORIGINAL PARAGRAPH:" in p.prompts[1]
    assert MATERIALITY in p.systems[1], "the reviewers see the same rule as the first reader"
    assert result.failing == ["added (number): 1834", "added (scene): a distant shore"]
    assert [(a.text, a.votes) for a in result.added if a.kind == "claim"] == [
        ("1834", "3-0"),
        ("a distant shore", "3-0"),
    ]
    assert [(c.text, c.votes) for c in result.cleared] == [
        ("went down with the ship", "0-3"),
        ("cattle finding nothing to graze", "1-2"),
    ]
    assert result.cleared[1].note == "the fields are sea, so nothing grows there"
    assert result.cleared[0].note == "restates the sinking" and result.cleared[0].sentence
    assert [a.kind for a in result.added] == ["claim", "citation", "claim"]


def test_strict_fidelity_keeps_the_old_behaviour():
    from deslop.semantic import check_meaning

    answer = json.dumps(
        {"original": [{"n": 1, "kept": True}], "added": [{"text": "we may enter", "kind": "claim"}]}
    )
    p = Canned("anthropic", answer)
    strict = check_meaning("orig", "cand", ["A"], p, fidelity="strict")
    assert strict.failing == ["added: we may enter"]
    assert "change of grammatical person" not in p.systems[0]


def test_proposition_extraction_asks_for_content_not_framing():
    from deslop.semantic import extract_propositions

    p = Canned("anthropic", "1. Tallow is the fat of the ox.")
    assert extract_propositions("text", p) == ["Tallow is the fat of the ox."]
    system = p.systems[0]
    assert "perspective-neutral" in system and "the speaker" in system
    strict = Canned("anthropic", "1. x")
    extract_propositions("text", strict, fidelity="strict")
    assert "perspective-neutral" not in strict.systems[0]


def test_judge_retries_once_with_a_stricter_format_before_giving_up(fixture_voice):
    from deslop.judge import compare_document

    class Flaky(CachingProvider):
        def __init__(self):
            super().__init__(use_cache=False, cache_only=False)
            self.name, self.default_model, self.systems = "anthropic", "m", []

        def _raw_complete(self, system, user, model, temperature):
            self.systems.append(system)
            text = (
                "Sure! Here is my verdict."
                if len(self.systems) == 1
                else json.dumps(
                    {
                        "voice": "A",
                        "coherent": "tie",
                        "tense_person": "tie",
                        "order": "tie",
                        "confidence": 0.9,
                    }
                )
            )
            return Completion(text=text, model=model, provider=self.name, usage=Usage(calls=1))

    judge = Flaky()
    v = compare_document("cand", "orig", "orig", fixture_voice, judge, exemplars=[])
    assert len(judge.systems) == 2 and "JSON object only" in judge.systems[1]
    assert v.note != "judge gave no JSON" and v.confidence == 0.9
    assert "headings and lists is not part of this item" in judge.systems[0]
    hopeless = Canned("anthropic", "no")
    v = compare_document("cand", "orig", "orig", fixture_voice, hopeless, exemplars=[])
    assert v.note == "judge gave no JSON" and len(hopeless.prompts) == 2


def test_neighbouring_paragraphs_reach_both_passes_only_when_given():
    from deslop.semantic import CONTEXT_NOTE, ENTAILMENT_SYSTEM, REVIEW_SYSTEM, check_meaning

    class Flagger(CachingProvider):
        def __init__(self):
            super().__init__(use_cache=False, cache_only=False)
            self.name, self.default_model, self.systems, self.prompts = "anthropic", "m", [], []

        def _raw_complete(self, system, user, model, temperature):
            self.systems.append(system)
            self.prompts.append(user)
            if "first reader flagged" in system:
                text = json.dumps({"new": "BEFORE:" not in user, "why": "seen above"})
            else:
                text = json.dumps(
                    {
                        "original": [{"n": 1, "kept": True}],
                        "added": [
                            {
                                "sentence": "Tallow becomes stearin.",
                                "text": "tallow",
                                "category": "entity",
                            }
                        ],
                    }
                )
            return Completion(text=text, model=model, provider=self.name, usage=Usage(calls=1))

    p = Flagger()
    bare = check_meaning("The conversion proceeds.", "Tallow becomes stearin.", ["It proceeds."], p)
    assert not bare.ok and p.systems[0] == ENTAILMENT_SYSTEM and p.systems[1] == REVIEW_SYSTEM
    assert "AROUND" not in p.prompts[0], "no context, no change to the cached prompt"

    p = Flagger()
    seen = check_meaning(
        "The conversion proceeds.",
        "Tallow becomes stearin.",
        ["It proceeds."],
        p,
        context=("Gay-Lussac converted tallow into stearin.", "1. Boil the tallow."),
    )
    assert seen.ok and [c.text for c in seen.cleared] == ["tallow"]
    for system, user in zip(p.systems, p.prompts, strict=True):
        assert system.endswith(CONTEXT_NOTE)
        assert "BEFORE:\nGay-Lussac converted tallow into stearin." in user
        assert "AFTER:\n1. Boil the tallow." in user
