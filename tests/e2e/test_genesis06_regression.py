# ABOUTME: Regression: v2 apply on the Genesis 6 lesson against the tutor voice must score
# ABOUTME: lower and judge higher than both v1 conversions. Replays from tests/cache.
from pathlib import Path

import pytest

from deslop import Voice, apply, score
from deslop.apply import ExemplarIndex
from deslop.judge import judge_paragraph

FIXTURES = Path(__file__).parent.parent / "fixtures"
TUTOR = FIXTURES / "tutor.voice"


@pytest.fixture(scope="module")
def tutor_voice():
    if not TUTOR.exists():
        pytest.skip("tests/fixtures/tutor.voice is not built")
    return Voice.load(TUTOR)


def judge_mean(text: str, voice: Voice, provider) -> float:
    """Mean judge score over the paragraphs of a text."""
    index = ExemplarIndex(voice)
    scores = []
    for para in [p for p in text.split("\n\n") if p.strip()]:
        scores.append(
            judge_paragraph(para, voice, provider, exemplars=index.closest(para, 2)).overall
        )
    return sum(scores) / len(scores)


def test_v2_beats_both_v1_conversions(tutor_voice, provider_factory, tmp_path):
    lesson = (FIXTURES / "genesis06_lesson.md").read_text(encoding="utf-8")
    v1_edit = (FIXTURES / "genesis06_v1_edit.md").read_text(encoding="utf-8")
    v1_rewrite = (FIXTURES / "genesis06_v1_rewrite.md").read_text(encoding="utf-8")
    res = apply(
        lesson,
        tutor_voice,
        mode="rewrite",
        n=3,
        pairs_path=tmp_path / "pairs.jsonl",
        provider_factory=provider_factory,
    )
    v2 = res.report_after.distance
    assert v2 < score(v1_edit, tutor_voice).distance
    assert v2 < score(v1_rewrite, tutor_voice).distance
    judge = provider_factory("claude")
    v2_judge = judge_mean(res.text, tutor_voice, judge)
    assert v2_judge > judge_mean(v1_edit, tutor_voice, judge)
    assert v2_judge > judge_mean(v1_rewrite, tutor_voice, judge)
