# ABOUTME: Unit tests for anchor metrics: citations, quotes, concrete tokens,
# ABOUTME: reader address, first person, committed claims.
import pytest

from deslop.metrics import METRICS
from deslop.metrics.anchors import count_citations, count_quotes
from deslop.nlp import analyze


def m(name, text):
    return METRICS[name].fn(analyze(text))


@pytest.mark.parametrize(
    "text,n",
    [
        ("See Gen 22:1-3 and Genesis 12:3.", 2),
        ("Plato argues this at Republic 514a and again at 331c.", 2),
        ("The passage (Alter 1996) differs from (Alter 1996, 12) and (Smith and Jones 2001).", 3),
        ("Compare p. 12 with pp. 40-41 and l. 5.", 3),
        ("The scene is Act III, scene ii in most editions.", 1),
        ("No citations here at all.", 0),
    ],
)
def test_count_citations(text, n):
    assert count_citations(text) == n


def test_citation_per_100w():
    text = "See Gen 22:1. " + " ".join(["word"] * 98)
    assert m("citation_per_100w", text) == pytest.approx(1.0)


def test_count_quotes_min_four_words():
    assert count_quotes('He said "no". She said "this is a long quote".') == 1
    assert count_quotes("“curly quotes count as well here”") == 1


def test_quote_per_para():
    text = 'One "quoted span of words" here.\n\nNothing quoted in this paragraph at all.'
    assert m("quote_per_para", text) == 0.5


def test_concrete_per_100w():
    text = (
        "In 1996 Robert Alter translated Genesis; the word *hesed* and αγάπη appear. "
        + " ".join(["word"] * 89)
    )
    # numbers: 1996; proper nouns: Robert, Alter, Genesis; italic: hesed; non-latin: αγάπη
    assert m("concrete_per_100w", text) == pytest.approx(6.0, abs=1.0)


def test_reader_address():
    text = "You should read this. Consider the opening line. The author wrote it."
    assert m("reader_address_per_1k", text) > 0
    a = analyze(text)
    from deslop.metrics.anchors import count_reader_address

    assert count_reader_address(a) == 2


def test_first_person():
    text = "I think my reading is right, and I'll defend it. He disagrees."
    assert m("first_person_per_1k", text) == pytest.approx(3 / (12 / 1000))


def test_committed_claims():
    text = "This reading is wrong. I take the second view. The better reading is Alter's.\n\nNothing evaluative here."
    assert m("committed_claim_per_para", text) == 1.5
