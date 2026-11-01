# ABOUTME: Tests for the primary-text module: marker parsing, verse and Stephanus resolution,
# ABOUTME: verbatim quotation lookup, and the added-quotation/citation verification.
from __future__ import annotations

from deslop.reference import (
    PrimaryText,
    normalize,
    verify_additions,
)

MARKED = """# title: Test Genesis
# book: Genesis
# aliases: Gen, Ge

{{Genesis 6:5}} And GOD saw that the wickedness of man was great in the earth, and that
every imagination of the thoughts of his heart was only evil continually.
{{Genesis 6:6}} And it repented the LORD that he had made man on the earth, and it grieved
him at his heart.
"""

STEPHANUS = """# title: Test Republic
# book: Republic

{{369a}} Then in the larger the quantity of justice is likely to be larger.
{{369b}} A State, I said, arises out of the needs of mankind; no one is self-sufficing,
but all of us have many wants. {{369c}} Then, as we have many wants, one takes a helper.
"""


def test_normalize_ignores_case_punctuation_and_curly_quotes():
    assert normalize("“Every imagination—of the Thoughts,” he said.") == normalize(
        "every imagination of the thoughts he said"
    )


def test_markers_become_segments_and_verses_resolve(tmp_path):
    p = tmp_path / "gen.txt"
    p.write_text(MARKED, encoding="utf-8")
    t = PrimaryText.load(p)
    assert t.name == "Test Genesis" and t.kinds == {"verse"} and len(t.segments) == 2
    assert t.resolve("Gen 6:5") == "Genesis 6:5"
    assert t.resolve("Genesis 6:5-6") == "Genesis 6:5"
    assert t.resolve("6:6") == "Genesis 6:6"
    assert t.resolve("Gen 6:7") is None, "verse not in the text"
    assert t.resolve("Gen 6:5-9") is None, "range end not in the text"
    assert t.resolve("Exod 6:5") is None, "another book"
    assert t.resolve("369b") is None, "no Stephanus markers here"


def test_quotes_locate_to_their_verse(tmp_path):
    p = tmp_path / "gen.txt"
    p.write_text(MARKED, encoding="utf-8")
    t = PrimaryText.load(p)
    assert t.find_quote("only evil continually") == "Genesis 6:5"
    assert t.find_quote("it grieved him at his heart") == "Genesis 6:6"
    assert t.find_quote("it grieved him in his heart") is None, "one word off is not verbatim"


def test_stephanus_resolution_and_ranges(tmp_path):
    p = tmp_path / "rep.txt"
    p.write_text(STEPHANUS, encoding="utf-8")
    t = PrimaryText.load(p)
    assert t.kinds == {"stephanus"}
    assert t.resolve("369b") == "369b"
    assert t.resolve("Republic II, 369b") == "369b"
    assert t.resolve("369b–c") == "369b"
    assert t.resolve("369a-369c") == "369a"
    assert t.resolve("369d") is None
    assert t.resolve("369b-370a") is None, "range end missing"


def test_verify_additions_without_texts_lists_everything_as_unverified():
    original = 'The flood comes. "Noah walked with God."'
    candidate = 'The flood comes (Gen 6:17). "Noah walked with God." "The earth was filled."'
    check = verify_additions(original, candidate, [])
    assert check.ok and not check.checked
    assert sorted(a.text for a in check.unverified) == ["Gen 6:17", "The earth was filled."]
    assert all("no primary text" in a.note for a in check.unverified)


def test_verify_additions_against_texts_rejects_fabrication(tmp_path):
    p = tmp_path / "gen.txt"
    p.write_text(MARKED, encoding="utf-8")
    texts = [PrimaryText.load(p)]
    original = "God sees the wickedness of man."
    good = 'God sees that "every imagination of the thoughts of his heart was only evil" (Gen 6:5).'
    check = verify_additions(original, good, texts)
    assert check.ok and check.checked and all(a.verified for a in check.additions)
    assert check.additions[0].where == "Test Genesis Genesis 6:5"
    bad_quote = 'God sees that "every thought of his heart was wicked" (Gen 6:5).'
    check = verify_additions(original, bad_quote, texts)
    assert not check.ok
    assert [a.text for a in check.rejected] == ["every thought of his heart was wicked"]
    bad_cite = "God sees the wickedness of man (Gen 7:5)."
    check = verify_additions(original, bad_cite, texts)
    assert not check.ok and check.rejected[0].text == "Gen 7:5"
    assert "does not resolve" in str(check.rejected[0])


def test_citation_kind_without_a_marked_text_is_unverifiable_not_rejected(tmp_path):
    p = tmp_path / "gen.txt"
    p.write_text(MARKED, encoding="utf-8")
    texts = [PrimaryText.load(p)]
    check = verify_additions("Socrates builds a city.", "Socrates builds a city (369b).", texts)
    assert check.ok and check.unverified and check.unverified[0].note == "unverifiable"


def test_short_quoted_terms_are_not_quotations():
    check = verify_additions("The ark.", 'The "ark" is a "chest" here.', [])
    assert not check.additions
