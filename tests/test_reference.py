# ABOUTME: Tests for the primary-text module: the sidecar of labelled spans, citation schemes,
# ABOUTME: verbatim quotation lookup with nearest runs, and the added-quotation/citation check.
from __future__ import annotations

import json

from deslop.reference import (
    CitationScheme,
    PrimaryText,
    PrimaryTextSpec,
    canonical_label,
    load_primary_texts,
    normalize,
    verify_additions,
)

SCRIPTURE = (
    "Chapter six.\n\nThe Lord saw that the wickedness of humankind was great in the earth, "
    "and that every imagination of the thoughts of the heart was only evil continually. "
    "And the Lord was sorry that he had made humankind, and it grieved him in his heart. "
    "But I will establish my covenant with you. You shall come into the vessel, you and your "
    "sons.\n"
)
VERSE = CitationScheme(
    name="verse",
    pattern=r"\b(?:Book|Bk)\.?\s+\d{1,3}:\d{1,3}(?:[-–]\d{1,3})?",
    bare_pattern=r"(?<![\d:])\d{1,3}:\d{1,3}(?:[-–]\d{1,3})?(?![\d:])",
)
STEPHANUS = CitationScheme(
    name="stephanus", pattern=r"\b(?:Dialogue\s+)?\d{3}[a-e](?:[-–](?:\d{3})?[a-e])?\b"
)


def spans(raw: str, *items: tuple[str, str]) -> list[dict[str, object]]:
    """Sidecar entries for (label, verbatim phrase) pairs: the phrase's span in `raw`."""
    out: list[dict[str, object]] = []
    for ref, phrase in items:
        start = raw.index(phrase)
        out.append({"ref": ref, "start": start, "end": start + len(phrase)})
    return out


def scripture(tmp_path):
    p = tmp_path / "book.md"
    p.write_text(SCRIPTURE, encoding="utf-8")
    locs = spans(
        SCRIPTURE,
        ("Book 6:5", "The Lord saw that the wickedness of humankind was great in the earth"),
        ("Bk 6:5", "The Lord saw that the wickedness of humankind was great in the earth"),
        ("6:5", "The Lord saw that the wickedness of humankind was great in the earth"),
        (
            "Book 6:6",
            "And the Lord was sorry that he had made humankind, and it grieved him in his heart.",
        ),
        (
            "6:6",
            "And the Lord was sorry that he had made humankind, and it grieved him in his heart.",
        ),
        ("Book 6:18", "But I will establish my covenant with you. You shall come into the vessel"),
        ("6:18", "But I will establish my covenant with you. You shall come into the vessel"),
    )
    (tmp_path / "book.locs.json").write_text(json.dumps({"locations": locs}), encoding="utf-8")
    return p


def test_normalize_ignores_case_punctuation_and_curly_quotes():
    assert normalize("“Every imagination—of the Thoughts,” he said.") == normalize(
        "every imagination of the thoughts he said"
    )
    assert normalize("Noah's ark; 'quoted' word") == "noah's ark quoted word"
    assert canonical_label(" Bk. 6:5-6,") == "bk 6:5-6"


def test_sidecar_spans_resolve_citations_and_locate_quotations(tmp_path):
    t = PrimaryText.load(scripture(tmp_path), schemes=[VERSE], title="Book")
    assert t.name == "Book" and t.has_locations and len(t.segments) == 7
    assert t.resolve("Book 6:5") == "Book 6:5" and t.resolve("Bk. 6:5") == "Bk 6:5"
    assert t.resolve("6:18") == "6:18", "a bare location, the book known from context"
    assert t.resolve("Book 6:5-6") == "Book 6:5", "a range resolves by its first label"
    assert t.resolve("6:99") is None and t.resolve("369b") is None
    assert t.find_quote("it grieved him in his heart") == "Book 6:6"
    assert t.find_quote("made humankind, and it grieved") == "Book 6:6"
    assert t.find_quote("nothing of the kind") is None
    assert t.segment_text("6:6").startswith("and the lord was sorry")
    assert t.locate(0) == "Book", "text outside every span locates to the name"


def test_a_text_without_a_sidecar_still_verifies_quotations(tmp_path):
    p = tmp_path / "plain.txt"
    p.write_text("The quick brown fox jumps over the lazy dog.", encoding="utf-8")
    t = PrimaryText.load(p)
    assert not t.has_locations and t.find_quote("brown fox jumps") == "plain"
    assert t.resolve("6:5") is None, "no scheme, no resolution"


def test_nearest_run_shows_why_a_quotation_was_rejected(tmp_path):
    t = PrimaryText.load(scripture(tmp_path), schemes=[VERSE], title="Book")
    where, run = t.nearest(
        "But with thee will I establish my covenant; and thou shalt come into the vessel"
    )
    assert where == "Book 6:18" and run.startswith("but i will establish my covenant")
    assert t.nearest("nothing like this anywhere at all") == ("", "")


def test_a_declared_scheme_without_labels_rejects_its_citations(tmp_path):
    p = tmp_path / "dialogue.md"
    p.write_text(
        "He coiled up and sprang at us like a wild beast. We were terrified.", encoding="utf-8"
    )
    (tmp_path / "dialogue.locs.json").write_text(
        json.dumps([{"ref": "ch. 2 ¶1", "start": 0, "end": 48}]), encoding="utf-8"
    )
    d = PrimaryText.load(p, schemes=[STEPHANUS], title="Dialogue")
    assert d.answers_to("369b") and d.answers_to("372c-e") and d.answers_to("Dialogue 369b")
    assert (
        d.resolve("369b") is None and d.find_quote("sprang at us like a wild beast") == "ch. 2 ¶1"
    )
    check = verify_additions("x", "See 369b and 372c-e.", [d])
    assert [a.note for a in check.additions] == ["does not resolve in Dialogue"] * 2
    assert not check.ok


def test_verify_additions_without_texts_lists_everything_as_unverified():
    check = verify_additions("x", 'He wrote "the quick brown fox" (Book 6:5; 369b).', [])
    assert not check.checked and check.ok
    assert [(a.kind, a.note) for a in check.additions] == [
        ("quote", "no primary text given"),
        ("citation", "no primary text given"),
        ("citation", "no primary text given"),
    ]


def test_verify_additions_against_texts(tmp_path):
    t = PrimaryText.load(scripture(tmp_path), schemes=[VERSE], title="Book")
    original = "The covenant is given before the rain (6:18)."
    cand = (
        'As the text says, "But with thee will I establish my covenant" (6:18; cf. 6:6, '
        '369b and p. 41). "It grieved him in his heart," we read.'
    )
    check = verify_additions(original, cand, [t])
    assert check.checked and not check.ok
    other, verbatim, verse, steph, page = check.additions
    assert other.kind == "quote" and not other.verified and other.nearest.startswith("but i will")
    assert "(Book Book 6:18)" in other.nearest
    assert verbatim.verified and verbatim.where == "Book Book 6:6"
    assert verse.verified and verse.where == "Book 6:6"
    assert steph.note == "unverifiable", "no scheme recognises it: listed, not rejected"
    assert page.note == "unverifiable"
    assert [a.text for a in check.rejected] == ["But with thee will I establish my covenant"]
    assert "6:18" not in [a.text for a in check.additions], "cited by the original already"
    assert verify_additions(
        original, 'He said "I will establish my covenant with you" (6:18).', [t]
    ).ok


def test_load_primary_texts_takes_specs_paths_and_directories(tmp_path):
    scripture(tmp_path)
    spec = PrimaryTextSpec(path="book.md", title="Book", schemes=[VERSE])
    (t,) = load_primary_texts([spec], relative_to=tmp_path)
    assert t.name == "Book" and t.resolve("6:5") == "6:5"
    (u,) = load_primary_texts([tmp_path / "book.md"], schemes=[VERSE])
    assert u.name == "book" and u.resolve("6:5") == "6:5"
    (d,) = load_primary_texts([str(tmp_path)])
    assert d.name == "book" and d.resolve("6:5") is None, "no scheme given for the directory"
    try:
        load_primary_texts(["missing.md"], relative_to=tmp_path)
    except FileNotFoundError as exc:
        assert "missing.md" in str(exc)
    else:
        raise AssertionError("missing text must raise")


def test_short_quoted_terms_are_not_quotations():
    check = verify_additions("x", 'The "ark" and "two words" here.', [])
    assert check.additions == []
