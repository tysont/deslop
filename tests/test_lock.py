# ABOUTME: Tests for the content-lock inventory and before/after check.
# ABOUTME: Citations, quotes, numbers, and proper nouns must survive a rewrite unchanged.
from deslop.lock import check, inventory

SPAN = (
    'Abraham climbs (Alter 1996, 12) in Gen 22:1-3 with "his only son, whom he loves" '
    "and 3 servants in 1820."
)


def test_inventory_classes():
    inv = inventory(SPAN)
    assert inv.citations == {"(Alter 1996, 12)": 1, "Gen 22:1-3": 1}
    assert inv.quotes == {"his only son, whom he loves": 1}
    assert inv.numbers == {"3": 1, "1820": 1}
    assert set(inv.proper_nouns) >= {"Abraham", "Alter"}
    assert "citations: (Alter 1996, 12); Gen 22:1-3" in inv.describe()


def test_numbers_inside_citations_and_quotes_not_double_counted():
    inv = inventory('He cites p. 12 and says "there were 40 of them".')
    assert inv.numbers == {}


def test_reordered_rewrite_passes():
    after = (
        "In Gen 22:1-3 (Alter 1996, 12) Abraham climbs with 3 servants in 1820, "
        '"his only son, whom he loves".'
    )
    assert check(SPAN, after) == []


def test_violations_reported():
    after = "Isaac climbs with 4 servants in 1820 (Alter 1996, 12) in Gen 22:1-3."
    msgs = [str(v) for v in check(SPAN, after)]
    assert any(m.startswith("quotes: missing his only son") for m in msgs)
    assert "numbers: missing 3; added 4" in msgs
    assert any(m == "proper_nouns: missing Abraham; added Isaac" for m in msgs)


def test_lock_classes_subset():
    after = "Isaac climbs in 1820."
    assert check("Abraham climbs in 1820.", after, ("numbers",)) == []
    assert check("Abraham climbs in 1820.", after, ("proper_nouns",))


def test_proper_noun_surface_match_tolerates_retagging():
    before = "The Danish prince hesitates; Hamlet speaks."
    after = "Hamlet, the Danish prince, hesitates and then speaks."
    assert check(before, after, ("proper_nouns",)) == []
