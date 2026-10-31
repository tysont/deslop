# ABOUTME: Tests for the content lock: document-scoped allowlist, removals-only violations,
# ABOUTME: sentence-initial capitals, and primary-text citations that may always be added.
from deslop.lock import DocumentLock, check, inventory, is_primary_citation

SPAN = (
    'Abraham climbs (Alter 1996, 12) in Gen 22:1-3 with "his only son, whom he loves" '
    "and 3 servants in 1820."
)
DOCUMENT = SPAN + " Later Isaac is bound. The knife is raised twice, at 22:9 and 22:10."


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
    doc = DocumentLock.build(DOCUMENT)
    after = (
        "In Gen 22:1-3 (Alter 1996, 12) Abraham climbs with 3 servants in 1820, "
        '"his only son, whom he loves".'
    )
    result = check(SPAN, after, doc)
    assert result.ok and result.additions == []


def test_removals_are_violations_and_document_items_may_move():
    doc = DocumentLock.build(DOCUMENT)
    after = 'Isaac climbs with 3 servants in 1820 (Alter 1996, 12) in Gen 22:1-3, "his only son, whom he loves".'
    result = check(SPAN, after, doc)
    msgs = [str(v) for v in result.violations]
    assert msgs == ["proper_nouns: missing Abraham"]
    # Isaac is named elsewhere in the document, so adding him is not a new item.
    assert result.additions == []


def test_new_items_are_reported_not_rejected():
    doc = DocumentLock.build(DOCUMENT)
    after = SPAN.replace("in 1820", "in 1820, as Kierkegaard noted in 1843")
    result = check(SPAN, after, doc)
    assert result.ok
    assert "numbers: 1843" in result.additions and "proper_nouns: Kierkegaard" in result.additions


def test_primary_text_citations_may_always_be_added():
    doc = DocumentLock.build(DOCUMENT)
    after = SPAN.replace("in 1820", "in 1820 (Gen 22:11; 331c)")
    result = check(SPAN, after, doc)
    assert result.ok and result.additions == []
    assert is_primary_citation("Gen 22:11") and is_primary_citation("331c")
    assert not is_primary_citation("(Alter 1996, 12)")
    scholarly = SPAN.replace("in 1820", "in 1820 (Westermann 1985, 355)")
    assert "citations: (Westermann 1985, 355)" in check(SPAN, scholarly, doc).additions


def test_sentence_initial_capitals_are_not_names_unless_named_elsewhere():
    text = "Grief comes before the sentence. The grief is a maker's sorrow. Noah builds."
    doc = DocumentLock.build(text)
    assert "Grief" not in doc.names and "Noah" in doc.names
    before = "The grief is a maker's sorrow."
    after = "Grief is a maker's sorrow."
    result = check(before, after, doc)
    assert result.ok and result.additions == []
