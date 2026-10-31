# ABOUTME: Unit tests for every catalogued tell detector, positive and negative cases.
# ABOUTME: Tests the detector functions through detect_tells with the built-in catalog.
import pytest

from deslop.metrics import detect_tells
from deslop.nlp import analyze
from deslop.patterns import Catalog, load_catalog


def ids(text, pattern=None):
    hits = detect_tells(analyze(text))
    return [h for h in hits if pattern is None or h.pattern_id == pattern]


def test_T1_text_as_agent():
    hits = ids(
        "The chapter asks what Abraham owes. The narrator refuses to say. Abraham asks nothing.",
        "T1",
    )
    assert [h.text for h in hits] == ["chapter asks", "narrator refuses"]
    assert hits[0].note == "asks"


def test_T1_titled_proper_noun():
    assert ids("The Odyssey insists on hospitality.", "T1")
    assert not ids("Homer insists on hospitality.", "T1")


def test_T2_negative_parallelism():
    text = "It is not a question of loyalty; it is a question of appetite. He chose silence rather than speech."
    assert len(ids(text, "T2")) == 2
    assert not ids("Not only did he come, but also he stayed.", "T2")


def test_T3_balance_markers():
    assert (
        len(ids("The scene is at once tender and cruel, and the tension never resolves.", "T3"))
        == 2
    )


def test_T4_triad():
    assert ids("He hesitates, he broods, and he stalls.", "T4")
    assert ids("Johnson wrote prefaces, dictionaries, and lives.", "T4")
    assert not ids("Johnson wrote prefaces and lives.", "T4")
    clauses = (
        "Abraham is told to take his son, whom he loves, and the text withholds any account of it."
    )
    assert not ids(clauses, "T4")


def test_T5_aphoristic_close():
    text = "Abraham climbs the mountain with Isaac in Genesis 22. Faith here is not a feeling but a geometry."
    hits = ids(text, "T5")
    assert len(hits) == 1 and hits[0].note == "antithesis"
    anchored = "Abraham climbs the mountain. Then Abraham returns to Beersheba with his servants."
    assert not ids(anchored, "T5")
    dense = "He went up the hill. The weight of history became the burden of memory."
    assert ids(dense, "T5")[0].note.startswith("abstract")


def test_T5_requires_paragraph_final_and_short():
    text = "Faith here is not a feeling but a geometry. Abraham climbs the mountain with Isaac."
    assert not ids(text, "T5")


def test_T6_invented_label():
    assert [h.text for h in ids("Appetite's arithmetic leaves no remainder.", "T6")] == [
        "Appetite's arithmetic"
    ]
    assert not ids("Abraham's son climbs the hill.", "T6")


def test_T7_meta_commentary():
    assert ids("This essay explores the limits of mercy. We now turn to the second act.", "T7")
    assert not ids("Johnson explores the limits of mercy.", "T7")


def test_T8_epithet_substitution():
    text = "Hamlet enters. The melancholy prince speaks to the ghost."
    hits = ids(text, "T8")
    assert [h.text for h in hits] == ["The melancholy prince"]
    assert not ids("The melancholy prince speaks to the ghost.", "T8")


def test_T9_unanswered_question():
    assert ids("What does Sarah want? Nobody says. The sun rose.", "T9")
    assert not ids("What does Sarah want? She wants her son back.", "T9")
    assert not ids("Is this fair? No, it is not.", "T9")


def test_T10_formula_openers():
    text = "So the chapter ends. Then, nothing. Against that, consider Job. It is worth noting the date. Importantly, he left."
    assert len(ids(text, "T10")) == 5


def test_T11_hedge_stack():
    assert ids("Perhaps this is, in some sense, a kind of riddle.", "T11")
    assert not ids("Perhaps this is a riddle.", "T11")


def test_T12_intensifiers():
    assert len(ids("It is deeply and genuinely strange.", "T12")) == 2


def test_T13_colon_reveal():
    assert ids("Here is the point: silence.", "T13")
    assert not ids("Here is the point: he left at dawn and never returned.", "T13")
    assert not ids("She listed three things: apples, pears, and plums.", "T13")


def test_T14_tidy_summary_close():
    body = (
        "Abraham climbs the mountain with Isaac and the servants wait below.\n\n"
        "Isaac carries the wood and asks about the lamb while Abraham answers.\n\n"
        "Abraham and Isaac climb the mountain; Isaac carries the wood and asks about the lamb."
    )
    hits = ids(body, "T14")
    assert len(hits) == 1 and hits[0].para_idx == 2
    with_new_anchor = body + " Then Sarah dies in Hebron."
    assert not ids(with_new_anchor, "T14")


def test_regex_pattern_extension_needs_no_code(tmp_path):
    extra = tmp_path / "extra.yaml"
    extra.write_text(
        "patterns:\n  - id: X1\n    name: tapestry\n    description: banned word\n"
        "    detector: regex\n    params:\n      patterns: ['\\btapestry\\b']\n"
    )
    catalog = load_catalog((str(extra),))
    assert catalog.by_id("X1").name == "tapestry"
    hits = detect_tells(analyze("A rich tapestry of meaning."), catalog)
    assert [h.pattern_id for h in hits] == ["X1"]


def test_catalog_extend_overrides_by_id():
    base = load_catalog()
    override = Catalog(patterns=[base.by_id("T12").model_copy(update={"description": "changed"})])
    merged = base.extend(override)
    assert merged.by_id("T12").description == "changed"
    assert len(merged.patterns) == len(base.patterns)


def test_hit_offsets_index_into_text():
    text = "Prelude here.\n\nThe chapter asks what Abraham owes, and the story holds together both promise and threat."
    for h in detect_tells(analyze(text)):
        assert text[h.start : h.end] == h.text


@pytest.mark.parametrize("pid", [f"T{i}" for i in range(1, 15)])
def test_every_catalog_pattern_has_detector(pid):
    from deslop.metrics.tells import DOCUMENT_DETECTORS, SENTENCE_DETECTORS

    p = load_catalog().by_id(pid)
    if p.detector == "dep-pattern":
        assert p.function in SENTENCE_DETECTORS
    elif p.detector == "metric":
        assert p.function in DOCUMENT_DETECTORS
    else:
        assert p.params["patterns"]
