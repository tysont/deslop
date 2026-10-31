# ABOUTME: Tests for paragraph/sentence segmentation and offset bookkeeping.
# ABOUTME: Offsets must index back into the original text exactly.
from deslop.nlp import analyze, count_words, split_paragraphs


def test_split_paragraphs_offsets():
    text = "First para here.\n\n  Second one.  \n\n\n\nThird."
    spans = split_paragraphs(text)
    assert [text[s:e] for s, e in spans] == ["First para here.", "Second one.", "Third."]


def test_analyze_sentence_offsets_round_trip():
    text = "One sentence. Two sentence here.\n\nAnother paragraph! Is it? Yes."
    a = analyze(text)
    assert len(a.paragraphs) == 2
    for s in a.sentences:
        assert text[s.start : s.end] == s.text
    assert [s.para_index for s in a.sentences] == [0, 0, 1, 1, 1]
    assert [s.index for s in a.sentences] == list(range(5))


def test_headings_and_fragments_skipped():
    text = "# Title\n\nShort.\n\nThis is a real paragraph of prose with several words."
    a = analyze(text)
    assert len(a.paragraphs) == 1


def test_count_words():
    assert count_words("Don't count 42 or -- dashes, but do count hyphen-words.") == 8
