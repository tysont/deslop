# ABOUTME: Tests for ingest normalization: quotes, hard wraps, Gutenberg boilerplate.
# ABOUTME: Output must be stable paragraphs separated by one blank line.
from deslop.corpus.normalize import normalize, strip_gutenberg, unwrap_lines


def test_straight_quotes_and_nbsp():
    assert normalize("“Hi,” she said ‘so’. ") == "\"Hi,\" she said 'so'.\n"


def test_unwrap_hard_wraps_keeps_paragraphs():
    text = "This is a line\nthat wraps here.\n\nSecond para\nalso wraps."
    assert unwrap_lines(text) == "This is a line that wraps here.\n\nSecond para also wraps."


def test_gutenberg_header_footer_stripped():
    text = (
        "junk\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\nBody text.\n\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK X ***\nlicense junk"
    )
    assert strip_gutenberg(text).strip() == "Body text."
    assert normalize(text) == "Body text.\n"


def test_multiple_blank_lines_collapse():
    assert normalize("a.\n\n\n\n\nb.") == "a.\n\nb.\n"
