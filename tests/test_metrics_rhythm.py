# ABOUTME: Unit tests for rhythm metrics on small hand-built texts.
# ABOUTME: Each test pins one metric to a value computable by hand.
import pytest

from deslop.metrics import METRICS, compute_all, rhythm
from deslop.nlp import analyze


def m(name, text):
    return METRICS[name].fn(analyze(text))


def test_registry_has_all_rhythm_metrics():
    names = {n for n, s in METRICS.items() if s.group == "rhythm"}
    assert {
        "sent_len_mean",
        "sent_len_std",
        "sent_len_cv",
        "pct_sent_lt8",
        "pct_sent_gt30",
        "max_run_similar_len",
        "pct_compound_multi",
        "para_len_mean",
        "para_len_std",
        "semicolons_per_1k",
        "colons_per_1k",
        "em_dashes_per_1k",
        "parens_per_1k",
        "questions_per_1k",
    } <= names


def test_sentence_length_stats():
    text = "One two three four. One two three four five six seven eight nine ten eleven twelve."
    assert m("sent_len_mean", text) == 8.0
    assert m("sent_len_std", text) == 4.0
    assert m("sent_len_cv", text) == 0.5
    assert m("pct_sent_lt8", text) == 50.0
    assert m("pct_sent_gt30", text) == 0.0


def test_max_run_similar_len():
    same = (
        "A b c d e f. A b c d e f g. A b c d e f g h. "
        + "Now here is a sentence that is very much longer than all of the others combined and more."
    )
    assert m("max_run_similar_len", same) == 3.0


def test_pct_compound_multi():
    text = "I came and I saw and I conquered. I left."
    assert m("pct_compound_multi", text) == 50.0


def test_paragraph_lengths():
    text = "One two. Three four. Five six.\n\nOne two. Three four.\n\nOne two three four."
    assert m("para_len_mean", text) == 2.0
    assert m("para_len_std", text) == pytest.approx(0.8165, abs=1e-3)


def test_punctuation_rates():
    text = " ".join(["word"] * 1000) + "; : — ( ?"
    assert m("semicolons_per_1k", text) == pytest.approx(1.0, rel=1e-3)
    assert m("colons_per_1k", text) == pytest.approx(1.0, rel=1e-3)
    assert m("em_dashes_per_1k", text) == pytest.approx(1.0, rel=1e-3)
    assert m("parens_per_1k", text) == pytest.approx(1.0, rel=1e-3)
    assert m("questions_per_1k", text) == pytest.approx(1.0, rel=1e-3)


def test_em_dash_double_hyphen_counts():
    assert m("em_dashes_per_1k", "a -- b " * 500) == pytest.approx(500.0)


def test_openers():
    text = "The dog ran. She ran. However, it stopped. Dogs run."
    a = analyze(text)
    assert rhythm.opener_pct_det(a) == 25.0
    assert rhythm.opener_pct_pron(a) == 25.0
    assert rhythm.opener_pct_adv_conj(a) == 25.0
    assert rhythm.opener_pct_noun(a) == 25.0
    assert rhythm.opener_top_word_share(a) == 25.0
    assert rhythm.opener_top_word_share(
        analyze("The dog ran. The cat sat. A bird flew.")
    ) == pytest.approx(66.67, abs=0.01)


def test_empty_text_is_safe():
    values = compute_all(analyze(""))
    assert all(v == 0.0 for v in values.values())
