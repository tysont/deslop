# ABOUTME: Tests for chunking, weighted aggregation, and the author share cap.
# ABOUTME: Pure arithmetic on hand-built samples.
import math

import pytest

from deslop.nlp import analyze
from deslop.stats import (
    AuthorSamples,
    aggregate,
    author_weights,
    chunk_analysis,
    sample_text,
    weighted_quantile,
    weighted_stat,
)


def test_chunk_analysis_groups_paragraphs():
    para = " ".join(["word"] * 120) + "."
    text = "\n\n".join([para] * 10)
    chunks = chunk_analysis(analyze(text), target_words=500)
    assert len(chunks) == 2
    assert all(c.word_count >= 500 for c in chunks)
    assert chunks[0].text.count("\n\n") == 4


def test_chunk_small_tail_merges():
    para = " ".join(["word"] * 100) + "."
    text = "\n\n".join([para] * 6)
    chunks = chunk_analysis(analyze(text), target_words=500)
    assert len(chunks) == 1


def test_weighted_stat():
    s = weighted_stat([1.0, 2.0, 3.0, 4.0], [1, 1, 1, 1])
    assert s.mean == 2.5 and s.p50 == 2.0 and s.n == 4
    assert s.std == pytest.approx(1.118, abs=1e-3)
    assert weighted_quantile([1.0, 10.0], [1, 9], 0.5) == 10.0


def test_author_weights_cap():
    per = {
        "big": AuthorSamples(words=900, chunks=9),
        "small": AuthorSamples(words=100, chunks=1),
    }
    w = author_weights(per, max_share=0.5)
    assert w["big"] * 9 == pytest.approx(0.5)
    assert w["small"] * 1 == pytest.approx(0.5)


def test_aggregate_profile():
    per = {
        "a": AuthorSamples(
            words=1000,
            chunks=2,
            metrics={"x": [1.0, 3.0]},
            tells={"T1": [0.0, 2.0]},
            tell_hits={"T1": 1},
        ),
        "b": AuthorSamples(
            words=1000,
            chunks=2,
            metrics={"x": [5.0, 7.0]},
            tells={"T1": [4.0, 6.0]},
            tell_hits={"T1": 5},
        ),
    }
    p = aggregate(per, max_share=0.5, chunk_words=500)
    assert p.metrics["x"].mean == 4.0
    assert p.tells["T1"].rate_per_1k == 3.0
    assert p.tells["T1"].hits == 6
    assert p.total_words == 2000


def test_sample_text_runs_all_metrics(heldout_texts):
    s = sample_text(heldout_texts[0])
    assert s.chunks >= 2
    assert "sent_len_mean" in s.metrics and len(s.metrics["sent_len_mean"]) == s.chunks
    assert "T1" in s.tells


def test_within_author_std_ignores_between_author_spread():
    from deslop.stats import DocSamples, within_author_std

    docs = [
        DocSamples(title="a1", author="a", metrics={"x": 10.0}),
        DocSamples(title="a2", author="a", metrics={"x": 12.0}),
        DocSamples(title="b1", author="b", metrics={"x": 100.0}),
        DocSamples(title="b2", author="b", metrics={"x": 102.0}),
    ]
    assert within_author_std(docs, "x") == pytest.approx(math.sqrt(2.0))
    singles = [DocSamples(title="c", author="c", metrics={"x": 1.0}),
               DocSamples(title="d", author="d", metrics={"x": 3.0})]
    assert within_author_std(singles, "x") == pytest.approx(math.sqrt(2.0))


def test_aggregate_with_documents_adds_within_std_and_ci():
    from deslop.stats import DocSamples

    per = {
        "a": AuthorSamples(words=1000, chunks=2, metrics={"x": [1.0, 3.0]}, tells={"T1": [0.0, 2.0]}),
        "b": AuthorSamples(words=1000, chunks=2, metrics={"x": [5.0, 7.0]}, tells={"T1": [4.0, 6.0]}),
    }
    docs = [
        DocSamples(title="a1", author="a", words=500, metrics={"x": 1.0}),
        DocSamples(title="a2", author="a", words=500, metrics={"x": 3.0}),
        DocSamples(title="b1", author="b", words=500, metrics={"x": 5.0}),
        DocSamples(title="b2", author="b", words=500, metrics={"x": 7.0}),
    ]
    p = aggregate(per, max_share=0.5, chunk_words=500, per_document=docs)
    stat = p.metrics["x"]
    assert stat.within_std == pytest.approx(math.sqrt(2.0))
    assert 1.0 <= stat.ci_low <= stat.mean <= stat.ci_high <= 7.0
    assert stat.ci_high - stat.ci_low > 0
    assert len(p.per_document) == 4
