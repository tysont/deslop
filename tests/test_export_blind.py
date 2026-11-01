# ABOUTME: Tests for fine-tuning export (pairs + summarized chunks, formats, cost) and the blind
# ABOUTME: test pairing, logging and summary.
import json
from pathlib import Path

import pytest

from deslop import blind
from deslop.providers.base import CachingProvider, Completion, Usage
from deslop.voice.export import chunk_text, estimate_tokens, export, load_pairs, write_jsonl

FIXTURES = Path(__file__).parent / "fixtures"


class Summarizer(CachingProvider):
    def __init__(self):
        super().__init__(use_cache=False, max_retries=1)
        self.name = "claude"
        self.default_model = "gpt-4.1"

    def _raw_complete(self, system, user, model, temperature):
        return Completion(
            text="A summary.",
            model=model,
            provider=self.name,
            usage=Usage(input_tokens=10, output_tokens=5, cost_usd=0.001, calls=1),
        )


def write_pairs(path: Path, voice: str):
    rows = [
        {"voice": voice, "before": "a b", "after": "b a", "accepted": True},
        {"voice": voice, "before": "a b", "after": "b a", "accepted": True},  # duplicate
        {"voice": voice, "before": "c", "after": "d", "accepted": False},
        {"voice": voice, "before": "e", "after": None, "accepted": True},
        {"voice": voice, "before": "f", "after": "g", "accepted": True},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_chunk_text_groups_paragraphs_between_bounds():
    short = "Word " * 100
    text = "\n\n".join([short.strip()] * 10 + ["# heading"] + ["Long " * 700])
    chunks = chunk_text(text, lo=300, hi=600)
    assert chunks and all(300 <= len(c.split()) <= 600 for c in chunks)
    assert sum(len(c.split()) for c in chunks) == 900  # 3 chunks of 300; leftover 100 dropped


def test_load_pairs_dedupes_and_keeps_accepted_only(fixture_voice, tmp_path):
    p = tmp_path / "pairs.jsonl"
    write_pairs(p, "fixture")
    assert load_pairs(fixture_voice, p) == [("a b", "b a"), ("f", "g")]
    assert load_pairs(fixture_voice, tmp_path / "missing.jsonl") == []


def test_export_formats_tokens_and_cost(fixture_voice, tmp_path, monkeypatch):
    monkeypatch.delenv("DESLOP_CACHE_ONLY", raising=False)
    p = tmp_path / "pairs.jsonl"
    write_pairs(p, "fixture")
    corpus = FIXTURES / "corpus_small"
    res = export(
        fixture_voice,
        fmt="openai-ft",
        pairs_path=p,
        corpus=corpus,
        provider=Summarizer(),
        max_chunks=3,
        cost_model="gpt-4.1-mini",
    )
    assert res.pairs == 2 and res.chunks == 3 and len(res.records) == 5
    assert res.summary_calls == 3 and res.summary_cost == pytest.approx(0.003)
    assert res.tokens > 0 and res.estimated_training_cost == pytest.approx(res.tokens * 5 / 1e6)
    out = tmp_path / "train.jsonl"
    assert write_jsonl(res, fixture_voice, out) == 5
    rows = [json.loads(ln) for ln in out.read_text().splitlines()]
    assert [m["role"] for m in rows[0]["messages"]] == ["system", "user", "assistant"]
    assert rows[0]["messages"][0]["content"] == fixture_voice.profile.strip()
    assert rows[-1]["messages"][1]["content"] == "A summary."
    plain = export(
        fixture_voice, fmt="jsonl", pairs_path=p, include_chunks=False, cost_model="gpt-5.5"
    )
    assert plain.estimated_training_cost is None and plain.chunks == 0
    write_jsonl(plain, fixture_voice, out)
    row = json.loads(out.read_text().splitlines()[0])
    assert row["input"] == "a b" and row["output"] == "b a" and row["kind"] == "pair"
    with pytest.raises(ValueError):
        export(fixture_voice, fmt="csv", include_chunks=False)
    assert estimate_tokens("abcd" * 10) == 10


def test_blind_pairs_match_length_and_overlap_without_reuse():
    pool = [
        ("p1", "Shakespeare wrote plays about kings and their sons. " * 5),
        ("p2", "The weather in the valley turned cold before the harvest came in. " * 4),
        ("p3", "Shakespeare and his kings; the sons of kings; plays on plays. " * 5),
        ("p4", "Shakespeare wrote plays about kings. " * 30),
    ]
    cands = [
        ("c1", "Shakespeare wrote his plays about kings and the sons of kings. " * 5),
        ("c2", "Shakespeare made plays of kings and sons. " * 6),
    ]
    pairing = blind.make_pairs(cands, pool, 2, seed=1)
    pairs = pairing.pairs
    assert len(pairs) == 2 and pairing.skipped == 0 and pairing.unpaired == 0
    assert {p.control_source for p in pairs} <= {"p1", "p3"}  # similar length, high overlap
    assert len({p.control_source for p in pairs}) == 2  # no reuse
    assert all(p.overlap > 0 for p in pairs)
    a, b = blind.shown(pairs[0])
    assert (a if pairs[0].candidate_position == "A" else b) == pairs[0].candidate
    positions = {
        blind.make_pairs(cands, pool, 1, seed=s).pairs[0].candidate_position for s in range(20)
    }
    assert positions == {"A", "B"}


def test_blind_min_overlap_skips_weak_pairs_and_counts_unpaired():
    pool = [
        ("p1", "Shakespeare wrote plays about kings and their sons. " * 5),
        ("p2", "The weather in the valley turned cold before the harvest came in. " * 4),
        ("p3", "Shakespeare and his kings; the sons of kings; plays on plays. " * 5),
    ]
    cands = [
        ("c1", "Shakespeare wrote his plays about kings and the sons of kings. " * 5),
        ("c2", "Granite quarries employ masons who cut blocks for cathedral walls. " * 5),
        ("c3", "Shakespeare made plays of kings and sons. " * 6),
    ]
    pairing = blind.make_pairs(cands, pool, 3, seed=1, min_overlap=0.15)
    assert {p.candidate_source for p in pairing.pairs} == {"c1", "c3"}
    assert all(p.overlap >= 0.15 for p in pairing.pairs)
    assert pairing.skipped == 1  # c2 overlaps nothing
    assert pairing.unpaired == 1  # c2 went unpaired
    strict = blind.make_pairs(cands, pool, 3, seed=1, min_overlap=0.99)
    assert strict.pairs == [] and strict.skipped == 3 and strict.unpaired == 3
    # Asking for more pairs than the pool can supply also leaves candidates unpaired.
    short = blind.make_pairs(cands, pool[:1], 3, seed=1, min_overlap=0.0)
    assert len(short.pairs) == 1 and short.unpaired == 2


def test_blind_record_and_summary(tmp_path):
    log = tmp_path / "v.jsonl"
    pair = blind.Pair(
        candidate="x",
        candidate_source="c",
        control="y",
        control_source="p",
        overlap=0.1,
        candidate_position="B",
    )
    session = [
        blind.record("v", pair, "b", "too tidy", log_path=log),
        blind.record("v", pair, "A", "Too tidy ", log_path=log),
    ]
    s = blind.summarize("v", session, log_path=log)
    assert s.session_total == 2 and s.session_correct == 1 and s.session_accuracy == 0.5
    assert s.total == 2 and s.accuracy == 0.5
    assert s.top_reasons == [("too tidy", 2)]
    assert json.loads(s.to_json())["accuracy"] == 0.5
    assert blind.summarize("v", [], log_path=tmp_path / "none.jsonl").accuracy is None


def test_candidates_from_dir_and_control_pool(tmp_path, fixture_voice):
    (tmp_path / "a.md").write_text("# Title\n\n" + "Plain prose words here. " * 12 + "\n\nshort\n")
    (tmp_path / "b.py").write_text("print('no')")
    cands = blind.candidates_from_dir(tmp_path)
    assert len(cands) == 1 and cands[0][0] == "a.md"
    pool = blind.control_pool(fixture_voice, FIXTURES / "corpus_small")
    assert len(pool) > 50 and all(blind.count_words(p) >= blind.MIN_WORDS for _, p in pool)
    from_exemplars = blind.control_pool(fixture_voice, None)
    assert from_exemplars and all(src.startswith("exemplar") for src, _ in from_exemplars)
