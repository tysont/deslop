# ABOUTME: Tests for log-odds vocabulary signatures and pattern banning against a
# ABOUTME: synthetic LLM reference profile (the shipped one needs API keys to build).
from collections import Counter

from deslop.baselines import ReferenceProfile, english_generic
from deslop.patterns import load_catalog
from deslop.stats import MetricsProfile, TellStat, WordCounts
from deslop.voice.lexicon import build_lexicon, log_odds_with_prior


def test_log_odds_direction():
    a = Counter({"whale": 50, "ship": 40, "the": 500})
    b = Counter({"whale": 1, "ship": 2, "the": 500, "garden": 60})
    prior = a + b
    z = log_odds_with_prior(a, 590, b, 563, prior, sum(prior.values()))
    assert z["whale"] > 2 and z["ship"] > 2
    assert z["garden"] < -2


def synthetic_llm(tell_rates: dict[str, float], words: dict[str, int]) -> ReferenceProfile:
    return ReferenceProfile(
        name="llm_reference",
        created="2026-01-01",
        source="synthetic",
        total_words=sum(words.values()),
        tells={
            k: TellStat(rate_per_1k=v, hits=int(v * 100), words=100000)
            for k, v in tell_rates.items()
        },
        word_freq=words,
    )


def test_build_lexicon_bans_patterns_and_discourages_words():
    catalog = load_catalog()
    counts = WordCounts()
    counts.add("the poet writes of nature and the critic reads the poem with care. " * 200)
    profile = MetricsProfile(
        tells={
            "T1": TellStat(rate_per_1k=0.1, hits=1, words=10000),
            "T2": TellStat(rate_per_1k=2.0, hits=20, words=10000),
        }
    )
    llm_words = {"the": 5000, "tapestry": 400, "delve": 300, "poet": 50, "nature": 50}
    llm = synthetic_llm({"T1": 5.0, "T2": 3.0, "T5": 2.0}, llm_words)
    lex = build_lexicon(counts, profile, english_generic(), llm, catalog, ban_ratio=10.0)
    by_id = {p.id: p for p in lex.patterns}
    assert by_id["T1"].banned and by_id["T1"].ratio == 50.0
    assert not by_id["T2"].banned and by_id["T2"].ratio == 1.5
    assert by_id["T5"].banned and by_id["T5"].ratio == 999.0  # absent from corpus
    assert by_id["T3"].llm_per_1k is None and not by_id["T3"].banned
    assert lex.banned_ids() == ["T1", "T5"]
    discouraged = {e.term for e in lex.discouraged_words}
    assert {"tapestry", "delve"} <= discouraged
    assert "poet" not in discouraged
    encouraged = {e.term for e in lex.encouraged_words}
    assert "poet" in encouraged and "critic" in encouraged
    assert all(len(e.term.split()) == 2 for e in lex.encouraged_bigrams)


def test_build_lexicon_without_llm_reference():
    catalog = load_catalog()
    counts = WordCounts()
    counts.add("the poet writes of nature. " * 100)
    lex = build_lexicon(counts, MetricsProfile(), english_generic(), None, catalog)
    assert lex.discouraged_words == []
    assert all(p.llm_per_1k is None and not p.banned for p in lex.patterns)
