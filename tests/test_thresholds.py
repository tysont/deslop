# ABOUTME: Tests for threshold calibration: bands from within-author spread, greedy tightening
# ABOUTME: that fails LLM-reference documents while passing held-out ones, and distance rescaling.
from __future__ import annotations

import random

from deslop.stats import DocSamples, MetricsProfile, MetricStat, TellStat
from deslop.thresholds import (
    CalibrationStats,
    Thresholds,
    calibrate_thresholds,
    passes,
    rescale,
    split_heldout,
)


def synthetic_profile(seed: int = 0) -> tuple[MetricsProfile, list[DocSamples]]:
    """A voice whose documents are low on em dashes and high on citations, against LLM
    documents that are the reverse; ten authors with three documents each."""
    rng = random.Random(seed)
    docs = []
    for a in range(10):
        bias = rng.uniform(-0.5, 0.5)
        for i in range(3):
            docs.append(
                DocSamples(
                    title=f"a{a}-{i}",
                    author=f"author{a}",
                    words=2000,
                    chunks=4,
                    metrics={
                        "em_dashes_per_1k": max(0.0, 2.0 + bias + rng.gauss(0, 0.6)),
                        "citation_per_100w": max(0.0, 1.0 + bias / 4 + rng.gauss(0, 0.2)),
                        "sent_len_mean": 22.0 + rng.gauss(0, 2),
                    },
                    tells={
                        "T1": max(0.0, rng.gauss(0.2, 0.15)),
                        "T5": max(0.0, rng.gauss(0.3, 0.2)),
                    },
                )
            )
    llm = [
        DocSamples(
            title=f"llm{i}",
            author="llm",
            words=700,
            chunks=1,
            metrics={
                "em_dashes_per_1k": max(0.0, 8.0 + rng.gauss(0, 2)),
                "citation_per_100w": max(0.0, rng.gauss(0.05, 0.05)),
                "sent_len_mean": 21.0 + rng.gauss(0, 2),
            },
            tells={"T1": max(0.0, rng.gauss(1.0, 0.4)), "T5": max(0.0, rng.gauss(1.8, 0.6))},
        )
        for i in range(60)
    ]

    def stat(name: str) -> MetricStat:
        vals = [d.metrics[name] for d in docs]
        m = sum(vals) / len(vals)
        sd = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5
        return MetricStat(mean=m, std=sd, p10=m, p50=m, p90=m, n=len(vals), within_std=sd * 0.8)

    profile = MetricsProfile(
        total_words=60000,
        metrics={n: stat(n) for n in ("em_dashes_per_1k", "citation_per_100w", "sent_len_mean")},
        tells={
            "T1": TellStat(rate_per_1k=0.2, hits=12, words=60000),
            "T5": TellStat(rate_per_1k=0.3, hits=18, words=60000),
        },
        per_document=docs,
    )
    return profile, llm


def test_split_heldout_is_by_author_and_about_a_fifth():
    profile, _ = synthetic_profile()
    held, method = split_heldout(profile.per_document)
    assert "held out by author" in method
    authors = {d.author for d in held}
    assert 4 <= len(held) <= 9 and all(
        d in held for d in profile.per_document if d.author in authors
    )
    small, method = split_heldout(profile.per_document[:6])
    assert len(small) == 6 and "too small" in method


def test_calibration_meets_both_targets_on_separable_data():
    profile, llm = synthetic_profile()
    th = calibrate_thresholds(profile, llm)
    assert th.stats is not None
    assert th.stats.llm_fail_rate >= 0.95
    assert th.stats.heldout_pass_rate >= 0.90
    assert "sent_len_mean" not in th.metrics, "slop_side none metrics get no band"
    assert th.metrics["em_dashes_per_1k"].low is None and th.metrics["em_dashes_per_1k"].high
    assert th.stats.raw_llm_median > th.stats.raw_heldout_p90
    held, _ = split_heldout(profile.per_document)
    assert sum(passes(d.metrics, d.tells, th) for d in held) / len(held) >= 0.9
    assert sum(not passes(d.metrics, d.tells, th) for d in llm) / len(llm) >= 0.95


def test_rescale_anchors():
    stats = CalibrationStats(
        method="x",
        heldout_docs=1,
        llm_docs=1,
        heldout_pass_rate=1,
        llm_fail_rate=1,
        raw_heldout_p90=0.1,
        raw_llm_median=0.5,
        raw_llm_p90=0.7,
    )
    assert rescale(0.1, stats) == 20.0
    assert rescale(0.5, stats) == 70.0
    assert rescale(0.7, stats) == 89.0
    assert 95.0 < rescale(1.5, stats) <= 100.0
    assert rescale(0.0, stats) == 0.0 and rescale(0.05, stats) == 10.0
    assert rescale(0.3, None) == 30.0
    assert Thresholds().stats is None
