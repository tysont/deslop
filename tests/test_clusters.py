# ABOUTME: Tests for register clustering: k-means with silhouette-chosen k, cluster summaries,
# ABOUTME: and the facts block that profile.md must agree with.
from __future__ import annotations

from deslop.baselines import english_generic, llm_reference
from deslop.stats import MetricsProfile, MetricStat
from deslop.voice.clusters import (
    DocumentVector,
    cluster_documents,
    kmeans,
    silhouette,
)
from deslop.voice.profile import facts_block, percentile_of


def two_groups() -> list[DocumentVector]:
    docs = []
    for i in range(6):
        docs.append(
            DocumentVector(
                title=f"short{i}", author=f"a{i % 2}", words=1000 + i, vector=[-2.0 + 0.1 * i, -2.0]
            )
        )
    for i in range(6):
        docs.append(
            DocumentVector(
                title=f"long{i}", author=f"b{i % 3}", words=9000 + i, vector=[2.0 + 0.1 * i, 2.0]
            )
        )
    return docs


def test_kmeans_separates_two_groups_and_silhouette_prefers_two():
    docs = two_groups()
    labels2, _ = kmeans([d.vector for d in docs], 2, seed=0)
    assert len({labels2[i] for i in range(6)}) == 1
    assert len({labels2[i] for i in range(6, 12)}) == 1
    assert labels2[0] != labels2[6]
    s2 = silhouette([d.vector for d in docs], labels2)
    labels3, _ = kmeans([d.vector for d in docs], 3, seed=0)
    assert s2 > silhouette([d.vector for d in docs], labels3)


def test_cluster_documents_picks_k_and_summarizes():
    names = ["sent_len_mean", "quote_per_para"]
    clusters, assignment = cluster_documents(two_groups(), names)
    assert len(clusters) == 2
    assert sorted(c.documents for c in clusters) == [6, 6]
    big = max(clusters, key=lambda c: c.median_doc_words)
    assert big.median_doc_words >= 9000 and big.top_authors[0] in {"b0", "b1", "b2"}
    assert big.distinguishing and big.distinguishing[0].metric in names
    assert set(assignment) == {d.title for d in two_groups()}


def test_too_few_documents_make_one_cluster():
    docs = two_groups()[:3]
    clusters, assignment = cluster_documents(docs, ["a", "b"])
    assert len(clusters) == 1 and all(v == 0 for v in assignment.values())


def test_percentile_and_facts_block_wording():
    generic = english_generic()
    assert (
        45
        <= percentile_of(generic.metrics["sent_len_mean"].mean, generic.metrics["sent_len_mean"])
        <= 55
    )
    profile = MetricsProfile(
        metrics={"sent_len_cv": MetricStat(mean=0.56, std=0.1, p10=0.4, p50=0.55, p90=0.7, n=10)}
    )
    block = facts_block(profile, generic, llm_reference())
    assert block.startswith("## Facts")
    line = next(ln for ln in block.splitlines() if "sentence-length variation" in ln)
    assert "0.56" in line and "percentile of generic English" in line and "LLM prose" in line
