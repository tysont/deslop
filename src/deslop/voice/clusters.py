# ABOUTME: Finds the registers in a corpus by clustering documents on their metric vectors
# ABOUTME: (k-means, k in 2-4 chosen by silhouette) and summarizes each cluster.
from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass

from deslop.stats import Cluster, ClusterMetric, MetricsProfile

MIN_DOCS_TO_CLUSTER = 4
K_RANGE = (2, 3, 4)
DISTINGUISHING = 4


@dataclass
class DocumentVector:
    title: str
    author: str
    words: int
    vector: list[float]  # z-scores against the voice, one per metric name


def _dist(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)))


def kmeans(
    points: list[list[float]], k: int, *, seed: int = 0, iterations: int = 50
) -> tuple[list[int], list[list[float]]]:
    """Lloyd's algorithm with k-means++ seeding; returns (labels, centroids)."""
    rng = random.Random(seed)
    centroids = [list(points[rng.randrange(len(points))])]
    while len(centroids) < k:
        d2 = [min(_dist(p, c) ** 2 for c in centroids) for p in points]
        total = sum(d2)
        if total <= 0:
            centroids.append(list(points[rng.randrange(len(points))]))
            continue
        r = rng.random() * total
        acc = 0.0
        for p, w in zip(points, d2, strict=True):
            acc += w
            if acc >= r:
                centroids.append(list(p))
                break
    labels = [0] * len(points)
    for _ in range(iterations):
        new_labels = [min(range(k), key=lambda j: _dist(p, centroids[j])) for p in points]
        if new_labels == labels and _ != 0:
            break
        labels = new_labels
        for j in range(k):
            members = [p for p, lab in zip(points, labels, strict=True) if lab == j]
            if members:
                centroids[j] = [sum(col) / len(members) for col in zip(*members, strict=True)]
    return labels, centroids


def silhouette(points: list[list[float]], labels: list[int]) -> float:
    """Mean silhouette coefficient; 1 is well separated, 0 is overlapping."""
    scores = []
    for i, p in enumerate(points):
        own = [q for j, q in enumerate(points) if labels[j] == labels[i] and j != i]
        if not own:
            scores.append(0.0)
            continue
        a = sum(_dist(p, q) for q in own) / len(own)
        b = math.inf
        for lab in set(labels):
            if lab == labels[i]:
                continue
            others = [q for j, q in enumerate(points) if labels[j] == lab]
            if others:
                b = min(b, sum(_dist(p, q) for q in others) / len(others))
        scores.append(0.0 if b == math.inf else (b - a) / max(a, b, 1e-12))
    return sum(scores) / len(scores) if scores else 0.0


def cluster_documents(
    docs: list[DocumentVector], metric_names: list[str]
) -> tuple[list[Cluster], dict[str, int]]:
    """Cluster documents into registers. Returns the clusters (largest first) and a map
    from document title to cluster index. Fewer than four documents form one cluster."""
    if len(docs) < MIN_DOCS_TO_CLUSTER:
        labels = [0] * len(docs)
    else:
        points = [d.vector for d in docs]
        best: tuple[float, list[int]] | None = None
        for k in K_RANGE:
            if k >= len(docs):
                break
            labels_k, _ = kmeans(points, k)
            score = silhouette(points, labels_k)
            if best is None or score > best[0]:
                best = (score, labels_k)
        labels = best[1] if best else [0] * len(docs)
    groups: dict[int, list[DocumentVector]] = {}
    for d, lab in zip(docs, labels, strict=True):
        groups.setdefault(lab, []).append(d)
    ordered = sorted(groups.values(), key=lambda g: -sum(d.words for d in g))
    clusters: list[Cluster] = []
    assignment: dict[str, int] = {}
    for index, members in enumerate(ordered):
        by_author: dict[str, int] = {}
        for d in members:
            by_author[d.author] = by_author.get(d.author, 0) + d.words
        dims = len(members[0].vector)
        centroid = [sum(d.vector[i] for d in members) / len(members) for i in range(dims)]
        ranked = sorted(range(dims), key=lambda i: -abs(centroid[i]))
        distinguishing = [
            ClusterMetric(
                metric=metric_names[i], cluster_mean=centroid[i], voice_mean=0.0, z=centroid[i]
            )
            for i in ranked[:DISTINGUISHING]
        ]
        clusters.append(
            Cluster(
                index=index,
                label=f"register {index + 1}",
                documents=len(members),
                words=sum(d.words for d in members),
                median_doc_words=int(statistics.median(d.words for d in members)),
                top_authors=[a for a, _ in sorted(by_author.items(), key=lambda kv: -kv[1])[:5]],
                titles=[d.title for d in members],
                distinguishing=distinguishing,
            )
        )
        for d in members:
            assignment[d.title] = index
    return clusters, assignment


def with_voice_means(clusters: list[Cluster], profile: MetricsProfile) -> list[Cluster]:
    """Fill cluster_mean/voice_mean in real units from the z-scores and the voice stats."""
    for c in clusters:
        for m in c.distinguishing:
            stat = profile.metrics.get(m.metric)
            if stat is not None:
                m.voice_mean = stat.mean
                m.cluster_mean = stat.mean + m.z * stat.std
    return clusters
