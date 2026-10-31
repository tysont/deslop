# ABOUTME: Registry and shared helpers for scalar metrics.
# ABOUTME: Each metric declares which direction counts as slop-side for scoring.
from __future__ import annotations

import statistics
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from deslop.nlp import Analysis

SlopSide = Literal["high", "low", "both", "none"]

MetricFn = Callable[[Analysis], float]


@dataclass(frozen=True)
class MetricSpec:
    name: str
    fn: MetricFn
    group: str
    slop_side: SlopSide
    description: str


METRICS: dict[str, MetricSpec] = {}


def metric(
    name: str, *, group: str, slop_side: SlopSide, description: str
) -> Callable[[MetricFn], MetricFn]:
    """Register a metric function under `name`."""

    def register(fn: MetricFn) -> MetricFn:
        if name in METRICS:
            raise ValueError(f"metric already registered: {name}")
        METRICS[name] = MetricSpec(name, fn, group, slop_side, description)
        return fn

    return register


def compute_all(analysis: Analysis) -> dict[str, float]:
    return {name: spec.fn(analysis) for name, spec in METRICS.items()}


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def stdev(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def pct(values: list[float], predicate: Callable[[float], bool]) -> float:
    if not values:
        return 0.0
    return 100.0 * sum(1 for v in values if predicate(v)) / len(values)
