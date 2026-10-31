# ABOUTME: Metric registry: every metric is a named function Analysis -> float.
# ABOUTME: Importing this package registers rhythm, anchor, and tell metrics.
from __future__ import annotations

from deslop.metrics.base import (
    METRICS,
    MetricSpec,
    compute_all,
    metric,
)
from deslop.metrics import anchors, rhythm  # noqa: F401  (registration side effects)
from deslop.metrics.tells import TellHit, detect_tells, tell_rates

__all__ = [
    "METRICS",
    "MetricSpec",
    "TellHit",
    "compute_all",
    "detect_tells",
    "metric",
    "tell_rates",
]
