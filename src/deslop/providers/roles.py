# ABOUTME: The writer and the judge: two different models on one provider, so no model checks
# ABOUTME: its own work, with the judge's stored calibration.
from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from deslop.judge import (
    JudgeCalibration,
)
from deslop.providers.base import (
    DEFAULT_JUDGE,
    DEFAULT_WRITER,
    CachingProvider,
    Usage,
    get_provider,
    resolve_model,
)

ProviderFactory = Callable[[str], CachingProvider]


@dataclass
class Models:
    """The writer and the judge: two different Anthropic models served by one provider. The
    judge also runs the semantic lock. A model never judges its own output."""

    writer: CachingProvider
    judge: CachingProvider
    calibration: JudgeCalibration | None = None

    @property
    def usage(self) -> Usage:
        total = Usage()
        total.add(self.writer.usage)
        if self.judge is not self.writer:
            total.add(self.judge.usage)
        return total

    def describe(self) -> str:
        cal = self.calibration
        if cal is None:
            measured = "uncalibrated"
        else:
            measured = f"{100 * cal.accuracy:.0f}% on {cal.pairs} pairs"
            if not cal.usable:
                measured += "; below 80%, its verdicts are unreliable"
        return (
            f"writer {self.writer.key}; judge {self.judge.key} ({measured}); "
            f"provider {self.writer.name}"
        )


def build_models(
    writer: str = DEFAULT_WRITER,
    judge: str = DEFAULT_JUDGE,
    *,
    provider: str | None = None,
    factory: ProviderFactory | None = None,
    calibration: dict[str, Any] | None = None,
) -> Models:
    """Resolve the writer and judge model names (aliases allowed) on the configured provider.
    `factory` maps a provider name to a `CachingProvider` (tests replay a cache with it)."""
    from deslop.providers.base import default_provider_name

    writer_model, judge_model = resolve_model(writer), resolve_model(judge)
    if writer_model == judge_model:
        raise ValueError(
            f"the judge must be a different model from the writer ({writer_model}): "
            "a model never judges its own output"
        )
    name = provider or default_provider_name()
    make = factory or (lambda n: get_provider(n))
    base = make(name)
    writer_p, judge_p = copy.copy(base), copy.copy(base)
    writer_p.default_model, writer_p.usage = writer_model, Usage()
    judge_p.default_model, judge_p.usage = judge_model, Usage()
    cal = None
    for key, data in (calibration or {}).items():
        if key in (judge_model, judge_p.key):
            try:
                cal = JudgeCalibration.model_validate(data)
            except ValueError:
                cal = None
    return Models(writer=writer_p, judge=judge_p, calibration=cal)
