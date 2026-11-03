# ABOUTME: Loads model prices from the packaged pricing.yaml plus a user override file and
# ABOUTME: answers cost questions by longest-prefix match on the model name.
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from deslop.home import deslop_home


@dataclass
class Pricing:
    models: dict[str, tuple[float, float]] = field(default_factory=dict)  # USD per MTok in/out
    training: dict[str, float] = field(default_factory=dict)  # USD per MTok of training data
    sources: dict[str, str] = field(default_factory=dict)
    fetched: str = ""

    def match(self, model: str, table: Mapping[str, object]) -> str | None:
        for prefix in sorted(table, key=len, reverse=True):
            if model.startswith(prefix):
                return prefix
        return None


def _read(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


@lru_cache(maxsize=1)
def load_pricing() -> Pricing:
    """Packaged table, with entries from $DESLOP_HOME/pricing.yaml added or overriding."""
    packaged = _read(Path(str(resources.files("deslop") / "pricing.yaml")))
    override_path = deslop_home() / "pricing.yaml"
    override = _read(override_path) if override_path.exists() else {}
    table = Pricing(
        sources=dict(packaged.get("sources", {})),
        fetched=str(packaged.get("fetched", "")),
    )
    for data in (packaged, override):
        for model, spec in (data.get("models") or {}).items():
            table.models[model] = (float(spec["input"]), float(spec["output"]))
        for model, rate in (data.get("training") or {}).items():
            table.training[model] = float(rate)
    return table


def is_priced(model: str) -> bool:
    table = load_pricing()
    return table.match(model, table.models) is not None


def price(model: str, input_tokens: int, output_tokens: int) -> float:
    """USD for a call; zero for models not in the table (callers flag these as unpriced)."""
    table = load_pricing()
    key = table.match(model, table.models)
    if key is None:
        return 0.0
    pin, pout = table.models[key]
    return (input_tokens * pin + output_tokens * pout) / 1e6


def training_price(model: str, tokens: int) -> float | None:
    """USD to train on `tokens` of data, or None when no training rate is published."""
    table = load_pricing()
    key = table.match(model, table.training)
    return None if key is None else tokens * table.training[key] / 1e6
