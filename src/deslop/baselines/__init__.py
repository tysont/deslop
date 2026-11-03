# ABOUTME: Built-in reference profiles: generic English and LLM-generated prose.
# ABOUTME: Both share the ReferenceProfile schema; only stats are stored, never source text.
from __future__ import annotations

from functools import lru_cache
from importlib import resources
from pathlib import Path

from pydantic import BaseModel, Field

from deslop.stats import DocSamples, MetricStat, TellStat


class ReferenceProfile(BaseModel):
    """A baseline: aggregate stats plus, for the LLM reference, one `DocSamples` per
    generated document (chunk-mean metrics and tell rates, never the text) so threshold
    calibration can check that reference documents fail."""

    name: str
    created: str
    source: str
    total_words: int
    metrics: dict[str, MetricStat] = Field(default_factory=dict)
    tells: dict[str, TellStat] = Field(default_factory=dict)
    documents: list[DocSamples] = Field(default_factory=list)
    word_freq: dict[str, int] = Field(default_factory=dict)
    bigram_freq: dict[str, int] = Field(default_factory=dict)

    def save(self, path: Path) -> None:
        path.write_text(self.model_dump_json(indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> ReferenceProfile:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


def baseline_path(name: str) -> Path:
    return Path(str(resources.files("deslop.baselines") / f"{name}.json"))


@lru_cache(maxsize=4)
def load_baseline(name: str) -> ReferenceProfile:
    path = baseline_path(name)
    if not path.exists():
        raise FileNotFoundError(f"baseline not built: {path.name}")
    return ReferenceProfile.load(path)


def english_generic() -> ReferenceProfile:
    return load_baseline("english_generic")


def llm_reference() -> ReferenceProfile:
    return load_baseline("llm_reference")


def has_baseline(name: str) -> bool:
    return baseline_path(name).exists()
