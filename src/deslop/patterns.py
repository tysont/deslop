# ABOUTME: Loads the tell catalog (slop_patterns.yaml) and optional user extensions.
# ABOUTME: Patterns are data; detectors are selected by detector type and function name.
from __future__ import annotations

from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

DetectorType = Literal["regex", "dep-pattern", "metric"]


class Pattern(BaseModel):
    id: str
    name: str
    description: str
    detector: DetectorType
    function: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class Catalog(BaseModel):
    lexicons: dict[str, list[str]] = Field(default_factory=dict)
    patterns: list[Pattern] = Field(default_factory=list)

    def by_id(self, pattern_id: str) -> Pattern:
        for p in self.patterns:
            if p.id == pattern_id:
                return p
        raise KeyError(pattern_id)

    def lexicon(self, name: str) -> set[str]:
        return set(self.lexicons.get(name, []))

    def extend(self, other: Catalog) -> Catalog:
        lexicons = {k: list(v) for k, v in self.lexicons.items()}
        for k, v in other.lexicons.items():
            lexicons[k] = sorted(set(lexicons.get(k, [])) | set(v))
        ids = {p.id for p in other.patterns}
        patterns = [p for p in self.patterns if p.id not in ids] + other.patterns
        return Catalog(lexicons=lexicons, patterns=patterns)


def builtin_catalog_path() -> Path:
    return Path(str(resources.files("deslop.baselines") / "slop_patterns.yaml"))


def load_catalog_file(path: Path) -> Catalog:
    with path.open(encoding="utf-8") as fh:
        return Catalog.model_validate(yaml.safe_load(fh) or {})


@lru_cache(maxsize=8)
def load_catalog(extra: tuple[str, ...] = ()) -> Catalog:
    """Built-in catalog, extended by zero or more user pattern files."""
    catalog = load_catalog_file(builtin_catalog_path())
    for p in extra:
        catalog = catalog.extend(load_catalog_file(Path(p)))
    return catalog
