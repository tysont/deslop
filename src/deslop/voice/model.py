# ABOUTME: The Voice artifact: a directory of yaml/json/md files that fully describes a register.
# ABOUTME: Load/save are lossless so voices can be diffed, versioned, and edited by hand.
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from pydantic import BaseModel, Field

from deslop.stats import MetricsProfile

if TYPE_CHECKING:
    from deslop.corpus import Corpus

SCHEMA_VERSION = 1


class VoiceMeta(BaseModel):
    name: str
    schema_version: int = SCHEMA_VERSION
    deslop_version: str
    created: str = Field(
        default_factory=lambda: dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    )
    corpus_name: str
    corpus_hash: str
    corpus_path: str | None = None
    total_words: int
    author_shares: dict[str, float] = Field(default_factory=dict)
    max_share: float = 0.25
    profile_source: str = "template"


class LexiconEntry(BaseModel):
    term: str
    log_odds: float
    corpus_per_million: float
    baseline_per_million: float


class PatternRate(BaseModel):
    id: str
    name: str
    corpus_per_1k: float
    llm_per_1k: float | None = None
    ratio: float | None = None
    banned: bool = False


class Lexicon(BaseModel):
    encouraged_words: list[LexiconEntry] = Field(default_factory=list)
    encouraged_bigrams: list[LexiconEntry] = Field(default_factory=list)
    discouraged_words: list[LexiconEntry] = Field(default_factory=list)
    patterns: list[PatternRate] = Field(default_factory=list)
    ban_ratio: float = 10.0

    def banned_ids(self) -> list[str]:
        return [p.id for p in self.patterns if p.banned]


class Voice(BaseModel):
    meta: VoiceMeta
    metrics: MetricsProfile
    lexicon: Lexicon
    rules: str
    profile: str
    exemplars: str
    path: Path | None = None

    @property
    def name(self) -> str:
        return self.meta.name

    def save(self, path: Path | str) -> Path:
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        (root / "voice.yaml").write_text(
            yaml.safe_dump(self.meta.model_dump(), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        (root / "metrics.json").write_text(self.metrics.model_dump_json(indent=1), encoding="utf-8")
        (root / "lexicon.json").write_text(self.lexicon.model_dump_json(indent=1), encoding="utf-8")
        (root / "rules.md").write_text(self.rules, encoding="utf-8")
        (root / "profile.md").write_text(self.profile, encoding="utf-8")
        (root / "exemplars.md").write_text(self.exemplars, encoding="utf-8")
        self.path = root
        return root

    @classmethod
    def load(cls, path: Path | str) -> Voice:
        root = Path(path)
        if not (root / "voice.yaml").exists():
            raise FileNotFoundError(f"not a voice directory: {root}")
        meta = VoiceMeta.model_validate(
            yaml.safe_load((root / "voice.yaml").read_text(encoding="utf-8"))
        )
        return cls(
            meta=meta,
            metrics=MetricsProfile.model_validate_json(
                (root / "metrics.json").read_text(encoding="utf-8")
            ),
            lexicon=Lexicon.model_validate_json(
                (root / "lexicon.json").read_text(encoding="utf-8")
            ),
            rules=(root / "rules.md").read_text(encoding="utf-8"),
            profile=(root / "profile.md").read_text(encoding="utf-8"),
            exemplars=(root / "exemplars.md").read_text(encoding="utf-8"),
            path=root,
        )

    @classmethod
    def calibrate(cls, corpus: Corpus, **opts: Any) -> Voice:
        from deslop.voice.calibrate import calibrate

        return calibrate(corpus, **opts)
