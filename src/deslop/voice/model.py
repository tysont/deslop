# ABOUTME: The Voice artifact: a zip (NAME.voice) of yaml/json/md files that fully describes a
# ABOUTME: register, built from texts, loaded read-only from the zip or from an unpacked directory.
from __future__ import annotations

import datetime as dt
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from pydantic import BaseModel, Field

from deslop.stats import MetricsProfile

if TYPE_CHECKING:
    from deslop.corpus import Corpus

SCHEMA_VERSION = 2
VOICE_FILES = ("metrics.json", "lexicon.json", "rules.md", "profile.md", "exemplars.md")


class AuthorProvenance(BaseModel):
    name: str
    words: int
    share: float
    documents: int
    sources: list[str] = Field(default_factory=list)  # sites or origin notes


class DocumentProvenance(BaseModel):
    title: str
    author: str
    words: int
    source: str = ""
    license: str = ""
    year: int | None = None


class VoiceMeta(BaseModel):
    """voice.yaml: identity plus provenance. `checksum` is set when the voice is packed and
    verified whenever the zip is read; `has_exemplars` is False for a voice packed with
    `--strip-exemplars`."""

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
    document_count: int = 0
    authors: list[AuthorProvenance] = Field(default_factory=list)
    author_shares: dict[str, float] = Field(default_factory=dict)
    documents: list[DocumentProvenance] = Field(default_factory=list)
    licenses: list[str] = Field(default_factory=list)
    max_share: float = 0.25
    profile_source: str = "template"
    checksum: str | None = None
    has_exemplars: bool = True


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
    """A calibrated voice: the six files of a `NAME.voice` zip held in memory.

    `metrics` holds every metric's mean, spread, bootstrap interval and calibrated band plus
    tell rates and limits; `lexicon` holds over-represented words and bigrams and the
    per-pattern rates against the LLM reference; `rules`, `profile` and `exemplars` are the
    markdown a model is shown when writing. Build one with
    [`Voice.from_texts`][deslop.Voice.from_texts], read one with
    [`Voice.load`][deslop.Voice.load], write one with [`Voice.save`][deslop.Voice.save].
    """

    meta: VoiceMeta
    metrics: MetricsProfile
    lexicon: Lexicon
    rules: str
    profile: str
    exemplars: str
    path: Path | None = None
    read_only: bool = False  # True when loaded from a packed .voice zip

    @property
    def name(self) -> str:
        return self.meta.name

    # ------------------------------------------------------------ files
    def write_dir(self, root: Path) -> Path:
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
        return root

    def save(self, path: Path | str) -> Path:
        """Write the voice. `path` becomes a `.voice` zip with a content checksum unless it is
        an existing directory, in which case the six files are written into it (the unpacked
        working form)."""
        from deslop.voice.pack import PackError, pack_voice

        target = Path(path)
        if self.read_only and self.path is not None and target.resolve() == self.path.resolve():
            raise PackError("this voice was loaded read-only from its zip; save to another path")
        if target.is_dir():
            self.write_dir(target)
            self.path = target
            return target
        with tempfile.TemporaryDirectory(prefix="deslop-voice-") as tmp:
            self.write_dir(Path(tmp))
            pack_voice(Path(tmp), target)
        self.path = target
        return target

    @classmethod
    def load(cls, path: Path | str) -> Voice:
        """Read a `.voice` zip (extracted read-only to a temporary directory, checksum
        verified) or an unpacked voice directory; raises `FileNotFoundError` otherwise."""
        from deslop.voice.pack import extract_read_only, is_packed

        root = Path(path)
        read_only = False
        if is_packed(root):
            root = extract_read_only(root)
            read_only = True
        if not (root / "voice.yaml").exists():
            raise FileNotFoundError(f"not a voice: {path}")
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
            path=Path(path) if read_only else root,
            read_only=read_only,
        )

    # ------------------------------------------------------------ building
    @classmethod
    def from_texts(
        cls,
        paths: list[Path | str],
        *,
        name: str,
        author_map: dict[str, str] | None = None,
        max_share: float = 0.15,
        exemplar_words: int = 2500,
        use_llm: bool = True,
        llm_model: str | None = None,
        patterns: tuple[str, ...] = (),
        base: Voice | None = None,
        progress: Callable[[str], None] | None = None,
        **opts: Any,
    ) -> Voice:
        """Measure files and directories into a voice. Authors come from `author_map`
        (filename glob to author, `default` for the rest), then YAML frontmatter, then the
        filename's first underscore-separated part; a directory with a `manifest.jsonl`
        (as the crawler writes) supplies author, title, source and license per row. The
        corpus is built in a temporary directory and only its provenance is kept. With
        `base`, the new texts are folded into that voice's measurements instead and its
        prose files are kept."""
        from deslop.voice.calibrate import calibrate, extend
        from deslop.voice.collect import collect

        corpus, report = collect(
            [Path(p) for p in paths], name=name, author_map=author_map, progress=progress
        )
        if not corpus.manifest.documents:
            raise ValueError("no documents could be read from the given paths")
        if progress and (report.duplicates or report.failed):
            progress(
                f"skipped {len(report.duplicates)} near-duplicate(s), "
                f"{len(report.failed)} unreadable file(s)"
            )
        if base is not None:
            return extend(base, corpus, max_share=max_share, progress=progress)
        return calibrate(
            corpus,
            name=name,
            max_share=max_share,
            exemplar_words=exemplar_words,
            use_llm=use_llm,
            llm_model=llm_model,
            patterns=patterns,
            progress=progress,
            **opts,
        )

    @classmethod
    def calibrate(cls, corpus: Corpus, **opts: Any) -> Voice:
        """Measure an existing corpus directory into a voice (internal; `from_texts` is the
        public way in)."""
        from deslop.voice.calibrate import calibrate

        return calibrate(corpus, **opts)
