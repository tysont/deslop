# ABOUTME: Voice maintenance: snapshot a voice into history/, fold accepted texts back into its
# ABOUTME: corpus and recalibrate the measured files, and resolve <voice>@prev / @<stamp> refs.
from __future__ import annotations

import datetime as dt
import shutil
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, Field

from deslop.corpus import Corpus, DuplicateError
from deslop.providers.base import CachingProvider
from deslop.voice.calibrate import calibrate
from deslop.voice.model import Voice

HISTORY = "history"
ACCEPTED_SUFFIX = "(accepted)"


class UpdateResult(BaseModel):
    voice: str
    snapshot: Path
    added: list[str] = Field(default_factory=list)
    skipped: list[str] = Field(default_factory=list)
    total_words: int = 0


def snapshot(voice_dir: Path) -> Path:
    """Copy the voice's files into <voice>/history/<UTC stamp>/ and return that directory."""
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    target = voice_dir / HISTORY / stamp
    n = 2
    while target.exists():
        target = voice_dir / HISTORY / f"{stamp}-{n}"
        n += 1
    target.mkdir(parents=True)
    for item in voice_dir.iterdir():
        if item.name == HISTORY:
            continue
        if item.is_dir():
            shutil.copytree(item, target / item.name)
        else:
            shutil.copy2(item, target / item.name)
    return target


def history(voice_dir: Path) -> list[Path]:
    root = voice_dir / HISTORY
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if (p / "voice.yaml").exists())


def resolve_ref(ref: str) -> Path:
    """`path`, `path@prev` (latest snapshot) or `path@<stamp>` (unique prefix of a snapshot)."""
    if "@" not in ref:
        return Path(ref)
    base, _, tag = ref.rpartition("@")
    voice_dir = Path(base)
    snaps = history(voice_dir)
    if tag == "prev":
        if not snaps:
            raise FileNotFoundError(f"{voice_dir} has no history yet")
        return snaps[-1]
    matches = [p for p in snaps if p.name.startswith(tag)]
    if len(matches) != 1:
        raise FileNotFoundError(f"{voice_dir}@{tag}: {len(matches)} snapshot(s) match")
    return matches[0]


def corpus_for(voice: Voice, corpus: Path | None) -> Corpus:
    path = corpus or (Path(voice.meta.corpus_path) if voice.meta.corpus_path else None)
    if path is None:
        raise FileNotFoundError(
            f"voice {voice.name} does not record its corpus path; pass --corpus"
        )
    if not (path / "corpus.yaml").exists():
        raise FileNotFoundError(f"corpus not found at {path}")
    return Corpus.load(path)


def update(
    voice_dir: Path,
    accept: list[Path],
    *,
    corpus: Path | None = None,
    weight: float = 0.5,
    regenerate: bool = False,
    use_llm: bool = True,
    provider: CachingProvider | None = None,
    progress: Callable[[str], None] | None = None,
) -> UpdateResult:
    """Add accepted texts to the voice's corpus at `weight`, snapshot the voice, and rewrite
    metrics.json and lexicon.json. profile.md, rules.md and exemplars.md are kept unless
    `regenerate` is set."""
    old = Voice.load(voice_dir)
    corp = corpus_for(old, corpus)
    result = UpdateResult(voice=old.name, snapshot=snapshot(voice_dir))
    author = f"{old.name} {ACCEPTED_SUFFIX}"
    for path in accept:
        try:
            corp.add(str(path), author=author, title=path.stem, weight=weight)
            result.added.append(path.name)
        except DuplicateError:
            result.skipped.append(path.name)
            if progress:
                progress(f"skipping {path.name}: already in the corpus")
    new = calibrate(
        corp,
        name=old.name,
        max_share=old.meta.max_share,
        use_llm=regenerate and use_llm,
        provider=provider,
        ban_ratio=old.lexicon.ban_ratio,
        chunk_words=old.metrics.chunk_words,
        progress=progress,
    )
    if not regenerate:
        new.profile, new.rules, new.exemplars = old.profile, old.rules, old.exemplars
        new.meta.profile_source = old.meta.profile_source
    new.meta.created = old.meta.created
    new.save(voice_dir)
    result.total_words = new.meta.total_words
    return result
