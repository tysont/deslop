# ABOUTME: Voice validation: files parse against the schema, the corpus hash still matches,
# ABOUTME: and the voice's own exemplars score PASS against it.
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from deslop.corpus import Corpus
from deslop.voice.model import SCHEMA_VERSION, Voice

REQUIRED_FILES = (
    "voice.yaml",
    "metrics.json",
    "lexicon.json",
    "rules.md",
    "profile.md",
    "exemplars.md",
)


class Check(BaseModel):
    name: str
    ok: bool
    detail: str = ""


class ValidationResult(BaseModel):
    voice: str
    checks: list[Check] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    def by_name(self, name: str) -> Check:
        return next(c for c in self.checks if c.name == name)

    def to_json(self, indent: int | None = 1) -> str:
        data = self.model_dump()
        data["ok"] = self.ok
        import json

        return json.dumps(data, indent=indent)


def validate(voice_dir: Path, *, corpus: Path | None = None) -> ValidationResult:
    from deslop.score import score
    from deslop.voice.exemplars import parse_exemplars

    result = ValidationResult(voice=str(voice_dir))
    missing = [f for f in REQUIRED_FILES if not (voice_dir / f).exists()]
    if missing:
        result.checks.append(Check(name="schema", ok=False, detail="missing " + ", ".join(missing)))
        return result
    try:
        voice = Voice.load(voice_dir)
    except Exception as exc:  # noqa: BLE001 - any parse failure is a schema failure
        result.checks.append(Check(name="schema", ok=False, detail=str(exc)[:300]))
        return result
    if voice.meta.schema_version != SCHEMA_VERSION:
        result.checks.append(
            Check(
                name="schema",
                ok=False,
                detail=f"schema_version {voice.meta.schema_version}, expected {SCHEMA_VERSION}",
            )
        )
    elif not voice.metrics.metrics or not voice.lexicon.patterns:
        result.checks.append(Check(name="schema", ok=False, detail="metrics or lexicon empty"))
    else:
        result.checks.append(Check(name="schema", ok=True, detail=f"schema v{SCHEMA_VERSION}"))

    path = corpus or (Path(voice.meta.corpus_path) if voice.meta.corpus_path else None)
    if path is None:
        result.checks.append(
            Check(name="corpus_hash", ok=True, detail="skipped: no corpus path recorded")
        )
    elif not (path / "corpus.yaml").exists():
        result.checks.append(Check(name="corpus_hash", ok=False, detail=f"corpus not at {path}"))
    else:
        actual = Corpus.load(path).content_hash()
        ok = actual == voice.meta.corpus_hash
        result.checks.append(
            Check(
                name="corpus_hash",
                ok=ok,
                detail=f"voice {voice.meta.corpus_hash}, corpus {actual}",
            )
        )

    bodies = parse_exemplars(voice.exemplars)
    if not bodies:
        result.checks.append(Check(name="exemplars_self_score", ok=False, detail="no exemplars"))
    else:
        report = score("\n\n".join(bodies), voice)
        result.checks.append(
            Check(
                name="exemplars_self_score",
                ok=report.verdict == "PASS",
                detail=f"{report.verdict} distance {report.distance:.1f}"
                + ("; " + "; ".join(report.fixes[:3]) if report.fixes else ""),
            )
        )
    return result
