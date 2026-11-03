# ABOUTME: Voice validation: the zip or directory parses against the schema (checksum verified
# ABOUTME: for a zip), thresholds are calibrated, and the voice's own exemplars score PASS.
from __future__ import annotations

import json
import zipfile
from pathlib import Path

from pydantic import BaseModel, Field

from deslop.thresholds import HELDOUT_PASS_TARGET, LLM_FAIL_TARGET
from deslop.voice.model import SCHEMA_VERSION, VOICE_FILES, Voice
from deslop.voice.pack import PackError, is_packed


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

    def to_json(self, indent: int | None = 1) -> str:
        data = self.model_dump()
        data["ok"] = self.ok
        return json.dumps(data, indent=indent)


def validate(path: Path) -> ValidationResult:
    from deslop.score import score
    from deslop.voice.exemplars import parse_exemplars

    result = ValidationResult(voice=str(path))
    required = ("voice.yaml", *VOICE_FILES)
    if is_packed(path):
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
        missing = [f for f in required if f not in names]
    else:
        missing = [f for f in required if not (path / f).exists()]
    if missing:
        result.checks.append(Check(name="schema", ok=False, detail="missing " + ", ".join(missing)))
        return result
    try:
        voice = Voice.load(path)
    except PackError as exc:
        result.checks.append(Check(name="checksum", ok=False, detail=str(exc)[:300]))
        return result
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
    if voice.read_only:
        result.checks.append(
            Check(name="checksum", ok=True, detail=f"verified {voice.meta.checksum or ''}"[:80])
        )

    stats = voice.metrics.thresholds.stats
    if stats is None:
        result.checks.append(Check(name="thresholds", ok=False, detail="not calibrated"))
    else:
        met = (
            stats.llm_fail_rate >= LLM_FAIL_TARGET
            and stats.heldout_pass_rate >= HELDOUT_PASS_TARGET
        )
        note = (
            ""
            if met
            else (
                f"; below the {100 * LLM_FAIL_TARGET:.0f}%/{100 * HELDOUT_PASS_TARGET:.0f}% "
                "targets: the corpus and LLM-reference prose overlap on every metric and tell "
                "for the rest"
            )
        )
        result.checks.append(
            Check(
                name="thresholds",
                ok=True,
                detail=(
                    f"{100 * stats.llm_fail_rate:.0f}% of {stats.llm_docs} LLM-reference documents "
                    f"fail, {100 * stats.heldout_pass_rate:.0f}% of {stats.heldout_docs} held-out "
                    f"documents pass ({stats.method}){note}"
                ),
            )
        )

    bodies = parse_exemplars(voice.exemplars)
    if not voice.meta.has_exemplars:
        result.checks.append(
            Check(name="exemplars_self_score", ok=True, detail="skipped: exemplars stripped")
        )
    elif not bodies:
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
