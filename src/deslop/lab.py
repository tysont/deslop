# ABOUTME: Helpers behind `deslop lab`: paragraph pools for judge calibration (held-out corpus
# ABOUTME: vs LLM-reference), the blind-set export, recitation rescoring and the fixture check.
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from deslop.apply import SYSTEM_TEMPLATE, ExemplarIndex, Models, clean_response
from deslop.blind import MIN_WORDS, Pair, Pairing, shown
from deslop.judge import JudgeCalibration, calibrate_judge
from deslop.nlp import count_words, is_prose_paragraph, split_paragraphs
from deslop.providers.base import CachingProvider, ProviderError
from deslop.recitation import FIXTURE_MAX_RECITATION, RecitationResult, recitation, words_of
from deslop.semantic import extract_propositions
from deslop.thresholds import split_heldout
from deslop.voice.collect import collect
from deslop.voice.exemplars import parse_exemplars
from deslop.voice.model import Voice


def prose_paragraphs(text: str, min_words: int = MIN_WORDS) -> list[str]:
    out = []
    for start, end in split_paragraphs(text):
        para = text[start:end].strip()
        if is_prose_paragraph(para) and count_words(para) >= min_words:
            out.append(para)
    return out


def heldout_paragraphs(voice: Voice, corpus_paths: list[Path]) -> tuple[list[str], str]:
    """Paragraphs from the corpus documents of the voice's held-out authors (the same
    author split threshold calibration used), with a note saying which split was used."""
    heldout, method = split_heldout(voice.metrics.per_document)
    authors = {_key(d.author) for d in heldout}
    corpus, _ = collect(corpus_paths, name="judge-calibration")
    docs = corpus.documents()
    matched = [d for d in docs if _same_author(_key(d.entry.author), authors)]
    if not matched:
        matched, method = docs, "all documents (no corpus author matched the voice's split)"
    out: list[str] = []
    for doc in matched:
        out.extend(prose_paragraphs(doc.text))
    return out, method


def _key(author: str) -> str:
    return "".join(ch for ch in author.lower() if ch.isalnum() or ch == " ").strip()


def _same_author(key: str, keys: set[str]) -> bool:
    if key in keys:
        return True
    return any(key and (key in k.split() or k in key.split()) for k in keys)


def llm_paragraphs(path: Path) -> list[str]:
    """Paragraphs from LLM-reference essays: a directory of cached responses (`.json` with
    a `text` key, as scripts/build_llm_reference.py writes) or of .md/.txt files."""
    out: list[str] = []
    files = [p for p in sorted(path.rglob("*")) if p.is_file()]
    for p in files:
        if p.suffix == ".json":
            try:
                text = str(json.loads(p.read_text(encoding="utf-8")).get("text", ""))
            except (ValueError, OSError):
                continue
        elif p.suffix in (".md", ".txt"):
            text = p.read_text(encoding="utf-8")
        else:
            continue
        out.extend(prose_paragraphs(text))
    return out


def calibrate_judges(
    voice: Voice,
    providers: list[CachingProvider],
    llm: list[str],
    corpus: list[str],
    *,
    pairs: int = 100,
    seed: int = 0,
) -> dict[str, JudgeCalibration]:
    """Run the comparative judge calibration with each judge; keyed by the judge's model."""
    exemplars = parse_exemplars(voice.exemplars)[:2]
    out: dict[str, JudgeCalibration] = {}
    for p in providers:
        try:
            out[p.default_model] = calibrate_judge(
                voice, p, llm, corpus, exemplars=exemplars, pairs=pairs, seed=seed
            )
        except ProviderError:
            continue
    return out


def export_blind(pairing: Pairing, out_dir: Path, voice: str) -> tuple[Path, Path]:
    """Write a blind set for a person to judge offline: `pairs.md` shows each pair as A and B
    in shuffled order with no answers; `key.json` holds which letter was the candidate."""
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Blind set against {voice}",
        "",
        "For each pair, decide which paragraph is machine-written (A or B) and note why. "
        "The answers are in key.json; do not open it until you have finished.",
        "",
    ]
    key = []
    for i, pair in enumerate(pairing.pairs, 1):
        a, b = shown(pair)
        lines += [
            f"## Pair {i}",
            "",
            "**A**",
            "",
            a,
            "",
            "**B**",
            "",
            b,
            "",
            "Answer: ___  Why:",
            "",
        ]
        key.append(
            {
                "pair": i,
                "candidate": pair.candidate_position,
                "candidate_source": pair.candidate_source,
                "control_source": pair.control_source,
                "overlap": pair.overlap,
            }
        )
    pairs_path = out_dir / "pairs.md"
    key_path = out_dir / "key.json"
    pairs_path.write_text("\n".join(lines), encoding="utf-8")
    key_path.write_text(json.dumps(key, indent=1), encoding="utf-8")
    return pairs_path, key_path


def rescore_recitation(result: dict[str, Any], reference: str) -> list[dict[str, Any]]:
    """The recitation check of every candidate in a saved apply result against `reference`,
    with each unit's original as the input; offline."""
    rows = []
    for unit in result.get("units", []):
        for c in unit.get("candidates", []):
            if not c.get("text"):
                continue
            r = recitation(c["text"], reference, unit["original"])
            status = (
                "winner"
                if unit.get("chosen") is not None and unit["candidates"][unit["chosen"]] is c
                else ("accepted" if c.get("accepted") else "rejected")
            )
            rows.append(
                {"unit": unit["index"], "attempt": c["attempt"], "status": status, **r.model_dump()}
            )
    return rows


FIXTURE_DRAFT_TEMPLATE = """# Exemplars of the voice
{exemplars}

# What the piece asserts (keep every one, add none)
{propositions}

# Instruction
Write a piece of about {words} words as this writer would, from these propositions alone and
in this order. Keep every proposition and add none. Return the piece and nothing else."""


class FixtureCheck(BaseModel):
    """Whether a text can serve as a held-out fixture: a draft written from its propositions
    alone must recite under `threshold` percent of it."""

    words: int
    propositions: list[str]
    draft: str
    recitation: RecitationResult
    threshold: float = FIXTURE_MAX_RECITATION
    novel_words: int = 0  # recited words that are not in the propositions' vocabulary

    @property
    def qualifies(self) -> bool:
        return self.recitation.share < self.threshold


def fixture_check(
    text: str, voice: Voice, models: Models, *, threshold: float = FIXTURE_MAX_RECITATION
) -> FixtureCheck:
    """Extract the text's propositions (one call), write one draft from them alone with the
    writer model (one call, temperature 0, cached), and measure how much of the text the
    draft recites in runs of eight or more words the propositions do not contain. The text
    itself is never shown to the writer."""
    props = extract_propositions(text, models.writer)
    listed = "\n".join(f"{i}. {p}" for i, p in enumerate(props, 1))
    system = SYSTEM_TEMPLATE.format(profile=voice.profile.strip(), rules=voice.rules.strip())
    user = FIXTURE_DRAFT_TEMPLATE.format(
        exemplars="\n\n".join(ExemplarIndex(voice).closest(listed)) or "(no exemplars)",
        propositions=listed or "(none)",
        words=count_words(text),
    )
    draft = clean_response(models.writer.complete(system, user, temperature=0.0).text)
    rec = recitation(draft, text, listed)
    vocabulary = set(words_of(listed))
    return FixtureCheck(
        words=count_words(text),
        propositions=props,
        draft=draft,
        recitation=rec,
        threshold=threshold,
        novel_words=sum(w not in vocabulary for run in rec.runs for w in words_of(run)),
    )


__all__ = [
    "FixtureCheck",
    "Pair",
    "calibrate_judges",
    "export_blind",
    "fixture_check",
    "heldout_paragraphs",
    "llm_paragraphs",
    "prose_paragraphs",
    "rescore_recitation",
]
