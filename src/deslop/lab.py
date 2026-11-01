# ABOUTME: Helpers behind `deslop lab`: paragraph pools for judge calibration (held-out corpus
# ABOUTME: vs LLM-reference) and the blind-set export a person can judge offline.
from __future__ import annotations

import json
from pathlib import Path

from deslop.blind import MIN_WORDS, Pair, Pairing, shown
from deslop.judge import JudgeCalibration, calibrate_judge
from deslop.nlp import count_words, is_prose_paragraph, split_paragraphs
from deslop.providers.base import CachingProvider, ProviderError
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


__all__ = [
    "Pair",
    "calibrate_judges",
    "export_blind",
    "heldout_paragraphs",
    "llm_paragraphs",
    "prose_paragraphs",
]
