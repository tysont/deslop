# ABOUTME: Blind test: pairs candidate paragraphs with corpus paragraphs of similar length and
# ABOUTME: highest lexical overlap, logs the judge's A/B calls, and reports accuracy and reasons.
from __future__ import annotations

import datetime as dt
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from deslop.corpus import Corpus
from deslop.nlp import WORD_RE, count_words, is_prose_paragraph, split_paragraphs

BLIND_DIR = Path(os.environ.get("DESLOP_HOME", Path.home() / ".deslop")) / "blind"
MIN_WORDS = 40
LENGTH_WINDOW = 0.25
MIN_OVERLAP = 0.15
CANDIDATE_SUFFIXES = (".md", ".txt")


class Pair(BaseModel):
    candidate: str
    candidate_source: str
    control: str
    control_source: str
    overlap: float
    candidate_position: str = "A"  # which letter the candidate was shown as


class Pairing(BaseModel):
    """The pairs made for one session plus the candidates that were tried and left out."""

    pairs: list[Pair] = Field(default_factory=list)
    skipped: int = 0  # best control overlapped the candidate by less than min_overlap
    unpaired: int = 0  # candidates tried that got no pair, for any reason


class Summary(BaseModel):
    voice: str
    session_total: int = 0
    session_correct: int = 0
    total: int = 0
    correct: int = 0
    skipped: int = 0
    unpaired: int = 0
    top_reasons: list[tuple[str, int]] = Field(default_factory=list)

    @property
    def session_accuracy(self) -> float | None:
        return self.session_correct / self.session_total if self.session_total else None

    @property
    def accuracy(self) -> float | None:
        return self.correct / self.total if self.total else None

    def to_json(self, indent: int | None = 1) -> str:
        data = self.model_dump()
        data["session_accuracy"] = self.session_accuracy
        data["accuracy"] = self.accuracy
        return json.dumps(data, indent=indent)


def _stopwords() -> set[str]:
    from spacy.lang.en.stop_words import STOP_WORDS

    return set(STOP_WORDS)


def content_words(text: str) -> frozenset[str]:
    stop = _stopwords()
    return frozenset(
        w.lower() for w in WORD_RE.findall(text) if w.lower() not in stop and len(w) > 2
    )


def paragraphs_of(text: str, source: str) -> list[tuple[str, str]]:
    out = []
    for start, end in split_paragraphs(text):
        para = text[start:end].strip()
        if is_prose_paragraph(para) and count_words(para) >= MIN_WORDS:
            out.append((source, para))
    return out


def candidates_from_dir(directory: Path) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() in CANDIDATE_SUFFIXES and path.is_file():
            out.extend(paragraphs_of(path.read_text(encoding="utf-8"), path.name))
    return out


def corpus_pool(corpus: Corpus) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for doc in corpus.documents():
        out.extend(paragraphs_of(doc.text, doc.entry.path))
    return out


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    return len(a & b) / len(a | b) if a or b else 0.0


def make_pairs(
    candidates: list[tuple[str, str]],
    pool: list[tuple[str, str]],
    n: int,
    *,
    seed: int | None = None,
    min_overlap: float = MIN_OVERLAP,
) -> Pairing:
    """Walk the candidates in random order until n pairs are made. Each candidate is paired
    with the unused pool paragraph of similar length (+/- LENGTH_WINDOW) and highest
    content-word Jaccard overlap; if nothing is within the length window, the nearest length
    among the ten most similar paragraphs is used. A candidate whose best control overlaps it
    by less than min_overlap is skipped."""
    rng = random.Random(seed)
    order = list(candidates)
    rng.shuffle(order)
    pool_words = [(src, text, content_words(text), count_words(text)) for src, text in pool]
    used: set[int] = set()
    pairing = Pairing()
    for src, text in order:
        if len(pairing.pairs) >= n:
            break
        words = count_words(text)
        cw = content_words(text)
        scored = [
            (jaccard(cw, pw), i)
            for i, (_, _, pw, n_words) in enumerate(pool_words)
            if i not in used
        ]
        if not scored:
            pairing.unpaired += 1
            continue
        in_window = [
            (s, i) for s, i in scored if abs(pool_words[i][3] - words) <= LENGTH_WINDOW * words
        ]
        if in_window:
            score, idx = max(in_window)
        else:
            top = sorted(scored, reverse=True)[:10]
            score, idx = min(top, key=lambda si: abs(pool_words[si[1]][3] - words))
        if score < min_overlap:
            pairing.skipped += 1
            pairing.unpaired += 1
            continue
        used.add(idx)
        control_src, control, _, _ = pool_words[idx]
        pairing.pairs.append(
            Pair(
                candidate=text,
                candidate_source=src,
                control=control,
                control_source=control_src,
                overlap=round(score, 3),
                candidate_position=rng.choice("AB"),
            )
        )
    return pairing


def shown(pair: Pair) -> tuple[str, str]:
    """(A, B) texts in the order the judge sees them."""
    if pair.candidate_position == "A":
        return pair.candidate, pair.control
    return pair.control, pair.candidate


def record(
    voice: str, pair: Pair, guess: str, reason: str, *, log_path: Path | None = None
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "timestamp": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "voice": voice,
        "candidate_source": pair.candidate_source,
        "control_source": pair.control_source,
        "overlap": pair.overlap,
        "candidate_position": pair.candidate_position,
        "guess": guess.upper(),
        "correct": guess.upper() == pair.candidate_position,
        "reason": reason.strip(),
    }
    target = log_path or (BLIND_DIR / f"{voice}.jsonl")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def summarize(
    voice: str, session: list[dict[str, Any]], *, log_path: Path | None = None
) -> Summary:
    target = log_path or (BLIND_DIR / f"{voice}.jsonl")
    history = []
    if target.exists():
        history = [json.loads(ln) for ln in target.read_text(encoding="utf-8").splitlines() if ln]
    reasons = Counter(
        " ".join(r["reason"].lower().split()) for r in history if r.get("reason", "").strip()
    )
    return Summary(
        voice=voice,
        session_total=len(session),
        session_correct=sum(1 for r in session if r["correct"]),
        total=len(history),
        correct=sum(1 for r in history if r.get("correct")),
        top_reasons=reasons.most_common(5),
    )
