# ABOUTME: Rebuilds tests/fixtures/corpus_small and held-out excerpts from Project
# ABOUTME: Gutenberg sources. Run with a directory of raw pg<ID>.txt files as argv[1].
from __future__ import annotations

import re
import sys
from pathlib import Path

from deslop.corpus.manifest import Corpus
from deslop.corpus.normalize import normalize
from deslop.nlp import count_words

FOOTNOTE_RE = re.compile(r"\[\d+\]")

# (gutenberg id, author, title, year, start marker, target words)
CORPUS_DOCS = [
    (
        "5429",
        "Samuel Johnson",
        "Preface to Shakespeare",
        1765,
        "That praises are without reason",
        16000,
    ),
    (
        "5085",
        "William Hazlitt",
        "Characters of Shakespeare's Plays",
        1817,
        "CYMBELINE is one of the most",
        16000,
    ),
    (
        "12628",
        "Matthew Arnold",
        "The Function of Criticism at the Present Time",
        1865,
        "Many objections have been made to a proposition",
        16000,
    ),
    (
        "16966",
        "A. C. Bradley",
        "Shakespearean Tragedy, Lecture I",
        1904,
        "THE SUBSTANCE OF SHAKESPEAREAN TRAGEDY",
        16000,
    ),
]

HELDOUT_DOCS = [
    (
        "16209",
        "William Hazlitt",
        "On Poetry in General",
        "The best general notion which I can give of poetry",
        1500,
        "heldout_hazlitt.md",
    ),
    (
        "16966",
        "A. C. Bradley",
        "Macbeth",
        "_Macbeth_, it is probable, was the last",
        1500,
        "heldout_bradley.md",
    ),
]


def is_heading(para: str) -> bool:
    stripped = para.strip()
    if "\n" in stripped:
        return False
    words = stripped.split()
    if len(words) <= 8 and not stripped.endswith((".", "!", "?", '"', "'")):
        return True
    return stripped.upper() == stripped and len(words) < 15


def excerpt(raw: Path, gid: str, marker: str, target_words: int) -> str:
    text = normalize((raw / f"{gid}.txt").read_text(encoding="utf-8"))
    text = FOOTNOTE_RE.sub("", text)
    paras = text.split("\n\n")
    start = next(i for i, p in enumerate(paras) if p.strip().startswith(marker))
    chosen: list[str] = []
    words = 0
    for para in paras[start:]:
        if is_heading(para):
            if chosen and words >= target_words * 0.9:
                break
            continue
        chosen.append(para.strip())
        words += count_words(para)
        if words >= target_words:
            break
    return "\n\n".join(chosen) + "\n"


def main(raw_dir: str) -> None:
    raw = Path(raw_dir)
    root = Path("tests/fixtures/corpus_small")
    root.mkdir(parents=True, exist_ok=True)
    for old in root.glob("*.txt"):
        old.unlink()
    manifest = root / "corpus.yaml"
    if manifest.exists():
        manifest.unlink()
    corpus = Corpus.init(root, name="critical_prose_small")
    scratch = Path("tests/fixtures/_scratch")
    scratch.mkdir(exist_ok=True)
    for gid, author, title, year, marker, target in CORPUS_DOCS:
        tmp = scratch / f"{gid}.txt"
        tmp.write_text(excerpt(raw, gid, marker, target), encoding="utf-8")
        entry = corpus.add(
            str(tmp),
            author=author,
            title=title,
            year=year,
            license="Public domain (Project Gutenberg)",
        )
        print(f"{author:20s} {entry.words:6d} words -> {entry.path}")
    for gid, author, _title, marker, target, name in HELDOUT_DOCS:
        text = excerpt(raw, gid, marker, target)
        (Path("tests/fixtures") / name).write_text(text, encoding="utf-8")
        print(f"heldout {author:20s} {count_words(text):6d} words -> {name}")
    for f in scratch.iterdir():
        f.unlink()
    scratch.rmdir()


if __name__ == "__main__":
    main(sys.argv[1])
