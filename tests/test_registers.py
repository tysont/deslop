# ABOUTME: The register fixture set (essay, explainer, product, fiction): each has a marked
# ABOUTME: original, a generated model-prose version and a small voice that loads and scores.
from __future__ import annotations

from pathlib import Path

import pytest

from deslop import Voice, score

REGISTERS = Path(__file__).parent / "fixtures" / "registers"
NAMES = ("essay", "explainer", "product", "fiction")


def body_of(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    return text.split("---", 2)[2].strip() if text.startswith("---") else text.strip()


@pytest.mark.parametrize("name", NAMES)
def test_register_fixture_is_complete_and_marked(name):
    folder = REGISTERS / name
    original = (folder / "original.md").read_text(encoding="utf-8")
    assert original.startswith("---") and "license:" in original and "author:" in original
    slop = (folder / "slop.md").read_text(encoding="utf-8")
    assert "generated_by: anthropic:" in slop and "scripts/build_register_slop.py" in slop
    words = len(body_of(folder / "original.md").split())
    assert (300 <= words <= 500) if name == "product" else (850 <= words <= 1100)
    corpus = sorted((folder / "corpus").glob("*.txt"))
    assert 3 <= len(corpus) <= 5
    for doc in corpus:
        head = doc.read_text(encoding="utf-8")[:400]
        assert "source: https://www.gutenberg.org/ebooks/" in head and "license:" in head


@pytest.mark.parametrize("name", NAMES)
def test_register_voice_loads_and_scores_both_versions(name):
    folder = REGISTERS / name
    voice = Voice.load(folder / f"{name}.voice")
    assert voice.name == name and 3 <= voice.meta.document_count <= 5
    assert "connective_opener_pct" in voice.metrics.metrics
    for file in ("original.md", "slop.md"):
        report = score(body_of(folder / file), voice)
        assert 0 <= report.distance <= 100 and report.verdict in ("PASS", "FAIL")
