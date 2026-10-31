# ABOUTME: Shared pytest fixtures: fixture paths and a session-scoped calibrated voice.
# ABOUTME: Calibration runs once per session because it parses ~60k words.
from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture(scope="session")
def corpus_small():
    from deslop.corpus import Corpus

    return Corpus.load(FIXTURES / "corpus_small")


@pytest.fixture(scope="session")
def fixture_voice(corpus_small, tmp_path_factory):
    from deslop.voice.calibrate import calibrate

    voice = calibrate(corpus_small, name="fixture", use_llm=False)
    voice.save(tmp_path_factory.mktemp("voice") / "fixture.voice")
    return voice


@pytest.fixture(scope="session")
def slop_text() -> str:
    return (FIXTURES / "slop_sample.md").read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def heldout_texts() -> list[str]:
    return [p.read_text(encoding="utf-8") for p in sorted(FIXTURES.glob("heldout_*.md"))]
