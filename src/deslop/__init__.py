# ABOUTME: Public API for deslop: corpus handling, voice calibration, scoring, conversion.
# ABOUTME: CLI commands are thin wrappers over the names exported here.
from __future__ import annotations

__version__ = "1.0.0"

from deslop.converter import ConvertResult, convert
from deslop.corpus import Corpus
from deslop.draft import Brief, DraftResult, draft, load_brief
from deslop.report import Report
from deslop.score import score
from deslop.voice import Voice

__all__ = [
    "Brief",
    "ConvertResult",
    "Corpus",
    "DraftResult",
    "Report",
    "Voice",
    "__version__",
    "convert",
    "draft",
    "load_brief",
    "score",
]
