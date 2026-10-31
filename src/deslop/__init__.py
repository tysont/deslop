# ABOUTME: Public API for deslop: corpus handling, voice calibration, scoring, conversion.
# ABOUTME: CLI commands are thin wrappers over the names exported here.
"""deslop: measure a writer's voice from a corpus, score any text against it, and rewrite the
spans that deviate while leaving the rest byte-identical.

The public API is the handful of names exported here:

- [`Corpus`][deslop.Corpus]: a directory of normalized documents with a manifest.
- [`Voice`][deslop.Voice]: the calibrated profile of a corpus (metrics, lexicon, rules,
  profile, exemplars), saved as a `name.voice/` directory.
- [`score`][deslop.score.score]: a [`Report`][deslop.Report] saying how far a text is from a voice
  and exactly where.
- [`convert`][deslop.convert]: rewrite flagged sentences or paragraphs with a model until the
  text scores PASS, returning a [`ConvertResult`][deslop.ConvertResult].
- [`draft`][deslop.draft.draft] and [`load_brief`][deslop.load_brief]: write a new piece from a
  [`Brief`][deslop.Brief], returning a [`DraftResult`][deslop.DraftResult].
"""

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
