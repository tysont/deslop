# ABOUTME: Public API for deslop: build a voice from texts, score a text against it, apply it.
# ABOUTME: CLI commands are thin wrappers over the names exported here.
"""deslop: measure a writer's voice from their texts, score any text against it, and write
or rewrite text in that voice with every proposition kept.

The public API is five names:

- [`Voice`][deslop.Voice]: build one with `Voice.from_texts(paths, name=...)`, read one with
  `Voice.load(path)`, write one with `Voice.save(path)`. A voice is a `NAME.voice` zip.
- [`score`][deslop.score.score]: a [`Report`][deslop.Report] saying how far a text is from a
  voice, against calibrated thresholds, and exactly where.
- [`apply`][deslop.apply.apply]: rewrite a text, or draft one from a [`Brief`][deslop.Brief], in the
  voice under a content lock, a semantic lock and a judge; returns a
  [`Result`][deslop.Result].
"""

from __future__ import annotations

__version__ = "2.5.0"

from deslop.apply import Result, apply
from deslop.draft import Brief, load_brief
from deslop.report import Report
from deslop.score import score
from deslop.voice import Voice

__all__ = ["Brief", "Report", "Result", "Voice", "__version__", "apply", "load_brief", "score"]
