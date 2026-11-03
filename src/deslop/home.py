# ABOUTME: deslop's home directory: $DESLOP_HOME, else ~/.deslop. Cache, spend log, pairs,
# ABOUTME: blind-test logs, config.yaml and pricing.yaml all live under it.
from __future__ import annotations

import os
from pathlib import Path


def deslop_home() -> Path:
    """$DESLOP_HOME, else ~/.deslop, read on every call so tests can point it elsewhere."""
    return Path(os.environ.get("DESLOP_HOME", Path.home() / ".deslop"))
