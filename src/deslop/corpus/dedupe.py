# ABOUTME: Near-duplicate detection for corpus documents using simhash.
# ABOUTME: Documents within a small Hamming distance of an existing one are rejected.
from __future__ import annotations

import re
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", SyntaxWarning)
    from simhash import Simhash

TOKEN_RE = re.compile(r"[a-z']+")


def fingerprint(text: str) -> int:
    tokens = TOKEN_RE.findall(text.lower())
    shingles = [" ".join(tokens[i : i + 3]) for i in range(max(len(tokens) - 2, 1))]
    return int(Simhash(shingles, f=64).value)


def distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def is_duplicate(fp: int, existing: list[int], max_distance: int = 3) -> bool:
    return any(distance(fp, other) <= max_distance for other in existing)
