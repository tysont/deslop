# ABOUTME: Semantic lock: extracts a unit's propositions, then has a second provider check
# ABOUTME: that a rewrite keeps every one and adds none, naming any proposition that fails.
from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field

from deslop.lock import is_primary_citation
from deslop.providers.base import CachingProvider

PROPOSITION_SYSTEM = """You list the propositions a passage asserts. A proposition is one
short declarative claim in plain words, stated so it can be checked against another text.
Include claims made in passing, claims made as aphorisms (restate the idea plainly), and
claims attributed to others ("X says Y" is a claim). Do not include questions, instructions to
the reader, or citations. Return a numbered list, one proposition per line, nothing else."""

ENTAILMENT_SYSTEM = """You compare an original passage with a rewrite. Answer in JSON only.

Given the original's numbered propositions, decide for each whether the rewrite still asserts
or clearly entails it (true) or drops, weakens or contradicts it (false). A plainer or
reordered statement of the same idea still counts as kept.

Then list every claim the rewrite makes that the original does not entail. Classify each as
"claim" (new substantive content), "reader_address" (an imperative or question addressed to
the reader that adds no content), or "citation" (a reference to the primary text under
discussion, such as a verse, page, line or Stephanus number).

Return exactly: {"original": [{"n": 1, "kept": true, "note": ""}, ...],
"added": [{"text": "...", "kind": "claim" | "reader_address" | "citation"}, ...]}"""


class PropositionCheck(BaseModel):
    proposition: str
    kept: bool
    note: str = ""


class AddedClaim(BaseModel):
    text: str
    kind: str  # claim | reader_address | citation


class SemanticResult(BaseModel):
    """Outcome of the semantic lock for one unit: each original proposition with whether the
    rewrite kept it, the rewrite's additions, and the failures that reject the candidate."""

    checks: list[PropositionCheck] = Field(default_factory=list)
    added: list[AddedClaim] = Field(default_factory=list)
    provider: str = ""
    model: str = ""

    @property
    def failing(self) -> list[str]:
        out = [f"dropped: {c.proposition}" for c in self.checks if not c.kept]
        out += [f"added: {a.text}" for a in self.added if a.kind == "claim"]
        return out

    @property
    def ok(self) -> bool:
        return not self.failing


def parse_numbered(text: str) -> list[str]:
    items = []
    for line in text.splitlines():
        m = re.match(r"\s*(?:\d+[.)]|[-*•])\s+(.*\S)", line)
        if m:
            items.append(m.group(1).strip())
    return items


def extract_propositions(text: str, provider: CachingProvider) -> list[str]:
    """The unit's claims as short declaratives. Deterministic (temperature 0), so the
    provider cache answers repeat calls for the same unit."""
    completion = provider.complete(PROPOSITION_SYSTEM, text.strip(), temperature=0.0)
    props = parse_numbered(completion.text)
    return props or [line.strip() for line in completion.text.splitlines() if line.strip()]


def _json_block(text: str) -> dict[str, Any]:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object in response")
    data = json.loads(text[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("response is not a JSON object")
    return data


def check_meaning(
    original: str,
    candidate: str,
    propositions: list[str],
    provider: CachingProvider,
) -> SemanticResult:
    """Ask `provider` (a different one from the writer) whether `candidate` keeps every
    proposition of `original` and adds none. Reader-address imperatives and primary-text
    citations are exempt from the no-additions rule."""
    numbered = "\n".join(f"{i}. {p}" for i, p in enumerate(propositions, 1))
    user = (
        f"ORIGINAL:\n{original.strip()}\n\nORIGINAL PROPOSITIONS:\n{numbered}\n\n"
        f"REWRITE:\n{candidate.strip()}"
    )
    completion = provider.complete(ENTAILMENT_SYSTEM, user, temperature=0.0)
    try:
        data = _json_block(completion.text)
    except (ValueError, json.JSONDecodeError):
        return SemanticResult(
            checks=[
                PropositionCheck(proposition=p, kept=False, note="judge gave no JSON")
                for p in propositions
            ],
            provider=completion.provider,
            model=completion.model,
        )
    kept: dict[int, tuple[bool, str]] = {}
    for item in data.get("original", []) or []:
        if isinstance(item, dict) and "n" in item:
            try:
                kept[int(item["n"])] = (bool(item.get("kept")), str(item.get("note", "")))
            except (TypeError, ValueError):
                continue
    checks = [
        PropositionCheck(
            proposition=p,
            kept=kept.get(i, (False, "not judged"))[0],
            note=kept.get(i, (False, ""))[1],
        )
        for i, p in enumerate(propositions, 1)
    ]
    added = []
    for item in data.get("added", []) or []:
        if not isinstance(item, dict) or not str(item.get("text", "")).strip():
            continue
        kind = str(item.get("kind", "claim"))
        text = str(item["text"]).strip()
        if is_primary_citation(text):
            kind = "citation"
        added.append(AddedClaim(text=text, kind=kind))
    return SemanticResult(
        checks=checks, added=added, provider=completion.provider, model=completion.model
    )
