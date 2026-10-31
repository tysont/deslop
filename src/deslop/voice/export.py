# ABOUTME: Fine-tuning export: accepted convert pairs plus summarized corpus chunks, written as
# ABOUTME: OpenAI chat-format or plain JSONL, with a token count and a cost estimate. Never trains.
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, Field

from deslop.converter import PAIRS_DIR
from deslop.corpus import Corpus
from deslop.nlp import count_words, is_prose_paragraph, split_paragraphs
from deslop.pricing import training_price
from deslop.providers.base import CachingProvider
from deslop.voice.model import Voice

FORMATS = ("openai-ft", "jsonl")
CHUNK_MIN, CHUNK_MAX = 300, 600
SUMMARY_SYSTEM = "You summarize passages for a writer's brief. Reply with the summary only."
SUMMARY_USER = (
    "Summarize what this passage says in two or three plain sentences: its subject, the claims "
    "it makes, and the names, works, or facts it mentions. Do not describe its style.\n\n{chunk}"
)


class ExportRecord(BaseModel):
    kind: str  # pair | chunk
    source: str
    user: str
    assistant: str


class ExportResult(BaseModel):
    voice: str
    format: str
    records: list[ExportRecord] = Field(default_factory=list)
    pairs: int = 0
    chunks: int = 0
    tokens: int = 0  # estimated from characters
    cost_model: str = ""
    estimated_training_cost: float | None = None
    summary_cost: float = 0.0
    summary_calls: int = 0

    def to_json(self, indent: int | None = 1) -> str:
        return self.model_dump_json(indent=indent, exclude={"records"})


def estimate_tokens(text: str) -> int:
    """Rough token count: about four characters per token for English prose."""
    return max(1, round(len(text) / 4))


def load_pairs(voice: Voice, path: Path | None = None) -> list[tuple[str, str]]:
    target = path or (PAIRS_DIR / f"{voice.name}.jsonl")
    if not target.exists():
        return []
    seen: set[tuple[str, str]] = set()
    pairs: list[tuple[str, str]] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if not rec.get("accepted") or not rec.get("after"):
            continue
        key = (rec["before"], rec["after"])
        if key not in seen and rec["before"] != rec["after"]:
            seen.add(key)
            pairs.append(key)
    return pairs


def chunk_text(text: str, lo: int = CHUNK_MIN, hi: int = CHUNK_MAX) -> list[str]:
    """Group consecutive prose paragraphs into chunks of lo..hi words. Paragraphs longer than
    hi are skipped; a trailing group under lo words is dropped."""
    chunks: list[str] = []
    current: list[str] = []
    words = 0
    for start, end in split_paragraphs(text):
        para = text[start:end]
        if not is_prose_paragraph(para):
            continue
        n = count_words(para)
        if n > hi:
            continue
        if words + n > hi:
            if words >= lo:
                chunks.append("\n\n".join(current))
            current, words = [], 0
        current.append(para)
        words += n
        if words >= lo:
            chunks.append("\n\n".join(current))
            current, words = [], 0
    return chunks


def to_line(record: ExportRecord, fmt: str, system: str) -> str:
    if fmt == "openai-ft":
        payload = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": record.user},
                {"role": "assistant", "content": record.assistant},
            ]
        }
    else:
        payload = {
            "system": system,
            "input": record.user,
            "output": record.assistant,
            "kind": record.kind,
            "source": record.source,
        }
    return json.dumps(payload, ensure_ascii=False)


def export(
    voice: Voice,
    *,
    fmt: str = "openai-ft",
    pairs_path: Path | None = None,
    corpus: Path | None = None,
    include_pairs: bool = True,
    include_chunks: bool = True,
    provider: CachingProvider | None = None,
    max_chunks: int | None = None,
    cost_model: str = "gpt-4.1",
    progress: Callable[[str], None] | None = None,
) -> ExportResult:
    if fmt not in FORMATS:
        raise ValueError(f"format must be one of {', '.join(FORMATS)}")
    result = ExportResult(voice=voice.name, format=fmt, cost_model=cost_model)
    if include_pairs:
        for before, after in load_pairs(voice, pairs_path):
            result.records.append(
                ExportRecord(kind="pair", source="pairs", user=before, assistant=after)
            )
        result.pairs = len(result.records)
    if include_chunks:
        from deslop.voice.update import corpus_for

        corp: Corpus = corpus_for(voice, corpus)
        if provider is None:
            from deslop.providers.base import first_available

            provider = first_available()
        if provider is None:
            raise RuntimeError("corpus chunks need a provider to write summaries")
        chunks = [
            (doc.entry.path, chunk) for doc in corp.documents() for chunk in chunk_text(doc.text)
        ]
        if max_chunks is not None:
            chunks = chunks[:max_chunks]
        before_usage = provider.usage.cost_usd, provider.usage.calls
        for i, (source, chunk) in enumerate(chunks, 1):
            if progress and (i == 1 or i % 10 == 0):
                progress(f"summarizing chunk {i}/{len(chunks)}")
            completion = provider.complete(
                SUMMARY_SYSTEM, SUMMARY_USER.format(chunk=chunk), temperature=0.2
            )
            result.records.append(
                ExportRecord(
                    kind="chunk", source=source, user=completion.text.strip(), assistant=chunk
                )
            )
        result.chunks = len(chunks)
        result.summary_cost = round(provider.usage.cost_usd - before_usage[0], 4)
        result.summary_calls = provider.usage.calls - before_usage[1]
    system = voice.profile.strip()
    result.tokens = sum(
        estimate_tokens(system) + estimate_tokens(r.user) + estimate_tokens(r.assistant)
        for r in result.records
    )
    result.estimated_training_cost = training_price(cost_model, result.tokens)
    return result


def write_jsonl(result: ExportResult, voice: Voice, out: Path) -> int:
    system = voice.profile.strip()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for record in result.records:
            fh.write(to_line(record, result.format, system) + "\n")
    return len(result.records)
