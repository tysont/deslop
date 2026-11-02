# ABOUTME: Generates, once, the model-prose ("slop") version of each register original: the
# ABOUTME: original's propositions are extracted, then a piece is written from them alone.
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

from deslop import Voice, score, spend
from deslop.baselines import llm_reference
from deslop.metrics import compute_all
from deslop.nlp import analyze
from deslop.providers.base import get_provider
from deslop.semantic import extract_propositions
from deslop.stats import MetricStat

REGISTERS = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "registers"
CAP = 3.0  # USD for the whole run
KINDS = {
    "essay": "a personal essay",
    "explainer": "a technical explainer",
    "product": "a product page",
    "fiction": "a short-story excerpt",
}
SYSTEM = "You are a writing assistant. Return only the piece you are asked for, as prose."


def body_of(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if text.startswith("---"):
        text = text.split("---", 2)[2]
    return text.strip()


def instruction(kind: str, words: int, propositions: list[str]) -> str:
    listed = "\n".join(f"{i}. {p}" for i, p in enumerate(propositions, 1))
    return (
        f"Write {kind} of roughly {words} words that makes every one of these points, in "
        f"this order, and no others.\n\n{listed}"
    )


def mean_abs_z(text: str, stats: dict[str, MetricStat]) -> float:
    """How far the text's metrics sit from a reference, as the mean |z| over the metrics the
    reference measures with a nonzero spread."""
    values = compute_all(analyze(text))
    zs = [
        abs((values[name] - st.mean) / st.std)
        for name, st in stats.items()
        if name in values and st.std > 0
    ]
    return sum(zs) / len(zs) if zs else 0.0


def main(provider_name: str = "anthropic", only: str | None = None) -> None:
    """Write slop.md and propositions.json for every register, or for `only`."""
    spend.set_context(command="build_register_slop")
    spend.set_budget(CAP)
    provider = get_provider(provider_name)
    llm = llm_reference()
    total = 0.0
    results = []
    for register, kind in KINDS.items():
        if only and register != only:
            continue
        original = body_of(REGISTERS / register / "original.md")
        props = extract_propositions(original, provider)  # temperature 0, cached
        words = len(original.split())
        completion = provider.complete(SYSTEM, instruction(kind, words, props), temperature=0.7)
        total += completion.usage.cost_usd
        text = completion.text.strip()
        voice = Voice.load(REGISTERS / register / f"{register}.voice")
        report = score(text, voice)
        z_voice = mean_abs_z(text, voice.metrics.metrics)
        z_llm = mean_abs_z(text, llm.metrics)
        verdict = "closer to the LLM baseline" if z_llm < z_voice else "closer to the human voice"
        header = (
            "---\n"
            f"title: {register} (model prose from propositions)\n"
            f"generated_by: {completion.provider}:{completion.model}\n"
            f"generated_on: {dt.date.today().isoformat()}\n"
            f"propositions: {len(props)}\n"
            f"score_vs_voice: {report.distance:.1f} ({report.verdict})\n"
            f"mean_abs_z_vs_voice: {z_voice:.2f}\n"
            f"mean_abs_z_vs_llm_reference: {z_llm:.2f}\n"
            f"closeness: {verdict}\n"
            "note: Written by scripts/build_register_slop.py from the original's proposition\n"
            "  list alone; the model never saw the original.\n"
            "---\n\n"
        )
        (REGISTERS / register / "slop.md").write_text(header + text + "\n", encoding="utf-8")
        (REGISTERS / register / "propositions.json").write_text(
            json.dumps(props, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        results.append(
            (register, len(text.split()), report.distance, report.verdict, z_voice, z_llm, verdict)
        )
        print(
            f"{register}: {len(text.split())} words, {len(props)} propositions; vs voice "
            f"{report.distance:.0f} ({report.verdict}); mean |z| voice {z_voice:.2f}, "
            f"LLM {z_llm:.2f}: {verdict}{' (cached)' if completion.from_cache else ''}"
        )
    print(f"cost ${total:.3f} (cap ${CAP:.2f})")


if __name__ == "__main__":
    main(*sys.argv[1:3])
