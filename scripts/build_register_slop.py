# ABOUTME: Generates, once, the model-prose ("slop") version of each register original under
# ABOUTME: tests/fixtures/registers/<name>/slop.md with one cached provider call per register.
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

from deslop.providers.base import get_provider

REGISTERS = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "registers"
SYSTEM = (
    "You are a capable writing assistant. Rewrite the piece you are given in your own natural "
    "style, as you would if asked to polish it: clear, balanced, well organised, with a "
    "satisfying close. Keep every fact, name, number, quotation and claim; keep the same "
    "number of paragraphs; do not add new information. Return only the rewritten piece."
)


def body_of(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if text.startswith("---"):
        text = text.split("---", 2)[2]
    return text.strip()


def main(provider_name: str = "anthropic") -> None:
    provider = get_provider(provider_name)
    total = 0.0
    for register in ("essay", "explainer", "product", "fiction"):
        original = body_of(REGISTERS / register / "original.md")
        completion = provider.complete(SYSTEM, original, temperature=0.7)
        total += completion.usage.cost_usd
        header = (
            "---\n"
            f"title: {register} (model prose)\n"
            f"generated_by: {completion.provider}:{completion.model}\n"
            f"generated_on: {dt.date.today().isoformat()}\n"
            "note: Generated once from original.md by scripts/build_register_slop.py; the\n"
            "  machine-written version the register's apply run starts from.\n"
            "---\n\n"
        )
        (REGISTERS / register / "slop.md").write_text(header + completion.text.strip() + "\n")
        print(
            f"{register}: {len(completion.text.split())} words from {completion.model}"
            f"{' (cached)' if completion.from_cache else ''}"
        )
    print(f"cost ${total:.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:2])
