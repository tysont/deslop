# ABOUTME: Runs apply on each register's model-prose version against its small voice and writes
# ABOUTME: reports/registers/<name>.md: original, slop and output as paragraphs, plus the checks.
from __future__ import annotations

import json
import sys
from pathlib import Path

from deslop import Voice, apply
from deslop.apply import coherence_report

ROOT = Path(__file__).resolve().parent.parent
REGISTERS = ROOT / "tests" / "fixtures" / "registers"
OUT = ROOT / "reports" / "registers"


def body_of(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if text.startswith("---"):
        text = text.split("---", 2)[2]
    return text.strip()


def metric_lines(result) -> list[str]:
    lines = [
        "| metric | before | after | band low | band high | before | after |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for r in result.metric_table:
        if r.band_low is None and r.band_high is None:
            continue
        lo = "-" if r.band_low is None else f"{r.band_low:.2f}"
        hi = "-" if r.band_high is None else f"{r.band_high:.2f}"
        lines.append(
            f"| {r.name} | {r.before:.2f} | {r.after:.2f} | {lo} | {hi} | "
            f"{r.status_before} | {r.status_after} |"
        )
    return lines


def run(register: str, budget: float, n: int, models: tuple[str, ...]) -> dict:
    voice = Voice.load(REGISTERS / register / f"{register}.voice")
    original = body_of(REGISTERS / register / "original.md")
    slop = body_of(REGISTERS / register / "slop.md")
    result = apply(
        slop,
        voice,
        n=n,
        budget=budget,
        models=models,
        progress=lambda m: print(f"  {m}", file=sys.stderr),
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{register}.json").write_text(result.to_json(), encoding="utf-8")
    before, after = result.report_before, result.report_after
    unit = result.units[0] if result.units else None
    lines = [
        f"# {register}: original, model prose, and the v2.2 rewrite",
        "",
        f"Voice: {voice.name} ({voice.meta.document_count} documents, "
        f"{voice.meta.total_words} words). Distance {before.distance:.0f} -> "
        f"{after.distance:.0f} ({before.verdict} -> {after.verdict}); growth "
        f"{result.growth_pct:+.0f}%; cost ${result.cost.usd:.3f} over {result.cost.calls} "
        f"call(s); stopped: {result.stopped}.",
        "",
        f"{result.judge_gate}",
        "",
        "## Original (human)",
        "",
        original,
        "",
        "## Model prose (the input to apply)",
        "",
        slop,
        "",
        "## v2.2 output"
        + (" (the original was kept: no candidate passed)" if unit and unit.kept_original else ""),
        "",
        result.text.strip(),
        "",
        "## Coherence and judge",
        "",
        coherence_report(result),
        "",
    ]
    if unit:
        lines.append(f"Unit: {unit.reason}")
        lines.append("")
        for c in unit.candidates:
            judge = "n/a" if c.judge_ok is None else ("ok" if c.judge_ok else "no")
            coherence = "n/a" if c.coherence is None else ("ok" if c.coherence.ok else "failed")
            lines.append(
                f"- candidate {c.attempt} ({c.provider}): "
                f"{'accepted' if c.accepted else 'rejected'}; win rate "
                f"{c.win_rate if c.win_rate is not None else 'n/a'}; judge {judge}; "
                f"coherence {coherence}; {c.reason}"
            )
        lines.append("")
    lines += [
        "## Metrics against the voice's band",
        "",
        *metric_lines(result),
        "",
        "## Meaning",
        "",
        "```",
        result.meaning_diff,
        "```",
        "",
    ]
    (OUT / f"{register}.md").write_text("\n".join(lines), encoding="utf-8")
    return {
        "register": register,
        "before": before.distance,
        "after": after.distance,
        "verdict": after.verdict,
        "kept": bool(unit and unit.kept_original),
        "cost": result.cost.usd,
        "calls": result.cost.calls,
        "stopped": result.stopped,
    }


def main() -> None:
    """usage: run_registers.py [BUDGET] [N] [--models claude,gpt] [REGISTER...]"""
    args = sys.argv[1:]
    models: tuple[str, ...] = ("claude", "gpt")
    if "--models" in args:
        i = args.index("--models")
        models = tuple(args[i + 1].split(","))
        del args[i : i + 2]
    budget = float(args[0]) if args else 5.0
    n = int(args[1]) if len(args) > 1 else 3
    only = args[2:] or ["essay", "explainer", "product", "fiction"]
    rows = []
    for register in only:
        print(f"== {register}", file=sys.stderr)
        rows.append(run(register, budget, n, models))
        print(json.dumps(rows[-1]), file=sys.stderr)
    (OUT / "summary.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"total ${sum(r['cost'] for r in rows):.3f}")


if __name__ == "__main__":
    main()
