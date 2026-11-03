# ABOUTME: Runs apply on each register fixture's model prose against its small voice and writes
# ABOUTME: reports/registers/<name>.json and <name>.html (the three-column comparison page).
from __future__ import annotations

import json
import sys
from pathlib import Path

from deslop import Voice, apply
from deslop.compare_page import build_page

ROOT = Path(__file__).resolve().parent.parent
REGISTERS = ROOT / "tests" / "fixtures" / "registers"
OUT = ROOT / "reports" / "registers"
NAMES = ["essay", "explainer", "product", "fiction"]
ESTIMATE_PER_DOCUMENT = 2.0  # USD per document at n=3, about $2.75 per 1,000 words


def body_of(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    return text.split("---", 2)[2].strip() if text.startswith("---") else text.strip()


def run(register: str, budget: float, n: int) -> dict[str, object]:
    """Apply the register's voice to its model prose, with the human original as the
    recitation reference, and write the result JSON and the comparison page."""
    voice = Voice.load(REGISTERS / register / f"{register}.voice")
    original = body_of(REGISTERS / register / "original.md")
    slop = body_of(REGISTERS / register / "slop.md")
    result = apply(
        slop,
        voice,
        n=n,
        budget=budget,
        reference=original,
        report_all=True,
        progress=lambda m: print(f"  {m}", file=sys.stderr),
    )
    OUT.mkdir(parents=True, exist_ok=True)
    payload = result.to_json()
    (OUT / f"{register}.json").write_text(payload, encoding="utf-8")
    page = build_page(json.loads(payload), reference=original)
    (OUT / f"{register}.html").write_text(page, encoding="utf-8")
    return {
        "register": register,
        "before": result.report_before.distance,
        "after": result.report_after.distance,
        "verdict": result.report_after.verdict,
        "kept": bool(result.units and result.units[0].kept_original),
        "cost": result.cost.usd,
        "calls": result.cost.calls,
        "stopped": result.stopped,
    }


def projection(registers: list[str], n: int, budget: float) -> tuple[float, str]:
    """The cost this run may reach: per document the smaller of its budget and the estimate
    scaled by n, summed; with a line explaining it."""
    per_doc = min(budget, ESTIMATE_PER_DOCUMENT * n / 3)
    total = per_doc * len(registers)
    line = (
        f"projection: {len(registers)} document(s) x up to ${per_doc:.2f} each "
        f"(estimate ${ESTIMATE_PER_DOCUMENT:.2f} at n=3, scaled to n={n}, capped by the "
        f"per-document budget ${budget:.2f}) = ${total:.2f}"
    )
    return total, line


def main() -> None:
    """usage: run_registers.py [BUDGET] [N] [--max-cost USD] [--confirm] [REGISTER...]. This
    fans out over several documents, so it refuses to run without --confirm (Makefile:
    CONFIRM=1) and when the projection exceeds --max-cost."""
    args = sys.argv[1:]
    max_cost: float | None = None
    if "--max-cost" in args:
        i = args.index("--max-cost")
        max_cost = float(args[i + 1])
        del args[i : i + 2]
    confirm = "--confirm" in args
    args = [a for a in args if a != "--confirm"]
    budget = float(args[0]) if args else 5.0
    n = int(args[1]) if len(args) > 1 else 3
    only = args[2:] or NAMES
    total, line = projection(only, n, budget)
    print(line, file=sys.stderr)
    if max_cost is None:
        print("refusing: give --max-cost USD (Makefile: MAX_COST=...)", file=sys.stderr)
        sys.exit(1)
    if total > max_cost:
        print(
            f"refusing: projection ${total:.2f} exceeds --max-cost ${max_cost:.2f}", file=sys.stderr
        )
        sys.exit(1)
    if not confirm:
        print(
            "refusing: this fans out over several documents; pass --confirm (CONFIRM=1)",
            file=sys.stderr,
        )
        sys.exit(1)
    rows = []
    for register in only:
        print(f"== {register}", file=sys.stderr)
        rows.append(run(register, budget, n))
        print(json.dumps(rows[-1]), file=sys.stderr)
    (OUT / "summary.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"total ${sum(float(str(r['cost'])) for r in rows):.3f}; pages in {OUT}")


if __name__ == "__main__":
    main()
