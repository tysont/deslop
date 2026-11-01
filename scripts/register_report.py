# ABOUTME: Writes reports/registers/<name>.md from an apply --json result: original, input, output
# ABOUTME: (or every rejection), meaning diff, coherence, judge items, two-sided metrics, calls.
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGISTERS = ROOT / "tests" / "fixtures" / "registers"


def body_of(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    return text.split("---", 2)[2].strip() if text.startswith("---") else text.strip()


def row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def judge_table(unit: dict) -> list[str]:
    lines = [row(["cand", "vs", "voice", "coherent", "tense", "order", "conf"]), row(["---"] * 7)]
    for c in unit["candidates"]:
        for v in c["verdicts"]:
            items = v["items"]
            lines.append(
                row(
                    [
                        str(c["attempt"]),
                        v["against"].replace("candidate ", "cand "),
                        items.get("voice", "-"),
                        items.get("coherent", "-"),
                        items.get("tense_person", "-"),
                        items.get("order", "-"),
                        f"{v['confidence']:.2f}",
                    ]
                )
            )
    if len(lines) == 2:
        lines.append(row(["-", "no judge call reached", "", "", "", "", ""]))
    return lines


def coherence_lines(c: dict | None, unit: dict) -> list[str]:
    out = []
    if c is None:
        out.append("The original was kept, so no winner was checked. Per candidate:")
        for cand in unit["candidates"]:
            cc = cand.get("coherence")
            if cc is None:
                out.append(f"- candidate {cand['attempt']}: not reached")
                continue
            out.append(
                f"- candidate {cand['attempt']}: {'ok' if _ok(cc) else 'failed'}; paragraphs "
                f"{cc['paragraphs_before']} -> {cc['paragraphs_after']}; tense drift "
                f"{len(cc['tense'])}; antecedents {len(cc['antecedents'])}; connective "
                f"openers {100 * cc['connectives']['before']:.0f}% -> "
                f"{100 * cc['connectives']['after']:.0f}%"
                + (f" ({cc['connectives']['note']})" if cc["connectives"]["note"] else "")
            )
            for d in cc["tense"]:
                out.append(
                    f"  - tense: paragraph {d['paragraph']} {d['before']} -> {d['after']}: "
                    f'"{d["example"][:90]}"'
                )
            for f in cc["antecedents"]:
                out.append(f'  - antecedent: "{f["text"][:90]}" ({f["word"]})')
        return out
    out.append(
        f"Winner: {'ok' if _ok(c) else 'failed'}; paragraphs {c['paragraphs_before']} -> "
        f"{c['paragraphs_after']}; tense drift in {len(c['tense'])} paragraph(s); "
        f"{len(c['antecedents'])} pronoun(s) without an antecedent; connective openers "
        f"{100 * c['connectives']['before']:.0f}% -> {100 * c['connectives']['after']:.0f}%"
        + (f" ({c['connectives']['note']})" if c["connectives"]["note"] else "")
    )
    return out


def _ok(c: dict) -> bool:
    return (
        c["paragraphs_before"] == c["paragraphs_after"]
        and not c["tense"]
        and not c["antecedents"]
        and c["connectives"]["ok"]
    )


def metric_table(rows: list[dict]) -> list[str]:
    out = [row(["metric", "before", "after", "low", "high", "was/is"]), row(["---"] * 6)]
    for r in rows:
        if r["band_low"] is None and r["band_high"] is None:
            continue
        lo = "-" if r["band_low"] is None else f"{r['band_low']:.2f}"
        hi = "-" if r["band_high"] is None else f"{r['band_high']:.2f}"
        out.append(
            row(
                [
                    r["name"],
                    f"{r['before']:.2f}",
                    f"{r['after']:.2f}",
                    lo,
                    hi,
                    f"{r['status_before']}/{r['status_after']}",
                ]
            )
        )
    return out


def main(register: str, result_path: Path, out: Path) -> None:
    d = json.loads(result_path.read_text(encoding="utf-8"))
    original = body_of(REGISTERS / register / "original.md")
    slop = body_of(REGISTERS / register / "slop.md")
    unit = d["units"][0]
    before, after = d["report_before"], d["report_after"]
    lines = [f"# {register}: original, model prose, and the v2.3 rewrite", ""]
    lines += [
        f"Distance {before['distance']:.0f} -> {after['distance']:.0f} "
        f"({before['verdict']} -> {after['verdict']}); growth {d['growth_pct']:+.0f}%; "
        f"stopped: {d['stopped']}.",
        "",
        d["judge_gate"],
        "",
        "## Original (Faraday)",
        "",
        original,
        "",
        "## Model prose (the input to apply)",
        "",
        slop,
        "",
    ]
    if unit["kept_original"]:
        lines += ["## Output: original kept", "", unit["reason"], ""]
        for c in unit["candidates"]:
            lines.append(f"- candidate {c['attempt']} ({c['model']}): {c['reason']}")
            for note in c.get("reader_notes", []):
                lines.append(f"  - a reader noticed: {note}")
        lines.append("")
    else:
        lines += ["## Output", "", d["text"].strip(), "", f"Unit: {unit['reason']}", ""]
        for c in unit["candidates"]:
            mark = (
                "winner"
                if unit["chosen"] is not None and c is unit["candidates"][unit["chosen"]]
                else ("accepted" if c["accepted"] else "rejected")
            )
            lines.append(f"- candidate {c['attempt']} ({c['model']}): {mark}; {c['reason']}")
        lines.append("")
    if d.get("report_all"):
        lines += ["## Every candidate in full", ""]
        for c in unit["candidates"]:
            status = (
                "winner"
                if unit["chosen"] is not None and c is unit["candidates"][unit["chosen"]]
                else ("accepted" if c["accepted"] else "rejected")
            )
            lines += [
                f"### Candidate {c['attempt']} ({c['model']}): {status}",
                "",
                c["text"].strip() or "(empty)",
                "",
            ]
            lines.append(
                f"Lock: {'ok' if c['lexical_ok'] else 'failed'}"
                + (f" ({c['lexical_detail']})" if c.get("lexical_detail") else "")
            )
            sem = c.get("semantic")
            if sem:
                dropped = [x["proposition"] for x in sem["checks"] if not x["kept"]]
                claims = [
                    f"{x['text']} ({x['category'] or 'claim'})"
                    for x in sem["added"]
                    if x["kind"] == "claim"
                ]
                permitted = [
                    f"{x['text']} ({x['category']})"
                    for x in sem["added"]
                    if x["kind"] == "permitted"
                ]
                lines.append(
                    f"Meaning: {len(sem['checks']) - len(dropped)}/{len(sem['checks'])} propositions kept; dropped: {len(dropped)}; protected additions: {len(claims)}; permitted changes: {len(permitted)}"  # noqa: E501
                )
                for x in dropped:
                    lines.append(f"- dropped: {x}")
                for x in claims:
                    lines.append(f"- added: {x}")
                for x in permitted:
                    lines.append(f"- permitted: {x}")
                for x in sem.get("cleared", []):
                    lines.append(
                        f"- cleared on review ({x['category']}): {x['text']} — {x['note']}"
                    )
            cc = c.get("coherence")
            if cc:
                lines.append(
                    f"Coherence: {'ok' if _ok(cc) else 'failed'}; paragraphs {cc['paragraphs_before']} -> {cc['paragraphs_after']}; tense drift {len(cc['tense'])}; antecedents {len(cc['antecedents'])}; connective openers {100 * cc['connectives']['before']:.0f}% -> {100 * cc['connectives']['after']:.0f}%"  # noqa: E501
                    + (f" ({cc['connectives']['note']})" if cc["connectives"]["note"] else "")
                )
            if c["verdicts"]:
                for v in c["verdicts"]:
                    items = v["items"]
                    lines.append(
                        f"Judge vs {v['against']}: voice {items.get('voice')}, coherent {items.get('coherent')}, tense {items.get('tense_person')}, order {items.get('order')}; confidence {v['confidence']:.2f}; {v['note']}"  # noqa: E501
                    )
            else:
                lines.append("Judge: not reached")
            lines.append(f"Reason: {c['reason']}")
            lines.append("")
    lines += ["## Meaning diff", "", "```", d["meaning_diff"], "```", ""]
    lines += ["## Coherence", "", *coherence_lines(d.get("coherence"), unit), ""]
    lines += ["## Judge items per candidate", "", *judge_table(unit), ""]
    lines += [
        "## Metrics against the voice's band (both sides)",
        "",
        *metric_table(d["metric_table"]),
        "",
    ]
    cost = d["cost"]
    lines += [
        "## Calls",
        "",
        f"{cost['calls']} live call(s), {cost['cached']} replayed from cache; "
        f"API cost ${cost['usd']:.3f} (subscription calls are not priced).",
        "",
    ]
    text = "\n".join(lines)
    wide = [ln for ln in text.splitlines() if ln.startswith("|") and len(ln) > 80]
    if wide:
        raise SystemExit(f"table line over 80 columns: {wide[0]}")
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out} ({len(text.split())} words)")


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]))
