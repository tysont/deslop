# ABOUTME: Typer CLI for deslop: `voice` (build and inspect), `apply`, `score`, `lab`, `doctor`.
# ABOUTME: Options resolve as flag > deslop.yaml command section > top-level key > default.
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich.table import Table

from deslop import __version__
from deslop.cli.common import (
    Settings,
    arm_budget,
    as_list,
    console,
    err,
    fail,
    load_voice,
    print_json,
    provider_name,
    read_input,
    report_budget,
    split_frontmatter,
)
from deslop.cli.config import ConfigError, load_config
from deslop.cli.lab import lab_app
from deslop.cli.voice import voice_app

app = typer.Typer(
    help="Build a voice from texts, score prose against it, apply it.", no_args_is_help=True
)
providers_app = typer.Typer(help="LLM provider configuration.", no_args_is_help=True)
app.add_typer(voice_app, name="voice")
app.add_typer(lab_app, name="lab")
app.add_typer(providers_app, name="providers")


def version_callback(value: bool) -> None:
    if value:
        console.print(f"deslop {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", callback=version_callback, is_eager=True),
    config: Path | None = typer.Option(
        None, "--config", help="Settings file (default: ./deslop.yaml or ~/.deslop/config.yaml)"
    ),
) -> None:
    try:
        ctx.obj = load_config(config)
    except ConfigError as exc:
        fail(str(exc))


def _citation_schemes(items: list[Any]) -> list[Any]:
    """`--citation-scheme` values: JSON objects with name, pattern and optional bare_pattern."""
    from deslop.reference import CitationScheme

    out = []
    for item in items:
        try:
            data = json.loads(item) if isinstance(item, str) else item
            out.append(CitationScheme.model_validate(data))
        except (ValueError, TypeError) as exc:
            fail(f"--citation-scheme must be JSON with name and pattern: {exc}")
    return out


def _declared_texts(v: Any) -> list[Any]:
    """The voice's `primary_texts`: paths, or specs with their own schemes, relative to the
    voice file."""
    from deslop.reference import PrimaryTextSpec

    base = v.path.parent if v.path is not None else Path(".")
    out = []
    for item in v.meta.primary_texts:
        spec = (
            PrimaryTextSpec(path=item)
            if isinstance(item, str)
            else PrimaryTextSpec.model_validate(item)
        )
        if not Path(spec.path).is_absolute():
            spec.path = str(base / spec.path)
        out.append(spec)
    return out


@app.command("score")
def score_cmd(
    ctx: typer.Context,
    file: Path | None = typer.Argument(None, help="Text file to score (default: stdin)"),
    voice: Path | None = typer.Option(None, "--voice", "-v"),
    patterns: list[Path] = typer.Option([], "--patterns"),
    annotate: Path | None = typer.Option(
        None, "--annotate", help="Write the input with inline [Tn] markers here"
    ),
    judge: bool = typer.Option(False, "--judge", help="Also ask an LLM judge (costs a call)"),
    judge_model: str | None = typer.Option(
        None, "--judge-model", help="Model for --judge (claude-sonnet-5-5; opus/sonnet/haiku)"
    ),
    provider: str | None = typer.Option(None, "--provider", help="anthropic or claude-cli"),
    json_out: bool = typer.Option(False, "--json"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Print only the verdict line"),
) -> None:
    """Score a text against a voice. Exit 0 on PASS, 2 on FAIL."""
    from deslop.score import score

    s = Settings(ctx, "score")
    text = read_input(file)
    v = load_voice(s.voice(voice))
    report = score(text, v, patterns=s.paths("patterns", patterns))
    if s.get("judge", judge or None, False):
        from deslop.judge import judge_paragraph
        from deslop.providers.base import DEFAULT_JUDGE, ProviderError, get_provider, resolve_model
        from deslop.voice.exemplars import ExemplarIndex

        arm_budget(s, None, command="score --judge", voice=v.name, input_file=str(file or ""))
        try:
            judge_provider = get_provider(provider_name(s, provider))
        except ProviderError as exc:
            fail(str(exc))
        model = resolve_model(str(s.get("judge_model", judge_model, DEFAULT_JUDGE)))
        judge_provider.default_model = model
        report.judge = judge_paragraph(
            text, v, judge_provider, exemplars=ExemplarIndex(v).closest(text, 2)
        )
    annotate = s.path("annotate", annotate)
    if annotate is not None:
        annotate.write_text(report.annotate(text), encoding="utf-8")
    if json_out:
        print(report.to_json())
    elif quiet:
        console.print(report.renderables(show_hits=False)[0])
    else:
        for item in report.renderables():
            console.print(item)
        if report.judge:
            j = report.judge
            console.print(
                f"judge ({j.provider}): voice {j.voice}/5, model prose {j.model_prose}/5, "
                f"fidelity {j.fidelity}/5, overall {j.overall:.1f}. {j.note}"
            )
    raise typer.Exit(0 if report.verdict == "PASS" else 2)


def print_metric_table(rows: list[Any]) -> None:
    """Every metric before and after against the voice's band, flagging under and over."""

    table = Table(title="Metrics against the voice's band (both sides)")
    for col in ("metric", "before", "after", "band low", "band high", "before", "after"):
        table.add_column(col, justify="left" if col == "metric" else "right")
    for r in rows:
        if r.band_low is None and r.band_high is None:
            continue
        table.add_row(
            r.name,
            f"{r.before:.2f}",
            f"{r.after:.2f}",
            "-" if r.band_low is None else f"{r.band_low:.2f}",
            "-" if r.band_high is None else f"{r.band_high:.2f}",
            r.status_before if r.status_before == "in" else f"[red]{r.status_before}[/]",
            r.status_after if r.status_after == "in" else f"[red]{r.status_after}[/]",
        )
    console.print(table)


def print_apply_summary(result: Any, show_diff: bool, out: Path | None) -> None:
    before, after = result.report_before, result.report_after
    color = "green" if after.verdict == "PASS" else "red"
    rewritten = sum(1 for u in result.units if not u.kept_original)
    fallbacks = [u for u in result.units if u.kept_original]
    unpriced = (
        f" ({result.cost.unpriced_calls} unpriced calls)" if result.cost.unpriced_calls else ""
    )
    sep = after.separation
    context = (
        f" (held-out p90 {sep.heldout_p90:.0f}, LLM median {sep.llm_median:.0f})" if sep else ""
    )
    if result.unverified_additions:
        console.print(
            f"[yellow]unverified additions ({len(result.unverified_additions)}):[/] "
            "quotations and citations the output added that no primary text confirmed"
        )
        for item in result.unverified_additions:
            console.print(f"  ? {item}", markup=False, highlight=False)
    console.print(
        f"[bold {color}]{after.verdict}[/]: distance {before.distance:.0f} -> "
        f"{after.distance:.0f}{context}; {rewritten}/{len(result.units)} unit(s) rewritten"
        + (f", {len(fallbacks)} kept original" if fallbacks else "")
        + f"; growth {result.growth_pct:+.0f}% (limit +{result.max_growth:.0f}%)"
        + f"; ${result.cost.usd:.3f} over {result.cost.calls} call(s){unpriced}; "
        f"stopped: {result.stopped}"
    )
    style = "yellow" if "unreliable" in result.judge_gate else "dim"
    console.print(f"[{style}]{result.judge_gate}[/]")
    if result.primary_texts:
        console.print(f"[dim]primary texts: {', '.join(result.primary_texts)}[/]")
    if not result.exemplars_used:
        console.print("[yellow]this voice has no exemplars; wrote from profile and rules alone[/]")
    for u in result.units:
        tried = len([c for c in u.candidates if c.text])
        console.print(f"  unit {u.index + 1} ({u.kind}): {tried} candidate(s); {u.reason}")
        for c in u.candidates:
            if c.win_rate is None:
                continue
            mark = "*" if u.chosen is not None and c is u.candidates[u.chosen] else " "
            console.print(
                f"   {mark} candidate {c.attempt} ({c.provider}): win rate "
                f"{100 * c.win_rate:.0f}% over {len(c.verdicts)}, distance "
                f"{c.distance if c.distance is not None else float('nan'):.1f}"
            )
    if result.mode == "rewrite":
        from deslop.result import coherence_report

        console.print(coherence_report(result), markup=False, highlight=False)
        print_metric_table(result.metric_table)
    if after.fixes:
        console.print("[bold]Remaining:[/] " + "; ".join(after.fixes[:5]))
    if show_diff:
        console.print(result.diff, markup=False, highlight=False)
        console.print("[bold]Meaning[/]")
        console.print(result.meaning_diff, markup=False, highlight=False)
    elif out is None:
        console.print(result.text, markup=False, highlight=False)


@app.command("apply")
def apply_cmd(
    ctx: typer.Context,
    file: Path | None = typer.Argument(None, help="Text file to rewrite (default: stdin)"),
    voice: Path | None = typer.Option(None, "--voice", "-v"),
    out: Path | None = typer.Option(None, "--out", "-o", help="Write the result here"),
    brief: Path | None = typer.Option(
        None, "--brief", help="Draft from this brief yaml instead of rewriting a file"
    ),
    mode: str | None = typer.Option(
        None, "--mode", help="rewrite (the whole document, default) or edit (flagged sentences)"
    ),
    writer: str | None = typer.Option(
        None, "--writer", help="Writer model (claude-opus-5-5; opus/sonnet/haiku)"
    ),
    judge: str | None = typer.Option(
        None, "--judge", help="Judge model, must differ from the writer (claude-sonnet-5-5)"
    ),
    provider: str | None = typer.Option(
        None, "--provider", help="anthropic (API, default) or claude-cli (local login)"
    ),
    max_calls: int | None = typer.Option(
        None, "--max-calls", help="Max live calls this invocation may make (200)"
    ),
    fidelity: str | None = typer.Option(
        None,
        "--fidelity",
        help="content (default: protect facts, permit changes of person and framing) or strict",
    ),
    report_all: bool = typer.Option(
        False, "--report-all", help="Judge and report every candidate, pass or fail"
    ),
    compare: Path | None = typer.Option(
        None, "--compare", help="Write a self-contained HTML comparison page here"
    ),
    reference: Path | None = typer.Option(
        None,
        "--reference",
        help="The human text the input was made from: candidates are checked for recitation "
        "against it (never shown to a model) and it fills the comparison page's first column",
    ),
    max_recitation: float | None = typer.Option(
        None,
        "--max-recitation",
        help="Reject a candidate when more than PCT of its words are recited: in runs of 8+ "
        "words shared with the reference and not with the input, and not among the input's "
        "words at all (2)",
    ),
    budget: float | None = typer.Option(
        None,
        "--budget",
        help="Max USD this invocation may spend on live calls (5); rewrite mode then returns the "
        "original, edit mode the best text so far",
    ),
    n: int | None = typer.Option(None, "--n", help="Candidates per document in rewrite mode (3)"),
    max_growth: float | None = typer.Option(
        None, "--max-growth", help="Reject candidates more than PCT longer than the original (15)"
    ),
    chunk_words: int | None = typer.Option(
        None, "--chunk-words", help="Rewrite longer documents section by section (2500)"
    ),
    primary_text: list[str] = typer.Option(
        [],
        "--primary-text",
        help="Primary texts (a text file with an optional NAME.locs.json sidecar, or a "
        "directory of them) added quotations and citations must verify against; default: "
        "the voice's declared primary texts; without any, additions are listed as unverified",
    ),
    citation_scheme: list[str] = typer.Option(
        [],
        "--citation-scheme",
        help='A citation scheme for the command-line primary texts, as JSON: {"name": "verse", '
        '"pattern": "<regex>", "bare_pattern": "<regex or null>"}; repeatable',
    ),
    preceding: Path | None = typer.Option(
        None,
        "--preceding",
        help="The already-converted section immediately before the input: the writer continues "
        "it, the continuity checks and the judge's fifth item hold the seam; never rewritten",
    ),
    following: Path | None = typer.Option(
        None,
        "--following",
        help="The unconverted section after the input: context only, never rewritten or checked",
    ),
    citations: str | None = typer.Option(
        None,
        "--citations",
        help="keep: the input's citations and quotations must reappear exactly; verified "
        "(default): additions must verify against the primary texts; none: as keep, and "
        "nothing may be added",
    ),
    quote_context_words: int | None = typer.Option(
        None,
        "--quote-context-words",
        help="Cap on the verbatim primary-text passages offered to the writer (1500)",
    ),
    allow_primary_facts: bool = typer.Option(
        False,
        "--allow-primary-facts",
        help="A fact the input lacks is not an addition when it is verifiable at a cited "
        "location of a primary text (off: it remains a material addition)",
    ),
    max_iter: int | None = typer.Option(None, "--max-iter", help="Edit-mode iterations (4)"),
    patterns: list[Path] = typer.Option([], "--patterns"),
    show_diff: bool = typer.Option(False, "--diff", help="Print the unified diff and meaning diff"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Write a text, or a piece from a brief, in a voice. Exit 0 on PASS, 2 on FAIL."""
    from deslop.apply import apply
    from deslop.draft import load_brief
    from deslop.providers.base import DEFAULT_JUDGE, DEFAULT_WRITER, BudgetExceeded

    s = Settings(ctx, "apply")
    v = load_voice(s.voice(voice))
    brief = s.path("brief", brief)
    front = ""
    if brief is not None:
        try:
            source: Any = load_brief(brief)
        except (ValueError, OSError) as exc:
            fail(str(exc))
    else:
        front, source = split_frontmatter(read_input(file))
    mode = s.get("mode", mode, "rewrite")
    if mode not in ("edit", "rewrite"):
        fail("--mode must be rewrite or edit")
    citation_mode = str(s.get("citations", citations, "verified"))
    if citation_mode not in ("keep", "verified", "none"):
        fail("--citations must be keep, verified or none")
    citation_policy: Any = citation_mode
    preceding_path = s.path("preceding", preceding)
    following_path = s.path("following", following)
    preceding_text = split_frontmatter(read_input(preceding_path))[1] if preceding_path else None
    following_text = split_frontmatter(read_input(following_path))[1] if following_path else None
    limit = arm_budget(
        s,
        budget,
        command="apply",
        voice=v.name,
        input_file=str(brief or file or "stdin"),
        max_calls=max_calls,
    )
    texts: list[Any] = [str(x) for x in as_list(s.get("primary_text", primary_text or None, []))]
    schemes = _citation_schemes(as_list(s.get("citation_scheme", citation_scheme or None, [])))
    if not texts and v.meta.primary_texts:
        texts = _declared_texts(v)
    reference = s.path("reference", reference)
    ref_text = split_frontmatter(reference.read_text(encoding="utf-8"))[1] if reference else None
    try:
        result = apply(
            source,
            v,
            mode="edit" if mode == "edit" else "rewrite",
            writer=str(s.get("writer", writer, DEFAULT_WRITER)),
            judge=str(s.get("judge", judge, DEFAULT_JUDGE)),
            provider=provider_name(s, provider),
            budget=limit,
            n=int(s.get("n", n, 3)),
            max_growth=float(s.get("max_growth", max_growth, 15.0)),
            chunk_words=int(s.get("chunk_words", chunk_words, 2500)),
            fidelity="strict" if s.get("fidelity", fidelity, "content") == "strict" else "content",
            report_all=bool(s.get("report_all", report_all or None, False)),
            reference=ref_text,
            reference_name=reference.name if reference else "reference",
            max_recitation=float(s.get("max_recitation", max_recitation, 2.0)),
            primary_texts=list(texts) or None,
            citation_schemes=schemes,
            quote_context_words=int(s.get("quote_context_words", quote_context_words, 1500)),
            citations=citation_policy,
            preceding=preceding_text,
            following=following_text,
            allow_primary_facts=bool(
                s.get("allow_primary_facts", allow_primary_facts or None, False)
            ),
            max_iter=int(s.get("max_iter", max_iter, 4)),
            patterns=s.paths("patterns", patterns),
            progress=s.progress(json_out),
        )
    except (FileNotFoundError, ValueError) as exc:
        fail(str(exc))
    except BudgetExceeded as exc:
        fail(str(exc), code=2)
    out = s.path("out", out)
    if out is not None:
        out.write_text(front + result.text, encoding="utf-8")
    compare = s.path("compare", compare)
    if compare is not None:
        from deslop.compare_page import build_page

        compare.write_text(
            build_page(json.loads(result.to_json()), reference=ref_text), encoding="utf-8"
        )
        err.print(f"[dim]comparison page: {compare}[/]")
    report_budget()
    if json_out:
        print_json(result.to_json())
    else:
        if result.draft is not None:
            incomplete = [x.heading for x in result.draft if not x.complete]
            console.print(
                f"drafted {len(result.draft)} section(s)"
                + (f"; incomplete: {', '.join(incomplete)}" if incomplete else "")
            )
        print_apply_summary(result, s.get("diff", show_diff or None, False), out)
    raise typer.Exit(0 if result.report_after.verdict == "PASS" else 2)


@app.command("report")
def report_cmd(
    result: Path = typer.Argument(..., help="An apply --json result file"),
    html_out: bool = typer.Option(False, "--html", help="Write the HTML comparison page"),
    out: Path | None = typer.Option(None, "--out", "-o", help="Where to write it"),
    reference: Path | None = typer.Option(
        None, "--reference", help="Human reference text for the first column"
    ),
) -> None:
    """Render a saved apply result: `--html` writes the three-column comparison page."""
    from deslop.compare_page import build_page

    try:
        data = json.loads(result.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read {result}: {exc}")
    if not html_out:
        fail("give --html (the only report format)")
    if out is None:
        fail("--out PATH is required")
    ref_text = split_frontmatter(reference.read_text(encoding="utf-8"))[1] if reference else None
    out.write_text(build_page(data, reference=ref_text), encoding="utf-8")
    console.print(f"wrote {out}")


@app.command("doctor")
def doctor_cmd(ctx: typer.Context, json_out: bool = typer.Option(False, "--json")) -> None:
    """Check the installation: spaCy model, providers, pricing age, cache dir, default voice."""
    from deslop import doctor

    s = Settings(ctx, "doctor")
    checks = doctor.run(s.cfg)
    checks.append(doctor.check_spend())
    ok = all(c.status != "fail" for c in checks)
    if json_out:
        print_json(json.dumps({"ok": ok, "checks": [c.model_dump() for c in checks]}))
    else:
        colors = {"ok": "green", "warn": "yellow", "fail": "red"}
        for c in checks:
            console.print(f"[{colors[c.status]}]{c.status:4s}[/] {c.name:10s} {c.detail}")
    raise typer.Exit(0 if ok else 1)


@providers_app.command("test")
def providers_test(json_out: bool = typer.Option(False, "--json")) -> None:
    from deslop.providers.base import ANTHROPIC_PROVIDERS, get_provider, list_providers

    results = {}
    for name, configured in list_providers().items():
        if not configured:
            results[name] = {"configured": False, "ok": False, "detail": "not configured"}
            continue
        if name not in ANTHROPIC_PROVIDERS:
            results[name] = {
                "configured": True,
                "ok": False,
                "detail": "configured but unused unless named",
            }
            continue
        try:
            provider = get_provider(name, use_cache=False)
            completion = provider.complete(
                "Reply with the single word OK.", "ping", temperature=0.0
            )
            results[name] = {
                "configured": True,
                "ok": "ok" in completion.text.lower(),
                "detail": completion.model,
            }
        except Exception as exc:  # noqa: BLE001
            results[name] = {"configured": True, "ok": False, "detail": str(exc)[:200]}
    if json_out:
        print_json(json.dumps(results))
        return
    for name, r in results.items():
        status = (
            "[green]ok[/]"
            if r["ok"]
            else ("[red]fail[/]" if r["configured"] else "[dim]unconfigured[/]")
        )
        console.print(f"{name:12s} {status}  {r['detail']}")


if __name__ == "__main__":
    app()
