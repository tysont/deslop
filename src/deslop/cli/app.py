# ABOUTME: Typer CLI for deslop. Every command wraps the public API and supports --json.
# ABOUTME: Options resolve as flag > deslop.yaml command section > top-level key > default.
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from deslop import __version__
from deslop.cli.config import Config, ConfigError, load_config
from deslop.corpus import Corpus, DuplicateError
from deslop.corpus.ingest import IngestError

app = typer.Typer(
    help="Voice calibration, scoring, and conversion for prose.", no_args_is_help=True
)
corpus_app = typer.Typer(help="Build and inspect a corpus directory.", no_args_is_help=True)
voice_app = typer.Typer(help="Inspect and maintain voices.", no_args_is_help=True)
providers_app = typer.Typer(help="LLM provider configuration.", no_args_is_help=True)
app.add_typer(corpus_app, name="corpus")
app.add_typer(voice_app, name="voice")
app.add_typer(providers_app, name="providers")

console = Console()
err = Console(stderr=True)


def print_json(payload: str) -> None:
    """Emit JSON untouched by rich wrapping or markup."""
    sys.stdout.write(payload if payload.endswith("\n") else payload + "\n")


def fail(message: str, code: int = 1) -> None:
    err.print(f"[red]error:[/] {message}")
    raise typer.Exit(code)


def version_callback(value: bool) -> None:
    if value:
        console.print(f"deslop {__version__}")
        raise typer.Exit()


class Settings:
    """Per-invocation option resolver bound to one command name."""

    def __init__(self, ctx: typer.Context, command: str):
        self.cfg: Config = ctx.obj if isinstance(ctx.obj, Config) else Config()
        self.command = command

    def get(self, key: str, value: Any, default: Any = None) -> Any:
        return self.cfg.get(self.command, key, value, default=default)

    def path(self, key: str, value: Path | None, default: Path | None = None) -> Path | None:
        got = self.get(key, value, default)
        return None if got is None else Path(got)

    def paths(self, key: str, value: list[Path] | None) -> tuple[str, ...]:
        got = self.get(key, value or None, [])
        return tuple(str(p) for p in _as_list(got))

    def voice(self, value: Path | None) -> Path:
        got = self.path("voice", value)
        if got is None:
            fail("no voice given: pass --voice or set `voice:` in deslop.yaml")
        return got

    def progress(self, json_out: bool):
        return None if json_out else (lambda msg: err.print(f"[dim]{msg}[/]"))


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]
    return list(value)


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


# ------------------------------------------------------------------ corpus


@corpus_app.command("init")
def corpus_init(
    directory: Path,
    name: str | None = typer.Option(None, help="Corpus name (default: directory name)"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    corpus = Corpus.init(directory, name=name)
    if json_out:
        print_json(corpus.manifest.model_dump_json())
    else:
        console.print(f"initialized corpus [bold]{corpus.name}[/] at {corpus.root}")


@corpus_app.command("add")
def corpus_add(
    ctx: typer.Context,
    directory: Path,
    sources: list[str] = typer.Argument(..., help="Files or URLs: txt, md, html, epub, pdf"),
    author: str | None = typer.Option(None, "--author"),
    title: str | None = typer.Option(None, "--title"),
    year: int | None = typer.Option(None, "--year"),
    weight: float | None = typer.Option(None, "--weight", help="Default 1.0"),
    license: str | None = typer.Option(None, "--license"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    s = Settings(ctx, "corpus_add")
    author = s.get("author", author)
    if not author:
        fail("--author is required")
    corpus = Corpus.load(directory)
    added = []
    for source in sources:
        try:
            entry = corpus.add(
                source,
                author=author,
                title=title if len(sources) == 1 else None,
                year=s.get("year", year),
                weight=float(s.get("weight", weight, 1.0)),
                license=s.get("license", license, ""),
            )
        except (DuplicateError, IngestError) as exc:
            fail(str(exc))
        added.append(entry)
        if not json_out:
            console.print(
                f"added [bold]{entry.title}[/] by {entry.author}: "
                f"{entry.words} words -> {entry.path}"
            )
    if json_out:
        print_json(json.dumps([e.model_dump() for e in added]))


@corpus_app.command("stats")
def corpus_stats(
    ctx: typer.Context,
    directory: Path,
    max_share: float | None = typer.Option(None, "--max-share", help="Default 0.25"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    s = Settings(ctx, "corpus_stats")
    max_share = float(s.get("max_share", max_share, 0.25))
    corpus = Corpus.load(directory)
    stats = corpus.stats(max_share=max_share)
    if json_out:
        print_json(stats.model_dump_json())
        return
    table = Table(title=f"{corpus.name}: {stats.total_words} words")
    table.add_column("author")
    table.add_column("docs", justify="right")
    table.add_column("words", justify="right")
    table.add_column("share", justify="right")
    table.add_column("")
    for a in stats.authors:
        flag = "[red]over cap[/]" if a.author in stats.over_cap else ""
        table.add_row(a.author, str(a.documents), str(a.words), f"{100 * a.share:.1f}%", flag)
    console.print(table)
    for doc in corpus.manifest.documents:
        console.print(
            f"  {doc.path}: {doc.author}, {doc.title} ({doc.words} words, weight {doc.weight})"
        )
    if stats.over_cap:
        console.print(
            f"[yellow]{len(stats.over_cap)} author(s) exceed {100 * max_share:.0f}%; "
            "calibration downweights them.[/]"
        )


# ------------------------------------------------------------------ calibrate


@app.command("calibrate")
def calibrate_cmd(
    ctx: typer.Context,
    corpus_dir: Path,
    out: Path | None = typer.Option(None, "--out", help="Output voice directory, e.g. name.voice"),
    name: str | None = typer.Option(None, "--name"),
    no_llm: bool = typer.Option(
        False, "--no-llm", help="Write a templated profile.md instead of calling a model"
    ),
    max_share: float | None = typer.Option(None, "--max-share", help="Default 0.25"),
    exemplar_words: int | None = typer.Option(None, "--exemplar-words", help="Default 2000"),
    llm_model: str | None = typer.Option(None, "--llm-model"),
    patterns: list[Path] = typer.Option([], "--patterns", help="Extra pattern catalog yaml"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    from deslop.voice.calibrate import calibrate

    s = Settings(ctx, "calibrate")
    out = s.path("out", out)
    if out is None:
        fail("--out is required")
    corpus = Corpus.load(corpus_dir)
    voice = calibrate(
        corpus,
        name=s.get("name", name) or out.name.removesuffix(".voice"),
        max_share=float(s.get("max_share", max_share, 0.25)),
        exemplar_words=int(s.get("exemplar_words", exemplar_words, 2000)),
        use_llm=not s.get("no_llm", no_llm or None, False),
        llm_model=s.get("llm_model", llm_model),
        patterns=s.paths("patterns", patterns),
        progress=s.progress(json_out),
    )
    voice.save(out)
    if json_out:
        print_json(voice.meta.model_dump_json())
    else:
        console.print(
            f"wrote voice [bold]{voice.name}[/] to {out} "
            f"({voice.meta.total_words} words, profile: {voice.meta.profile_source})"
        )


# ------------------------------------------------------------------ score


def read_input(path: Path | None) -> str:
    if path is None or str(path) == "-":
        return sys.stdin.read()
    return path.read_text(encoding="utf-8")


def score_options(s: Settings, sigma, tell_mult, tell_floor, patterns) -> dict[str, Any]:
    return dict(
        sigma=float(s.get("sigma", sigma, 1.5)),
        tell_mult=float(s.get("tell_mult", tell_mult, 2.0)),
        tell_floor=float(s.get("tell_floor", tell_floor, 1.0)),
        patterns=s.paths("patterns", patterns),
    )


@app.command("score")
def score_cmd(
    ctx: typer.Context,
    file: Path | None = typer.Argument(None, help="Text file to score (default: stdin)"),
    voice: Path | None = typer.Option(None, "--voice", "-v"),
    sigma: float | None = typer.Option(None, "--sigma", help="Default 1.5"),
    tell_mult: float | None = typer.Option(None, "--tell-mult", help="Default 2.0"),
    tell_floor: float | None = typer.Option(
        None, "--tell-floor", help="Minimum tell rate per 1k words before flagging (1.0)"
    ),
    patterns: list[Path] = typer.Option([], "--patterns"),
    annotate: Path | None = typer.Option(
        None, "--annotate", help="Write the input with inline [Tn] markers here"
    ),
    json_out: bool = typer.Option(False, "--json"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Print only the verdict line"),
) -> None:
    from deslop.score import score
    from deslop.voice import Voice

    s = Settings(ctx, "score")
    text = read_input(file)
    v = Voice.load(s.voice(voice))
    report = score(text, v, **score_options(s, sigma, tell_mult, tell_floor, patterns))
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
    raise typer.Exit(0 if report.verdict == "PASS" else 2)


# ------------------------------------------------------------------ convert


def convert_options(
    s: Settings, mode, max_iter, models, budget, lock, preserve_length
) -> dict[str, Any]:
    from deslop.lock import LOCK_CLASSES

    mode = s.get("mode", mode, "edit")
    if mode not in ("edit", "rewrite"):
        fail("--mode must be edit or rewrite")
    lock = s.get("lock", lock, ",".join(LOCK_CLASSES))
    lock_classes: tuple[str, ...] = ()
    if not (isinstance(lock, str) and lock.strip().lower() == "none"):
        lock_classes = tuple(_as_list(lock))
        bad = [c for c in lock_classes if c not in LOCK_CLASSES]
        if bad:
            fail(f"unknown lock class(es): {', '.join(bad)}; choose from {', '.join(LOCK_CLASSES)}")
    budget = s.get("budget", budget)
    return dict(
        mode=mode,
        max_iter=int(s.get("max_iter", max_iter, 4)),
        models=tuple(_as_list(s.get("models", models, "claude,gpt,claude,gpt"))),
        budget=None if budget is None else float(budget),
        lock=lock_classes,
        preserve_length=float(s.get("preserve_length", preserve_length, 15.0)) / 100.0,
    )


def print_convert_summary(result, show_diff: bool, out: Path | None) -> None:
    before, after = result.report_before, result.report_after
    color = "green" if after.verdict == "PASS" else "red"
    accepted = sum(1 for u in result.units if u.accepted)
    unpriced = (
        f" ({result.cost.unpriced_calls} unpriced calls)" if result.cost.unpriced_calls else ""
    )
    console.print(
        f"[bold {color}]{after.verdict}[/] after {len(result.iterations)} iteration(s): "
        f"distance {before.distance:.0f} -> {after.distance:.0f}, "
        f"{accepted}/{len(result.units)} rewrites accepted, "
        f"${result.cost.usd:.3f}{unpriced}, stopped: {result.stopped}"
    )
    if after.fixes:
        console.print("[bold]Remaining:[/] " + "; ".join(after.fixes[:5]))
    if show_diff:
        console.print(result.diff, markup=False, highlight=False)
    elif out is None:
        console.print(result.text, markup=False, highlight=False)


@app.command("convert")
def convert_cmd(
    ctx: typer.Context,
    file: Path | None = typer.Argument(None, help="Text file to convert (default: stdin)"),
    voice: Path | None = typer.Option(None, "--voice", "-v"),
    mode: str | None = typer.Option(
        None, "--mode", help="edit (sentences, default) or rewrite (paragraphs)"
    ),
    max_iter: int | None = typer.Option(None, "--max-iter", help="Default 4"),
    models: str | None = typer.Option(
        None, "--models", help="Providers to alternate per iteration (claude,gpt,claude,gpt)"
    ),
    budget: float | None = typer.Option(None, "--budget", help="Max USD; returns best-so-far"),
    sigma: float | None = typer.Option(None, "--sigma"),
    tell_mult: float | None = typer.Option(None, "--tell-mult"),
    tell_floor: float | None = typer.Option(None, "--tell-floor"),
    lock: str | None = typer.Option(
        None, "--lock", help="Lock classes (default all four), or 'none'"
    ),
    preserve_length: float | None = typer.Option(
        None, "--preserve-length", help="Max +/- percent (15)"
    ),
    patterns: list[Path] = typer.Option([], "--patterns"),
    out: Path | None = typer.Option(None, "--out", "-o", help="Write converted text here"),
    show_diff: bool = typer.Option(False, "--diff", help="Print the unified diff"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    from deslop.converter import convert
    from deslop.voice import Voice

    s = Settings(ctx, "convert")
    text = read_input(file)
    v = Voice.load(s.voice(voice))
    result = convert(
        text,
        v,
        **convert_options(s, mode, max_iter, models, budget, lock, preserve_length),
        **score_options(s, sigma, tell_mult, tell_floor, patterns),
        progress=s.progress(json_out),
    )
    out = s.path("out", out)
    if out is not None:
        out.write_text(result.text, encoding="utf-8")
    if json_out:
        print_json(result.to_json())
    else:
        print_convert_summary(result, s.get("diff", show_diff or None, False), out)
    raise typer.Exit(0 if result.report_after.verdict == "PASS" else 2)


# ------------------------------------------------------------------ draft


@app.command("draft")
def draft_cmd(
    ctx: typer.Context,
    brief: Path = typer.Argument(..., help="Brief yaml: title and sections with claim/refs/quotes"),
    voice: Path | None = typer.Option(None, "--voice", "-v"),
    out: Path | None = typer.Option(None, "--out", "-o", help="Write the finished draft here"),
    max_iter: int | None = typer.Option(None, "--max-iter", help="Convert iterations (2)"),
    models: str | None = typer.Option(None, "--models"),
    budget: float | None = typer.Option(None, "--budget"),
    sigma: float | None = typer.Option(None, "--sigma"),
    tell_mult: float | None = typer.Option(None, "--tell-mult"),
    tell_floor: float | None = typer.Option(None, "--tell-floor"),
    lock: str | None = typer.Option(None, "--lock"),
    preserve_length: float | None = typer.Option(None, "--preserve-length"),
    patterns: list[Path] = typer.Option([], "--patterns"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    from deslop.draft import draft, load_brief
    from deslop.voice import Voice

    s = Settings(ctx, "draft")
    try:
        b = load_brief(brief)
    except (ValueError, OSError) as exc:
        fail(str(exc))
    v = Voice.load(s.voice(voice))
    opts = convert_options(s, "rewrite", max_iter, models, budget, lock, preserve_length)
    opts["convert_max_iter"] = (
        opts.pop("max_iter") if max_iter is not None else int(s.get("max_iter", None, 2))
    )
    opts.pop("mode")
    result = draft(
        b,
        v,
        **opts,
        **score_options(s, sigma, tell_mult, tell_floor, patterns),
        progress=s.progress(json_out),
    )
    out = s.path("out", out)
    if out is not None:
        out.write_text(result.text, encoding="utf-8")
    if json_out:
        print_json(result.to_json())
    else:
        incomplete = [x.heading for x in result.sections if not x.complete]
        stripped = sum(len(x.stripped) for x in result.sections)
        console.print(
            f"drafted {len(result.sections)} section(s); stripped {stripped} unlisted item(s)"
            + (f"; incomplete: {', '.join(incomplete)}" if incomplete else "")
        )
        print_convert_summary(result.convert, False, out)
        console.print(f"total cost ${result.cost.usd:.3f} over {result.cost.calls} call(s)")
    raise typer.Exit(0 if result.report.verdict == "PASS" else 2)


# ------------------------------------------------------------------ voice


@voice_app.command("show")
def voice_show(voice: Path, json_out: bool = typer.Option(False, "--json")) -> None:
    from deslop.voice import Voice

    v = Voice.load(voice)
    if json_out:
        print_json(
            json.dumps(
                {
                    "meta": v.meta.model_dump(),
                    "metrics": v.metrics.model_dump(exclude={"per_author"}),
                    "lexicon": v.lexicon.model_dump(),
                }
            )
        )
        return
    console.print(
        f"[bold]{v.name}[/]  corpus {v.meta.corpus_name} ({v.meta.corpus_hash})  "
        f"{v.meta.total_words} words  created {v.meta.created}"
    )
    console.print(
        "author shares: "
        + ", ".join(f"{a} {100 * s:.0f}%" for a, s in v.meta.author_shares.items())
    )
    table = Table(title="Metrics")
    for col in ("metric", "mean", "std", "p10", "p50", "p90"):
        table.add_column(col, justify="right" if col != "metric" else "left")
    for name, s in v.metrics.metrics.items():
        table.add_row(
            name, f"{s.mean:.2f}", f"{s.std:.2f}", f"{s.p10:.2f}", f"{s.p50:.2f}", f"{s.p90:.2f}"
        )
    console.print(table)
    tt = Table(title="Tell rates (/1k words)")
    for col in ("id", "pattern", "voice", "llm", "ratio", "banned"):
        tt.add_column(col)
    for p in v.lexicon.patterns:
        tt.add_row(
            p.id,
            p.name,
            f"{p.corpus_per_1k:.2f}",
            "-" if p.llm_per_1k is None else f"{p.llm_per_1k:.2f}",
            "-" if p.ratio is None else f"{p.ratio:.1f}",
            "yes" if p.banned else "",
        )
    console.print(tt)
    console.print(
        "[bold]Encouraged words:[/] " + ", ".join(e.term for e in v.lexicon.encouraged_words[:40])
    )
    if v.lexicon.discouraged_words:
        console.print(
            "[bold]Discouraged words:[/] "
            + ", ".join(e.term for e in v.lexicon.discouraged_words[:40])
        )
    console.print()
    console.print(v.rules)


@voice_app.command("edit")
def voice_edit(voice: Path) -> None:
    import os
    import subprocess

    editor = os.environ.get("VISUAL") or os.environ.get("EDITOR") or "vi"
    files = [str(voice / f) for f in ("profile.md", "rules.md", "exemplars.md")]
    subprocess.run([editor, *files], check=False)


# ------------------------------------------------------------------ providers


@providers_app.command("test")
def providers_test(json_out: bool = typer.Option(False, "--json")) -> None:
    from deslop.providers.base import get_provider, list_providers

    results = {}
    for name, configured in list_providers().items():
        if not configured:
            results[name] = {"configured": False, "ok": False, "detail": "no credentials"}
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
