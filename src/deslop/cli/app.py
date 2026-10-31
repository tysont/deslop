# ABOUTME: Typer CLI for deslop. Every command wraps the public API and supports --json.
# ABOUTME: Subcommands: corpus, calibrate, score, voice, providers.
from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from deslop import __version__
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


@app.callback()
def main(
    version: bool = typer.Option(False, "--version", callback=version_callback, is_eager=True),
) -> None:
    pass


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
    directory: Path,
    sources: list[str] = typer.Argument(..., help="Files or URLs: txt, md, html, epub, pdf"),
    author: str = typer.Option(..., "--author"),
    title: str | None = typer.Option(None, "--title"),
    year: int | None = typer.Option(None, "--year"),
    weight: float = typer.Option(1.0, "--weight"),
    license: str = typer.Option("", "--license"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    corpus = Corpus.load(directory)
    added = []
    for source in sources:
        try:
            entry = corpus.add(
                source,
                author=author,
                title=title if len(sources) == 1 else None,
                year=year,
                weight=weight,
                license=license,
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
    directory: Path,
    max_share: float = typer.Option(0.25, "--max-share"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
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
    corpus_dir: Path,
    out: Path = typer.Option(..., "--out", help="Output voice directory, e.g. name.voice"),
    name: str | None = typer.Option(None, "--name"),
    no_llm: bool = typer.Option(
        False, "--no-llm", help="Write a templated profile.md instead of calling a model"
    ),
    max_share: float = typer.Option(0.25, "--max-share"),
    exemplar_words: int = typer.Option(2000, "--exemplar-words"),
    llm_model: str | None = typer.Option(None, "--llm-model"),
    patterns: list[Path] = typer.Option([], "--patterns", help="Extra pattern catalog yaml"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    from deslop.voice.calibrate import calibrate

    corpus = Corpus.load(corpus_dir)
    progress = None if json_out else (lambda msg: err.print(f"[dim]{msg}[/]"))
    voice = calibrate(
        corpus,
        name=name or out.name.removesuffix(".voice"),
        max_share=max_share,
        exemplar_words=exemplar_words,
        use_llm=not no_llm,
        llm_model=llm_model,
        patterns=tuple(str(p) for p in patterns),
        progress=progress,
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


@app.command("score")
def score_cmd(
    file: Path | None = typer.Argument(None, help="Text file to score (default: stdin)"),
    voice: Path = typer.Option(..., "--voice", "-v"),
    sigma: float = typer.Option(1.5, "--sigma"),
    tell_mult: float = typer.Option(2.0, "--tell-mult"),
    tell_floor: float = typer.Option(
        1.0, "--tell-floor", help="Minimum tell rate per 1k words before flagging"
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

    text = read_input(file)
    v = Voice.load(voice)
    report = score(
        text,
        v,
        sigma=sigma,
        tell_mult=tell_mult,
        tell_floor=tell_floor,
        patterns=tuple(str(p) for p in patterns),
    )
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
