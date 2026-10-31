# ABOUTME: Typer CLI for deslop. Every command wraps the public API and supports --json.
# ABOUTME: Options resolve as flag > deslop.yaml command section > top-level key > default.
from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, NoReturn

import typer
from rich.console import Console
from rich.table import Table

from deslop import __version__
from deslop.cli.config import Config, ConfigError, load_config, resolve_voice
from deslop.converter import ConvertResult
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


def fail(message: str, code: int = 1) -> NoReturn:
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
        try:
            return resolve_voice(self.cfg, self.command, value)
        except ConfigError as exc:
            fail(str(exc))

    def progress(self, json_out: bool) -> Callable[[str], None] | None:
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
    from_manifest: bool = typer.Option(
        False, "--from-manifest", help="SOURCE is a directory with a manifest.jsonl"
    ),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    s = Settings(ctx, "corpus_add")
    corpus = Corpus.load(directory)
    if from_manifest:
        if len(sources) != 1:
            fail("--from-manifest takes exactly one directory")
        try:
            report = corpus.add_manifest(Path(sources[0]))
        except IngestError as exc:
            fail(str(exc))
        if json_out:
            print_json(report.model_dump_json())
        else:
            for entry in report.added:
                console.print(
                    f"added [bold]{entry.title}[/] by {entry.author}: {entry.words} words"
                )
            for title, reason in report.failed:
                console.print(f"[red]failed[/] {title}: {reason}")
            console.print(
                f"added {len(report.added)}, {len(report.duplicates)} duplicate(s) skipped, "
                f"{len(report.failed)} failed"
            )
        return
    author = s.get("author", author)
    if not author:
        fail("--author is required")
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
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        fail(f"cannot read {path}: {exc.strerror or exc}")


def score_options(
    s: Settings,
    sigma: float | None,
    tell_mult: float | None,
    tell_floor: float | None,
    patterns: list[Path],
) -> dict[str, Any]:
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
    s: Settings,
    mode: str | None,
    max_iter: int | None,
    models: str | None,
    budget: float | None,
    lock: str | None,
    preserve_length: float | None,
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


def print_convert_summary(result: ConvertResult, show_diff: bool, out: Path | None) -> None:
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
    opts.pop("mode")
    opts.pop("max_iter")  # convert defaults to 4 rounds; draft defaults to 2
    opts["convert_max_iter"] = int(s.get("max_iter", max_iter, 2))
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


@voice_app.command("update")
def voice_update(
    ctx: typer.Context,
    voice: Path,
    accept: list[Path] = typer.Option(..., "--accept", help="Accepted texts to fold in"),
    corpus: Path | None = typer.Option(None, "--corpus", help="Corpus dir (default: recorded)"),
    weight: float | None = typer.Option(None, "--weight", help="Weight for accepted docs (0.5)"),
    regenerate: bool = typer.Option(
        False, "--regenerate", help="Also rewrite profile.md, rules.md and exemplars.md"
    ),
    no_llm: bool = typer.Option(False, "--no-llm", help="Template profile when regenerating"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    from deslop.voice.update import update

    s = Settings(ctx, "voice_update")
    try:
        res = update(
            voice,
            accept,
            corpus=s.path("corpus", corpus),
            weight=float(s.get("weight", weight, 0.5)),
            regenerate=bool(s.get("regenerate", regenerate or None, False)),
            use_llm=not s.get("no_llm", no_llm or None, False),
            progress=s.progress(json_out),
        )
    except (FileNotFoundError, ValueError) as exc:
        fail(str(exc))
    if json_out:
        print_json(res.model_dump_json())
        return
    console.print(
        f"updated [bold]{res.voice}[/]: added {len(res.added)}, skipped {len(res.skipped)}, "
        f"now {res.total_words} words; previous voice saved to {res.snapshot}"
    )


@voice_app.command("diff")
def voice_diff(
    a: str = typer.Argument(..., help="Voice dir, or dir@prev, or dir@<snapshot stamp>"),
    b: str = typer.Argument(...),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    from deslop.voice import Voice
    from deslop.voice.diff import diff_voices
    from deslop.voice.update import resolve_ref

    try:
        va, vb = Voice.load(resolve_ref(a)), Voice.load(resolve_ref(b))
    except FileNotFoundError as exc:
        fail(str(exc))
    d = diff_voices(va, vb)
    if json_out:
        print_json(d.to_json())
        return
    console.print(f"[bold]{a}[/] ({d.words_before} words) -> [bold]{b}[/] ({d.words_after} words)")
    table = Table(title="Metric means")
    for col in ("metric", "before", "after", "delta", "z", "direction"):
        table.add_column(col, justify="left" if col in ("metric", "direction") else "right")
    for m in d.metrics:
        if m.delta == 0:
            continue
        color = {"toward_slop": "red", "away_from_slop": "green"}.get(m.direction, "")
        table.add_row(
            m.name,
            f"{m.before:.2f}",
            f"{m.after:.2f}",
            f"{m.delta:+.2f}",
            f"{m.z:+.2f}",
            f"[{color}]{m.direction}[/]" if color else m.direction,
        )
    console.print(table)
    changed_tells = [t for t in d.tells if t.before != t.after or t.banned_before != t.banned_after]
    if changed_tells:
        tt = Table(title="Tell rates (/1k)")
        for col in ("id", "pattern", "before", "after", "banned"):
            tt.add_column(col)
        for t in changed_tells:
            ban = (
                ("yes" if t.banned_before else "no") + " -> " + ("yes" if t.banned_after else "no")
            )
            tt.add_row(t.id, t.name, f"{t.before:.2f}", f"{t.after:.2f}", ban)
        console.print(tt)
    for label, items in (
        ("encouraged +", d.encouraged_added),
        ("encouraged -", d.encouraged_removed),
        ("discouraged +", d.discouraged_added),
        ("discouraged -", d.discouraged_removed),
        ("banned +", d.banned_added),
        ("banned -", d.banned_removed),
    ):
        if items:
            console.print(f"[bold]{label}[/] " + ", ".join(items[:40]))
    shares = {k: v for k, v in d.author_shares.items() if v[0] != v[1]}
    if shares:
        console.print(
            "[bold]author shares:[/] "
            + ", ".join(f"{k} {100 * v[0]:.0f}% -> {100 * v[1]:.0f}%" for k, v in shares.items())
        )
    if d.rules_diff:
        console.print(d.rules_diff, markup=False, highlight=False)
    for name, flag in (("profile.md", d.profile_changed), ("exemplars.md", d.exemplars_changed)):
        if flag:
            console.print(f"{name} changed")
    if not d.changed:
        console.print("no differences")


@voice_app.command("validate")
def voice_validate(
    ctx: typer.Context,
    voice: Path,
    corpus: Path | None = typer.Option(None, "--corpus", help="Corpus dir (default: recorded)"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    from deslop.voice.validate import validate

    s = Settings(ctx, "voice_validate")
    result = validate(voice, corpus=s.path("corpus", corpus))
    if json_out:
        print_json(result.to_json())
    else:
        for c in result.checks:
            mark = "[green]ok[/]" if c.ok else "[red]FAIL[/]"
            console.print(f"{c.name:22s} {mark}  {c.detail}")
    raise typer.Exit(0 if result.ok else 1)


@voice_app.command("export")
def voice_export(
    ctx: typer.Context,
    voice: Path,
    out: Path | None = typer.Option(None, "--out", "-o", help="JSONL file to write"),
    fmt: str | None = typer.Option(None, "--format", help="openai-ft (default) or jsonl"),
    corpus: Path | None = typer.Option(None, "--corpus"),
    no_pairs: bool = typer.Option(False, "--no-pairs", help="Skip accepted convert pairs"),
    no_chunks: bool = typer.Option(False, "--no-chunks", help="Skip summarized corpus chunks"),
    max_chunks: int | None = typer.Option(None, "--max-chunks"),
    cost_model: str | None = typer.Option(
        None, "--cost-model", help="Model whose training price to estimate with (gpt-4.1)"
    ),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    from deslop.voice import Voice
    from deslop.voice.export import export, write_jsonl

    s = Settings(ctx, "voice_export")
    out = s.path("out", out)
    if out is None:
        fail("--out is required")
    v = Voice.load(voice)
    try:
        res = export(
            v,
            fmt=s.get("format", fmt, "openai-ft"),
            corpus=s.path("corpus", corpus),
            include_pairs=not s.get("no_pairs", no_pairs or None, False),
            include_chunks=not s.get("no_chunks", no_chunks or None, False),
            max_chunks=s.get("max_chunks", max_chunks),
            cost_model=s.get("cost_model", cost_model, "gpt-4.1"),
            progress=s.progress(json_out),
        )
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        fail(str(exc))
    n = write_jsonl(res, v, out)
    if json_out:
        print_json(res.to_json())
        return
    cost = (
        f"${res.estimated_training_cost:.2f} to train {res.cost_model}"
        if res.estimated_training_cost is not None
        else f"no training price for {res.cost_model} in pricing.yaml"
    )
    console.print(
        f"wrote {n} record(s) to {out} ({res.pairs} pairs, {res.chunks} chunks); "
        f"about {res.tokens:,} tokens, estimated {cost}; "
        f"summaries cost ${res.summary_cost:.3f}. No job was launched."
    )


# ------------------------------------------------------------------ blind


@app.command("blind")
def blind_cmd(
    ctx: typer.Context,
    voice: Path | None = typer.Option(None, "--voice", "-v"),
    candidates: Path | None = typer.Option(None, "--candidates", help="Dir of .md/.txt drafts"),
    n: int | None = typer.Option(None, "--n", help="Pairs to judge (5)"),
    corpus: Path | None = typer.Option(None, "--corpus"),
    seed: int | None = typer.Option(None, "--seed"),
    min_overlap: float | None = typer.Option(
        None, "--min-overlap", help="Skip pairs whose lexical overlap is below this (0.03)"
    ),
    json_out: bool = typer.Option(False, "--json", help="Print the summary as JSON at the end"),
) -> None:
    from deslop import blind
    from deslop.voice import Voice
    from deslop.voice.update import corpus_for

    s = Settings(ctx, "blind")
    v = Voice.load(s.voice(voice))
    candidates = s.path("candidates", candidates)
    if candidates is None or not candidates.is_dir():
        fail("--candidates must be a directory of .md or .txt files")
    try:
        corp = corpus_for(v, s.path("corpus", corpus))
    except FileNotFoundError as exc:
        fail(str(exc))
    cands = blind.candidates_from_dir(candidates)
    if not cands:
        fail(f"no paragraphs of {blind.MIN_WORDS}+ words under {candidates}")
    min_overlap = float(s.get("min_overlap", min_overlap, blind.MIN_OVERLAP))
    pairing = blind.make_pairs(
        cands,
        blind.corpus_pool(corp),
        int(s.get("n", n, 5)),
        seed=s.get("seed", seed),
        min_overlap=min_overlap,
    )
    pairs = pairing.pairs
    session = []
    for i, pair in enumerate(pairs, 1):
        a, b = blind.shown(pair)
        console.rule(f"pair {i}/{len(pairs)}  (overlap {pair.overlap:.2f})")
        for letter, text in (("A", a), ("B", b)):
            console.print(f"[bold]{letter}[/]")
            console.print(text + "\n", markup=False, highlight=False)
        guess = ""
        while guess.upper() not in ("A", "B"):
            guess = typer.prompt("Which one is machine-written? (A/B)")
        reason = typer.prompt("One-line reason", default="", show_default=False)
        entry = blind.record(v.name, pair, guess, reason)
        session.append(entry)
        verdict = "[green]right[/]" if entry["correct"] else "[red]wrong[/]"
        console.print(
            f"{verdict}: the candidate was {pair.candidate_position} "
            f"({pair.candidate_source} vs {pair.control_source})"
        )
    summary = blind.summarize(v.name, session)
    summary.skipped = pairing.skipped
    summary.unpaired = pairing.unpaired
    if json_out:
        print_json(summary.to_json())
        return
    console.rule("results")
    if pairing.skipped:
        console.print(
            f"skipped {pairing.skipped} pair(s) with overlap below {min_overlap:.2f}; "
            "lower --min-overlap or add candidates closer to the corpus"
        )
    if pairing.unpaired:
        console.print(f"{pairing.unpaired} candidate(s) could not be paired")
    if summary.session_accuracy is not None:
        console.print(
            f"this session: {summary.session_correct}/{summary.session_total} "
            f"({100 * summary.session_accuracy:.0f}%)"
        )
    if summary.accuracy is not None:
        console.print(
            f"cumulative: {summary.correct}/{summary.total} ({100 * summary.accuracy:.0f}%)"
        )
    if summary.top_reasons:
        console.print(
            "most frequent reasons: " + "; ".join(f"{r} ({c})" for r, c in summary.top_reasons)
        )


# ------------------------------------------------------------------ doctor


@app.command("doctor")
def doctor_cmd(ctx: typer.Context, json_out: bool = typer.Option(False, "--json")) -> None:
    """Check the installation: spaCy model, providers, pricing age, cache dir, default voice."""
    from deslop import doctor

    s = Settings(ctx, "doctor")
    checks = doctor.run(s.cfg)
    ok = all(c.status != "fail" for c in checks)
    if json_out:
        print_json(json.dumps({"ok": ok, "checks": [c.model_dump() for c in checks]}))
    else:
        colors = {"ok": "green", "warn": "yellow", "fail": "red"}
        for c in checks:
            console.print(f"[{colors[c.status]}]{c.status:4s}[/] {c.name:10s} {c.detail}")
    raise typer.Exit(0 if ok else 1)


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
