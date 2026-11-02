# ABOUTME: Typer CLI for deslop: `voice` (build and inspect), `apply`, `score`, `lab`, `doctor`.
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
from typer.core import TyperGroup

from deslop import __version__
from deslop.cli.config import Config, ConfigError, load_config, resolve_voice
from deslop.providers.base import Usage


class VoiceGroup(TyperGroup):
    """`deslop voice FILES... -o NAME.voice` builds; `deslop voice show|diff|...` inspect.
    Anything that is not a subcommand name is routed to the hidden `build` command."""

    def parse_args(self, ctx: Any, args: list[str]) -> list[str]:
        if args and args[0] not in self.commands and args[0] not in ("--help", "-h"):
            args = ["build", *args]
        return super().parse_args(ctx, args)


app = typer.Typer(
    help="Build a voice from texts, score prose against it, apply it.", no_args_is_help=True
)
voice_app = typer.Typer(
    cls=VoiceGroup,
    help="Build a voice: `deslop voice FILES... -o NAME.voice`. Subcommands inspect one.",
    no_args_is_help=True,
)
lab_app = typer.Typer(
    help="Experiments: blind tests, exports, threshold calibration.", no_args_is_help=True
)
providers_app = typer.Typer(help="LLM provider configuration.", no_args_is_help=True)
app.add_typer(voice_app, name="voice")
app.add_typer(lab_app, name="lab")
app.add_typer(providers_app, name="providers")

console = Console()
err = Console(stderr=True)


def arm_budget(
    s: Settings,
    budget: float | None,
    *,
    command: str,
    voice: str = "",
    input_file: str = "",
    max_calls: int | None = None,
) -> float:
    """Start the spend guard for this invocation: the context every live call is logged with,
    the USD limit (--budget, config, else $5) and the live-call limit (--max-calls, 200)."""
    from deslop import spend

    limit = float(s.get("budget", budget, spend.DEFAULT_BUDGET))
    spend.set_context(command=command, voice=voice, input_file=input_file)
    spend.set_budget(limit)
    spend.set_max_calls(int(s.get("max_calls", max_calls, spend.DEFAULT_MAX_CALLS)))
    return limit


def provider_name(s: Settings, explicit: str | None) -> str:
    """Which Anthropic provider serves this invocation: --provider, else the config file's
    `provider:` (anthropic | claude-cli), else DESLOP_PROVIDER, else the API."""
    from deslop.providers.base import ANTHROPIC_PROVIDERS, default_provider_name

    name = str(s.get("provider", explicit, default_provider_name())).strip().lower()
    if name not in (*ANTHROPIC_PROVIDERS, "openai", "compatible"):
        fail(f"unknown provider {name!r}; use anthropic or claude-cli")
    return name


def report_budget() -> None:
    from deslop import spend

    if spend.budget_hit():
        err.print(f"[yellow]{spend.over_budget()}[/]")


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


def load_voice(path: Path) -> Any:
    from deslop.voice import Voice
    from deslop.voice.pack import PackError

    try:
        return Voice.load(path)
    except FileNotFoundError as exc:
        fail(str(exc))
    except PackError as exc:
        fail(str(exc))


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


def read_input(path: Path | None) -> str:
    if path is None or str(path) == "-":
        return sys.stdin.read()
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        fail(f"cannot read {path}: {exc.strerror or exc}")


def split_frontmatter(text: str) -> tuple[str, str]:
    """YAML frontmatter and the body: the frontmatter is stripped on ingest and put back on
    output, never rewritten or treated as prose."""
    from deslop.voice.collect import FRONTMATTER_RE

    m = FRONTMATTER_RE.match(text)
    if not m:
        return "", text
    return text[: m.end()], text[m.end() :]


# ------------------------------------------------------------------ voice (build)


@voice_app.command("build", hidden=True)
def voice_build(
    ctx: typer.Context,
    paths: list[Path] = typer.Argument(None, help="Text files or directories to measure"),
    out: Path | None = typer.Option(None, "--out", "-o", help="NAME.voice to write"),
    author_map: Path | None = typer.Option(
        None, "--author-map", help="yaml: filename glob -> author; `default` for the rest"
    ),
    max_share: float | None = typer.Option(None, "--max-share", help="Default 0.15"),
    exemplar_words: int | None = typer.Option(None, "--exemplar-words", help="Default 2500"),
    no_llm: bool = typer.Option(False, "--no-llm", help="Template profile.md instead of a model"),
    llm_model: str | None = typer.Option(None, "--llm-model"),
    provider: str | None = typer.Option(
        None, "--provider", help="anthropic or claude-cli for the profile (config, else API)"
    ),
    base: Path | None = typer.Option(
        None, "--from", help="Fold --add texts into this voice instead of building anew"
    ),
    add: list[Path] = typer.Option([], "--add", help="Texts to add with --from"),
    patterns: list[Path] = typer.Option([], "--patterns", help="Extra pattern catalog yaml"),
    min_docs: int | None = typer.Option(
        None,
        "--min-docs",
        help="Below this many documents the bands shrink toward generic English (10)",
    ),
    profile_from: Path | None = typer.Option(
        None, "--profile-from", help="Reuse this voice's profile.md instead of writing one"
    ),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Build NAME.voice from texts, or extend an existing voice with --from/--add."""
    from deslop.voice import Voice
    from deslop.voice.collect import load_author_map

    s = Settings(ctx, "voice")
    out = s.path("out", out)
    if out is None:
        fail("--out NAME.voice is required")
    sources = list(paths or []) + list(add or [])
    if base is None and not sources:
        fail("give files or directories to measure, or --from OLD.voice --add FILES")
    if base is not None and not add:
        fail("--from needs --add FILES")
    try:
        amap = load_author_map(s.path("author_map", author_map))
    except (OSError, ValueError) as exc:
        fail(str(exc))
    name = out.name.removesuffix(".voice").removesuffix(".zip")
    use_llm = not s.get("no_llm", no_llm or None, False) and profile_from is None
    profile_provider = None
    if use_llm:
        from deslop.providers.base import ProviderError, get_provider

        arm_budget(s, None, command="voice build", voice=name, input_file=str(sources[0]))
        try:
            profile_provider = get_provider(provider_name(s, provider))
        except ProviderError as exc:
            err.print(f"[yellow]no provider for the profile ({exc}); using the template[/]")
            use_llm = False
    try:
        voice = Voice.from_texts(
            [str(p) for p in sources],
            name=name,
            author_map=amap,
            max_share=float(s.get("max_share", max_share, 0.15)),
            exemplar_words=int(s.get("exemplar_words", exemplar_words, 2500)),
            use_llm=use_llm,
            llm_model=s.get("llm_model", llm_model),
            provider=profile_provider,
            patterns=s.paths("patterns", patterns),
            base=load_voice(base) if base is not None else None,
            min_docs=int(s.get("min_docs", min_docs, 10)),
            progress=s.progress(json_out),
        )
    except ValueError as exc:
        fail(str(exc))
    if profile_from is not None:
        donor = load_voice(profile_from)
        voice.profile = donor.profile
        voice.meta.profile_source = f"{donor.meta.profile_source} (from {profile_from.name})"
    voice.save(out)
    if json_out:
        print_json(voice.meta.model_dump_json())
        return
    st = voice.metrics.thresholds.stats
    sep = (
        f"; thresholds: {100 * st.llm_fail_rate:.0f}% of LLM-reference documents fail, "
        f"{100 * st.heldout_pass_rate:.0f}% of held-out documents pass"
        if st
        else ""
    )
    console.print(
        f"wrote voice [bold]{voice.name}[/] to {out} ({voice.meta.document_count} documents, "
        f"{voice.meta.total_words} words, {len(voice.meta.authors)} authors, "
        f"{len(voice.metrics.clusters)} register(s), profile: {voice.meta.profile_source}{sep})"
    )


# ------------------------------------------------------------------ voice (inspect)


@voice_app.command("show")
def voice_show(
    voice: Path,
    clusters: bool = typer.Option(False, "--clusters", help="Show the registers found"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Metadata, provenance, metrics with confidence intervals and bands, tells, rules."""
    v = load_voice(voice)
    if json_out:
        print_json(
            json.dumps(
                {
                    "meta": v.meta.model_dump(),
                    "metrics": v.metrics.model_dump(exclude={"per_author", "per_document"}),
                    "lexicon": v.lexicon.model_dump(),
                }
            )
        )
        return
    m = v.meta
    console.print(
        f"[bold]{v.name}[/]  {m.document_count} documents, {m.total_words} words, "
        f"{len(m.authors)} authors  created {m.created}  deslop {m.deslop_version}"
        + (f"  checksum {m.checksum[:12]}" if m.checksum else "")
    )
    if m.licenses:
        console.print("licenses: " + "; ".join(m.licenses))
    small = v.metrics.small_corpus
    if small is not None:
        console.print(
            f"[yellow]small corpus: {small.documents} documents (fewer than {small.min_docs}); "
            f"bands were shrunk toward generic English with weight {small.weight:.2f}; about "
            f"{small.words_needed:,} more words of corpus would leave small-corpus mode[/]"
        )
    top = ", ".join(
        f"{a.name} {100 * a.share:.0f}%" + (f" ({', '.join(a.sources)})" if a.sources else "")
        for a in m.authors[:12]
    )
    console.print(f"top authors: {top}" + (" ..." if len(m.authors) > 12 else ""))
    if not m.has_exemplars:
        console.print("[yellow]exemplars stripped from this voice[/]")
    if clusters:
        ct = Table(title="Registers")
        for col in ("register", "docs", "words", "median doc", "top authors", "distinguished by"):
            ct.add_column(col)
        for c in v.metrics.clusters:
            marks = "; ".join(
                f"{x.metric} {x.cluster_mean:.2f} vs {x.voice_mean:.2f} (z {x.z:+.1f})"
                for x in c.distinguishing
            )
            ct.add_row(
                c.label,
                str(c.documents),
                str(c.words),
                str(c.median_doc_words),
                ", ".join(c.top_authors),
                marks,
            )
        console.print(ct)
    table = Table(title="Metrics (band: calibrated PASS range; CI: 95% document bootstrap)")
    for col in ("metric", "mean", "95% CI", "within-author sd", "band", "CI width / band"):
        table.add_column(col, justify="right" if col != "metric" else "left")
    warnings = []
    for name, stat in v.metrics.metrics.items():
        band = v.metrics.thresholds.metrics.get(name)
        band_text = "-"
        ratio_text = "-"
        if band is not None:
            lo = f"{band.low:.2f}" if band.low is not None else "-inf"
            hi = f"{band.high:.2f}" if band.high is not None else "inf"
            band_text = f"{lo} .. {hi}"
            if band.high is not None:
                half = band.high - stat.mean
            elif band.low is not None:
                half = stat.mean - band.low
            else:
                half = 0.0
            ci_width = stat.ci_high - stat.ci_low
            if half > 0:
                ratio = ci_width / half
                ratio_text = f"{ratio:.2f}"
                if ratio > 1.0:
                    needed = int(m.total_words * ratio * ratio) - m.total_words
                    warnings.append(
                        f"{name}: CI width {ci_width:.2f} exceeds the band half-width {half:.2f}; "
                        f"about {max(needed, 0):,} more words would narrow it to fit"
                    )
        table.add_row(
            name,
            f"{stat.mean:.2f}",
            f"{stat.ci_low:.2f} .. {stat.ci_high:.2f}",
            f"{stat.within_std:.2f}",
            band_text,
            ratio_text,
        )
    console.print(table)
    for w in warnings:
        console.print(f"[yellow]warning:[/] {w}")
    tt = Table(title="Tell rates (/1k words)")
    for col in ("id", "pattern", "voice", "limit", "llm", "ratio", "banned"):
        tt.add_column(col)
    for p in v.lexicon.patterns:
        tt.add_row(
            p.id,
            p.name,
            f"{p.corpus_per_1k:.2f}",
            f"{v.metrics.thresholds.tells.get(p.id, 0.0):.2f}",
            "-" if p.llm_per_1k is None else f"{p.llm_per_1k:.2f}",
            "-" if p.ratio is None else f"{p.ratio:.1f}",
            "yes" if p.banned else "",
        )
    console.print(tt)
    st = v.metrics.thresholds.stats
    if st:
        console.print(
            f"thresholds: {100 * st.llm_fail_rate:.0f}% of {st.llm_docs} LLM-reference documents "
            f"fail, {100 * st.heldout_pass_rate:.0f}% of {st.heldout_docs} held-out documents "
            f"pass ({st.method})"
        )
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


@voice_app.command("diff")
def voice_diff(
    a: Path = typer.Argument(..., help="A voice (.voice zip or directory)"),
    b: Path = typer.Argument(...),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Metric, tell, lexicon, share and rule differences between two voices."""
    from deslop.voice.diff import diff_voices

    va, vb = load_voice(a), load_voice(b)
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
def voice_validate(voice: Path, json_out: bool = typer.Option(False, "--json")) -> None:
    """Schema and checksum, calibrated thresholds, and the exemplars' own score."""
    from deslop.voice.validate import validate

    result = validate(voice)
    if json_out:
        print_json(result.to_json())
    else:
        for c in result.checks:
            mark = "[green]ok[/]" if c.ok else "[red]FAIL[/]"
            console.print(f"{c.name:22s} {mark}  {c.detail}")
    raise typer.Exit(0 if result.ok else 1)


@voice_app.command("pack")
def voice_pack(
    voice: Path,
    out: Path | None = typer.Option(None, "--out", "-o", help="Default: <dir>.voice"),
    strip_exemplars: bool = typer.Option(
        False, "--strip-exemplars", help="Share the voice without its corpus excerpts"
    ),
) -> None:
    """Zip an unpacked voice directory (plumbing; `voice` writes zips directly)."""
    import tempfile

    from deslop.voice.pack import PackError, pack_voice

    target = out or voice.with_name(voice.name.removesuffix(".voice") + ".voice")
    if voice.is_file():
        work = Path(tempfile.mkdtemp(prefix="deslop-voice-"))
        load_voice(voice).write_dir(work)
        voice = work
    try:
        pack_voice(voice, target, strip_exemplars=strip_exemplars)
    except PackError as exc:
        fail(str(exc))
    console.print(f"packed -> {target}" + (" (exemplars stripped)" if strip_exemplars else ""))


@voice_app.command("unpack")
def voice_unpack(
    archive: Path,
    out: Path | None = typer.Option(None, "--out", "-o", help="Default: <name>.voice.d"),
) -> None:
    """Extract a .voice zip into a working directory, verifying its checksum (plumbing)."""
    from deslop.voice.pack import PackError, unpack_voice

    target = out or archive.with_name(archive.name + ".d")
    try:
        unpack_voice(archive, target)
    except PackError as exc:
        fail(str(exc))
    console.print(f"unpacked {archive} -> {target}")


@voice_app.command("edit")
def voice_edit(voice: Path) -> None:
    """Open profile.md, rules.md and exemplars.md in $VISUAL or $EDITOR; a zip is unpacked
    to a temporary directory and repacked when the editor closes."""
    import os
    import subprocess
    import tempfile

    from deslop.voice.pack import PackError, is_packed, pack_voice, unpack_voice

    editor = os.environ.get("VISUAL") or os.environ.get("EDITOR") or "vi"
    work = voice
    if is_packed(voice):
        work = Path(tempfile.mkdtemp(prefix="deslop-edit-"))
        try:
            unpack_voice(voice, work)
        except PackError as exc:
            fail(str(exc))
    files = [str(work / f) for f in ("profile.md", "rules.md", "exemplars.md")]
    subprocess.run([editor, *files], check=False)
    if work != voice:
        pack_voice(work, voice)
        console.print(f"repacked {voice}")


# ------------------------------------------------------------------ score


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
        from deslop.apply import ExemplarIndex
        from deslop.judge import judge_paragraph
        from deslop.providers.base import DEFAULT_JUDGE, ProviderError, get_provider, resolve_model

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


# ------------------------------------------------------------------ apply


def print_metric_table(rows: list[Any]) -> None:
    """Every metric before and after against the voice's band, flagging under and over."""
    from rich.table import Table

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
    fallbacks = [u for u in result.units if u.kept_original and "already" not in u.reason]
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
        from deslop.apply import coherence_report

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
        help="Max USD this invocation may spend on live calls (5); returns best-so-far",
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
        help="Primary texts (file or directory) added quotations and citations must verify "
        "against; without one, additions are listed as unverified",
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
    limit = arm_budget(
        s,
        budget,
        command="apply",
        voice=v.name,
        input_file=str(brief or file or "stdin"),
        max_calls=max_calls,
    )
    texts = [str(x) for x in _as_list(s.get("primary_text", primary_text or None, []))]
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


# ------------------------------------------------------------------ lab


@lab_app.command("calibrate-thresholds")
def lab_calibrate_thresholds(
    ctx: typer.Context,
    voice: Path,
    out: Path | None = typer.Option(None, "--out", "-o", help="Write the retuned voice here"),
    budget: float | None = typer.Option(None, "--budget", help="Max USD on live calls (5)"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Retune the voice's per-metric bands and per-tell limits against its held-out documents
    and the LLM-reference documents, and report the separation. Makes no live calls."""
    from deslop.baselines import has_baseline, llm_reference
    from deslop.thresholds import calibrate_thresholds

    s = Settings(ctx, "lab")
    v = load_voice(voice)
    arm_budget(s, budget, command="lab calibrate-thresholds", voice=v.name)
    llm = llm_reference() if has_baseline("llm_reference") else None
    v.metrics.thresholds = calibrate_thresholds(
        v.metrics, llm.documents if llm else [], banned=set(v.lexicon.banned_ids())
    )
    st = v.metrics.thresholds.stats
    target = out or voice
    v.read_only = False
    v.save(target)
    if json_out:
        print_json(v.metrics.thresholds.model_dump_json())
        return
    assert st is not None
    console.print(
        f"{100 * st.llm_fail_rate:.0f}% of {st.llm_docs} LLM-reference documents fail; "
        f"{100 * st.heldout_pass_rate:.0f}% of {st.heldout_docs} held-out documents pass "
        f"({st.method}); raw held-out p90 {st.raw_heldout_p90:.3f}, LLM median "
        f"{st.raw_llm_median:.3f}; written to {target}"
    )


@lab_app.command("calibrate-judge")
def lab_calibrate_judge(
    ctx: typer.Context,
    voice: Path,
    corpus: Path = typer.Option(
        ..., "--corpus", help="The voice's source texts (held-out authors)"
    ),
    llm_texts: Path = typer.Option(
        ...,
        "--llm-texts",
        help="LLM-reference essays: a directory of .json (text key), .md or .txt",
    ),
    pairs: int = typer.Option(100, "--pairs", help="Pairs per judge (100)"),
    judges: str | None = typer.Option(
        None, "--judges", help="Judge models to measure (sonnet,opus; aliases allowed)"
    ),
    provider: str | None = typer.Option(None, "--provider", help="anthropic or claude-cli"),
    max_calls: int | None = typer.Option(None, "--max-calls", help="Max live calls (200)"),
    seed: int = typer.Option(0, "--seed"),
    out: Path | None = typer.Option(None, "--out", "-o", help="Write the voice here"),
    budget: float | None = typer.Option(None, "--budget", help="Max USD on live calls (5)"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Measure each judge model as the comparative judge on pairs of one LLM-reference
    paragraph and one held-out corpus paragraph, and store the accuracy in the voice, keyed
    by model. A judge below 80% is reported as unreliable."""
    from deslop import lab
    from deslop.judge import CALIBRATION_MIN_ACCURACY
    from deslop.providers.base import ProviderError, get_provider, resolve_model

    s = Settings(ctx, "calibrate-judge")
    v = load_voice(voice)
    arm_budget(
        s,
        budget,
        command="lab calibrate-judge",
        voice=v.name,
        input_file=str(corpus),
        max_calls=max_calls,
    )
    corpus_paras, method = lab.heldout_paragraphs(v, [corpus])
    llm_paras = lab.llm_paragraphs(llm_texts)
    if not corpus_paras or not llm_paras:
        fail(
            f"need paragraphs of both kinds; got {len(corpus_paras)} held-out and "
            f"{len(llm_paras)} LLM-reference"
        )
    name = provider_name(s, provider)
    providers = []
    for model in _as_list(s.get("judges", judges, "sonnet,opus")):
        try:
            judge_provider = get_provider(name)
        except ProviderError as exc:
            fail(str(exc))
        judge_provider.default_model = resolve_model(str(model))
        providers.append(judge_provider)
    results = lab.calibrate_judges(v, providers, llm_paras, corpus_paras, pairs=pairs, seed=seed)
    usage = Usage()
    for jp in providers:
        usage.add(jp.usage)
    report_budget()
    for model_name, cal in results.items():
        v.meta.judge_calibration[model_name] = cal.model_dump()
    target = out or voice
    v.read_only = False
    v.save(target)
    if json_out:
        print_json(json.dumps({k: c.model_dump() for k, c in results.items()}, indent=1))
        return
    console.print(
        f"{len(corpus_paras)} held-out paragraphs ({method}), {len(llm_paras)} LLM-reference "
        f"paragraphs; ${usage.cost_usd:.3f} over {usage.calls} call(s) via {name}"
    )
    for model_name, cal in results.items():
        verdict = (
            "usable"
            if cal.usable
            else f"below {100 * CALIBRATION_MIN_ACCURACY:.0f}%: unreliable as judge"
        )
        console.print(
            f"  {model_name}: {100 * cal.accuracy:.0f}% picked the corpus paragraph as the "
            f"writer's over {cal.pairs} pairs, {100 * cal.model_prose_accuracy:.0f}% picked the "
            f"LLM paragraph as model prose, mean confidence {cal.mean_confidence:.2f}, "
            f"position bias {cal.position_bias:+.2f}; {verdict}"
        )
    if results and not any(c.usable for c in results.values()):
        console.print("[yellow]no provider reached 80%; apply will not gate on the judge[/]")
    console.print(f"written to {target}" if target else "not written (read-only voice; use --out)")


@lab_app.command("recitation")
def lab_recitation(
    result: Path = typer.Argument(..., help="An apply --json result file"),
    reference: Path = typer.Option(..., "--reference", help="The human text to compare against"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Re-score every candidate of a saved result for recitation of a reference: runs of
    eight or more words shared with the reference and not with the input. Offline."""
    from deslop.lab import rescore_recitation

    try:
        data = json.loads(result.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read {result}: {exc}")
    ref_text = split_frontmatter(reference.read_text(encoding="utf-8"))[1]
    rows = rescore_recitation(data, ref_text)
    if json_out:
        print_json(json.dumps(rows, indent=1))
        return
    for r in rows:
        console.print(
            f"candidate {r['attempt']} ({r['status']}): {r['ngrams']} shared run(s), "
            f"{r['covered']}/{r['words']} words covered, {r['share']:.1f}%; "
            f"{r['novel']} outside the input's words, {r['novel_share']:.1f}%"
        )
        for span in r["spans"]:
            console.print(f"    {len(span.split())} words: {span}", markup=False)


@lab_app.command("fixture-check")
def lab_fixture_check(
    ctx: typer.Context,
    original: Path = typer.Argument(..., help="The candidate fixture text"),
    voice: Path = typer.Option(..., "--voice", "-v"),
    writer: str | None = typer.Option(None, "--writer", help="Writer model (claude-opus-5-5)"),
    judge: str | None = typer.Option(None, "--judge", help="Judge model (claude-sonnet-5-5)"),
    provider: str | None = typer.Option(None, "--provider", help="anthropic or claude-cli"),
    max_calls: int | None = typer.Option(None, "--max-calls", help="Max live calls (200)"),
    budget: float | None = typer.Option(None, "--budget", help="Max USD on live calls (5)"),
    show_draft: bool = typer.Option(False, "--draft", help="Print the draft too"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Can this text be a held-out fixture? Its propositions are extracted, one draft is
    written from them alone (the text is never shown), and the draft is checked for
    recitation of the text. It qualifies only when the recited words outside the
    propositions' vocabulary are under 1% of the draft. Two live calls, cached."""
    from deslop.apply import build_models
    from deslop.lab import fixture_check
    from deslop.providers.base import DEFAULT_JUDGE, DEFAULT_WRITER, BudgetExceeded, ProviderError

    s = Settings(ctx, "fixture-check")
    v = load_voice(voice)
    arm_budget(
        s,
        budget,
        command="lab fixture-check",
        voice=v.name,
        input_file=str(original),
        max_calls=max_calls,
    )
    body = split_frontmatter(read_input(original))[1]
    try:
        models = build_models(
            str(s.get("writer", writer, DEFAULT_WRITER)),
            str(s.get("judge", judge, DEFAULT_JUDGE)),
            provider=provider_name(s, provider),
            calibration=v.meta.judge_calibration,
        )
        check = fixture_check(body, v, models)
    except (ValueError, BudgetExceeded, ProviderError) as exc:
        fail(str(exc), code=2)
    report_budget()
    if json_out:
        print_json(check.model_dump_json(indent=1))
        return
    r = check.recitation
    console.print(
        f"{original}: {check.words} words, {len(check.propositions)} propositions; draft of "
        f"{r.words} words recites {r.share:.1f}% ({r.ngrams} shared run(s), {r.covered} words)"
    )
    for span in r.spans:
        console.print(f"    {len(span.split())} words: {span}", markup=False)
    console.print(
        f"{r.novel} of the {r.covered} recited words are outside the propositions' vocabulary: "
        f"{r.novel_share:.1f}% of the draft (reassembly of the propositions gives 0; memory of "
        "the text gives more)"
    )
    verdict = "qualifies" if check.qualifies else "does not qualify"
    console.print(
        f"{verdict} as a fixture (recited words outside the propositions under "
        f"{check.threshold:.0f}% of the draft)"
    )
    if show_draft:
        console.print("", check.draft, markup=False)
    raise typer.Exit(0 if check.qualifies else 2)


@lab_app.command("blind")
def lab_blind(
    ctx: typer.Context,
    voice: Path | None = typer.Option(None, "--voice", "-v"),
    candidates: Path | None = typer.Option(None, "--candidates", help="Dir of .md/.txt drafts"),
    texts: Path | None = typer.Option(
        None, "--texts", help="Human texts to pair against (default: the voice's exemplars)"
    ),
    n: int | None = typer.Option(None, "--n", help="Pairs to judge (5)"),
    seed: int | None = typer.Option(None, "--seed"),
    min_overlap: float | None = typer.Option(
        None, "--min-overlap", help="Skip pairs whose lexical overlap is below this (0.03)"
    ),
    export: Path | None = typer.Option(
        None, "--export", help="Write the shuffled pairs and a key to this directory instead"
    ),
    budget: float | None = typer.Option(None, "--budget", help="Max USD on live calls (5)"),
    json_out: bool = typer.Option(False, "--json", help="Print the summary as JSON at the end"),
) -> None:
    """Can you tell a draft from the real thing? Pairs candidate paragraphs with human ones.
    Makes no live calls."""
    from deslop import blind

    s = Settings(ctx, "blind")
    v = load_voice(s.voice(voice))
    arm_budget(s, budget, command="lab blind", voice=v.name)
    candidates = s.path("candidates", candidates)
    if candidates is None or not candidates.is_dir():
        fail("--candidates must be a directory of .md or .txt files")
    cands = blind.candidates_from_dir(candidates)
    if not cands:
        fail(f"no paragraphs of {blind.MIN_WORDS}+ words under {candidates}")
    pool = blind.control_pool(v, s.path("texts", texts))
    if not pool:
        fail("no human paragraphs to pair with; give --texts DIR or a voice with exemplars")
    min_overlap = float(s.get("min_overlap", min_overlap, blind.MIN_OVERLAP))
    pairing = blind.make_pairs(
        cands, pool, int(s.get("n", n, 5)), seed=s.get("seed", seed), min_overlap=min_overlap
    )
    pairs = pairing.pairs
    if export is not None:
        from deslop.lab import export_blind

        pairs_path, key_path = export_blind(pairing, export, v.name)
        console.print(
            f"wrote {len(pairs)} pair(s) to {pairs_path} (answers in {key_path}); "
            f"skipped {pairing.skipped}, unpaired {pairing.unpaired}"
        )
        return
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


@lab_app.command("export")
def lab_export(
    ctx: typer.Context,
    voice: Path,
    out: Path | None = typer.Option(None, "--out", "-o", help="JSONL file to write"),
    fmt: str | None = typer.Option(None, "--format", help="openai-ft (default) or jsonl"),
    corpus: Path | None = typer.Option(None, "--corpus", help="Texts to summarize into chunks"),
    no_pairs: bool = typer.Option(False, "--no-pairs", help="Skip accepted apply pairs"),
    no_chunks: bool = typer.Option(False, "--no-chunks", help="Skip summarized corpus chunks"),
    max_chunks: int | None = typer.Option(None, "--max-chunks"),
    budget: float | None = typer.Option(None, "--budget", help="Max USD on live calls (5)"),
    cost_model: str | None = typer.Option(
        None, "--cost-model", help="Model whose training price to estimate with (gpt-4.1)"
    ),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Write fine-tuning data from accepted pairs and summarized corpus chunks."""
    from deslop.voice.export import export, write_jsonl

    s = Settings(ctx, "voice_export")
    arm_budget(s, budget, command="lab export", voice=str(voice))
    out = s.path("out", out)
    if out is None:
        fail("--out is required")
    v = load_voice(voice)
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


# ------------------------------------------------------------------ doctor, providers


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
        fail("give --html (the only report format so far)")
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
