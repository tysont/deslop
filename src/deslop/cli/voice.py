# ABOUTME: `deslop voice`: build a voice from files, and show, diff, validate, pack, unpack,
# ABOUTME: edit or add tells to one.
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich.table import Table
from typer.core import TyperGroup

from deslop.cli.common import (
    Settings,
    arm_budget,
    console,
    err,
    fail,
    load_voice,
    print_json,
    provider_name,
)


class VoiceGroup(TyperGroup):
    """`deslop voice FILES... -o NAME.voice` builds; `deslop voice show|diff|...` inspect.
    Anything that is not a subcommand name is routed to the hidden `build` command."""

    def parse_args(self, ctx: Any, args: list[str]) -> list[str]:
        if args and args[0] not in self.commands and args[0] not in ("--help", "-h"):
            args = ["build", *args]
        return super().parse_args(ctx, args)


voice_app = typer.Typer(
    cls=VoiceGroup,
    help="Build a voice: `deslop voice FILES... -o NAME.voice`. Subcommands inspect one.",
    no_args_is_help=True,
)


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
    if m.primary_texts:
        names = [t if isinstance(t, str) else str(t.get("path", "")) for t in m.primary_texts]
        console.print("primary texts: " + "; ".join(names))
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


@voice_app.command("add-tells")
def voice_add_tells(
    ctx: typer.Context,
    voice: Path = typer.Argument(..., help="The voice to extend"),
    paths: list[Path] = typer.Argument(..., help="Its corpus: text files or directories"),
    out: Path = typer.Option(..., "--out", "-o", help="NAME.voice to write"),
    author_map: Path | None = typer.Option(None, "--author-map"),
    patterns: list[Path] = typer.Option([], "--patterns", help="Extra pattern catalog yaml"),
) -> None:
    """Measure, on the voice's own corpus, the tells it predates (a pattern added to the
    catalog after the voice was built) and add only those: rate, starting limit and the
    lexicon's comparison with the LLM reference. Offline; nothing else changes."""
    from deslop.voice.calibrate import add_tells
    from deslop.voice.collect import collect, load_author_map

    s = Settings(ctx, "voice")
    v = load_voice(voice)
    try:
        amap = load_author_map(author_map)
    except (OSError, ValueError) as exc:
        fail(str(exc))
    corpus, _ = collect(list(paths), name=v.name, author_map=amap)
    added = add_tells(
        v, corpus, patterns=tuple(str(p) for p in patterns), progress=s.progress(False)
    )
    v.save(out)
    if not added:
        console.print(f"{voice}: no tell to add; wrote {out} unchanged")
        return
    for tid in added:
        rate = v.metrics.tells[tid].rate_per_1k
        limit = v.metrics.thresholds.tells[tid]
        console.print(f"{tid}: {rate:.2f} per 1k words in the corpus, limit {limit:.2f}")
    console.print(f"wrote {out}")


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
