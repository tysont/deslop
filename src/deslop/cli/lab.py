# ABOUTME: `deslop lab`: calibration, recitation, fixture checks, model prose from propositions,
# ABOUTME: blind tests and fine-tuning exports.
from __future__ import annotations

import json
from pathlib import Path

import typer

from deslop.cli.common import (
    Settings,
    arm_budget,
    as_list,
    console,
    fail,
    load_voice,
    print_json,
    provider_name,
    read_input,
    report_budget,
    split_frontmatter,
)
from deslop.providers.base import Usage

lab_app = typer.Typer(
    help="Calibration, recitation and fixture checks, blind tests and fine-tuning exports.",
    no_args_is_help=True,
)


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
    for model in as_list(s.get("judges", judges, "sonnet,opus")):
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
        console.print(
            "[yellow]no judge reached 80%; apply will still use it and mark its verdicts "
            "unreliable[/]"
        )
    console.print(f"written to {target}")


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
    from deslop.lab import fixture_check
    from deslop.providers.base import DEFAULT_JUDGE, DEFAULT_WRITER, BudgetExceeded, ProviderError
    from deslop.providers.roles import build_models

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


@lab_app.command("slop")
def lab_slop(
    ctx: typer.Context,
    original: Path = typer.Argument(..., help="The human text to write model prose from"),
    out: Path = typer.Option(..., "--out", "-o", help="SLOP.md to write"),
    kind: str = typer.Option("a piece of prose", "--kind", help='e.g. "a personal essay"'),
    provider: str | None = typer.Option(None, "--provider", help="anthropic or claude-cli"),
    max_calls: int | None = typer.Option(None, "--max-calls", help="Max live calls (200)"),
    budget: float | None = typer.Option(None, "--budget", help="Max USD on live calls (5)"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Model prose from a text's propositions alone (the text is never shown): the input for
    an apply run that holds the text out as the reference. Two live calls, cached."""
    import datetime as dt

    from deslop.lab import slop_from_propositions
    from deslop.providers.base import BudgetExceeded, ProviderError, get_provider

    s = Settings(ctx, "slop")
    arm_budget(s, budget, command="lab slop", input_file=str(original), max_calls=max_calls)
    body = split_frontmatter(read_input(original))[1].strip()
    try:
        p = get_provider(provider_name(s, provider))
        draft = slop_from_propositions(body, p, kind=kind)
    except (ValueError, BudgetExceeded, ProviderError) as exc:
        fail(str(exc), code=2)
    report_budget()
    header = (
        "---\n"
        f"title: {original.name} (model prose from propositions)\n"
        f"generated_by: {p.name}:{p.default_model}\n"
        f"generated_on: {dt.date.today().isoformat()}\n"
        f"propositions: {len(draft.propositions)}\n"
        f"reference: {original}\n"
        "note: Written by `deslop lab slop` from the text's proposition list alone; the model\n"
        "  never saw the text.\n"
        "---\n\n"
    )
    out.write_text(header + draft.text + "\n", encoding="utf-8")
    if json_out:
        print_json(draft.model_dump_json(indent=1))
        return
    console.print(
        f"wrote {out}: {len(draft.text.split())} words from {len(draft.propositions)} "
        f"propositions of {original}"
    )


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
