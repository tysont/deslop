# Changelog

All notable changes to deslop. Dates are release dates; versions are git tags.

## v2.2.1 - 2026-11-01

- Spend guard: every live provider call is appended to `$DESLOP_HOME/spend.jsonl` with its
  command, provider, model, tokens, cost, voice and input file. `apply` and every `lab`
  command take `--budget` (default $5 per invocation) and stop with the best result so far
  when it is spent. `make registers` prints its cost projection and refuses without
  `CONFIRM=1` and `MAX_COST`. `deslop doctor` prints spend today and over the last 7 days.
- Register slop fixtures are now written from each original's proposition list alone (the
  model never sees the original) and carry their score against the register voice and the
  LLM-reference baseline; the v2.2.0 paraphrases moved to `registers/paraphrase/`.
- Small-corpus calibration: with fewer than `--min-docs` (10) documents, metric means and
  spreads shrink toward generic English weighted by document count, using within-document
  paragraph variance; band lower bounds clip at 0; every own document must pass and banned
  tells relax until they do; `voice show` warns and says how many words would leave
  small-corpus mode. `voice build --profile-from OLD.voice` reuses a profile without a
  model call. A metric generic English does not measure (`connective_opener_pct`) keeps the
  corpus estimate with the paragraph spread.
- `--allow-same-provider-judge` lets a different model of the writer's vendor judge
  (`--models claude,claude:claude-sonnet-5-5`), with a warning in the report header; cross-
  vendor remains the supported configuration and the default.

## v2.2.0 - 2026-11-01

- `apply` rewrites the document as a whole by default. The writer sees the profile, the
  rules, the exemplars nearest the piece, the full original and the propositions by
  paragraph, and is told to write the piece as this writer would, keep every proposition and
  add none, and keep the author's tense, person, paragraphing and order of argument unless
  the profile says otherwise. Nothing in the prompt names a metric, sentence length, openers
  or citations; linter findings reach only the retry prompt, as what a reader noticed, with
  the span. Documents over `--chunk-words` (2500) are rewritten section by section with the
  already-rewritten section in context and the next original section visible. Paragraph-
  isolated rewriting, the per-paragraph judge gate and `--min-confidence` are gone.
- Coherence checks, all required: the dominant finite-verb tense of each paragraph unchanged
  (drift reported as paragraph, before, after, example sentence); every sentence-initial
  pronoun or this/that/it with a noun phrase in the previous two sentences; the share of
  connective-opened sentences within the voice's band and within half of the original's; and
  a document-level judge answering four items separately (reads as the voice, coherent,
  consistent tense and person, preserves the argument's order and emphasis), on every one of
  which the candidate must win or tie against the original. `--n` candidates are judged
  round-robin on the same items; the winner must pass every check and both locks, else the
  result is the original with the report, never a partial.
- Two-sided anchors: `citation_per_100w`, `quote_per_para`, `concrete_per_100w`,
  `sent_len_cv`, `opener_top_word_share` and `reader_address_per_1k` are matched to the
  voice's band in both directions; overshoot fails like undershoot. New metric
  `connective_opener_pct`. The LLM-reference baseline was rebuilt from the cached responses
  (no new generation) and the shipped voices retuned with `lab calibrate-thresholds`.
- The report adds the coherence results, the judge's item-by-item verdict, and a table of
  every metric before and after against the voice's band.
- A fixture set of four registers (essay, explainer, product, fiction) under
  `tests/fixtures/registers/` replaces the Paideia lesson as the test case.
- The bundled reference texts (KJV Genesis, Jowett's Republic) and every default primary-text
  selection (`voice.yaml` `primary_texts`, a brief's `primary_texts`, `voice build
  --primary-text`) are removed. `--primary-text` is user-supplied only: a file or directory.
- `lab blind --texts DIR` now loads a crawl directory with a `manifest.jsonl` through the
  corpus collector instead of silently skipping it, so the control pool is no longer empty
  for a manifest-backed corpus.

## v2.1.0 - 2026-11-01

- Primary-text verification: `apply --primary-text FILE...` (and `primary_texts:` in a brief
  or in `voice.yaml`) names plain-text sources with `{{Genesis 6:5}}` or `{{369b}}` markers.
  Every added quotation must occur verbatim in one, every added verse or Stephanus citation
  must resolve to a marker, else the candidate is rejected and the span named. Without a
  primary text, additions pass and are listed as "unverified additions" at the top of the
  report and in `--diff`. `deslop.reference` ships KJV Genesis (Gutenberg #10) and Jowett's
  Republic with Stephanus pages (Gutenberg #55201) as `kjv-genesis` and `jowett-republic`.
- Comparative judge: the absolute rubric gate is replaced by pairwise judging. A provider
  that wrote neither text sees two paragraphs and two exemplars and says which reads more
  like the writer and which more like model prose, with a confidence. Each candidate is
  judged against the original and round-robin against the other candidates, giving a win
  rate; the gate is "beats the original with confidence >= 0.7" (`--min-confidence`), and
  candidates rank by win rate, then confidence, then linter distance. `deslop lab
  calibrate-judge` measures each provider on 100 held-out-vs-LLM pairs and stores the
  accuracy in the voice; a provider under 80% is not used as a judge and the gate is off when
  none qualifies. The 1-to-5 rubric stays as a secondary trend score. `--judge-min` is gone.
- Length control: `apply --max-growth PCT` (15) rejects candidates that outgrow their
  paragraph; the report gives document growth.
- `lab blind --export DIR` writes a shuffled blind set and its key for a session run by hand.
- Rewrite mode no longer skips paragraphs the absolute judge liked; every paragraph is
  sampled and the original wins unless a candidate beats it.

## v2.0.0 - 2026-11-01

Breaking release: two primary verbs, zip voices, calibrated thresholds, and a
generate-then-verify apply loop. No compatibility with v1 voices or commands.

- CLI: `deslop voice FILES... -o NAME.voice` builds a voice straight from files or
  directories (authors from `--author-map`, frontmatter, a `manifest.jsonl`, or the
  filename); `--from OLD.voice --add FILES` extends one. `deslop apply` replaces `convert`
  and `draft` (`--brief` selects drafting). `corpus init/add/stats`, `calibrate`, `convert`,
  `draft`, `voice update` and `voice export` are gone; `blind`, `export` and the new
  `calibrate-thresholds` live under `deslop lab`.
- Library: `Voice.from_texts`, `Voice.load`, `Voice.save`, `apply(text_or_brief, voice)`
  returning `Result`, and `score(text, voice)` returning `Report`. `Corpus` is internal.
- Voice file: `NAME.voice` is a zip and the canonical form; every command accepts it, the
  directory is only the unpacked working copy, `voice edit` unpacks and repacks. `voice.yaml`
  carries provenance (document and word counts, authors with shares and sources, per-document
  title/author/words/source/license, license notes, checksum). Exemplar headings carry the
  source URL; `voice pack --strip-exemplars` makes a shareable voice and `apply` says when it
  wrote without exemplars. `metrics.json` carries a 95% document-bootstrap interval per metric
  and `voice show` warns when an interval is wider than its band, with the words needed.
- Thresholds: the fixed 1.5σ rule is replaced by per-metric bands and per-tell limits tuned so
  that at least 95% of the 267 LLM-reference documents fail and at least 90% of held-out
  corpus documents (20% by author) pass; bands use within-author spread so between-author
  differences do not widen them. The composite distance is rescaled so held-out documents sit
  below 20 and LLM-reference prose centres on 70, and reports carry the separation stats. The
  LLM reference baseline now stores per-document stats (never text) for this.
- Content lock: a document-scoped allowlist. Removals are violations; additions pass when
  the document already has them or they cite the primary text; everything else new goes to the
  semantic lock. Sentence-initial capitals are proper nouns only when the document names them
  elsewhere (spaCy NER plus document frequency).
- Semantic lock: each unit's propositions are extracted and a different provider checks every
  one survives and none is added (reader address and primary-text citations exempt); a failure
  names the proposition. `--diff` prints a meaning section with ✓/✗ per proposition.
- Judge: a rubric-scored LLM judge (voice, model prose, fidelity; 1 to 5) from a provider other
  than the writer, stored in the report; `--judge-min` (4) gates rewrite mode; `score --judge`.
- Apply loop: per paragraph (or a two-to-three paragraph section when T14/T15 need room), `--n`
  candidates at temperature 0.9 across providers in rotation, verified by both locks, the
  linter in document context and the judge, ranked by distance then judge; one retry naming
  the failures, else the original is kept and recorded; a document-level pass for T14, T15 and
  aphoristic closes. Edit mode keeps its mechanism under the new locks.
- Tests: `tests/fixtures/genesis06_lesson.md` with both v1 conversions as regression
  references; the e2e suite asserts the v2 output scores lower and judges higher than both.

## v1.0.2 - 2026-10-31

- `profile.md` now opens with a facts block rendered from `metrics.json`: every metric in
  words with its value, median, and percentile against the generic and LLM baselines, then
  the tell rates. Calibration clusters documents on their metric vectors (k-means, k in 2-4 by
  silhouette) into registers, summarizes each by top authors, median length and distinguishing
  metrics, and the model writes the description under the instruction that every claim agree
  with the facts and that the registers be named. `voice show --clusters` lists the registers.
- Exemplars are drawn from every register in proportion to its share of the corpus; each
  exemplar heading names its register.
- `voice pack` and `voice unpack`: a zip of the voice directory with a sha256 checksum in
  `voice.yaml`, verified on unpack. Every `--voice` accepts the `.zip`, extracted read-only to
  a temporary directory.
- `corpus add DIR SOURCE --from-manifest` bulk-adds every row of `SOURCE/manifest.jsonl`
  (`file`, `author`, `title`, optional `date`/`year`, `weight`, `license`); near duplicates are
  skipped and counted instead of aborting.

## v1.0.1 - 2026-10-31

- `blind --min-overlap` defaults to 0.03 instead of 0.15. Content-word Jaccard between a
  candidate paragraph and its best corpus match runs 0.03-0.06 on a small corpus, so the old
  default skipped every real pair.

## v1.0.0 - 2026-10-31

Hardening release.

- `deslop doctor`: checks the spaCy model, provider reachability, the age of `pricing.yaml`
  (warns past 90 days), cache directory writability, and that the default voice resolves and
  validates. Exits nonzero on any failure.
- Default voice resolution is explicit and documented: `--voice`, then `DESLOP_VOICE`, then
  `default_voice` in the config file, then an error listing that order. The top-level
  `voice:` config key from v0.3.0 is replaced by `default_voice`.
- `blind --min-overlap` (default 0.15) skips pairs whose lexical overlap is below the
  threshold; each pair's overlap is printed with the prompt, and the skip and unpaired counts
  are reported at the end.
- Providers gain a cache-only mode (`cache_only=True`, or `DESLOP_CACHE_ONLY=1`) that needs no
  credentials and raises `CacheMissError` for any prompt not already cached;
  `DESLOP_CACHE_DIR` overrides the cache location. The end-to-end tests replay from the
  committed `tests/cache/` and no longer skip without keys.
- Worked example `examples/convert_example.py` builds a voice from the fixture corpus, converts
  `examples/slop_sample.md`, and prints the before/after report from cache.
- Packaging: full project metadata, MIT license file, `py.typed`, wheel builds with `uv build`
  and installs with pipx; the spaCy model is a documented post-install step.
- `mypy --strict` passes on `src/`; a GitHub Actions workflow runs ruff, mypy, pytest with an
  85 percent coverage gate on Python 3.12 and 3.13, the worked example, the docs build, and the
  wheel build.
- Documentation: README rewritten with a quickstart that runs on the shipped fixture corpus,
  the full command reference, the voice directory format, the metrics and tells catalog, and
  exit codes; mkdocs-material site with a page per concept, an examples page, and the API
  reference rendered from docstrings.
- Fixed: `deslop draft` crashed when `--max-iter` was omitted; a missing input file now
  exits 1 with a message instead of a traceback; a DRM-protected epub is refused before pandoc
  is looked for; the A/B headings in `blind` were printed as raw markup;
  `ConvertResult.changed` compared the text with a report object.

## v0.3.0 - 2026-10-31

Phase 3: drafting, voice lifecycle, blind test, configuration file.

- `deslop draft <brief.yaml>`: writes a piece section by section, one claim per section,
  required references and quotations verbatim; unlisted citations and quotations are stripped
  before the draft is converted in rewrite mode.
- `voice update --accept`: folds accepted texts into the corpus at weight 0.5, recalibrates
  `metrics.json` and `lexicon.json`, snapshots the previous voice to `history/<timestamp>/`
  (addressable as `@prev` or `@<timestamp>`). Voices record their corpus path.
- `voice diff`, `voice validate` (schema, corpus hash, exemplar self-score), `voice export`
  (`openai-ft` or `jsonl`; accepted pairs plus summarized corpus chunks; token count and
  training-cost estimate; never launches a job).
- `deslop blind`: pairs candidate paragraphs with corpus paragraphs of similar length and
  highest lexical overlap, shuffles, prompts for A/B and a reason, logs to
  `~/.deslop/blind/<voice>.jsonl`, and reports session and cumulative accuracy.
- Every option can come from `deslop.yaml` (or `~/.deslop/config.yaml`, or `--config`), with
  flags overriding a per-command section, which overrides top-level keys.

## v0.2.1 - 2026-10-31

- T15 `monotone_run`: three or more consecutive sentences within five words of each other,
  with reference rates measured for the LLM reference and the fixture voice.
- Edit mode merges adjacent flagged sentences into one passage the model may split or merge;
  the length band applies to the passage total, and all text outside a unit stays
  byte-identical. Rhythm-only flags target every similar-length run.
- Rejected attempts are fed back into the retry prompt so a cached identical prompt cannot
  return the same rejected answer.
- Prices move to `pricing.yaml` (packaged, with `~/.deslop/pricing.yaml` overriding) populated
  from the providers' published pricing pages, including OpenAI fine-tuning training rates;
  calls to unlisted models are counted as unpriced.

## v0.2.0 - 2026-10-31

Phase 2: conversion.

- `deslop convert`: edit mode (sentences) and rewrite mode (paragraphs), alternating providers
  per iteration, content lock on citations, quotations, numbers, and proper nouns,
  `--preserve-length`, context-repeat and new-tell rejection, `--max-iter` and `--budget`,
  best-iteration selection, unified diff, pairs log at `~/.deslop/pairs/<voice>.jsonl`.
- Anthropic and OpenAI providers with a disk cache keyed on the full prompt, exponential
  backoff, and fail-fast on authentication errors; an OpenAI-compatible endpoint provider.
- Live end-to-end tests that convert the slop sample to PASS.

## v0.1.0 - 2026-10-31

Phase 1: corpus, metrics, calibration, scoring.

- `deslop corpus init|add|stats`: ingestion of text, markdown, HTML, epub, PDF and URLs with
  normalization and simhash de-duplication; author shares and the `--max-share` cap.
- Twenty-five rhythm and anchor metrics and fourteen tells (T1 through T14) detected by regex,
  dependency pattern, or document-level function; catalog in `slop_patterns.yaml`.
- Baselines: `english_generic.json` from seventeen Gutenberg books and `llm_reference.json`
  from 200,000 words of machine prose.
- `deslop calibrate`: the six-file `name.voice/` directory with metrics, lexicon, rules,
  profile (model-written or templated), and exemplars.
- `deslop score`: per-metric and per-tell thresholds, distance 0 to 100, PASS/FAIL verdict,
  fixes, `--annotate`, `--json`, exit code 2 on FAIL.
