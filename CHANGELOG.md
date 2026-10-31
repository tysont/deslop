# Changelog

All notable changes to deslop. Dates are release dates; versions are git tags.

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
