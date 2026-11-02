# deslop

deslop measures a writer's voice from their texts, scores any text against that voice, and
writes or rewrites text in it with every proposition kept. Its first job is removing the
structural and lexical tells of machine-written prose, but a voice can be any writer or
register you have enough text for. Two verbs do the work: `voice` builds a voice from files,
`apply` writes a text or a brief in it; `score` reports how far any text is from the voice.

## Install

With pipx:

```sh
pipx install deslop
pipx inject deslop https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
deslop doctor
```

With uv, in a project:

```sh
uv add deslop
uv add https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
uv run deslop doctor
```

The spaCy model is a GitHub release wheel, which package metadata cannot reference, so it is
a second step; `deslop doctor` points at it when it is missing. `apply` needs an Anthropic
provider: the API (`ANTHROPIC_API_KEY`, the default) or the Claude Code CLI (`claude`, logged
in; select it with `provider: claude-cli` in the config file or `DESLOP_PROVIDER=claude-cli`).
The writer and the judge are two different Claude models (claude-opus-5-5 writes,
claude-sonnet-5-5 judges and runs the semantic lock), so no model checks its own work.
OpenAI stays in the provider layer but is never selected unless named. `voice` needs a
provider only to write `profile.md`; `--no-llm` keeps it offline.

## Quickstart

```sh
# 1. A voice from four essays on Shakespeare (Johnson, Hazlitt, Bradley, Arnold), shipped as
#    fixtures. Authors come from the author map; frontmatter or filenames also work.
deslop voice tests/fixtures/corpus_small -o critics.voice --author-map examples/authors.yaml
deslop voice show critics.voice --clusters
export DESLOP_VOICE=critics.voice

# 2. Score a machine-written sample: FAIL, with every flagged metric and tell located, and
#    the distance read against the voice's own held-out documents and LLM-reference prose.
deslop score examples/slop_sample.md
deslop score examples/slop_sample.md --annotate annotated.md

# 3. Apply the voice (needs keys). Each paragraph gets three candidates, checked by the
#    content lock, the semantic lock, the linter and a judge; the diff shows what changed and
#    the meaning section shows which propositions survived.
deslop apply examples/slop_sample.md -o applied.md --diff

# 4. Draft a new piece from a brief, in the voice, then check it.
deslop apply --brief tests/fixtures/brief_sample.yaml -o piece.md
deslop score piece.md

# 5. Can you tell the sample from the critics?
deslop lab blind --candidates examples --n 3
```

Without a key, `make example` runs the apply step from the cache committed under
`tests/cache`.

## Command reference

Every command reads `deslop.yaml` (see Configuration file) and takes `--json` for machine
output. Where a flag has a default it is in the table.

### `deslop voice FILES-OR-DIRS... -o NAME.voice [options]`

Measure texts into a voice. Files may be text, markdown, HTML, epub, PDF or docx; directories
are walked. A directory holding a `manifest.jsonl` (one JSON object per line with `file`,
`author`, `title`, and optional `url`, `license`, `date`) is read through the manifest, which
is how a crawler hands over provenance. The corpus is built in a temporary directory, near
duplicates are dropped, and only provenance is kept in the voice.

| flag | default | meaning |
| --- | --- | --- |
| `-o, --out` | required | the `NAME.voice` zip to write; the name is the voice's name |
| `--author-map` | none | yaml of filename glob to author; `default` names the author for files nothing matches |
| `--max-share` | `0.15` | cap on any author's share at calibration |
| `--exemplar-words` | `2500` | size of `exemplars.md` |
| `--no-llm` | off | template `profile.md` instead of asking a model |
| `--llm-model` | provider default | model for `profile.md` |
| `--from OLD.voice --add FILES` | | fold new texts into an existing voice: measurements, registers and thresholds are recomputed, the prose files kept |
| `--patterns` | none | extra pattern catalog yaml, repeatable |

Author resolution per file: the author map, then `author:` in YAML frontmatter, then the
filename's first underscore-separated part. At the end of calibration the thresholds are tuned
(see Thresholds) and the command prints the separation it achieved.

### `deslop apply [FILE] -o OUT.md [options]`

Write FILE (or stdin) in the voice. Exit 0 when the result scores PASS, 2 when FAIL.

| flag | default | meaning |
| --- | --- | --- |
| `-v, --voice` | resolved | the voice (see Default voice) |
| `-o, --out` | stdout | where to write the result |
| `--brief BRIEF.yaml` | none | draft from a brief instead of rewriting a file |
| `--mode` | `rewrite` | `rewrite` the document as a whole, or `edit` only flagged sentences |
| `--writer` | `claude-opus-5-5` | the model that writes candidates; `opus`, `sonnet`, `haiku` are aliases |
| `--judge` | `claude-sonnet-5-5` | the model that judges and runs the semantic lock; must differ from the writer |
| `--provider` | config, else `anthropic` | `anthropic` (API) or `claude-cli` (Claude Code CLI, local login) |
| `--max-calls` | `200` | live calls this invocation may make, so subscription runs have a cap too |
| `--fidelity` | `content` | `content` protects facts, entities, numbers, quotations, causal and attributed evaluative claims and permits changes of person, discourse words, emphasis, figurative restatement, hedging and reordering; `strict` fails any new claim |
| `--report-all` | off | judge and report every candidate in full, pass or fail |
| `--compare PATH` | none | write a self-contained HTML comparison page (reference, input, candidate; sentence-aligned diff; checks pinned to paragraphs) |
| `--reference FILE` | none | the human text the input was made from: every candidate is checked for recitation of it (it is never shown to a model), and it fills the comparison page's first column |
| `--max-recitation` | `2` | reject a candidate when more than this percent of its words are recited: in runs of eight or more words shared with the reference and not with the input, and not among the input's words at all (the raw share is reported alongside) |
| `--n` | `3` | candidates sampled per document (or per section) in rewrite mode |
| `--chunk-words` | `2500` | longer documents are rewritten section by section |
| `--max-growth` | `15` | reject a candidate more than this percent longer than its paragraph |
| `--primary-text` | none | reference text (file or directory) added quotations and citations must verify against; repeatable |
| `--budget` | `5` | USD this invocation may spend on live calls; past it the run stops with the best text so far |
| `--max-iter` | `4` | edit-mode iterations |
| `--diff` | off | print the unified diff and the meaning diff |
| `--patterns` | none | extra pattern catalog yaml |

Rewrite mode writes the document as a whole: every paragraph's propositions are extracted
(cached); `--n` candidates are sampled at temperature 0.9 from the providers in rotation
with the profile, the rules, the exemplars nearest the piece, the full original and the
propositions by paragraph (each with a texture note: what the original paragraph also did,
such as its address, questions, images and exclamations, as a hint and never a lock), and
told to write the piece as this writer would, keep every proposition and add none, adopt the
writer's person and address, let paragraph length and count follow the exemplars rather than
the input, and keep the author's tense and order of argument unless the profile says
otherwise. Each candidate must pass the content lock (nothing locked removed), the length
limit (`--max-growth`), the primary-text check, the coherence checks (sentences per
paragraph inside the exemplars' range; the dominant tense unchanged; every sentence-initial
pronoun with an antecedent in the two sentences before it; connective openers in the voice's
band and near the original's share) and the semantic lock paragraph by paragraph (every
proposition kept, nothing material added: no new entity, number, event, unstated cause, or
scene, object or action the original does not present; judged by a different provider). Then a
provider that wrote neither text judges it against the original on four items, each
answered separately: reads as the voice, coherent, consistent tense and person, preserves
the argument's order and emphasis. A candidate must win or tie on every item; survivors are
judged round-robin and the best win rate wins. If none passes, one retry carries what a
reader noticed in the earlier drafts (the linter's finds with their spans, never a metric
name); if that fails too the result is the original with the report, never a partial
rewrite. A document longer than `--chunk-words` is rewritten section by section, each with
the already-rewritten section before it in context and the original section after it
visible.

Without a primary text, added quotations and citations are allowed and listed as
"unverified additions" at the top of the report and in `--diff`. deslop ships no reference
texts; you supply them with `--primary-text`. A reference text is a plain-text file with `# title:`, `# book:` and `# aliases:`
header lines and `{{Genesis 6:5}}` or `{{369b}}` markers in the text; a file without markers
still verifies quotations.

The judge gate is only used with a provider that has been measured: `deslop lab
calibrate-judge` stores each provider's accuracy at telling held-out corpus paragraphs
from LLM-reference paragraphs in the voice, a provider below 80% is never used as a judge,
and when no provider reaches 80% the gate is off and the report says so.

Edit mode rewrites only flagged sentences, with adjacent flagged sentences merged into one
passage, under the same two locks; everything outside a unit stays byte-identical.

A brief (`--brief`) is drafted section by section first (one claim per section, listed
references and quotations verbatim, anything unlisted stripped), then the draft goes through
rewrite mode.

### `deslop score [FILE] [options]`

| flag | default | meaning |
| --- | --- | --- |
| `-v, --voice` | resolved | the voice |
| `--annotate PATH` | none | write the input with inline `[Tn]` markers |
| `--judge` | off | also ask the absolute rubric judge (one call) and print its three scores, a trend figure, not a gate |
| `--judge-model` | `claude-sonnet-5-5` | model for `--judge` |
| `--provider` | config, else `anthropic` | `anthropic` or `claude-cli` |
| `-q, --quiet` | off | print only the verdict line |

Exit 0 on PASS, 2 on FAIL.

### `deslop voice show VOICE [--clusters] [--json]`

Provenance (documents, words, authors with shares and sources, licenses), the registers found
by clustering (`--clusters`), every metric with its mean, 95% bootstrap interval,
within-author spread and calibrated band, a warning for any metric whose interval is wider
than its band with the words needed to narrow it, the tell rates with their limits, the
separation achieved, and the rules.

### `deslop voice diff A B [--json]`

Metric deltas with their direction, tell and ban changes, lexicon and author-share changes,
and a diff of `rules.md`.

### `deslop voice validate VOICE [--json]`

Schema and checksum, that thresholds are calibrated and meet their targets, and that the
voice's own exemplars score PASS. Exit 1 if any check fails.

### `deslop voice pack DIR [-o NAME.voice] [--strip-exemplars]` and `deslop voice unpack NAME.voice [-o DIR]`

Plumbing between the zip and its unpacked working directory. `--strip-exemplars` writes a
shareable voice without the corpus excerpts; `apply` then works from `profile.md` and
`rules.md` alone and says so.

### `deslop voice edit VOICE`

Unpack to a temporary directory, open `profile.md`, `rules.md` and `exemplars.md` in
`$VISUAL` or `$EDITOR`, and repack when the editor closes.

### `deslop lab calibrate-thresholds VOICE [-o OUT.voice]`

Retune the bands and limits (see Thresholds) and print the separation. `voice` runs this
itself; the command exists to rerun it after editing a voice.

### `deslop lab calibrate-judge VOICE --corpus DIR --llm-texts DIR [--pairs 100] [--judges sonnet,opus] [--provider P] [-o OUT.voice]`

Measure each judge model as the comparative judge on pairs of one LLM-reference paragraph and
one paragraph from the voice's held-out authors (the same author split threshold calibration
uses), report the accuracy, and store it in the voice. `--llm-texts` is a directory of
`.md`/`.txt` essays or of `.json` files with a `text` key, as `scripts/build_llm_reference.py`
caches them.

### `deslop lab recitation RESULT.json --reference FILE [--json]`

Re-score every candidate of a saved `apply --json` result for recitation of a reference text,
offline: the runs of eight or more words (lowercased, punctuation stripped) it shares with the
reference and not with the input, their count, the share of the candidate's words they cover,
the recited words outside the input's vocabulary (the novel share, which the limit applies
to) and the longest three.

### `deslop lab fixture-check ORIGINAL.md --voice VOICE [--draft] [--json]`

Can a text serve as a held-out fixture? Its propositions are extracted, one draft is written
from them alone in the voice (the text itself is never shown; two live calls, cached), and the
draft is checked for recitation of the text. A text qualifies only when the recited words
outside the propositions' vocabulary are under 1% of the draft (the raw share is reported
alongside); exit 2 when it does not. `tests/fixtures/registers/README.md` states the rule for
the fixture set.

### `deslop lab blind --candidates DIR [--texts DIR] [--n 5] [--seed N] [--min-overlap 0.03] [--export DIR]`

Pair paragraphs from your drafts with human paragraphs (from `--texts`, else the voice's
exemplars) matched on length and lexical overlap, show each pair in random order, and ask
which is machine-written. Results append to `$DESLOP_HOME/blind/<voice>.jsonl`. With
`--export DIR` the shuffled pairs are written to `DIR/pairs.md` and the answers to
`DIR/key.json` for a session run by hand.

### `deslop lab export VOICE --out FILE [--format openai-ft|jsonl] [--corpus DIR] [--no-pairs] [--no-chunks]`

Fine-tuning data from accepted apply pairs and summarized corpus chunks, with a token count
and cost estimate. Never launches a job.

### `deslop report --html RESULT.json -o PAGE.html [--reference FILE]`

Render a saved `apply --json` result as the three-column comparison page: human reference
(undiffed), input, and the selected candidate (a selector when the result was made with
`--report-all`), with changed, moved and new sentences marked, a word-level diff on hover or
toggle, a read-clean toggle, and every lock, meaning, coherence and judge finding pinned to
the paragraph it concerns. One HTML file, nothing external.

### `deslop doctor [--json]` and `deslop providers test [--json]`

`doctor` checks the spaCy model, each configured provider, pricing age, cache writability and
the default voice, and prints live spend today and over the last seven days; exit 1 on any
failure. `providers test` pings each configured provider.

### The spend guard

Every live provider call is appended to `$DESLOP_HOME/spend.jsonl` (timestamp, command,
provider, model, tokens in and out, cost, voice, input file); cached replays are not spend.
`apply` and every `lab` command take `--budget` (default $5 per invocation) and stop with the
best result so far and a clear message when it is spent. A Makefile target that fans out over
several documents (`make registers`) prints its cost projection first and refuses to run
without `CONFIRM=1` and a `MAX_COST` it stays under.

## Default voice

`apply`, `score` and `lab blind` resolve their voice in this order, and the first hit wins:

1. `--voice` on the command line.
2. The `DESLOP_VOICE` environment variable.
3. `default_voice` in the config file, in the command's section first, then at top level.

If none is set the command exits 1 with a message listing this order.

## Configuration file

Every option can live in `deslop.yaml` (also `deslop.yml` or `.deslop.yaml`) in the working
directory, else `$DESLOP_HOME/config.yaml`, else the file named with `--config`. Top-level keys
apply to every command; a key named after a command holds that command's settings; flags win
over both. Keys may use hyphens or underscores; lists may be yaml lists or comma strings.

```yaml
default_voice: voices/critics.voice
provider: anthropic        # or claude-cli
apply:
  writer: claude-opus-5-5
  judge: claude-sonnet-5-5
  n: 3
  chunk-words: 2500
  max-growth: 15
  budget: 5.0
voice:
  max-share: 0.15
blind:
  min-overlap: 0.05
```

## Environment

| variable | meaning |
| --- | --- |
| `DESLOP_HOME` | base directory for cache, pairs, blind logs, `config.yaml` and `pricing.yaml` (default `~/.deslop`) |
| `DESLOP_VOICE` | default voice directory |
| `DESLOP_CACHE_DIR` | response cache directory (default `$DESLOP_HOME/cache`) |
| `DESLOP_CACHE_ONLY` | `1` makes providers answer from the cache only and fail on a miss; no credentials needed |
| `ANTHROPIC_API_KEY` | the Anthropic API key (provider `anthropic`) |
| `DESLOP_PROVIDER` | `anthropic` (default) or `claude-cli`; the config file's `provider:` key does the same |
| `OPENAI_API_KEY` | OpenAI credentials; the provider is never selected unless named |
| `DESLOP_COMPATIBLE_BASE_URL`, `DESLOP_COMPATIBLE_API_KEY`, `DESLOP_COMPATIBLE_MODEL` | an OpenAI-compatible endpoint, selected as provider `compatible` |

A `.env` file in the working directory is loaded and overrides the shell.

## Cost and pricing

Every model call is cached under `$DESLOP_HOME/cache` keyed on the full prompt, so repeating a
run is free and reproducible. Costs come from `src/deslop/pricing.yaml`, which records USD per
million tokens per model, fine-tuning training rates, and the pricing pages it was taken from
with the fetch date. To add a model or change a price, create `$DESLOP_HOME/pricing.yaml` with
the same layout; its entries merge over the packaged table. Calls to models in neither file are
counted as `unpriced_calls` and add nothing to the reported cost. `deslop doctor` warns when
the packaged table is more than 90 days old.

## What a voice is

`NAME.voice` is a zip of six files; `voice unpack` gives you the directory form to edit and
`voice pack` zips it again. Every command accepts either.

| file | contents |
| --- | --- |
| `voice.yaml` | name, deslop version, created date, document and word counts, the author list with shares and sources, licenses carried from ingest, every document's title, author, words, source and license, and the sha256 of the other five files |
| `metrics.json` | for every metric: mean, standard deviation, percentiles, within-author spread and a 95% bootstrap interval over documents; tell rates; per-author and per-document samples; the registers; the calibrated thresholds with their separation statistics |
| `lexicon.json` | words and bigrams over-represented against generic English; each tell's rate in the voice and in the LLM reference, the ratio, and whether it is banned |
| `rules.md` | the default rules plus rules the numbers justify |
| `profile.md` | a facts block rendered from `metrics.json` (every metric in words with its percentile against the generic and LLM baselines), the registers, then a description written by a model that must agree with the facts |
| `exemplars.md` | verbatim excerpts drawn from every register in proportion to its share, each headed with author, title, register and source URL when known |

## Thresholds

A fixed sigma rule produced false PASSes: a text could avoid one tell and sail through. Each
voice therefore carries tuned thresholds. For every metric with a slop side the band is
`mean ± k · s`, where `s` is the pooled standard deviation of document means within one
author (so the spread between authors does not widen the band) and `k` is chosen per metric;
every tell has a rate limit per 1,000 words. Calibration holds out about 20% of documents by
author, starts loose, then tightens one criterion at a time, taking the step that fails the
most LLM-reference documents per held-out document lost, until at least 95% of the 267
LLM-reference documents fail while at least 90% of the held-out documents pass; with a real
split it then keeps tightening wherever a step costs no held-out document, down to `k = 2`
and twice the voice's tell rate. Corpora with fewer than five authors or ten documents use all
their documents as the held-out set and skip that second phase.

The composite distance is rescaled against the same calibration: the 90th-percentile held-out
document lands at 20, the median LLM-reference document at 70, and the scale saturates toward
100 above that. `score --json` carries the separation (`heldout_pass_rate`, `llm_fail_rate`,
`heldout_p90`, `llm_median`) so the number can be read.

## Metrics

Scored in chunks of the size the voice was measured in and compared with the voice's mean and
standard deviation. "Slop side" is the direction in which machine prose tends to deviate; a
metric is flagged on that side only, a metric with side `both` is matched to the voice's band
in both directions (overshoot is a failure like undershoot), and metrics with side `none` are
reported but never flagged.

| metric | group | slop side | definition |
| --- | --- | --- | --- |
| `citation_per_100w` | anchors | both | Citations (verse, Stephanus, page/line, author-year) per 100 words |
| `quote_per_para` | anchors | both | Quoted spans of four or more words per paragraph |
| `concrete_per_100w` | anchors | both | Numbers, proper nouns, non-Latin script, italic terms per 100 words |
| `reader_address_per_1k` | anchors | both | Second-person pronouns and reader imperatives per 1k words |
| `first_person_per_1k` | anchors | low | First-person singular pronouns per 1k words |
| `committed_claim_per_para` | anchors | low | Evaluative predicates asserting a reading right or wrong, per paragraph |
| `sent_len_mean` | rhythm | none | Mean words per sentence |
| `sent_len_std` | rhythm | low | Standard deviation of words per sentence |
| `sent_len_cv` | rhythm | both | Coefficient of variation of sentence length |
| `pct_sent_lt8` | rhythm | none | Percent of sentences under 8 words |
| `pct_sent_gt30` | rhythm | none | Percent of sentences over 30 words |
| `max_run_similar_len` | rhythm | high | Longest run of consecutive sentences within 5 words of each other |
| `pct_compound_multi` | rhythm | none | Percent of sentences with two or more coordinating joins |
| `para_len_mean` | rhythm | none | Mean sentences per paragraph |
| `para_len_std` | rhythm | low | Standard deviation of sentences per paragraph |
| `opener_pct_det` | rhythm | none | Percent of sentences opening with a determiner |
| `opener_pct_pron` | rhythm | none | Percent of sentences opening with a pronoun |
| `opener_pct_adv_conj` | rhythm | high | Percent of sentences opening with an adverb or conjunction |
| `opener_pct_noun` | rhythm | none | Percent of sentences opening with a noun or proper noun |
| `opener_top_word_share` | rhythm | both | Percent of sentences that begin with the single most common first word |
| `connective_opener_pct` | rhythm | both | Percent of sentences opening with and, but, so, then, because, since, while or yet |
| `semicolons_per_1k` | rhythm | none | Semicolons per 1k words |
| `colons_per_1k` | rhythm | high | Colons per 1k words |
| `em_dashes_per_1k` | rhythm | high | Em dashes per 1k words |
| `parens_per_1k` | rhythm | none | Open parentheses per 1k words |
| `questions_per_1k` | rhythm | high | Question marks per 1k words |

## Tells

Counted per 1k words and flagged above `max(voice_rate * tell_mult, tell_floor)`, or at any
rate when the voice bans the tell (it uses the tell less than a tenth as often as the LLM
reference). The catalog is `src/deslop/baselines/slop_patterns.yaml`; regex patterns can be
added there or in a file passed with `--patterns`.

| id | name | definition |
| --- | --- | --- |
| T1 | `text_as_agent` | A text noun is the subject of an intentional verb (the chapter asks, the passage insists) |
| T2 | `negative_parallelism` | Contrastive framing by negation (not X; Y / not X but Y / X, not Y / rather than / less X than Y) |
| T3 | `balance_markers` | Words that assert a poised both-ness instead of taking a side |
| T4 | `triad` | Three parallel conjuncts in one sentence |
| T5 | `aphoristic_close` | Paragraph-final sentence with no anchor and an antithetical shape or high abstract-noun density |
| T6 | `invented_label` | Possessive abstract noun modifying another abstract noun (appetite's arithmetic) |
| T7 | `meta_commentary` | The predicate describes what the text or author is doing rather than asserting content |
| T8 | `epithet_substitution` | A definite descriptive phrase stands in for an already-introduced proper name (the Danish prince) |
| T9 | `unanswered_question` | A question the next two sentences do not answer |
| T10 | `formula_openers` | Stock sentence openers (So, Then, Against that, Set X beside Y, It is worth noting, Importantly) |
| T11 | `hedge_stack` | Two or more hedges in one sentence |
| T12 | `intensifier_cluster` | Sincerity intensifiers (deeply, truly, fundamentally, inherently, genuinely) |
| T13 | `colon_reveal` | A colon followed by a single abstract noun phrase as the reveal |
| T14 | `tidy_summary_close` | Final paragraph restates earlier paragraphs with no new anchors |
| T15 | `monotone_run` | Three or more consecutive sentences whose word counts all fall within five of each other |

## Exit codes

| code | meaning |
| --- | --- |
| 0 | success; for `score` and `apply`, the text scores PASS |
| 1 | error: bad arguments or input, no voice resolvable, unreadable files, provider failure, or a failed `voice validate` or `doctor` check |
| 2 | `score` or `apply` finished and the text scores FAIL |

## Python API

```python
from deslop import Voice, apply, load_brief, score

voice = Voice.from_texts(["essays/"], name="critics", author_map={"default": "A. Writer"})
voice.save("critics.voice")

voice = Voice.load("critics.voice")
report = score(open("draft.md").read(), voice)
print(report.verdict, report.distance, report.separation.llm_median, report.fixes)

result = apply(open("draft.md").read(), voice, n=3, primary_texts=["genesis.txt"])
print(result.report_after.verdict, result.cost.usd, result.growth_pct)
print(result.unverified_additions, result.meaning_diff)

piece = apply(load_brief("brief.yaml"), voice)
print(piece.text)
```

`Voice.from_texts`, `Voice.load`, `Voice.save`, `apply` and `score` are the public API;
`Report`, `Result` and `Brief` are their types. `make docs` renders their docstrings with
mkdocs-material along with a page per concept.

## Development

```sh
make install          # uv sync --all-groups (includes the spaCy model)
make check            # ruff, mypy --strict, pytest with 85% coverage gate
make test             # pytest only
make example          # the worked example, from cache
make docs             # mkdocs build --strict
make build            # uv build -> dist/
make llm-reference    # rebuild baselines/llm_reference.json (needs keys)
make english-generic  # rebuild baselines/english_generic.json from Gutenberg
```

End-to-end tests replay responses recorded under `tests/cache`; a prompt that is not cached
skips. To record, run them with `DESLOP_RECORD=1` and a key: the misses go live, are written
to the cache, and are logged to the spend file. CI replays without keys (`DESLOP_CACHE_ONLY=1`).
