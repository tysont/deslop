# deslop

deslop measures a writer's voice from a corpus of their prose, scores any text against that
voice, and rewrites only the spans that deviate while leaving the rest byte-identical. Its first
job is removing the structural and lexical tells of machine-written text, but a voice can be
any writer or register you have enough text for. It is a Python library with a CLI on top, and
every model call it makes is cached, priced, and logged.

Documentation: this README is the command reference; the concept pages and the API reference
are built with `make docs` from `docs/`.

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

The spaCy model `en_core_web_sm` is a GitHub release wheel, which package indexes cannot
reference as a dependency, so it is a second install step; `deslop doctor` tells you if it is
missing. pandoc is needed only to ingest epub and docx.

Conversion and drafting need a model. Put `ANTHROPIC_API_KEY` and/or `OPENAI_API_KEY` in the
environment or a `.env` file in the working directory; everything else works offline.

## Quickstart

This runs on the four public-domain essays shipped in the repository, so you need no corpus of
your own. From a checkout:

```sh
# 1. A corpus: four essays on Shakespeare by Johnson, Hazlitt, Bradley and Arnold.
deslop corpus init critics.corpus
deslop corpus add critics.corpus tests/fixtures/corpus_small/johnson_preface_to_shakespeare.txt \
  --author "Samuel Johnson" --title "Preface to Shakespeare" --year 1765
deslop corpus add critics.corpus tests/fixtures/corpus_small/hazlitt_characters_of_shakespeare_s_plays.txt \
  --author "William Hazlitt" --title "Characters of Shakespeare's Plays" --year 1817
deslop corpus add critics.corpus tests/fixtures/corpus_small/bradley_shakespearean_tragedy_lecture_i.txt \
  --author "A. C. Bradley" --title "Shakespearean Tragedy, Lecture I" --year 1904
deslop corpus add critics.corpus tests/fixtures/corpus_small/arnold_the_function_of_criticism_at_the_present_time.txt \
  --author "Matthew Arnold" --title "The Function of Criticism at the Present Time" --year 1865
deslop corpus stats critics.corpus

# 2. A voice. --no-llm keeps this step offline (the profile is templated instead of written).
deslop calibrate critics.corpus --out critics.voice --no-llm
deslop voice show critics.voice
export DESLOP_VOICE=critics.voice

# 3. Score a machine-written sample: FAIL, with every flagged metric and tell located.
deslop score examples/slop_sample.md
deslop score examples/slop_sample.md --annotate annotated.md

# 4. Convert it (needs a key): only the flagged sentences are rewritten.
deslop convert examples/slop_sample.md --out converted.md --diff

# 5. Draft a new piece from a brief (needs a key), then check it.
deslop draft tests/fixtures/brief_sample.yaml --out piece.md
deslop score piece.md

# 6. Can you tell the sample from the critics? Overlap on this small corpus is low, so lower the floor.
deslop blind --candidates examples --n 3 --min-overlap 0.03
```

Without a key, `make example` runs the conversion step from the cache committed under
`tests/cache` and prints the before and after reports.

## Command reference

Every option can also be set in a config file (see Configuration); a flag always wins. Paths
to voices are directories ending in `.voice`. `--json` on any command prints the full result
as JSON on stdout and nothing else.

Global options: `--config PATH` names the settings file; `--version` prints the version.

### `deslop corpus init DIR [--name NAME]`

Create a corpus directory with an empty manifest. `--name` defaults to the directory name.
Loading an existing corpus is a no-op.

### `deslop corpus add DIR SOURCE... --author AUTHOR [options]`

Ingest one or more files or URLs (text, markdown, HTML, epub, PDF) into the corpus.

| flag | default | meaning |
| --- | --- | --- |
| `--author` | required | author name, used for share accounting |
| `--title` | from the source | document title (only with a single source) |
| `--year` | none | publication year |
| `--weight` | `1.0` | share multiplier at calibration |
| `--license` | empty | free-text license note |

A near-duplicate of an existing document is refused with exit 1.

### `deslop corpus stats DIR [--max-share 0.25]`

Word counts and shares per author, and which authors exceed `--max-share` and will be
downweighted at calibration.

### `deslop calibrate CORPUS --out NAME.voice [options]`

Measure the corpus into a voice directory.

| flag | default | meaning |
| --- | --- | --- |
| `--out` | required | voice directory to write |
| `--name` | the `--out` name | voice name |
| `--no-llm` | off | template `profile.md` instead of asking a model |
| `--llm-model` | provider default | model that writes the profile |
| `--max-share` | `0.25` | author share cap |
| `--exemplar-words` | `2000` | size of `exemplars.md` |
| `--patterns` | none | extra tell catalog yaml, repeatable |

### `deslop score [FILE] [options]`

Score a file (or stdin) against a voice. Prints the verdict, distance, flagged metrics and
tells with their thresholds, the hits by paragraph, and one-line fixes.

| flag | default | meaning |
| --- | --- | --- |
| `--voice`, `-v` | resolved (see Default voice) | voice directory |
| `--sigma` | `1.5` | standard deviations from the voice mean before a metric is flagged |
| `--tell-mult` | `2.0` | multiple of the voice's own rate before a tell is flagged |
| `--tell-floor` | `1.0` | minimum rate per 1k words before a tell is flagged |
| `--patterns` | none | extra tell catalog yaml, repeatable |
| `--annotate PATH` | none | write the input with inline `[Tn]` markers |
| `--quiet`, `-q` | off | print only the verdict line |

Exit 0 on PASS, 2 on FAIL.

### `deslop convert [FILE] [options]`

Rewrite the flagged spans with a model until the text passes.

| flag | default | meaning |
| --- | --- | --- |
| `--voice`, `-v` | resolved | voice directory |
| `--mode` | `edit` | `edit` rewrites flagged sentences (adjacent ones as one passage); `rewrite` redoes flagged paragraphs |
| `--max-iter` | `4` | score/rewrite rounds |
| `--models` | `claude,gpt,claude,gpt` | providers to alternate per iteration (`claude`, `gpt`, `compatible`) |
| `--budget` | none | stop after this many USD and return the best iteration |
| `--lock` | `citations,quotes,numbers,proper_nouns` | content classes a rewrite may not change; `none` to disable |
| `--preserve-length` | `15` | max percent a unit's word count may change |
| `--sigma`, `--tell-mult`, `--tell-floor`, `--patterns` | as for score | scoring options |
| `--out`, `-o` | stdout | write the converted text here |
| `--diff` | off | print the unified diff instead of the text |

Prints the verdict, iterations, distance before and after, rewrites accepted, cost, and why
the loop stopped (`pass`, `max_iter`, `budget`, `nothing_flagged`, `no_progress`). Every
attempt is appended to `$DESLOP_HOME/pairs/<voice>.jsonl`. Exit 0 on PASS, 2 on FAIL.

### `deslop draft BRIEF.yaml [options]`

Write a piece from a brief (see `tests/fixtures/brief_sample.yaml`: a title, optional audience,
`words_per_section`, and sections with `heading`, `claim`, `refs`, `quotes`, `notes`). Each
section makes one claim and uses its references and quotations verbatim; anything unlisted is
stripped; the result is converted in rewrite mode.

| flag | default | meaning |
| --- | --- | --- |
| `--voice`, `-v` | resolved | voice directory |
| `--out`, `-o` | stdout | write the finished piece here |
| `--max-iter` | `2` | conversion rounds after drafting |
| `--models`, `--budget`, `--lock`, `--preserve-length` | as for convert | conversion options |
| `--sigma`, `--tell-mult`, `--tell-floor`, `--patterns` | as for score | scoring options |

Prints sections drafted, items stripped, incomplete sections, the conversion summary, and total
cost. Exit 0 on PASS, 2 on FAIL.

### `deslop voice show VOICE [--json]`

Print the metadata, the metric table, the tell rates against the LLM reference, and the rules.

### `deslop voice edit VOICE`

Open `profile.md`, `rules.md` and `exemplars.md` in `$VISUAL` or `$EDITOR` (default `vi`).

### `deslop voice update VOICE --accept FILE... [options]`

Fold accepted texts into the voice. The previous voice is snapshotted to
`VOICE/history/<timestamp>/` first; `metrics.json` and `lexicon.json` are recalculated;
`profile.md`, `rules.md` and `exemplars.md` are untouched unless `--regenerate`.

| flag | default | meaning |
| --- | --- | --- |
| `--accept` | required | accepted text file, repeatable |
| `--corpus` | recorded in the voice | corpus directory to add the texts to |
| `--weight` | `0.5` | weight of the accepted documents |
| `--regenerate` | off | also rewrite profile, rules and exemplars |
| `--no-llm` | off | template the profile when regenerating |

### `deslop voice diff A B [--json]`

Metric deltas with their direction (toward or away from slop), rule and lexicon changes, and
author-share changes between two voices. Either side may be `VOICE@prev` or
`VOICE@<timestamp>` to address a snapshot.

### `deslop voice validate VOICE [--corpus DIR] [--json]`

Check the schema, that the corpus hash still matches, and that the voice's own exemplars score
PASS. Exit 1 if any check fails.

### `deslop voice export VOICE --out FILE [options]`

Write fine-tuning data: accepted convert pairs (system = `profile.md`, user = before,
assistant = after) and 300 to 600 word corpus chunks with a model-written summary as the user
turn. Prints the token count and a training-cost estimate from `pricing.yaml`. Never launches
a job.

| flag | default | meaning |
| --- | --- | --- |
| `--out`, `-o` | required | JSONL file to write |
| `--format` | `openai-ft` | `openai-ft` (messages) or `jsonl` (system/input/output/kind/source) |
| `--corpus` | recorded | corpus to chunk |
| `--no-pairs` | off | skip convert pairs |
| `--no-chunks` | off | skip corpus chunks |
| `--max-chunks` | all | cap on chunks |
| `--cost-model` | `gpt-4.1` | model whose training rate prices the estimate |

### `deslop blind --candidates DIR [options]`

Pair each candidate paragraph with the unused corpus paragraph of similar length and highest
content-word overlap, shuffle, and ask which is machine-written, with a one-line reason.
Records go to `$DESLOP_HOME/blind/<voice>.jsonl`.

| flag | default | meaning |
| --- | --- | --- |
| `--voice`, `-v` | resolved | voice directory |
| `--candidates` | required | directory of `.md` or `.txt` drafts |
| `--n` | `5` | pairs to judge |
| `--min-overlap` | `0.15` | skip pairs whose Jaccard overlap is below this |
| `--corpus` | recorded | corpus to draw controls from |
| `--seed` | random | make the sampling reproducible |

Prints each pair's overlap, right or wrong after each answer, then session and cumulative
accuracy, the most frequent reasons, pairs skipped, and candidates that could not be paired.

### `deslop doctor [--json]`

Check the spaCy model, that each configured provider answers, the age of the pricing table
(warns past 90 days), that the cache directory is writable, and that the default voice resolves
and validates. Exit 1 if any check fails; warnings do not change the exit code.

### `deslop providers test [--json]`

Send a one-word prompt to each configured provider and report the model that answered.

## Default voice

`score`, `convert`, `draft` and `blind` resolve their voice in this order, and the first hit
wins:

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
sigma: 1.5
convert:
  mode: rewrite
  max-iter: 3
  models: [claude, gpt]
  budget: 2.0
blind:
  min-overlap: 0.03
```

## Environment

| variable | meaning |
| --- | --- |
| `DESLOP_HOME` | base directory for cache, pairs, blind logs, `config.yaml` and `pricing.yaml` (default `~/.deslop`) |
| `DESLOP_VOICE` | default voice directory |
| `DESLOP_CACHE_DIR` | response cache directory (default `$DESLOP_HOME/cache`) |
| `DESLOP_CACHE_ONLY` | `1` makes providers answer from the cache only and fail on a miss; no credentials needed |
| `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` | provider credentials |
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

`name.voice/` holds six hand-editable files:

| file | contents |
| --- | --- |
| `voice.yaml` | name, schema and deslop version, created date, corpus name, path and content hash, total words, author shares, profile source |
| `metrics.json` | for every metric: mean, standard deviation, p10, p50, p90 over 500-word chunks; tell rates per 1k words; per-author samples |
| `lexicon.json` | words and bigrams over-represented against generic English; each tell's rate in the voice and in the LLM reference, the ratio, and whether it is banned |
| `rules.md` | the default rules plus rules the numbers justify (for example a ban on a tell the voice never uses) |
| `profile.md` | a positive description of the voice, written by a model or templated with `--no-llm` |
| `exemplars.md` | 1,500 to 3,000 words of verbatim excerpts chosen to cover the authors |

`voice update` adds `history/<timestamp>/` snapshots of all six files.

## Metrics

Scored in chunks of the size the voice was measured in and compared with the voice's mean and
standard deviation. "Slop side" is the direction in which machine prose tends to deviate; a
metric is only ever flagged on that side, and metrics with side `none` are reported but never
flagged.

| metric | group | slop side | definition |
| --- | --- | --- | --- |
| `citation_per_100w` | anchors | low | Citations (verse, Stephanus, page/line, author-year) per 100 words |
| `quote_per_para` | anchors | low | Quoted spans of four or more words per paragraph |
| `concrete_per_100w` | anchors | low | Numbers, proper nouns, non-Latin script, italic terms per 100 words |
| `reader_address_per_1k` | anchors | none | Second-person pronouns and reader imperatives per 1k words |
| `first_person_per_1k` | anchors | low | First-person singular pronouns per 1k words |
| `committed_claim_per_para` | anchors | low | Evaluative predicates asserting a reading right or wrong, per paragraph |
| `sent_len_mean` | rhythm | none | Mean words per sentence |
| `sent_len_std` | rhythm | low | Standard deviation of words per sentence |
| `sent_len_cv` | rhythm | low | Coefficient of variation of sentence length |
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
| `opener_top_word_share` | rhythm | high | Percent of sentences that begin with the single most common first word |
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
| 0 | success; for `score`, `convert` and `draft`, the text scores PASS |
| 1 | error: bad arguments or input, no voice resolvable, duplicate document, provider failure, or a failed `voice validate` or `doctor` check |
| 2 | `score`, `convert` or `draft` finished and the text scores FAIL |

## Python API

```python
from deslop import Corpus, Voice, convert, draft, load_brief, score

corpus = Corpus.init("critics.corpus")
corpus.add("essay.txt", author="A. Writer", title="Essay", year=1911)
voice = Voice.calibrate(corpus, use_llm=False)
voice.save("critics.voice")

report = score(open("draft.md").read(), Voice.load("critics.voice"))
print(report.verdict, round(report.distance), report.fixes)

result = convert(open("draft.md").read(), Voice.load("critics.voice"), mode="edit")
print(result.report_after.verdict, result.cost.usd)
print(result.diff)

piece = draft(load_brief("brief.yaml"), Voice.load("critics.voice"))
print(piece.report.verdict, piece.cost.usd)
print(piece.text)
```

The docstrings on these names are the API reference; `make docs` renders them with
mkdocs-material along with a page per concept and an examples page.

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

The end-to-end tests under `tests/e2e` call real providers and record every response under
`tests/cache`, which is committed. Without credentials they replay from that cache and fail
loudly on any uncached prompt; with credentials, uncached prompts go live. CI runs on Python
3.12 and 3.13 and uses repository secrets for the keys when present.
