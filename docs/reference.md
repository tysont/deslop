# Command reference

Every command reads the configuration file (see [Configuration](#configuration)) and takes
`--help`. Commands that report take `--json` for machine-readable output. Defaults are in
the tables.

## `deslop voice FILES... -o NAME.voice`

Measure texts into a voice. Files may be text, markdown, HTML, epub, PDF or docx;
directories are walked. A directory holding a `manifest.jsonl` (one JSON object per line
with `file`, `author`, `title`, and optional `url`, `license`, `date`) is read through the
manifest, which is how a crawler hands over provenance. Near-duplicate documents are
dropped, and the voice keeps only provenance, never the corpus itself.

| flag | default | meaning |
| --- | --- | --- |
| `-o, --out` | required | the `NAME.voice` file to write; its name is the voice's name |
| `--author-map` | none | yaml of filename glob to author; `default` names the author of every other file |
| `--max-share` | `0.15` | cap on any one author's share at calibration |
| `--exemplar-words` | `2500` | size of `exemplars.md` |
| `--min-docs` | `10` | below this many documents the voice is a small corpus and its bands shrink toward generic English ([voice.md](voice.md#small-corpora)) |
| `--no-llm` | off | write a template `profile.md` instead of asking a model |
| `--llm-model` | provider default | the model that writes `profile.md` |
| `--provider` | config, else `anthropic` | the provider for `profile.md` |
| `--profile-from` | none | reuse another voice's `profile.md` instead of writing one |
| `--from OLD.voice --add FILES` | | fold new texts into an existing voice: measurements, registers and thresholds are recomputed, the prose files kept |
| `--patterns` | none | an extra tell catalog yaml, repeatable |

Each file's author comes from the author map, else `author:` in its YAML frontmatter, else
the first underscore-separated part of its filename. At the end of calibration the
thresholds are tuned ([voice.md](voice.md#thresholds)) and the command prints the separation
it achieved.

### `deslop voice show VOICE [--clusters]`

Provenance (documents, words, authors with shares and sources, licenses), the registers
found by clustering (`--clusters`), and every metric with its mean, 95% bootstrap interval,
within-author spread and calibrated band. A metric whose interval is wider than its band is
flagged, with an estimate of the words needed to narrow it. Then the tell rates with their
limits, the separation achieved, and the rules.

### `deslop voice diff A B`

Metric deltas with their direction, tell and ban changes, lexicon and author-share changes,
and a diff of `rules.md`.

### `deslop voice validate VOICE`

Checks the schema and checksum, that the thresholds are calibrated and meet their targets,
and that the voice's own exemplars score PASS. Exits 1 if any check fails.

### `deslop voice edit VOICE`

Unpacks the voice to a temporary directory, opens `profile.md`, `rules.md` and
`exemplars.md` in `$VISUAL` or `$EDITOR`, and repacks it when the editor closes.

### `deslop voice add-tells VOICE CORPUS... -o OUT.voice`

For a tell added to the catalog after the voice was built: measures it on the voice's own
corpus and adds only that tell's rate, its starting limit and its lexicon comparison with
the LLM reference. Offline. Every other measurement, threshold and prose file is unchanged.
The new tell's limit is the loose starting limit a fresh build would begin from; run
`deslop lab calibrate-thresholds` to tighten it.

### `deslop voice pack DIR [-o NAME.voice] [--strip-exemplars]` and `deslop voice unpack NAME.voice [-o DIR]`

Convert between the zip and the unpacked directory. Every command accepts either form.
`--strip-exemplars` writes a shareable voice without the corpus excerpts; `apply` then
works from `profile.md` and `rules.md` alone and says so.

## `deslop score [FILE]`

Score FILE (or stdin) against the voice. Exits 0 on PASS, 2 on FAIL.

| flag | default | meaning |
| --- | --- | --- |
| `-v, --voice` | resolved | the voice ([Default voice](#default-voice)) |
| `--annotate PATH` | none | write the input with inline `[Tn]` markers at every tell |
| `--judge` | off | also ask a model for an absolute 1-to-5 rubric score (one call); a trend figure, never a gate |
| `--judge-model` | `claude-sonnet-5-5` | the model for `--judge` |
| `--provider` | config, else `anthropic` | `anthropic` or `claude-cli` |
| `--patterns` | none | an extra tell catalog yaml |
| `-q, --quiet` | off | print only the verdict line |

## `deslop apply [FILE] -o OUT.md`

Write FILE (or stdin) in the voice. Exits 0 when the result scores PASS, 2 when it scores
FAIL. [apply.md](apply.md) describes the loop.

| flag | default | meaning |
| --- | --- | --- |
| `-v, --voice` | resolved | the voice |
| `-o, --out` | stdout | where to write the result |
| `--brief BRIEF.yaml` | none | draft from a brief instead of rewriting a file |
| `--mode` | `rewrite` | `rewrite` the document as a whole, or `edit` only the flagged sentences |
| `--writer` | `claude-opus-5-5` | the model that writes candidates; `opus`, `sonnet` and `haiku` are aliases |
| `--judge` | `claude-sonnet-5-5` | the model that judges and runs the semantic lock; must differ from the writer |
| `--provider` | config, else `anthropic` | `anthropic` (the API) or `claude-cli` (the Claude Code CLI) |
| `--n` | `3` | candidates per document, or per section of a long one |
| `--chunk-words` | `2500` | documents longer than this are rewritten section by section |
| `--max-growth` | `15` | reject a candidate more than this percent longer than the original |
| `--fidelity` | `content` | `content` protects facts, names, numbers, quotations and causal and evaluative claims, and permits changes of person, framing, emphasis and order, and inferences that follow from stated facts; `strict` fails any new claim |
| `--report-all` | off | judge and report every candidate in full, pass or fail |
| `--diff` | off | print the unified diff and the meaning diff |
| `--compare PAGE.html` | none | write a self-contained comparison page (reference, input, candidate, with a sentence-aligned diff and every check pinned to its paragraph) |
| `--reference FILE` | none | the human text the input was made from: never shown to a model, but every candidate is checked for recitation of it, and it fills the comparison page's first column |
| `--max-recitation` | `2` | reject a candidate when more than this percent of its words are recited from the reference ([apply.md](apply.md#recitation)) |
| `--primary-text` | the voice's | a primary text that added quotations and citations must verify against: a text file with an optional `NAME.locs.json` sidecar, or a directory of them; repeatable ([primary-texts.md](primary-texts.md)) |
| `--citation-scheme` | none | JSON `{"name", "pattern", "bare_pattern"}`: how the command-line primary texts are cited; repeatable |
| `--citations` | `verified` | `verified`: added citations and quotations must verify against the primary texts; `keep`: every citation and quotation in the input must reappear exactly, each in the sentence that makes the same point; `none`: as `keep`, and nothing may be added |
| `--quote-context-words` | `1500` | cap on the verbatim primary-text passages shown to the writer |
| `--allow-primary-facts` | off | a fact the input lacks is not an addition when its names and numbers all occur at a cited location of a primary text |
| `--preceding FILE` | none | the already-converted section before the input: the writer continues it, the seam is checked, and the judge asks whether the same writer is continuing ([apply.md](apply.md#sections-of-a-longer-work)) |
| `--following FILE` | none | the unconverted section after the input: shown to the writer as context only |
| `--budget` | `5` | USD this invocation may spend on live calls; when it runs out, rewrite mode returns the original and edit mode the best text so far |
| `--max-calls` | `200` | live calls this invocation may make; caps subscription runs, which have no dollar cost |
| `--max-iter` | `4` | edit-mode iterations |
| `--patterns` | none | an extra tell catalog yaml |

## `deslop report RESULT.json --html -o PAGE.html [--reference FILE]`

Render a saved `apply --json` result as the comparison page: the human reference, the
input and the selected candidate side by side (a selector picks any candidate when the
result was made with `--report-all`). Changed, moved and new sentences are marked, with a
word-level diff on hover, and every lock, meaning, coherence and judge finding is pinned to
its paragraph. One HTML file with nothing external.

## `deslop lab`

Tools for calibrating voices, building test fixtures and running experiments.

### `deslop lab calibrate-thresholds VOICE [-o OUT.voice]`

Retune the voice's bands and limits ([voice.md](voice.md#thresholds)) and print the
separation. `deslop voice` runs this itself; run it again after editing a voice. Offline.

### `deslop lab calibrate-judge VOICE --corpus DIR --llm-texts DIR`

Measure each judge model on pairs of one LLM-reference paragraph and one paragraph by a
held-out author of the voice: accuracy is how often it picks the corpus paragraph as the
writer's. The accuracy is stored in the voice, keyed by model, and printed in every `apply`
report header; a judge below 80% is marked unreliable there. `--llm-texts` is a directory of
`.md` or `.txt` essays, or of `.json` files with a `text` key as
`scripts/build_llm_reference.py` caches them.

| flag | default | meaning |
| --- | --- | --- |
| `--pairs` | `100` | pairs per judge |
| `--judges` | `sonnet,opus` | judge models to measure |
| `--provider`, `--max-calls`, `--budget` | | as for `apply` |
| `--seed` | `0` | pair sampling seed |
| `-o, --out` | the input voice | where to write the voice |

### `deslop lab recitation RESULT.json --reference FILE`

Re-score every candidate of a saved `apply --json` result for recitation of a reference,
offline: the runs of eight or more words it shares with the reference but not with the
input, the share of its words they cover, the recited words outside the input's vocabulary
(the novel share, which the limit applies to), and the longest three runs.

### `deslop lab fixture-check ORIGINAL.md --voice VOICE [--draft]`

Can a text serve as a held-out test fixture? Its propositions are extracted and one draft
is written from them alone in the voice (the text itself is never shown; two live calls,
cached), then the draft is checked for recitation of the text. The text qualifies only when
the recited words outside the propositions' vocabulary are under 1% of the draft. Exits 2
when it doesn't. A text the writer model can recite from memory is no test of writing in a
voice.

### `deslop lab slop ORIGINAL.md -o SLOP.md [--kind "a personal essay"]`

Write model prose from a text's propositions alone: the propositions are extracted, then a
piece of the same length is written from them (two live calls, cached; the text is never
shown). This turns a held-out human piece into an `apply` input that carries its content and
none of its prose, with the piece itself as `--reference`. [personal-voice.md](personal-voice.md)
walks through it.

### `deslop lab blind --candidates DIR [--texts DIR] [--n 5] [--export DIR]`

Can you tell a draft from the real thing? Pairs paragraphs from your drafts with human
paragraphs (from `--texts`, else the voice's exemplars) matched on length and lexical
overlap (`--min-overlap`, default 0.03), shows each pair in random order, and asks which is
machine-written. Answers are appended to `~/.deslop/blind/<voice>.jsonl`. `--export DIR`
writes the shuffled pairs to `DIR/pairs.md` and the answers to `DIR/key.json` for a session
run by hand. Offline.

### `deslop lab export VOICE -o FILE [--format openai-ft|jsonl] [--corpus DIR]`

Write fine-tuning data from accepted `apply` pairs and summarized corpus chunks
(`--no-pairs`, `--no-chunks` and `--max-chunks` choose), with a token count and a cost
estimate at `--cost-model`'s training price. It never launches a job.

## `deslop doctor` and `deslop providers test`

`doctor` checks the spaCy model, each configured provider, the age of the pricing table,
that the cache is writable, and the default voice, and prints live spend today and over the
last seven days. Exits 1 on any failure. `providers test` pings each configured provider.

## Exit codes

| code | meaning |
| --- | --- |
| 0 | success; for `score` and `apply`, the text scores PASS |
| 1 | error: bad arguments or input, no voice found, unreadable files, a provider failure, or a failed `voice validate` or `doctor` check |
| 2 | `score` or `apply` finished and the text scores FAIL; `lab fixture-check` found the text unfit |

## Default voice

`apply`, `score` and `lab blind` look for their voice in this order:

1. `--voice` on the command line.
2. The `DESLOP_VOICE` environment variable.
3. `default_voice` in the configuration file, in the command's section first, then at the
   top level.

If none is set, the command exits 1 and lists this order.

## Configuration

Settings are read from `deslop.yaml` (or `deslop.yml`, or `.deslop.yaml`) in the working
directory, else `$DESLOP_HOME/config.yaml`, or the file named with `--config`. Top-level
keys apply to every command; a key named after a command holds that command's settings;
flags win over both. Keys may use hyphens or underscores, and lists may be yaml lists or
comma-separated strings.

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
| `DESLOP_HOME` | home for the cache, spend log, pairs, blind-test logs, `config.yaml` and `pricing.yaml` (default `~/.deslop`) |
| `DESLOP_VOICE` | the default voice |
| `DESLOP_PROVIDER` | `anthropic` (default) or `claude-cli`; the config file's `provider:` does the same |
| `ANTHROPIC_API_KEY` | credentials for the `anthropic` provider |
| `DESLOP_CACHE_DIR` | the response cache (default `$DESLOP_HOME/cache`) |
| `DESLOP_CACHE_ONLY` | `1` answers from the cache only and fails on a miss; no credentials needed |
| `DESLOP_RECORD` | `1` lets the end-to-end tests make live calls for uncached prompts and record them |
| `OPENAI_API_KEY` | credentials for the `openai` provider, used only when named |
| `DESLOP_COMPATIBLE_BASE_URL`, `DESLOP_COMPATIBLE_API_KEY`, `DESLOP_COMPATIBLE_MODEL` | an OpenAI-compatible endpoint, used as provider `compatible` |

A `.env` file in the working directory is loaded and overrides the shell.

## Spend, cost and pricing

Every live provider call is appended to `$DESLOP_HOME/spend.jsonl` (timestamp, command,
provider, model, tokens in and out, cost, voice, input file). Replays from the cache are
not spend. `apply` and every `lab` command stop at `--budget` dollars (default $5 per
invocation) or `--max-calls` live calls (default 200). Calls through `claude-cli` are
logged as subscription calls with no dollar cost, so `--max-calls` is their cap.

Costs come from `src/deslop/pricing.yaml`, which records USD per million tokens per model,
fine-tuning training rates, and the pricing pages they were taken from with the fetch date.
To add a model or change a price, create `$DESLOP_HOME/pricing.yaml` with the same layout;
its entries merge over the packaged table. Calls to a model in neither file are counted as
`unpriced_calls` and add nothing to the reported cost. `deslop doctor` warns when the
packaged table is more than 90 days old.
