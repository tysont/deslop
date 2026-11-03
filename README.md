# deslop

deslop learns a writer's voice from their texts, tells you where a piece of prose falls
outside it, and rewrites text into it without changing what the text says.

Its first use is removing the tells of machine-written prose: the "not X but Y" frames, the
chapter that "asks" and the passage that "insists", the tidy triads, the aphoristic closes,
the runs of same-length sentences. But a voice can be any writer or register you have
enough text for: your own essays, a house style, a period, a genre.

```text
$ deslop score examples/slop_sample.md --voice critics.voice --quiet      # model-written
FAIL against voice critics  distance 100/100  582 words, 45 sentences, 8 paragraphs

$ deslop score tests/fixtures/heldout_hazlitt.md --voice critics.voice --quiet   # Hazlitt
PASS against voice critics  distance 9/100  1454 words, 42 sentences, 5 paragraphs
```

(`critics.voice` is built in the [Quickstart](#quickstart) from four 18th- and 19th-century
essays on Shakespeare; the Hazlitt passage comes from another of his works, not in the
corpus.)

## How it works

| command | what it does |
| --- | --- |
| `deslop voice` | Measures a corpus into a `NAME.voice` file: 26 rhythm and anchoring metrics, 16 machine-prose tells, a lexicon, exemplar passages, a written profile, and thresholds calibrated so that the corpus's own held-out documents pass and LLM-written reference essays fail. |
| `deslop score` | Lints a text against a voice. It flags each metric outside its band and each tell over its limit, with the exact spans. It exits 0 on PASS and 2 on FAIL, so it works as a CI check. Offline. |
| `deslop apply` | Rewrites a text, or drafts one from a brief, in the voice. Several candidates are written and each must pass a content lock, a semantic lock, coherence checks and a comparative judge. If none passes, the input comes back unchanged with the reasons. |

`score` is deterministic and needs no model. `voice` uses a model once, to write the voice's
profile (`--no-llm` skips it). `apply` uses two different Claude models, one to write and
one to judge, so no model checks its own work.

## Install

deslop needs Python 3.12 or later and the spaCy English model. Install it from GitHub. The
`deslop` package on PyPI is a different project.

```sh
pipx install git+https://github.com/tysont/deslop.git
pipx inject deslop https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
deslop doctor
```

With uv:

```sh
uv tool install git+https://github.com/tysont/deslop.git \
  --with https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
deslop doctor
```

`deslop doctor` checks the spaCy model, the providers, the cache and the default voice, and
prints what you have spent on model calls today and over the last seven days.

`apply`, and `voice` without `--no-llm`, need a Claude provider. There are two:

- **`anthropic`** (the default): the Messages API. Set `ANTHROPIC_API_KEY`.
- **`claude-cli`**: the Claude Code CLI with your local login. Set `provider: claude-cli` in
  the config file, set `DESLOP_PROVIDER=claude-cli`, or pass `--provider claude-cli`.

## Quickstart

The repository ships four public-domain essays on Shakespeare (Johnson, Hazlitt, Bradley and
Arnold) to try deslop on.

```sh
git clone https://github.com/tysont/deslop.git && cd deslop

# 1. Build a voice from the essays. --no-llm writes a template profile instead of asking a
#    model, so this step is offline.
deslop voice tests/fixtures/corpus_small -o critics.voice \
  --author-map examples/authors.yaml --no-llm
deslop voice show critics.voice --clusters
export DESLOP_VOICE=critics.voice

# 2. Score a machine-written essay against it: FAIL, with every flagged metric and tell
#    located. --annotate writes a copy with inline [Tn] markers.
deslop score examples/slop_sample.md
deslop score examples/slop_sample.md --annotate annotated.md

# 3. Rewrite it in the voice (live model calls). --diff prints what changed and which
#    propositions survived; --compare writes a side-by-side HTML page.
deslop apply examples/slop_sample.md -o applied.md --diff --compare compare.html

# 4. Draft a new piece from a brief, then check it.
deslop apply --brief tests/fixtures/brief_sample.yaml -o piece.md
deslop score piece.md
```

Without credentials, `make example` runs a worked example (an edit-mode rewrite of the same
essay) from the responses cached in `tests/cache`.

To build a voice from your own writing and test it fairly, follow
[docs/personal-voice.md](docs/personal-voice.md).

## What `apply` will and won't do

`apply` keeps every claim the input makes and adds nothing material. Its limits follow from
that.

- **It can't restore what the input lacks.** If an earlier rewrite stripped out the
  questions to the reader, the images or the asides, they are gone. The output is in the
  voice but carries only the content the input still had.
- **It declines rather than returning a partial rewrite.** If no candidate passes every
  check, you get the input back unchanged, with a report on why each candidate failed.
- **Its ceiling is the voice.** The writer imitates the voice's profile, rules and
  exemplars, and the judge compares against those exemplars. The output gets no closer to
  the writer than the exemplars are. You choose and edit those files (`deslop voice edit`),
  and a voice built from thirty documents sets a higher ceiling than one built from three.

`apply` run on a human original in its own voice should keep the original, or make only
small changes. If it "improves" a human original, look closely: the judge is preferring
something to the writer's own prose.

## Commands

| command | purpose |
| --- | --- |
| `deslop voice FILES... -o NAME.voice` | build a voice from text, markdown, HTML, epub, PDF or docx files |
| `deslop voice show \| diff \| validate \| edit` | inspect, compare, check or hand-edit a voice |
| `deslop voice add-tells \| pack \| unpack` | maintain a voice file |
| `deslop score [FILE]` | lint a text against a voice |
| `deslop apply [FILE] -o OUT.md` | rewrite a text in a voice (`--brief` drafts one, `--mode edit` touches only flagged sentences) |
| `deslop report RESULT.json --html -o PAGE.html` | render a saved `apply --json` result as a comparison page |
| `deslop lab ...` | calibration, recitation checks, test-fixture tools, blind tests, fine-tuning export |
| `deslop doctor`, `deslop providers test` | check the installation and the providers |

Every command takes `--help`, and those that report take `--json`. The full flag reference is in
[docs/reference.md](docs/reference.md).

## Configuration

Flags can be set in `deslop.yaml` in the working directory, or in `~/.deslop/config.yaml`.
Top-level keys apply to every command, and a section named after a command applies to that
command only. A flag on the command line overrides both.

```yaml
default_voice: voices/critics.voice
provider: claude-cli
apply:
  n: 3
  budget: 5.0
```

A command looks for its voice in `--voice`, then `$DESLOP_VOICE`, then `default_voice` in the
config file. [docs/reference.md](docs/reference.md#configuration) covers every key and
environment variable.

## Cost and caching

Every model response is cached under `~/.deslop/cache`, keyed on the full prompt, so a
repeated run is free and reproducible. Every live call is logged to `~/.deslop/spend.jsonl`.
`apply` and the `lab` commands stop at `--budget` dollars (default $5) or `--max-calls` live
calls (default 200, which also caps subscription runs through `claude-cli`). Expect 50 to 80
calls to rewrite a thousand-word piece with `--n 3`.

## Python API

```python
from deslop import Voice, apply, load_brief, score

voice = Voice.from_texts(["essays/"], name="critics", author_map={"default": "A. Writer"})
voice.save("critics.voice")

voice = Voice.load("critics.voice")
report = score(open("draft.md").read(), voice)
print(report.verdict, report.distance, report.fixes)

result = apply(open("draft.md").read(), voice, n=3)
print(result.text)
print(result.meaning_diff)

piece = apply(load_brief("brief.yaml"), voice)
```

`Voice`, `score`, `apply`, `load_brief` and the `Report`, `Result` and `Brief` types are the
public API. [docs/api.md](docs/api.md) documents them.

## Documentation

- [Voice](docs/voice.md): what a voice file holds, and how it is measured and calibrated
- [Score](docs/score.md): the metrics, the tells and the distance
- [Apply](docs/apply.md): the rewrite loop, the locks, the judge, and sections of longer works
- [Primary texts](docs/primary-texts.md): verifying added quotations and citations against a source
- [A voice from your own writing](docs/personal-voice.md): a walkthrough
- [Command reference](docs/reference.md)

`make docs` builds the site with mkdocs.

## Development

```sh
make install     # uv sync --all-groups, including the spaCy model
make check       # ruff, mypy --strict, and pytest with an 85% coverage gate
make test        # pytest only
make example     # the worked example, replayed from tests/cache
make docs        # mkdocs build --strict
make build       # wheel and sdist into dist/
make clean       # remove build outputs and caches
```

The tests never call a model. The end-to-end tests under `tests/e2e` replay responses
recorded in `tests/cache`, and a prompt that isn't cached skips its test. To record new
responses, run with `DESLOP_RECORD=1` and a key: misses go live and are written to the cache.
CI replays with `DESLOP_CACHE_ONLY=1` and no keys.

```text
src/deslop/
  cli/            the typer app: app.py (score, apply, report, doctor), voice.py, lab.py
  voice/          building, calibrating, packing and editing a voice
  metrics/        the rhythm and anchoring metrics and the tell detectors
  baselines/      the tell catalog, generic-English and LLM-reference statistics, default rules
  providers/      Anthropic API, Claude Code CLI and OpenAI clients behind one cached interface
  apply.py        the rewrite and edit loops
  score.py        scoring a text against a voice
  lock.py, semantic.py, coherence.py, continuity.py, judge.py, recitation.py   the checks
scripts/          rebuild the baselines and test fixtures, and run the register benchmark
tests/            unit tests, end-to-end replays, fixtures and the response cache
```

The scripts under `scripts/` rebuild the shipped data: `make llm-reference` (live calls),
`make english-generic`, and `make register-fixtures`. `make registers` runs `apply` across
the four register fixtures and writes comparison pages to `reports/`. It refuses to start
without `CONFIRM=1` and a `MAX_COST` above its projected cost.

## License

MIT. The corpus fixtures are public-domain texts from Project Gutenberg.
