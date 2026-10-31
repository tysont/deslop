# deslop

Voice calibration, scoring, and conversion for prose. deslop compiles a corpus
of human writing into a reusable **voice**, scores any text against that voice
(how far it deviates and exactly where), and in later phases rewrites only the
deviating spans while locking content.

The primary use is removing the structural and lexical tells of LLM-generated
prose, but a voice is any measured writer or register.

## Install

```sh
make install          # uv sync; installs the spaCy en_core_web_sm model too
```

pandoc is required to ingest epub/docx; pypdf handles PDF.

## Quick start

```sh
deslop corpus init my.corpus
deslop corpus add my.corpus essays.txt --author "A. Writer" --title "Essays" --year 1911
deslop corpus stats my.corpus
deslop calibrate my.corpus --out writer.voice --no-llm
deslop score draft.md --voice writer.voice            # exit 0 PASS, 2 FAIL
deslop score draft.md --voice writer.voice --json
deslop score draft.md --voice writer.voice --annotate draft.annotated.md
deslop voice show writer.voice
deslop convert draft.md --voice writer.voice --out draft.voiced.md --diff
```

`convert` rewrites only the flagged sentences (`--mode edit`) or paragraphs
(`--mode rewrite`), alternating providers per iteration (`--models
claude,gpt,claude,gpt`), and rejects any rewrite that changes a locked
citation, quotation, number, or proper noun, drifts outside `--preserve-length`,
repeats its context, or introduces a new tell. It stops on PASS, `--max-iter`,
or `--budget`, returns the best iteration by distance, and appends every
(before, after, tells, accepted) pair to `~/.deslop/pairs/<voice>.jsonl`.
Responses are cached under `~/.deslop/cache`, so a rerun with the same inputs
is free and reproducible.

`calibrate` without `--no-llm` asks a configured provider (ANTHROPIC_API_KEY,
OPENAI_API_KEY, or DESLOP_COMPATIBLE_BASE_URL in the environment or `.env`) to
write `profile.md`; everything else is deterministic.

## Cost and pricing

Every model call is cached under `~/.deslop/cache`, so rerunning the same conversion is free.
Cost estimates come from `src/deslop/pricing.yaml`, which records USD per million tokens
for each model and the pricing pages it was taken from. To add a model or change a price,
create `~/.deslop/pricing.yaml` with the same `models:` layout; its entries are merged over
the packaged table. Calls to models missing from both files are counted as `unpriced_calls`
and contribute nothing to the reported cost.

## What a voice is

`name.voice/` holds six hand-editable files:

| file | contents |
| --- | --- |
| `voice.yaml` | name, version, corpus hash, author shares |
| `metrics.json` | mean/std/p10/p50/p90 for every metric over 500-word chunks, tell rates, per-author samples |
| `lexicon.json` | words and bigrams over-represented vs generic English; per-pattern rates vs the LLM reference; banned patterns |
| `rules.md` | default rules plus rules justified by the numbers |
| `profile.md` | positive description of the voice |
| `exemplars.md` | 1,500-3,000 words of verbatim excerpts |

## Python API

```python
from deslop import Corpus, Voice, score

corpus = Corpus.load("my.corpus")
voice = Voice.calibrate(corpus, use_llm=False)
voice.save("writer.voice")
report = score(open("draft.md").read(), Voice.load("writer.voice"))
print(report.verdict, report.fixes)
print(report.to_json())

from deslop import convert
result = convert(open("draft.md").read(), Voice.load("writer.voice"), mode="edit")
print(result.report_after.verdict, result.cost.usd)
print(result.diff)
```

## Development

```sh
make test             # pytest (tests/e2e needs provider keys; skipped otherwise)
make cov              # coverage report
make lint             # ruff
make english-generic  # rebuild baselines/english_generic.json from Gutenberg
make llm-reference    # rebuild baselines/llm_reference.json (needs API keys)
uv run python scripts/build_fixtures.py .cache/gutenberg   # rebuild test corpus
```

The pattern catalog lives in `src/deslop/baselines/slop_patterns.yaml`; regex
patterns can be added there or via `--patterns extra.yaml` with no code change.
