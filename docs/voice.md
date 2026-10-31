# Voice

A voice is what calibration measures from a corpus. It lives in a `name.voice/` directory of
six hand-editable files:

| file | contents |
| --- | --- |
| `voice.yaml` | name, version, corpus path and hash, author shares |
| `metrics.json` | mean, standard deviation and percentiles of every metric over 500-word chunks; tell rates |
| `lexicon.json` | words and bigrams over-represented against generic English; per-tell rates against the LLM reference; banned tells |
| `rules.md` | the default rules plus rules the numbers justify |
| `profile.md` | a facts block from the metrics, the registers found by clustering, then a model-written description that must agree with the facts (templated with `--no-llm`) |
| `exemplars.md` | 1,500 to 3,000 words of verbatim excerpts drawn from every register |

```sh
deslop calibrate essays.corpus --out essays.voice
deslop voice show essays.voice
deslop voice edit essays.voice        # opens profile, rules and exemplars in $EDITOR
```

Only `profile.md` needs a model; everything else is deterministic, so `--no-llm` gives a
reproducible voice with a templated profile.

Calibration clusters the documents on their metric vectors (k-means, k from 2 to 4 chosen by
silhouette) and calls the clusters registers. `voice show --clusters` lists them. The facts
block at the top of `profile.md` is the check on the prose: every claim below it has to be
consistent with the numbers above it.

```sh
deslop voice pack essays.voice --out essays.voice.zip   # checksum written into voice.yaml
deslop score draft.md --voice essays.voice.zip          # any --voice accepts the zip, read-only
deslop voice unpack essays.voice.zip --out essays.voice # verifies the checksum
```

## Baselines

Two baselines ship with the package. `english_generic.json` is measured from seventeen
Project Gutenberg books and gives the lexicon something to be over-represented against.
`llm_reference.json` is measured from 200,000 words of machine-written prose and gives every
tell a reference rate; a tell the voice uses less than a tenth as often as the reference is
banned outright.

## Lifecycle

```sh
deslop voice update essays.voice --accept piece.md     # fold accepted text in at weight 0.5
deslop voice diff essays.voice@prev essays.voice       # metric, rule and lexicon deltas
deslop voice validate essays.voice                     # schema, corpus hash, exemplar self-score
deslop voice export essays.voice --format openai-ft --out train.jsonl
```

`update` snapshots the voice to `history/<timestamp>/` first; `@prev` and `@<timestamp>`
address those snapshots. Only `metrics.json` and `lexicon.json` are rewritten unless you pass
`--regenerate`. `export` writes fine-tuning pairs and summarized corpus chunks and prints a
token count and cost estimate; it never launches a job.

## From Python

```python
from deslop import Corpus, Voice

voice = Voice.calibrate(Corpus.load("essays.corpus"), use_llm=False)
voice.save("essays.voice")
voice = Voice.load("essays.voice")
print(voice.name, voice.meta.total_words, voice.lexicon.banned_ids())
```
