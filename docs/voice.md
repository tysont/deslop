# Voice

A voice is what `deslop voice` measures from texts. It is a `NAME.voice` zip of six files
with a sha256 checksum in `voice.yaml`; `voice unpack` gives the directory form for editing
and `voice pack` zips it again. Every command accepts either.

| file | contents |
| --- | --- |
| `voice.yaml` | name, version, created, document and word counts, authors with shares and sources, licenses, every document's provenance, checksum, default primary texts, judge calibration per provider |
| `metrics.json` | mean, spread, within-author spread and bootstrap interval of every metric; tell rates; per-document samples; registers; calibrated thresholds |
| `lexicon.json` | over-represented words and bigrams; per-tell rates against the LLM reference; banned tells |
| `rules.md` | the default rules plus rules the numbers justify |
| `profile.md` | a facts block from the metrics, the registers, then a model-written description that must agree with the facts |
| `exemplars.md` | verbatim excerpts from every register, each with author, title, register and source |

```sh
deslop voice essays/ -o essays.voice --author-map authors.yaml
deslop voice show essays.voice --clusters
deslop voice edit essays.voice            # unpacks, opens profile/rules/exemplars, repacks
deslop voice --from essays.voice --add new/*.md -o essays2.voice
deslop voice pack essays.voice.d -o shared.voice --strip-exemplars
```

## Authors and provenance

Each file's author is resolved from the author map (filename glob to author, `default` for
the rest), then `author:` in YAML frontmatter, then the filename's first underscore-separated
part. A directory with a `manifest.jsonl` (as the crawler in the voices repo writes) supplies
author, title, URL and license per row. The voice keeps every document's title, author,
words, source and license, the authors' shares and the sites they came from, and the distinct
license notes, so a shared voice says where it came from.

## Registers and the facts block

Calibration clusters the documents on their metric vectors (k-means, k from 2 to 4 chosen by
silhouette) and calls the clusters registers. Exemplars are drawn from every register in
proportion to its share. The facts block at the top of `profile.md` is the check on the prose:
every claim below it has to be consistent with the numbers above it, and the model is told to
name the registers.

## Confidence intervals

Every metric's mean carries a 95% interval from 200 bootstrap resamples at document level.
`voice show` prints the interval beside the calibrated band and warns when the interval is
wider than the band's half-width, with an estimate of how many more words would narrow it to
fit (the width shrinks with the square root of the corpus size).

## Thresholds

See the README's Thresholds section for the calibration method: within-author spread, the
held-out split by author, the greedy tightening against the LLM-reference documents, and the
rescaled distance. `voice validate` reports the separation the voice achieves.
