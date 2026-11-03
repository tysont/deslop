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
part. A directory with a `manifest.jsonl` (as a crawler might write) supplies
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

Each voice carries thresholds tuned against its own documents and against 267 LLM-written
reference essays, so that a text cannot pass by avoiding a single tell.

- **Metric bands.** For every metric with a slop side the band is `mean ± k · s`, where `s`
  is the pooled standard deviation of document means within one author (so the spread
  between authors does not widen the band) and `k` is chosen per metric. Lower bounds clip
  at 0, since every metric is a rate or a count.
- **Tell limits.** Every tell has a rate limit per 1,000 words, `max(voice_rate · m,
  floor)`, with `m` and the floor chosen per tell.

Calibration holds out about 20% of documents by author and starts loose. It then tightens
one criterion at a time, each time taking the step that fails the most LLM-reference
documents per held-out document lost, until at least 95% of the LLM-reference documents
fail while at least 90% of the held-out documents pass. With a real held-out split it then
keeps tightening wherever a step costs no held-out document, down to `k = 2` and twice the
voice's tell rate. A corpus with fewer than five authors or ten documents uses all its
documents as the held-out set and skips that second phase.

The distance `score` reports is rescaled against the same calibration: the 90th-percentile
held-out document lands at 20, the median LLM-reference document at 70, and the scale
saturates toward 100 above that. `score --json` carries the separation
(`heldout_pass_rate`, `llm_fail_rate`, `heldout_p90`, `llm_median`) so the number can be
read, and `voice validate` reports whether the voice meets its targets.
`deslop lab calibrate-thresholds` reruns the calibration after a voice is edited.

## Small corpora

With fewer than `--min-docs` documents (default 10) a voice cannot estimate between-document
spread, so each metric is shrunk toward generic English with weight `w = documents /
min_docs`:

```
mean      = w * corpus_mean      + (1 - w) * generic_mean
within_sd = sqrt(w * paragraph_variance + (1 - w) * generic_sd^2)
sd        = sqrt(w * corpus_sd^2 + (1 - w) * generic_sd^2)
```

where `paragraph_variance` is the pooled within-document variance of the metric over
paragraphs; a metric generic English does not measure keeps the corpus mean with the
paragraph spread. The threshold search then requires every one of the voice's own documents to
pass (not 90%), and a tell the lexicon banned is relaxed floor by floor until they do.
`voice show` prints a small-corpus warning with the document count and roughly how many more
words of corpus would leave small-corpus mode.
