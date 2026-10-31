# Corpus

A corpus is a directory of plain-text documents plus a `corpus.yaml` manifest. Each document
records its author, title, year, weight, license, word count, and a simhash fingerprint used
to reject near-duplicates.

```sh
deslop corpus init essays.corpus --name essays
deslop corpus add essays.corpus hazlitt.txt --author "William Hazlitt" --title "Characters" --year 1817
deslop corpus add essays.corpus https://example.org/essay.html --author "A. Writer"
deslop corpus stats essays.corpus
```

Sources may be text, markdown, HTML, epub, PDF, or a URL. Text is normalized on the way in:
hard-wrapped lines are joined, quotes and dashes are standardized, and front matter is dropped.

## Author balance

Calibration measures the corpus as a whole, so one prolific author would otherwise dominate.
`stats` shows each author's share of words; authors above `--max-share` (default 25 percent)
are flagged and downweighted when the voice is calibrated. A document's `weight` scales its
contribution; texts accepted back into the corpus by `voice update` carry weight 0.5.

## From Python

```python
from deslop import Corpus

corpus = Corpus.init("essays.corpus")
corpus.add("hazlitt.txt", author="William Hazlitt", title="Characters", year=1817)
print(corpus.stats().authors)
```

The shipped fixture corpus under `tests/fixtures/corpus_small` holds four public-domain
essays on Shakespeare (Johnson, Hazlitt, Bradley, Arnold) and is enough to try every command.
