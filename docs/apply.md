# Apply

`apply` writes a text in a voice. The default, rewrite mode, is generate-then-verify: the
document is written afresh as a whole several times and each candidate has to prove itself.

1. Every paragraph's propositions are extracted as a numbered list of short claims (cached),
   and apart from them a texture note: what the paragraph does beyond asserting (grammatical
   person and address, rhetorical questions, images or figures, exclamations, direct appeals
   to the reader), each quoted briefly. The note is a hint to the writer only: it is never
   locked, required or treated as a proposition, and the semantic lock never sees it.
2. `--n` candidates (default 3) are sampled at temperature 0.9 from the configured providers in
   rotation. The writer sees the profile, the rules, the exemplars nearest the piece, the
   full original and the propositions by paragraph (each with "the original paragraph also
   did these things; use them where this writer would"), and is told: write this piece as
   this writer would; keep every proposition and add none; adopt this writer's person and
   address; paragraph length and count follow the exemplars, not the input, so merge or split
   the input's paragraphs freely; keep the author's tense and order of argument unless the
   voice's profile says otherwise. Nothing in the prompt mentions metrics, sentence length,
   openers or citations.
3. Each candidate is verified without a judge: the content lock (nothing locked removed), the
   length limit (no more than `--max-growth` percent, default 15, longer than the original),
   the primary-text check (below), and the coherence checks: the output's mean sentences per
   paragraph inside the exemplars' range (their 10th to 90th percentile; without a range the
   paragraph count must match the original's); the dominant finite-verb tense of each
   paragraph unchanged (paired by position, or against the original's dominant tense as a
   whole when the output is re-paragraphed); every sentence-initial pronoun
   (and this, that, it) with a noun phrase in the two sentences before it; and the share of
   sentences opening with a connective (and, but, so, then, because, since, while, yet)
   inside the voice's band and within half of the original's share.
4. The semantic lock runs paragraph by paragraph when the paragraphing is kept, else once
   over the section: a different provider checks that every proposition is kept and nothing
   material is added. A material addition is a new named entity, number, event, a causal
   claim whose cause the original neither states nor implies, or a scene, object or action
   the original does not present (cattle finding nothing to graze, when the original says
   only that the fields are now sea at high tide); restating a stated fact in other words is
   permitted.
5. A provider that wrote neither text judges the candidate against the original on four
   items, each answered separately: reads as the voice; coherent (each sentence follows
   from the last); consistent tense and person; preserves the argument's order and emphasis
   (paragraph breaks are not part of it: the same content in the same order, merged or split
   into different paragraphs, is the same order).
   The candidate must win or tie on every item. Survivors are judged round-robin on the same
   items, giving a win rate; the best win rate, then the linter distance, wins.
6. If no candidate passes, one retry carries what a reader noticed in the earlier drafts:
   the missing name, the dropped claim, the paragraph that changed tense, the linter's finds
   with their spans ("a reader noticed ..."), never a metric name. If that fails too, the
   result is the original with the report. A partial rewrite is never produced: with a long
   document, a section that fails reverts the sections already rewritten.

A document longer than `--chunk-words` (default 2500) is rewritten section by section, in
order, each section with the already-rewritten section before it in context and the original
section after it visible, and the full original always shown.

The report carries the meaning diff (propositions by paragraph with ✓ or ✗, added quotations
and citations with ✓ verified or ? unverified, unverified additions at the top), the
coherence results, the item-by-item judge verdict against the original, a table of every
metric before and after against the voice's band (under, in or over), each candidate's win
rate, the document's growth and the cost.

## Fidelity

YAML frontmatter is stripped on ingest and put back on output. Markdown structure is not
content: headings, rules, list markers and emphasis are never locked, so a list may be
rendered as prose and prose as a list; a word the document also writes in lower case, or one
capitalised only at a sentence or item start, is not a proper noun. Propositions are content
statements about the subject with framing dropped and no author as grammatical subject.
Under `--fidelity content` (the default) the semantic lock keeps the proposition list for
the dropped check and runs the added check against the original paragraph itself, sentence
by sentence: an addition is reported only if it introduces a new named entity, a new number
or quantity, a new event, a cause or condition the original neither states nor implies, or
a scene, object or image that the stated facts do not entail. A statement that follows
necessarily from stated facts is an inference, not an addition, and is permitted: cattle
have nothing to graze where the fields are now sea; a cliff is made of gravel and clay when
its gravel and clay are drawn into the sea. A distant shore, when the original names no
shore, is an addition. Restatement, elaboration of a stated claim in other words, and
making an implicit causal link explicit are not additions. Everything the first pass flags
is re-asked in a second pass with the original shown ("would a careful reader of the
original learn anything new from this sentence?"), as a majority vote of three samples at
temperature 0.3, cached as a set; only a majority yes survives, the rest are listed as
cleared on review, and every decision carries its vote split (`3-0`, `2-1`) so the
borderline cases can be seen. Each rejection names its category. `--fidelity strict` fails
any new claim, for originals already in the target voice. `--report-all` judges and reports
every candidate, pass or fail. The lexical lock compares words case-insensitively in both
directions. The coherence checks exempt first and second person from the antecedent rule
and report tense drift only between two different dominant tenses (a paragraph is mixed
when no tense holds more than half of its finite verbs). `--compare PAGE.html` writes the
comparison page described under `deslop report`.

## Recitation

A writer that has read the human original can recite it instead of writing from the
propositions. `--reference FILE` names that original when there is one: it is never shown
to the writer or the judge, only compared against. For each candidate the check collects
the runs of eight or more words (lowercased, punctuation stripped) shared with the reference
and not present in the input, and reports their count, the share of the candidate's words
they cover, the longest three spans, and the recited words the input does not contain at all
(the novel share). Eight-word runs can be reassembled from the input's own words in the
reference's order, so the raw share has a floor even for a text no model has seen; the novel
share separates memory of the reference from reassembly of the input, and it is the figure
the limits apply to. A candidate whose novel share is over `--max-recitation` (2% of its
words) is rejected for recitation; the retry prompt says only that the draft recited a
published text, never which words. `deslop lab recitation` re-scores a saved result offline,
and `deslop lab fixture-check` applies the same measure to a prospective fixture text (the
novel share under 1% of the draft to qualify; the raw share is reported alongside).

## Primary texts

`--primary-text` names a source that added quotations and citations must verify against,
or a voice declares its own in `voice.yaml`. The format is deslop's: a plain text, an
optional `<name>.locs.json` sidecar of labelled spans, and a citation scheme (a regex) per
text, documented in [primary-texts.md](primary-texts.md). A verbatim quotation and a
resolving citation are permitted and listed as verified additions with their locations; a
quotation that is not verbatim is rejected with the nearest run of the text shown; a
citation a scheme recognises but no label answers is rejected; one no scheme recognises is
listed as unverifiable. Facts drawn from a primary text that the input does not state stay
material additions unless `--allow-primary-facts` is on. The writer is shown, verbatim, the
spans the input's own citations resolve to (the whole chapter when it fits under
`--quote-context-words`), told to quote only from them, and told how each text is cited. Without a primary text everything
added is listed as unverified and nothing is rejected.

## The comparative judge

The judge sees the original and two texts, A and B, in a fixed but arbitrary order, plus two
exemplars and the profile, and answers four questions separately with A, B or tie: which
reads more like the writer; in which each sentence follows from the last; which keeps a
consistent tense and person; which better preserves the original's order and emphasis. The
provider that wrote a candidate never judges it; a pair of candidates that no provider is
clear of is not compared. Against the original a candidate must win or tie on every item.
The win rate counts a net win as 1 and a draw as 0.5 over the comparisons made.

`deslop lab calibrate-judge` measures each provider on 100 pairs of one LLM-reference
paragraph and one paragraph by a held-out author of the voice: accuracy is how often the
corpus paragraph was picked as the writer's. The result is stored in `voice.yaml` and read
by `apply`: a provider below 80% is not used as a judge (it may still write, judged by the
other), and when no provider reaches 80% no candidate can pass, the original is kept, and
the report says so. The absolute 1-to-5 rubric (`deslop score --judge`,
`Result.units[].candidates[].judge` on the winner) is kept as a trend figure only.

```sh
deslop apply draft.md --voice essays.voice -o out.md --diff
deslop apply draft.md --voice essays.voice --mode edit         # only flagged sentences
deslop apply --brief brief.yaml --voice essays.voice -o piece.md
```

Edit mode keeps the v1 mechanism (flagged sentences only, everything else byte-identical)
under the new locks. A brief is drafted section by section, then rewritten.

## Writer, judge and provider

The writer and the judge are two different Claude models: by default claude-opus-5-5 writes
and claude-sonnet-5-5 judges and runs the semantic lock. `--writer` and `--judge` swap them
(`opus`, `sonnet` and `haiku` are aliases); naming the same model for both is an error,
because a model never judges its own output. The judge's calibration (`lab calibrate-judge`,
stored in `voice.yaml` by model) is printed in the report header; below 80% it is marked
unreliable.

Both models are served by one provider: `anthropic`, the Messages API with
`ANTHROPIC_API_KEY`, or `claude-cli`, the Claude Code CLI in headless mode using the local
login (one subprocess per call, 120 s timeout, same cache and retries as the API, so a
response cached by one provider replays through the other). Choose with `--provider`, the
config file's `provider:` key or `DESLOP_PROVIDER`. CLI calls are logged to `spend.jsonl` as
subscription calls with cost null; `--budget` counts dollars only, so `--max-calls` (200)
caps those runs. OpenAI remains in the provider layer and is used only when named.

## Budget

`--budget` (default $5) caps what one invocation may spend on live calls; past it the run
stops with the best result so far and says so. Every live call is logged to
`$DESLOP_HOME/spend.jsonl`; `deslop doctor` sums today and the last seven days.

## The brief

```yaml
title: On the Binding of Isaac
audience: readers who know the story
words_per_section: 180
sections:
  - heading: The command
    claim: The chapter withholds the reason for the command.
    refs: ["Gen 22:1-2"]
    quotes: ["Take your son, your only son Isaac, whom you love"]
```

Each section makes one claim and uses exactly the references and quotations it lists; anything
unlisted is stripped before the rewrite loop.

## From Python

```python
from deslop import Voice, apply, load_brief

voice = Voice.load("essays.voice")
result = apply(open("draft.md").read(), voice, n=3, primary_texts=["genesis.txt"])
for unit in result.units:
    print(unit.index, unit.kind, unit.reason)
print(result.growth_pct, result.unverified_additions)
print(result.meaning_diff)
piece = apply(load_brief("brief.yaml"), voice)
```
