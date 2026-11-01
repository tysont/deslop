# Apply

`apply` writes a text in a voice. The default, rewrite mode, is generate-then-verify: the
document is written afresh as a whole several times and each candidate has to prove itself.

1. Every paragraph's propositions are extracted as a numbered list of short claims (cached).
2. `--n` candidates (default 3) are sampled at temperature 0.9 from the configured providers in
   rotation. The writer sees the profile, the rules, the exemplars nearest the piece, the
   full original and the propositions by paragraph, and is told: write this piece as this
   writer would; keep every proposition and add none; keep the author's tense, person,
   paragraphing and order of argument unless the voice's profile says otherwise. Nothing in
   the prompt mentions metrics, sentence length, openers or citations.
3. Each candidate is verified without a judge: the content lock (nothing locked removed), the
   length limit (no more than `--max-growth` percent, default 15, longer than the original),
   the primary-text check (below), and the coherence checks: the same number of paragraphs;
   the dominant finite-verb tense of each paragraph unchanged; every sentence-initial pronoun
   (and this, that, it) with a noun phrase in the two sentences before it; and the share of
   sentences opening with a connective (and, but, so, then, because, since, while, yet)
   inside the voice's band and within half of the original's share.
4. The semantic lock runs paragraph by paragraph: a different provider checks that every
   proposition is kept and nothing new is asserted.
5. A provider that wrote neither text judges the candidate against the original on four
   items, each answered separately: reads as the voice; coherent (each sentence follows
   from the last); consistent tense and person; preserves the argument's order and emphasis.
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

## Primary texts

`--primary-text` (repeatable) names plain-text sources for the work under discussion. Any
quoted span of three or more words the output adds must occur verbatim (whitespace and
punctuation normalised) in one of them, or the candidate is rejected and the span named. Any
verse reference or Stephanus page the output adds must resolve to a marker in a text that
carries markers of that kind, or the candidate is rejected; a kind no text has markers for
is listed as unverifiable. Without any primary text, added quotations and citations are
allowed and listed as unverified additions.

deslop ships no reference texts. A file of your own looks like this:

```
# title: Genesis (World English Bible)
# book: Genesis
# aliases: Gen, Ge

{{Genesis 6:5}} Yahweh saw that the wickedness of man was great in the earth...
{{Genesis 6:6}} Yahweh was sorry that he had made man on the earth...
```

Markers are `{{Book chapter:verse}}` or `{{369b}}`; `Gen 6:5`, `Genesis 6:5-7`, `6:5`,
`369b`, `369b-c` and `Republic II, 369b` all resolve. A file without markers still verifies
quotations.

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
