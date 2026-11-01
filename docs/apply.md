# Apply

`apply` writes a text in a voice. The default, rewrite mode, is generate-then-verify: every
paragraph is written afresh several times and each candidate has to prove itself.

For each paragraph:

1. Its propositions are extracted as a numbered list of short claims (cached).
2. `--n` candidates (default 3) are sampled at temperature 0.9 from the configured providers in
   rotation, with the profile, the rules, three exemplars chosen cluster-proportionally and
   nearest in subject, the paragraph, its propositions, and the document's allowlist of names,
   numbers, quotations and citations. The instruction: write this paragraph as this writer
   would; keep every proposition, add none; change any sentence, reorder, split, merge; cite
   the primary text and address the reader where the writer would.
3. Each candidate is verified: the content lock (nothing locked removed; additions the document
   already has, or citations to the primary text, are fine), the length limit (no more than
   `--max-growth` percent, default 15, longer than the paragraph), the primary-text check
   (below), the semantic lock (a different provider checks every original proposition is kept
   and nothing new is asserted, except reader address and primary-text citations), and the
   linter score of the paragraph in document context.
4. The survivors are judged comparatively (below): each against the original paragraph and,
   round-robin, against the other survivors, giving a win rate. A candidate is accepted when
   it beats the original with confidence at least `--min-confidence` (0.7). The best by win
   rate, then confidence over the original, then linter distance wins.
5. If none passes, one retry names the failures (the dropped propositions, the unverified
   quotation, the judge's note). If that fails too, the original paragraph is kept and the
   report says why.
6. After every paragraph, one document-level pass rewrites units still flagged for a summary
   close (T14), a monotone run across paragraphs (T15) or an aphoristic close.

The report lists, per paragraph, the candidates tried with their win rates, which won and
why, the cost, the document's growth, a meaning diff of propositions with ✓ or ✗ and of
added quotations and citations with ✓ (verified) or ? (unverified), and the before and after
score with its separation context. Unverified additions are listed at the top.

## Primary texts

`--primary-text` (repeatable), a brief's `primary_texts:` list or the voice's
`primary_texts:` in `voice.yaml` name plain-text sources for the work under discussion. Any
quoted span of three or more words the output adds must occur verbatim (whitespace and
punctuation normalised) in one of them, or the candidate is rejected and the span named. Any
verse reference or Stephanus page the output adds must resolve to a marker in a text that
carries markers of that kind, or the candidate is rejected; a kind no text has markers for
is listed as unverifiable. Without any primary text, added quotations and citations are
allowed and listed as unverified additions.

Two texts ship with deslop and are named by key: `kjv-genesis` (Genesis, King James
Version, Project Gutenberg #10, verse markers) and `jowett-republic` (Jowett's Republic,
Project Gutenberg #55201, Stephanus pages). A file of your own looks like this:

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

The judge sees two paragraphs, A and B, in a fixed but arbitrary order, plus two exemplars
and the profile, and answers which reads more like the exemplars' writer and which reads
more like model prose, with a confidence from 0.5 to 1. The provider that wrote a candidate
never judges it; a pair of candidates that no provider is clear of is not compared. The win
rate is wins over comparisons made.

`deslop lab calibrate-judge` measures each provider on 100 pairs of one LLM-reference
paragraph and one paragraph by a held-out author of the voice: accuracy is how often the
corpus paragraph was picked as the writer's. The result is stored in `voice.yaml` and read
by `apply`: a provider below 80% is not used as a judge (it may still write, judged by the
other), and when no provider reaches 80% the gate is off, candidates are ranked by the
linter alone, and the report says so. The absolute 1-to-5 rubric (`deslop score --judge`,
`Result.units[].candidates[].judge` on the winner) is kept as a trend figure only.

```sh
deslop apply draft.md --voice essays.voice -o out.md --diff
deslop apply draft.md --voice essays.voice --mode edit         # only flagged sentences
deslop apply --brief brief.yaml --voice essays.voice -o piece.md
```

Edit mode keeps the v1 mechanism (flagged sentences only, everything else byte-identical)
under the new locks. A brief is drafted section by section, then rewritten.

## The brief

```yaml
title: On the Binding of Isaac
audience: readers who know the story
words_per_section: 180
primary_texts: [kjv-genesis]
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
result = apply(open("draft.md").read(), voice, n=3, primary_texts=["kjv-genesis"])
for unit in result.units:
    print(unit.index, unit.kind, unit.reason)
print(result.growth_pct, result.unverified_additions)
print(result.meaning_diff)
piece = apply(load_brief("brief.yaml"), voice)
```
