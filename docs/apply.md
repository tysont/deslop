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
   already has, or citations to the primary text, are fine), the semantic lock (a different
   provider checks every original proposition is kept and nothing new is asserted, except
   reader address and primary-text citations), the linter score of the paragraph in document
   context, and a rubric judge (voice, model prose, fidelity, 1 to 5 each). Candidates are
   ranked by linter distance then judge score; the best that passes both locks and
   `--judge-min` wins.
4. If none passes, one retry names the failures (the dropped propositions, the new tells, the
   judge's note). If that fails too, the original paragraph is kept and the report says why.
5. After every paragraph, one document-level pass rewrites units still flagged for a summary
   close (T14), a monotone run across paragraphs (T15) or an aphoristic close.

A paragraph that already reads in the voice (judge at or above the gate, no flagged tell) is
kept as it is. The report lists, per paragraph, the candidates tried, which won and why, the
cost, a meaning diff of propositions with ✓ or ✗, and the before and after score with its
separation context.

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
result = apply(open("draft.md").read(), voice, n=3, judge_min=4.0)
for unit in result.units:
    print(unit.index, unit.kind, unit.reason)
print(result.meaning_diff)
piece = apply(load_brief("brief.yaml"), voice)
```
