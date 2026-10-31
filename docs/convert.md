# Convert

`convert` rewrites a text toward a voice. It scores, picks the spans to rewrite, asks a model
for each one, validates the answer, splices the accepted answers in, and scores again, until
the text passes or a limit is hit.

## Units

In **edit** mode (the default) the units are flagged sentences. Adjacent flagged sentences in
one paragraph are merged into a passage the model may split or merge, which is how rhythm
problems like monotone runs get fixed. Every sentence outside a unit is byte-identical in the
output. In **rewrite** mode the units are whole flagged paragraphs. When only document-level
metrics are off and no sentence is flagged, the runs of similar-length sentences become the
units.

## Validation

A rewrite is rejected, and the original kept, when it:

- changes a locked citation, quotation, number, or proper noun (`--lock`, default all four;
  proper nouns already present elsewhere in the document are allowed);
- drifts outside `--preserve-length` (default 15 percent, applied to a passage's total);
- repeats its surrounding context, or contains line breaks in a sentence unit;
- introduces a tell that was not in the paragraph before.

Rejected attempts are shown to the model on retry so an identical answer cannot come back.
The retry goes to the other provider.

## Providers, cost, cache

`--models` names the providers to alternate per iteration, default `claude,gpt,claude,gpt`,
read from `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` (or a `.env` file). Every response is
cached under `$DESLOP_HOME/cache` keyed on the full prompt, so repeating a run is free and
reproducible. Costs come from `pricing.yaml`; `--budget` stops the loop and returns the
best iteration so far. Every attempt, accepted or not, is appended to
`$DESLOP_HOME/pairs/<voice>.jsonl`, which `voice export` later turns into training data.

```sh
deslop convert draft.md --voice essays.voice --out draft.voiced.md --diff
deslop convert draft.md --voice essays.voice --mode rewrite --max-iter 2 --budget 1.00
```

## From Python

```python
from deslop import Voice, convert

result = convert(open("draft.md").read(), Voice.load("essays.voice"), mode="edit")
print(result.report_before.verdict, "->", result.report_after.verdict, result.stopped)
print(result.cost.usd, "USD over", result.cost.calls, "calls")
print(result.diff)
open("draft.voiced.md", "w").write(result.text)
```
