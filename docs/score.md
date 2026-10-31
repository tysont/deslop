# Score

`score` compares a text with a voice on two axes and returns a report.

**Metrics** are twenty-five numbers about rhythm and anchoring: sentence-length spread,
openers, punctuation rates, citation and quotation density, concreteness, first person,
committed claims. The text is measured in chunks of the same size the voice was measured in,
and each metric with a slop side is compared with the band the voice calibrated for it. A
metric is flagged when it falls outside its band; a metric with no slop side is reported but
never flagged.

**Tells** are fifteen patterns of machine prose, T1 through T15, from text-as-agent
("the chapter asks") through negative parallelism, triads, aphoristic closes, hedge stacks,
and monotone runs of same-length sentences. Each is counted per 1,000 words and flagged when
its rate exceeds the limit the voice calibrated for it.

The **distance** is a single 0 to 100 number rescaled so the voice's own held-out documents
sit below 20 and LLM-reference prose centres on 70; the report's `separation` carries the
calibration behind that scale. The **verdict** is PASS when nothing is flagged.

```sh
deslop score draft.md --voice essays.voice
deslop score draft.md --voice essays.voice --json
deslop score draft.md --voice essays.voice --annotate draft.annotated.md
deslop score draft.md --voice essays.voice --judge     # adds the three rubric scores
```

The exit code is 0 for PASS and 2 for FAIL, so `score` works as a check in a pipeline.

## From Python

```python
from deslop import Voice, score

report = score(open("draft.md").read(), Voice.load("essays.voice"))
print(report.verdict, report.distance, report.separation.heldout_p90, report.fixes)
```
