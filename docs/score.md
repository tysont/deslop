# Score

`score` compares a text with a voice on two axes and returns a report.

**Metrics** are twenty-five numbers about rhythm and anchoring: sentence-length spread,
openers, punctuation rates, citation and quotation density, concreteness, first person,
committed claims. The text is measured in chunks of the same size the voice was measured in,
and each metric is compared with the voice's mean and standard deviation. A metric is flagged
when it sits more than `--sigma` (default 1.5) standard deviations away on its slop side; a
metric with no slop side is reported but never flagged.

**Tells** are fifteen patterns of machine prose, T1 through T15, from text-as-agent
("the chapter asks") through negative parallelism, triads, aphoristic closes, hedge stacks,
and monotone runs of same-length sentences. Each is detected by regex, dependency pattern, or
a document-level function, and counted per 1,000 words. A tell is flagged when its rate
exceeds `max(voice_rate * --tell-mult, --tell-floor)`, or at all if the voice bans it.

The **distance** is a single 0 to 100 number: 0 means every metric sits at the voice's mean
and no tell exceeds its threshold. The **verdict** is PASS when nothing is flagged.

```sh
deslop score draft.md --voice essays.voice
deslop score draft.md --voice essays.voice --json
deslop score draft.md --voice essays.voice --annotate draft.annotated.md
```

The exit code is 0 for PASS and 2 for FAIL, so `score` works as a check in a pipeline.

## From Python

```python
from deslop import Voice, score

report = score(open("draft.md").read(), Voice.load("essays.voice"))
print(report.verdict, round(report.distance))
for fix in report.fixes:
    print("-", fix)
for tell in report.flagged_tells():
    print(tell.id, tell.name, tell.rate, tell.threshold)
```

Extra tell patterns can be added without code: write a yaml catalog in the shape of
`src/deslop/baselines/slop_patterns.yaml` and pass it with `--patterns` or `patterns=`.
