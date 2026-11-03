# Score

`score` compares a text with a voice and returns a report: a PASS or FAIL verdict, a
distance from 0 to 100, every flagged metric and tell, and the spans that triggered them.
It makes no model calls.

```sh
deslop score draft.md --voice essays.voice
deslop score draft.md --voice essays.voice --json
deslop score draft.md --voice essays.voice --annotate draft.annotated.md
deslop score draft.md --voice essays.voice --judge     # adds an absolute rubric (trend only)
```

The exit code is 0 for PASS and 2 for FAIL, so `score` works as a check in a pipeline.

## Metrics

Twenty-six numbers about rhythm and anchoring. The text is measured in chunks of the same
size the voice was measured in, and each metric is compared with the band the voice
calibrated for it ([voice.md](voice.md#thresholds)).

The **slop side** is the direction in which machine prose tends to deviate. A metric is
flagged only when it falls outside its band on that side. A metric whose side is `both` is
flagged on either side, so overshooting the voice fails like undershooting it. A metric
whose side is `none` is reported but never flagged.

| metric | group | slop side | definition |
| --- | --- | --- | --- |
| `citation_per_100w` | anchors | both | Citations (verse, Stephanus, page/line, author-year) per 100 words |
| `quote_per_para` | anchors | both | Quoted spans of four or more words per paragraph |
| `concrete_per_100w` | anchors | both | Numbers, proper nouns, non-Latin script and italic terms per 100 words |
| `reader_address_per_1k` | anchors | both | Second-person pronouns and reader imperatives per 1k words |
| `first_person_per_1k` | anchors | low | First-person singular pronouns per 1k words |
| `committed_claim_per_para` | anchors | low | Evaluative predicates asserting a reading right or wrong, per paragraph |
| `sent_len_mean` | rhythm | none | Mean words per sentence |
| `sent_len_std` | rhythm | low | Standard deviation of words per sentence |
| `sent_len_cv` | rhythm | both | Coefficient of variation of sentence length |
| `pct_sent_lt8` | rhythm | none | Percent of sentences under 8 words |
| `pct_sent_gt30` | rhythm | none | Percent of sentences over 30 words |
| `max_run_similar_len` | rhythm | high | Longest run of consecutive sentences within 5 words of each other |
| `pct_compound_multi` | rhythm | none | Percent of sentences with two or more coordinating joins |
| `para_len_mean` | rhythm | none | Mean sentences per paragraph |
| `para_len_std` | rhythm | low | Standard deviation of sentences per paragraph |
| `opener_pct_det` | rhythm | none | Percent of sentences opening with a determiner |
| `opener_pct_pron` | rhythm | none | Percent of sentences opening with a pronoun |
| `opener_pct_adv_conj` | rhythm | high | Percent of sentences opening with an adverb or conjunction |
| `opener_pct_noun` | rhythm | none | Percent of sentences opening with a noun or proper noun |
| `opener_top_word_share` | rhythm | both | Percent of sentences that begin with the single most common first word |
| `connective_opener_pct` | rhythm | both | Percent of sentences opening with and, but, so, then, because, since, while or yet |
| `semicolons_per_1k` | rhythm | none | Semicolons per 1k words |
| `colons_per_1k` | rhythm | high | Colons per 1k words |
| `em_dashes_per_1k` | rhythm | high | Em dashes per 1k words |
| `parens_per_1k` | rhythm | none | Open parentheses per 1k words |
| `questions_per_1k` | rhythm | high | Question marks per 1k words |

## Tells

Sixteen patterns of machine prose, each counted per 1,000 words. A tell is flagged when its
rate exceeds the limit the voice calibrated for it, `max(voice_rate * m, floor)`, with the
multiple `m` (6 down to 1) and the floor (2 down to 0.5 per 1k words) tuned per tell. A tell the voice **bans** is flagged at any rate: the
voice bans a tell when it uses it less than a tenth as often as the LLM reference does.

The catalog is `src/deslop/baselines/slop_patterns.yaml`. Add regex patterns there or in a
file passed with `--patterns`. To add a new tell to a voice built before the tell existed,
use `deslop voice add-tells`.

| id | name | what it catches |
| --- | --- | --- |
| T1 | `text_as_agent` | A text noun as the subject of an intentional verb: the chapter asks, the passage insists, the sentence turns on, the verse is about |
| T2 | `negative_parallelism` | Contrast framed by negation: not X; Y, not X but Y, X, not Y, rather than, less X than Y |
| T3 | `balance_markers` | Words that assert a poised both-ness instead of taking a side |
| T4 | `triad` | Three or more parallel clauses or predicates, or three abstract nouns, adjectives or adverbs in rhetorical parallel; a list of concrete nouns (gravel, chalk and clay) is not a triad |
| T5 | `aphoristic_close` | A paragraph-final sentence with no anchor and an antithetical shape or a high density of abstract nouns |
| T6 | `invented_label` | A possessive abstract noun modifying another abstract noun (appetite's arithmetic) |
| T7 | `meta_commentary` | The predicate describes what the text or author is doing rather than asserting content |
| T8 | `epithet_substitution` | A definite descriptive phrase standing in for an already-introduced name (the Danish prince) |
| T9 | `unanswered_question` | A question the next two sentences do not answer |
| T10 | `formula_openers` | Stock sentence openers: So, Then, Against that, Set X beside Y, It is worth noting, Importantly |
| T11 | `hedge_stack` | Two or more hedges in one sentence |
| T12 | `intensifier_cluster` | Sincerity intensifiers: deeply, truly, fundamentally, inherently, genuinely |
| T13 | `colon_reveal` | A colon followed by a single abstract noun phrase as the reveal |
| T14 | `tidy_summary_close` | A final paragraph that restates earlier paragraphs with no new anchors |
| T15 | `monotone_run` | Three or more consecutive sentences whose word counts all fall within five of each other |
| T16 | `cleft_inversion` | A cleft or inversion that holds the point back: "What X is Y", "It is X that Y" (not "It is clear that"), "X is what/why Y", and a fronted object ("Grief Plato will not allow") |

## Distance and verdict

The **distance** is a single number from 0 to 100, rescaled against the voice's calibration
so that the voice's own held-out documents sit below 20 and LLM-reference prose centres on
70 ([voice.md](voice.md#thresholds)). The report's `separation` field carries the
calibration behind the scale. The **verdict** is PASS when no metric or tell is flagged.

## From Python

```python
from deslop import Voice, score

report = score(open("draft.md").read(), Voice.load("essays.voice"))
print(report.verdict, report.distance, report.separation.heldout_p90, report.fixes)
```
