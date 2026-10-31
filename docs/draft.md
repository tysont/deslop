# Draft

`draft` writes a new piece from a brief, in a voice, with every citation and quotation under
control. A brief is yaml:

```yaml
title: What Johnson Asked of Shakespeare
audience: readers who know the plays but not the Preface
words_per_section: 180
sections:
  - heading: General nature
    claim: Johnson praises Shakespeare for drawing species, not individuals.
    refs: ["(Johnson 1765)"]
    quotes:
      - "just representations of general nature"
  - heading: No heroes
    claim: Johnson's Shakespeare has no heroes, only men.
    refs: ["(Johnson 1765)"]
    quotes:
      - "Shakespeare has no heroes"
```

Each section is drafted on its own and must make exactly its one claim and use each listed
reference and quotation verbatim; a section missing one gets a second attempt with feedback.
Any quotation or citation the brief did not list is stripped: a sentence carrying an unlisted
quotation is dropped whole, and an unlisted citation is cut out of its sentence. The assembled
draft is then run through `convert` in rewrite mode.

```sh
deslop draft brief.yaml --voice essays.voice --out piece.md
```

The sample brief at `tests/fixtures/brief_sample.yaml` drafts against the fixture corpus in
one conversion iteration for about a quarter of a dollar.

## From Python

```python
from deslop import Voice, draft, load_brief

piece = draft(load_brief("brief.yaml"), Voice.load("essays.voice"))
for section in piece.sections:
    print(section.heading, section.attempts, section.missing, section.stripped)
print(piece.report.verdict, piece.cost.usd)
print(piece.text)
```
