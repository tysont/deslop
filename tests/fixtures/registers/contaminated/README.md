# Retired fixtures

The originals the essay and fiction registers used before v2.4.0, kept for reference and
never used as fixtures. Both fail the validity rule in `../README.md`: a draft written from
their propositions alone, in the register's voice, recites them in runs the propositions do
not contain, which means the writer model knows the text and a rewrite of it would not be a
test of writing in a voice.

| file | author | raw recitation | novel recitation | limit |
| --- | --- | ---: | ---: | ---: |
| `essay.md` | Robert Louis Stevenson, An Apology for Idlers | 18.4% | 4.2% | 1% novel |
| `fiction.md` | Saki (H. H. Munro), The Open Window | 54.6% | 14.9% | 1% novel |

The figures are from `deslop lab fixture-check` under v2.3.4's measure (eight-word runs;
novel means recited words the propositions do not contain at all). Each file's frontmatter
carries the same numbers under `retired:`. Their replacements, chosen as the lowest of three
candidates each, are in `../essay/` and `../fiction/`.
