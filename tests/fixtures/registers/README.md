# Register fixtures

Four register pairs (essay, explainer, product, fiction). Each directory holds a marked
public-domain original (`original.md`, or one written for the set), `slop.md` written from
the original's proposition list alone by `scripts/build_register_slop.py` (the model never
sees the original), `propositions.json`, a 3-4 document same-register corpus under `corpus/`
and the small voice built from it (`<name>.voice`). The original is never in its voice's
corpus.

## Validity rule

A text qualifies as a fixture original only when `deslop lab fixture-check ORIGINAL.md
--voice <name>.voice` reports under 1% novel recitation: in a draft written from the text's
propositions alone, the words that lie in runs of eight or more words shared with the text
and that the propositions do not contain at all must be fewer than one in a hundred of the
draft. The raw share (every word in a shared run) is reported alongside but does not decide:
eight-word runs reassemble from the propositions' own words even for a text no model has
seen. A text the writer model can recite from memory is not a test of writing in a voice,
whatever its author's standing. Prefer little-known authors; the check decides.

`paraphrase/` keeps the v2.2.0 paraphrases for reference only.
