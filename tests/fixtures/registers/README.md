# Register fixtures

Four register pairs (essay, explainer, product, fiction). Each directory holds a marked
public-domain original (`original.md`, or one written for the set), `slop.md` written from
the original's proposition list alone by `scripts/build_register_slop.py` (the model never
sees the original), `propositions.json`, a 3-4 document same-register corpus under `corpus/`
and the small voice built from it (`<name>.voice`). The original is never in its voice's
corpus.

## Validity rule

A text qualifies as a fixture original only when `deslop lab fixture-check ORIGINAL.md
--voice <name>.voice` reports under 1% recitation: a draft written from the text's
propositions alone must share with the text fewer than one word in a hundred in runs of eight
or more words. A text the writer model can recite from memory is not a test of writing in a
voice, whatever its author's standing. Prefer little-known authors; the check decides.

`paraphrase/` keeps the v2.2.0 paraphrases for reference only.
