# A voice from your own writing

One page: build a voice from a directory of your own pieces, hold one piece out, turn it
into model prose, check it is a fair test, run `apply` on it against your voice, and read the
result. Every live step is cached, so a repeat costs nothing. Provider: the Claude Code CLI
(`provider: claude-cli` in `~/.deslop/config.yaml`) or an API key; `deslop doctor` says which.

## 1. Build the voice, with one piece held out

```sh
mkdir -p mine/corpus mine/heldout
cp writing/*.md mine/corpus/              # your pieces, one file each (md, txt, docx, html)
mv mine/corpus/one-good-piece.md mine/heldout/original.md   # never in the corpus
deslop voice mine/corpus -o mine/me.voice --author-map mine/authors.yaml --min-docs 4
deslop voice show mine/me.voice
```

`authors.yaml` is one line, `default: Your Name`, so every file is yours. With fewer than
`--min-docs` documents the voice runs in small-corpus mode: its bands are shrunk toward
generic English and every one of your own documents must pass its thresholds (see
`voice.md`, Small corpora). Three or four pieces of a thousand words each is enough to
start; `voice show` says how many more words would leave small-corpus mode. Building asks a
model to write `profile.md` once; `--no-llm` uses a template instead.

## 2. Model prose from the held-out piece

```sh
deslop lab slop mine/heldout/original.md -o mine/heldout/slop.md --kind "a personal essay"
```

Two calls: the piece's propositions are extracted, then a piece is written from them alone.
The model never sees your text, so `slop.md` carries your content and none of your prose.

## 3. Is it a fair test?

```sh
deslop lab fixture-check mine/heldout/original.md --voice mine/me.voice
```

One draft is written from the propositions in your voice and checked for recitation of the
piece. It qualifies when the recited words outside the propositions' vocabulary are under 1%
of the draft; for unpublished writing this is a formality, and it still pins the number.

## 4. Apply, with the original as the reference

```sh
deslop apply mine/heldout/slop.md --voice mine/me.voice --n 3 --report-all \
  --reference mine/heldout/original.md --compare mine/heldout/compare.html \
  -o mine/heldout/out.md --json > mine/heldout/result.json
```

The reference is only compared against, never shown to a model: each candidate is checked
for recitation of it and it fills the comparison page's first column. Expect 50 to 80 calls
for a thousand-word piece; `--max-calls` caps them.

## 5. Read the result

Open `compare.html`: your original, the model prose, and the selected candidate side by side
with a sentence-aligned diff and every check pinned to its paragraph. The report under
`out.md` says, per candidate, what the lock dropped or added (with the review's vote split),
the coherence checks, the recitation numbers, and the judge's four items against the input.
Exit 0 is PASS against the voice, 2 is FAIL; if nothing passed, the output is the input
unchanged and the report says why each candidate failed. Then read the candidate against
your original: what came back is the register and the paragraphing; what cannot come back
is anything the model prose had already lost.

## 6. Edit the voice

```sh
deslop voice edit mine/me.voice      # opens profile.md, rules.md, exemplars.md
```

- `profile.md`: the model's description of how you write. Strike anything that is not you,
  and name what it missed: your person and address, the sentence lengths you favour, the
  moves you never make. The writer reads this first.
- `exemplars.md`: the passages the writer imitates and the judge compares against. Replace
  the generic ones with the paragraphs you would hand to someone learning your voice; the
  output can be no closer to you than these are.
- `rules.md`: the lexicon's bans and preferences; add the words you never use.

## 7. Fold in what you accept

When an output is one you would have written, add it to the voice so the next build has
more of you to measure:

```sh
deslop voice --from mine/me.voice --add mine/heldout/out.md -o mine/me2.voice
```

Measurements, registers and thresholds are recomputed; the edited prose files are kept.
Keep the held-out original out of the corpus for as long as you use it as a test.
