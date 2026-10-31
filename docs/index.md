# deslop

deslop measures a writer's voice from a corpus of their prose, scores any text against that
voice, and rewrites only the spans that deviate while leaving everything else byte-identical.
Its first job is removing the structural and lexical tells of machine-written text, but a
voice can be any writer or register you have enough text for.

The library has five verbs, and the CLI mirrors them:

| verb | what it does |
| --- | --- |
| `corpus` | collect and normalize the writer's texts |
| `calibrate` | measure the corpus into a `name.voice/` directory |
| `score` | report how far a text is from the voice and exactly where |
| `convert` | rewrite the flagged spans with a model until the text passes |
| `draft` | write a new piece from a brief, in the voice, with every citation controlled |

Three maintenance commands keep a voice honest over time (`voice update`, `voice diff`,
`voice validate`), `voice export` turns accepted rewrites into fine-tuning data, and `blind`
runs an A/B test of your own ability to tell the voice from its imitation.

## Install

```sh
pipx install deslop
pipx inject deslop https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
deslop doctor
```

The README on GitHub has the full command reference. These pages explain the concepts and
show the Python API.
