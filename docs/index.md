# deslop

deslop measures a writer's voice from their texts, scores any text against that voice, and
writes or rewrites text in it with every proposition kept. Its first job is removing the
structural and lexical tells of machine-written prose, but a voice can be any writer or
register you have enough text for.

The library has two primary verbs and a check, and the CLI mirrors them:

| verb | what it does |
| --- | --- |
| `voice` | measure files or directories into a `NAME.voice` zip, with provenance |
| `apply` | write a text, or a piece from a brief, in the voice under a content lock, a semantic lock and a judge |
| `score` | report how far a text is from the voice, against calibrated thresholds, and exactly where |

`voice show`, `voice diff`, `voice validate`, `voice pack`, `voice unpack` and `voice edit`
inspect and maintain a voice; `lab blind`, `lab export` and `lab calibrate-thresholds` are
experiments.

## Install

```sh
pipx install deslop
pipx inject deslop https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
deslop doctor
```

The README on GitHub has the full command reference. These pages explain the concepts and
show the Python API.
