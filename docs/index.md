# deslop

deslop learns a writer's voice from their texts, tells you where a piece of prose falls
outside it, and rewrites text into it without changing what the text says. Its first use is
removing the tells of machine-written prose, but a voice can be any writer or register you
have enough text for.

| command | what it does |
| --- | --- |
| `voice` | measure files or directories into a `NAME.voice` file, with provenance and calibrated thresholds |
| `score` | report how far a text is from the voice, and exactly where, against those thresholds |
| `apply` | rewrite a text, or draft one from a brief, in the voice, under a content lock, a semantic lock, coherence checks and a judge |

`voice show`, `voice diff`, `voice validate` and `voice edit` inspect and maintain a voice;
the `lab` commands calibrate judges and thresholds, check test fixtures and run blind tests.

## Install

deslop is installed from GitHub (the `deslop` package on PyPI is a different project), with
the spaCy English model beside it:

```sh
pipx install git+https://github.com/tysont/deslop.git
pipx inject deslop https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
deslop doctor
```

## These pages

- [Voice](voice.md): what a voice file holds, and how it is measured and calibrated
- [Score](score.md): the metrics, the tells and the distance
- [Apply](apply.md): the rewrite loop, the locks, the judge, and sections of longer works
- [Primary texts](primary-texts.md): verifying added quotations and citations against a source
- [A voice from your own writing](personal-voice.md): a walkthrough from corpus to result
- [Examples](examples.md): the worked example and small recipes
- [Command reference](reference.md): every command, flag, setting and environment variable
- [API reference](api.md): the Python API
