# Examples

## The worked example

`examples/apply_example.py` builds a voice from the fixture essays, applies it to
`examples/slop_sample.md` in edit mode, and prints the before and after reports with a diff.
It replays every model call from the cache committed under `tests/cache`, so it runs without
API keys:

```sh
make example
# or
uv run python examples/apply_example.py
```

## A pipeline check

```python
import sys
from deslop import Voice, score

voice = Voice.load("essays.voice")
report = score(sys.stdin.read(), voice)
print(report.verdict, report.distance)
sys.exit(0 if report.verdict == "PASS" else 2)
```

## Building a voice from a crawl

A crawler that writes a `manifest.jsonl` beside the texts it fetches (one JSON object per
line with `file`, `author`, `title`, and optional `url`, `license` and `date`) hands its
provenance to deslop: `deslop voice raw/<name> -o name.voice` reads the manifest for each
document's author, title, URL and license, so the voice records where every document came
from.
