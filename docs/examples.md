# Examples

## The worked example

`examples/convert_example.py` builds a voice from the fixture corpus, converts
`examples/slop_sample.md` in edit mode, and prints the before and after reports with a diff.
It replays every model call from the cache committed under `tests/cache`, so it runs without
API keys:

```sh
make example
# or
uv run python examples/convert_example.py
```

The sample starts at FAIL with a distance in the forties and ends at PASS after two
iterations, with fourteen of seventeen rewrites accepted and the rest of the text untouched.

## A pipeline check

```python
import sys
from deslop import Voice, score

voice = Voice.load("essays.voice")
report = score(sys.stdin.read(), voice)
print(report.pretty())
sys.exit(0 if report.verdict == "PASS" else 2)
```

## Converting with a budget and a custom provider cache

```python
from pathlib import Path
from deslop import Voice, convert
from deslop.providers.base import get_provider

def provider(name: str):
    return get_provider(name, cache_dir=Path("build/cache"))

result = convert(
    Path("draft.md").read_text(),
    Voice.load("essays.voice"),
    mode="rewrite",
    max_iter=2,
    budget=0.50,
    provider_factory=provider,
)
print(result.stopped, result.cost.usd)
```

`cache_only=True` on `get_provider` makes a provider answer from its cache alone and raise
`CacheMissError` for anything else, which is how the test suite runs in CI without secrets.

## Drafting several briefs

```python
from pathlib import Path
from deslop import Voice, draft, load_brief

voice = Voice.load("essays.voice")
for brief_path in sorted(Path("briefs").glob("*.yaml")):
    piece = draft(load_brief(brief_path), voice, convert_max_iter=2)
    Path("out", brief_path.stem + ".md").write_text(piece.text)
    print(brief_path.name, piece.report.verdict, f"${piece.cost.usd:.2f}")
```

## Folding accepted work back in

```python
from pathlib import Path
from deslop.voice.update import update
from deslop.voice.diff import diff_voices
from deslop.voice.validate import validate
from deslop import Voice

res = update(Path("essays.voice"), [Path("out/piece.md")])
print("snapshot:", res.snapshot, "added:", res.added)
d = diff_voices(Voice.load(res.snapshot), Voice.load("essays.voice"))
for m in d.metrics:
    if m.changed:
        print(m.name, m.before, "->", m.after, m.direction)
assert validate(Path("essays.voice")).ok
```
