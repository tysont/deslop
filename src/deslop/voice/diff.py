# ABOUTME: Structural diff between two voices: metric means with slop direction, tell rates and
# ABOUTME: bans, lexicon membership, rules.md text, and author shares.
from __future__ import annotations

import difflib

from pydantic import BaseModel, Field

from deslop.metrics import METRICS
from deslop.voice.model import Voice


class MetricDelta(BaseModel):
    name: str
    before: float
    after: float
    delta: float
    z: float  # delta in units of the before-voice's std
    direction: str  # toward_slop | away_from_slop | neutral


class TellDelta(BaseModel):
    id: str
    name: str
    before: float
    after: float
    banned_before: bool
    banned_after: bool


class VoiceDiff(BaseModel):
    before: str
    after: str
    words_before: int
    words_after: int
    metrics: list[MetricDelta] = Field(default_factory=list)
    tells: list[TellDelta] = Field(default_factory=list)
    encouraged_added: list[str] = Field(default_factory=list)
    encouraged_removed: list[str] = Field(default_factory=list)
    discouraged_added: list[str] = Field(default_factory=list)
    discouraged_removed: list[str] = Field(default_factory=list)
    banned_added: list[str] = Field(default_factory=list)
    banned_removed: list[str] = Field(default_factory=list)
    rules_diff: str = ""
    profile_changed: bool = False
    exemplars_changed: bool = False
    author_shares: dict[str, tuple[float, float]] = Field(default_factory=dict)

    @property
    def changed(self) -> bool:
        return bool(
            any(m.delta for m in self.metrics)
            or any(t.before != t.after or t.banned_before != t.banned_after for t in self.tells)
            or self.encouraged_added
            or self.encouraged_removed
            or self.discouraged_added
            or self.discouraged_removed
            or self.banned_added
            or self.banned_removed
            or self.rules_diff
            or self.profile_changed
            or self.exemplars_changed
            or any(a != b for a, b in self.author_shares.values())
        )

    def to_json(self, indent: int | None = 1) -> str:
        return self.model_dump_json(indent=indent)


def _direction(name: str, delta: float) -> str:
    side = METRICS[name].slop_side if name in METRICS else "none"
    if delta == 0 or side in ("none", "both"):
        return "neutral"
    toward = delta > 0 if side == "high" else delta < 0
    return "toward_slop" if toward else "away_from_slop"


def diff_voices(a: Voice, b: Voice) -> VoiceDiff:
    d = VoiceDiff(
        before=a.name,
        after=b.name,
        words_before=a.meta.total_words,
        words_after=b.meta.total_words,
    )
    for name in sorted(set(a.metrics.metrics) | set(b.metrics.metrics)):
        sa, sb = a.metrics.metrics.get(name), b.metrics.metrics.get(name)
        if sa is None or sb is None:
            continue
        delta = sb.mean - sa.mean
        d.metrics.append(
            MetricDelta(
                name=name,
                before=round(sa.mean, 4),
                after=round(sb.mean, 4),
                delta=round(delta, 4),
                z=round(delta / sa.std, 3) if sa.std > 0 else 0.0,
                direction=_direction(name, delta),
            )
        )
    bans_a = {p.id: p for p in a.lexicon.patterns}
    bans_b = {p.id: p for p in b.lexicon.patterns}
    for pid in sorted(set(bans_a) | set(bans_b), key=lambda x: (len(x), x)):
        pa, pb = bans_a.get(pid), bans_b.get(pid)
        d.tells.append(
            TellDelta(
                id=pid,
                name=(pb or pa).name,
                before=pa.corpus_per_1k if pa else 0.0,
                after=pb.corpus_per_1k if pb else 0.0,
                banned_before=bool(pa and pa.banned),
                banned_after=bool(pb and pb.banned),
            )
        )
    d.banned_added = [t.id for t in d.tells if t.banned_after and not t.banned_before]
    d.banned_removed = [t.id for t in d.tells if t.banned_before and not t.banned_after]
    enc_a = {e.term for e in a.lexicon.encouraged_words}
    enc_b = {e.term for e in b.lexicon.encouraged_words}
    dis_a = {e.term for e in a.lexicon.discouraged_words}
    dis_b = {e.term for e in b.lexicon.discouraged_words}
    d.encouraged_added = sorted(enc_b - enc_a)
    d.encouraged_removed = sorted(enc_a - enc_b)
    d.discouraged_added = sorted(dis_b - dis_a)
    d.discouraged_removed = sorted(dis_a - dis_b)
    d.rules_diff = "".join(
        difflib.unified_diff(
            a.rules.splitlines(keepends=True),
            b.rules.splitlines(keepends=True),
            fromfile=f"{a.name}/rules.md",
            tofile=f"{b.name}/rules.md",
        )
    )
    d.profile_changed = a.profile != b.profile
    d.exemplars_changed = a.exemplars != b.exemplars
    for author in sorted(set(a.meta.author_shares) | set(b.meta.author_shares)):
        d.author_shares[author] = (
            a.meta.author_shares.get(author, 0.0),
            b.meta.author_shares.get(author, 0.0),
        )
    return d
