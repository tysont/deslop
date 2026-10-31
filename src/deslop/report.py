# ABOUTME: Report model produced by score(): per-metric z-scores, tell hits, paragraph
# ABOUTME: deviations, verdict, ranked fixes. Renders to JSON, rich tables, or annotated text.
from __future__ import annotations

from io import StringIO
from typing import Literal

from pydantic import BaseModel, Field
from rich.console import Console, RenderableType
from rich.table import Table

Verdict = Literal["PASS", "FAIL"]


class MetricResult(BaseModel):
    name: str
    group: str
    value: float
    voice_mean: float
    voice_std: float
    z: float
    slop_side: str
    flagged: bool


class TellHitResult(BaseModel):
    pattern_id: str
    sentence_idx: int
    para_idx: int
    start: int
    end: int
    text: str
    note: str = ""


class TellResult(BaseModel):
    id: str
    name: str
    count: int
    rate_per_1k: float
    voice_rate: float
    llm_rate: float | None
    threshold: float
    ratio_vs_voice: float | None
    banned: bool
    flagged: bool
    hits: list[TellHitResult] = Field(default_factory=list)


class ParagraphResult(BaseModel):
    index: int
    start: int
    end: int
    words: int
    deviation: float
    flagged_sentences: list[int] = Field(default_factory=list)
    tells: list[str] = Field(default_factory=list)


class ScoreOptions(BaseModel):
    sigma: float = 1.5
    tell_mult: float = 2.0
    tell_floor: float = 1.0


class Report(BaseModel):
    voice: str
    verdict: Verdict
    distance: float
    words: int
    sentences: int
    paragraphs_count: int
    metrics: list[MetricResult]
    tells: list[TellResult]
    paragraphs: list[ParagraphResult]
    fixes: list[str]
    options: ScoreOptions

    # ------------------------------------------------------------ output
    def to_json(self, indent: int | None = 1) -> str:
        return self.model_dump_json(indent=indent)

    def flagged_metrics(self) -> list[MetricResult]:
        return [m for m in self.metrics if m.flagged]

    def flagged_tells(self) -> list[TellResult]:
        return [t for t in self.tells if t.flagged]

    def all_hits(self) -> list[TellHitResult]:
        return sorted((h for t in self.tells for h in t.hits), key=lambda h: h.start)

    def renderables(self, *, show_hits: bool = True, max_hits: int = 40) -> list[RenderableType]:
        items: list[RenderableType] = []
        color = "green" if self.verdict == "PASS" else "red"
        items.append(
            f"[bold {color}]{self.verdict}[/] against voice [bold]{self.voice}[/]  "
            f"distance {self.distance:.0f}/100  {self.words} words, {self.sentences} sentences, "
            f"{self.paragraphs_count} paragraphs"
        )
        mt = Table(title="Metrics", show_lines=False)
        for col in ("metric", "value", "voice mean", "std", "z", "slop side", "flag"):
            mt.add_column(
                col, justify="right" if col not in ("metric", "slop side", "flag") else "left"
            )
        for m in self.metrics:
            flag = "[red]FLAG[/]" if m.flagged else ""
            mt.add_row(
                m.name,
                f"{m.value:.2f}",
                f"{m.voice_mean:.2f}",
                f"{m.voice_std:.2f}",
                f"{m.z:+.2f}",
                m.slop_side,
                flag,
            )
        items.append(mt)
        tt = Table(title="Tells")
        for col in ("id", "pattern", "hits", "/1k", "voice /1k", "llm /1k", "threshold", "flag"):
            tt.add_column(col, justify="right" if col not in ("id", "pattern", "flag") else "left")
        for t in self.tells:
            flag = "[red]FLAG[/]" if t.flagged else ("[yellow]banned[/]" if t.banned else "")
            tt.add_row(
                t.id,
                t.name,
                str(t.count),
                f"{t.rate_per_1k:.2f}",
                f"{t.voice_rate:.2f}",
                "-" if t.llm_rate is None else f"{t.llm_rate:.2f}",
                f"{t.threshold:.2f}",
                flag,
            )
        items.append(tt)
        if show_hits:
            hits = self.all_hits()
            if hits:
                ht = Table(title=f"Hits ({len(hits)})")
                ht.add_column("id")
                ht.add_column("para", justify="right")
                ht.add_column("sent", justify="right")
                ht.add_column("span")
                for h in hits[:max_hits]:
                    label = f"{h.pattern_id}" + (f":{h.note}" if h.note else "")
                    ht.add_row(
                        label, str(h.para_idx), str(h.sentence_idx), h.text[:90].replace("\n", " ")
                    )
                if len(hits) > max_hits:
                    ht.add_row("...", "", "", f"{len(hits) - max_hits} more")
                items.append(ht)
        if self.fixes:
            items.append("[bold]Fix first:[/]")
            for i, fix in enumerate(self.fixes, 1):
                items.append(f"  {i}. {fix}")
        return items

    def pretty(self, width: int = 110) -> str:
        buf = StringIO()
        console = Console(file=buf, width=width, force_terminal=False, color_system=None)
        for item in self.renderables():
            console.print(item)
        return buf.getvalue()

    def annotate(self, text: str) -> str:
        """Insert [Tn:note] markers before every hit span."""
        hits = self.all_hits()
        out = []
        pos = 0
        for h in sorted(hits, key=lambda h: (h.start, h.pattern_id)):
            marker = f"[{h.pattern_id}" + (f":{h.note}" if h.note else "") + "]"
            if h.start < pos:
                out.append(marker)
                continue
            out.append(text[pos : h.start])
            out.append(marker)
            pos = h.start
        out.append(text[pos:])
        return "".join(out)
