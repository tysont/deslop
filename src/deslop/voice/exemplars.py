# ABOUTME: Chooses a voice's exemplars (verbatim excerpts, typical and diverse, per-author cap)
# ABOUTME: and finds the exemplars nearest a passage for the writer's prompt.
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from deslop.metrics import compute_all
from deslop.nlp import WORD_RE, Analysis, analyze
from deslop.stats import MetricsProfile, subset_analysis

if TYPE_CHECKING:
    from deslop.voice.model import Voice

MIN_WORDS = 80
MAX_WORDS = 350
DIVERSITY_WEIGHT = 1.0


@dataclass
class Candidate:
    author: str
    title: str
    text: str
    words: int
    vector: list[float]
    typicality: float
    content: frozenset[str]
    register: str = ""
    source: str = ""


def _zvector(values: dict[str, float], profile: MetricsProfile, names: list[str]) -> list[float]:
    vec = []
    for n in names:
        stat = profile.metrics[n]
        vec.append((values[n] - stat.mean) / stat.std if stat.std > 0 else 0.0)
    return vec


def _content(text: str) -> frozenset[str]:
    return frozenset(w.lower() for w in WORD_RE.findall(text) if len(w) > 4)


def _content_distance(a: frozenset[str], b: frozenset[str]) -> float:
    """1 - Jaccard overlap of content words: 0 for the same passage, 1 for disjoint topics."""
    union = len(a | b)
    return 1.0 - len(a & b) / union if union else 1.0


def candidates_from(
    author: str, title: str, analysis: Analysis, profile: MetricsProfile
) -> list[Candidate]:
    names = [n for n in profile.metrics if profile.metrics[n].std > 0]
    out = []
    for p in analysis.paragraphs:
        if not (MIN_WORDS <= p.word_count <= MAX_WORDS):
            continue
        sub = subset_analysis([p])
        vec = _zvector(compute_all(sub), profile, names)
        out.append(
            Candidate(
                author=author,
                title=title,
                text=p.text,
                words=p.word_count,
                vector=vec,
                typicality=math.sqrt(sum(v * v for v in vec) / max(len(vec), 1)),
                content=_content(p.text),
            )
        )
    return out


def select_exemplars(
    cands: list[Candidate], *, target_words: int, max_share: float
) -> list[Candidate]:
    if not cands:
        return []
    author_cap = max(
        int(target_words * max(max_share, 1.0 / max(len({c.author for c in cands}), 1)) * 1.3),
        MAX_WORDS,
    )
    chosen: list[Candidate] = []
    author_words: dict[str, int] = {}
    total = 0
    pool = sorted(cands, key=lambda c: c.typicality)
    while pool and total < target_words:
        best = None
        best_score = -math.inf
        for c in pool:
            if author_words.get(c.author, 0) + c.words > author_cap:
                continue
            diversity = min((_content_distance(c.content, x.content) for x in chosen), default=1.0)
            score = DIVERSITY_WEIGHT * diversity - c.typicality
            if score > best_score:
                best, best_score = c, score
        if best is None:
            break
        chosen.append(best)
        pool.remove(best)
        total += best.words
        author_words[best.author] = author_words.get(best.author, 0) + best.words
    return chosen


def render_exemplars(chosen: list[Candidate]) -> str:
    lines = ["# Exemplars", "", "Verbatim excerpts chosen for typicality and diversity.", ""]
    for c in chosen:
        suffix = f" ({c.register})" if c.register else ""
        source = f" <{c.source}>" if c.source else ""
        lines.append(f"## {c.author}, _{c.title}_{suffix}{source}")
        lines.append("")
        lines.append(c.text.strip())
        lines.append("")
    return "\n".join(lines)


def build_exemplars(
    docs: list[tuple[str, str, str, str]],
    profile: MetricsProfile,
    *,
    target_words: int = 2000,
    max_share: float = 0.25,
    registers: dict[str, str] | None = None,
) -> str:
    """Select exemplars from (author, title, source, text) documents. With `registers`
    (document title -> register label), the word budget is split across registers in
    proportion to the documents' words, so every register the corpus contains is
    represented, not only the most typical one. Each heading names author, title, register
    and, when known, the source URL."""
    by_register: dict[str, list[Candidate]] = {}
    words_by_register: dict[str, int] = {}
    for author, title, source, text in docs:
        label = (registers or {}).get(title, "")
        cands = candidates_from(author, title, analyze(text), profile)
        for c in cands:
            c.register = label
            c.source = source
        by_register.setdefault(label, []).extend(cands)
        words_by_register[label] = words_by_register.get(label, 0) + len(text.split())
    total = sum(words_by_register.values()) or 1
    chosen: list[Candidate] = []
    for label, cands in sorted(by_register.items()):
        budget = max(int(target_words * words_by_register[label] / total), MIN_WORDS * 2)
        chosen.extend(select_exemplars(cands, target_words=budget, max_share=max_share))
    return render_exemplars(chosen)


def parse_exemplars(markdown: str) -> list[str]:
    """Return exemplar bodies (without headings) from exemplars.md; an exemplars file with
    no `## ` headings (for example one stripped for sharing) yields none."""
    if not any(line.startswith("## ") for line in markdown.splitlines()):
        return []
    bodies: list[str] = []
    current: list[str] = []
    for line in markdown.splitlines():
        if line.startswith("## "):
            if current:
                bodies.append("\n".join(current).strip())
            current = []
        elif line.startswith("# ") or (not current and not line.strip()):
            continue
        else:
            current.append(line)
    if current:
        bodies.append("\n".join(current).strip())
    return [b for b in bodies if b and not b.startswith("Verbatim excerpts")]


def _registers(markdown: str) -> list[str]:
    out = []
    for line in markdown.splitlines():
        if line.startswith("## "):
            start, end = line.rfind("(register"), line.rfind(")")
            out.append(line[start + 1 : end] if start > 0 and end > start else "")
    return out


class ExemplarIndex:
    """Exemplar bodies with their register, metric vector and content words, for choosing
    the three nearest a unit: cluster-proportionally and by lexical overlap, not metrics alone."""

    def __init__(self, voice: Voice):
        self.names = [n for n, s in voice.metrics.metrics.items() if s.std > 0]
        self.voice = voice
        self.bodies = parse_exemplars(voice.exemplars)
        self.registers = _registers(voice.exemplars)
        self.vectors = [
            _zvector(compute_all(analyze(b)), voice.metrics, self.names) for b in self.bodies
        ]
        self.contents = [_content(b) for b in self.bodies]

    @property
    def available(self) -> bool:
        return bool(self.bodies)

    def closest(self, paragraph_text: str, k: int = 3) -> list[str]:
        if not self.bodies:
            return []
        content = _content(paragraph_text)
        v = _zvector(compute_all(analyze(paragraph_text)), self.voice.metrics, self.names)
        scores = []
        for i in range(len(self.bodies)):
            overlap = len(content & self.contents[i]) / max(len(content | self.contents[i]), 1)
            metric = math.sqrt(
                sum((x - y) ** 2 for x, y in zip(v, self.vectors[i], strict=True))
            ) / max(len(self.names), 1)
            scores.append((-(overlap * 10.0) + metric, i))
        scores.sort()
        # One exemplar from each register first, in proportion to register size, then the
        # remaining nearest ones.
        chosen: list[int] = []
        seen_registers: set[str] = set()
        for _, i in scores:
            reg = self.registers[i] if i < len(self.registers) else ""
            if reg not in seen_registers:
                chosen.append(i)
                seen_registers.add(reg)
            if len(chosen) >= k:
                break
        for _, i in scores:
            if len(chosen) >= k:
                break
            if i not in chosen:
                chosen.append(i)
        return [self.bodies[i] for i in chosen[:k]]
