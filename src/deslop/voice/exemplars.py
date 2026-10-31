# ABOUTME: Chooses verbatim corpus excerpts that are typical of the voice (near the metric
# ABOUTME: centroid) and diverse (far from each other), with a per-author cap.
from __future__ import annotations

import math
from dataclasses import dataclass

from deslop.metrics import compute_all
from deslop.nlp import WORD_RE, Analysis, analyze
from deslop.stats import MetricsProfile, subset_analysis

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
        lines.append(f"## {c.author}, _{c.title}_{suffix}")
        lines.append("")
        lines.append(c.text.strip())
        lines.append("")
    return "\n".join(lines)


def build_exemplars(
    docs: list[tuple[str, str, str]],
    profile: MetricsProfile,
    *,
    target_words: int = 2000,
    max_share: float = 0.25,
    registers: dict[str, str] | None = None,
) -> str:
    """Select exemplars. With `registers` (document title -> register label), the word
    budget is split across registers in proportion to the documents' words, so every
    register the corpus contains is represented, not only the most typical one."""
    by_register: dict[str, list[Candidate]] = {}
    words_by_register: dict[str, int] = {}
    for author, title, text in docs:
        label = (registers or {}).get(title, "")
        cands = candidates_from(author, title, analyze(text), profile)
        for c in cands:
            c.register = label
        by_register.setdefault(label, []).extend(cands)
        words_by_register[label] = words_by_register.get(label, 0) + len(text.split())
    total = sum(words_by_register.values()) or 1
    chosen: list[Candidate] = []
    for label, cands in sorted(by_register.items()):
        budget = max(int(target_words * words_by_register[label] / total), MIN_WORDS * 2)
        chosen.extend(select_exemplars(cands, target_words=budget, max_share=max_share))
    return render_exemplars(chosen)


def parse_exemplars(markdown: str) -> list[str]:
    """Return exemplar bodies (without headings) from exemplars.md."""
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
