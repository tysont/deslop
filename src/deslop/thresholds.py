# ABOUTME: Per-metric bands and per-tell limits tuned so LLM-reference documents fail and
# ABOUTME: held-out corpus documents pass; also the composite distance and its rescaling.
from __future__ import annotations

import math
import random
from dataclasses import dataclass

from deslop.metrics import METRICS
from deslop.stats import Band, CalibrationStats, DocSamples, MetricsProfile, Thresholds

LLM_FAIL_TARGET = 0.95
HELDOUT_PASS_TARGET = 0.90
HELDOUT_FRACTION = 0.20
MIN_AUTHORS_FOR_SPLIT = 5
MIN_DOCS_FOR_SPLIT = 10
K_STEPS = (6.0, 4.0, 3.0, 2.5, 2.0, 1.5, 1.25, 1.0)  # band half-widths in within-author std
TELL_STEPS = (6.0, 4.0, 3.0, 2.0, 1.5, 1.0)  # multiples of the voice rate
TELL_FLOORS = (2.0, 1.5, 1.0, 0.75, 0.5)  # per 1k words, for tells the voice barely uses
K_FREE_MIN = 2.0  # the second, zero-cost tightening phase never goes below this
TELL_FREE_MIN = 2.0
METRIC_CAP = 3.0  # overshoots beyond this many band widths count no further
TELL_CAP = 2.0  # rates beyond this many limits count no further
DISTANCE_HELDOUT = 20.0  # where the 90th percentile of held-out documents lands
DISTANCE_LLM = 70.0  # where the median LLM-reference document lands


# ---------------------------------------------------------------- evaluation


@dataclass
class MetricDeviation:
    name: str
    value: float
    overshoot: float  # in band half-widths on the slop side; 0 when inside the band
    flagged: bool


def band_for(stat_mean: float, within_std: float, std: float, side: str, k: float) -> Band:
    width = within_std if within_std > 0 else std
    low = max(0.0, stat_mean - k * width) if side in ("low", "both") else None  # rates are >= 0
    high = stat_mean + k * width if side in ("high", "both") else None
    return Band(low=low, high=high, k=k)


def metric_deviation(name: str, value: float, band: Band, mean: float) -> MetricDeviation:
    overshoot = 0.0
    if band.high is not None and value > band.high:
        width = max(band.high - mean, 1e-9)
        overshoot = (value - band.high) / width
    elif band.low is not None and value < band.low:
        width = max(mean - band.low, 1e-9)
        overshoot = (band.low - value) / width
    return MetricDeviation(name=name, value=value, overshoot=overshoot, flagged=overshoot > 0)


def raw_distance(
    metric_values: dict[str, float],
    tell_rates: dict[str, float],
    thresholds: Thresholds,
    means: dict[str, float],
) -> float:
    """Composite before rescaling: mean capped band overshoot over slop-side metrics, plus
    mean capped rate-over-limit over tells, each half the weight."""
    devs = []
    for name, band in thresholds.metrics.items():
        if name not in metric_values:
            continue
        d = metric_deviation(name, metric_values[name], band, means.get(name, 0.0))
        devs.append(min(d.overshoot, METRIC_CAP) / METRIC_CAP)
    ratios = []
    for tid, limit in thresholds.tells.items():
        rate = tell_rates.get(tid, 0.0)
        ratios.append(min(rate / max(limit, 1e-9), TELL_CAP) / TELL_CAP)
    m = sum(devs) / len(devs) if devs else 0.0
    t = sum(ratios) / len(ratios) if ratios else 0.0
    return 0.5 * m + 0.5 * t


def passes(
    metric_values: dict[str, float], tell_rates: dict[str, float], thresholds: Thresholds
) -> bool:
    for name, band in thresholds.metrics.items():
        v = metric_values.get(name)
        if v is None:
            continue
        if (band.high is not None and v > band.high) or (band.low is not None and v < band.low):
            return False
    return all(tell_rates.get(tid, 0.0) <= limit for tid, limit in thresholds.tells.items())


def rescale(raw: float, stats: CalibrationStats | None) -> float:
    """Map a raw composite onto 0-100: linear from 0 to the 90th-percentile held-out document
    (DISTANCE_HELDOUT), linear from there to the median LLM-reference document (DISTANCE_LLM),
    then saturating toward 100 with the LLM 90th percentile landing near 89."""
    if stats is None or stats.raw_llm_median <= stats.raw_heldout_p90:
        return round(min(100.0, 100.0 * raw), 1)
    span = stats.raw_llm_median - stats.raw_heldout_p90
    if raw <= stats.raw_heldout_p90:
        value = DISTANCE_HELDOUT * raw / max(stats.raw_heldout_p90, 1e-9)
    elif raw <= stats.raw_llm_median:
        scale = (DISTANCE_LLM - DISTANCE_HELDOUT) / span
        value = DISTANCE_HELDOUT + scale * (raw - stats.raw_heldout_p90)
    else:
        tail = max(stats.raw_llm_p90 - stats.raw_llm_median, span * 0.5, 1e-9)
        value = DISTANCE_LLM + (100.0 - DISTANCE_LLM) * (
            1.0 - math.exp(-(raw - stats.raw_llm_median) / tail)
        )
    return round(max(0.0, min(100.0, value)), 1)


# ---------------------------------------------------------------- calibration


def split_heldout(docs: list[DocSamples], *, seed: int = 0) -> tuple[list[DocSamples], str]:
    """Hold out about 20% of documents by author. Small corpora use every document."""
    authors = sorted({d.author for d in docs})
    if len(authors) < MIN_AUTHORS_FOR_SPLIT or len(docs) < MIN_DOCS_FOR_SPLIT:
        return list(docs), "all documents (corpus too small to split by author)"
    rng = random.Random(seed)
    rng.shuffle(authors)
    chosen: set[str] = set()
    count = 0
    for a in authors:
        chosen.add(a)
        count += sum(1 for d in docs if d.author == a)
        if count >= HELDOUT_FRACTION * len(docs):
            break
    return [d for d in docs if d.author in chosen], "20% of documents held out by author"


def initial_thresholds(profile: MetricsProfile) -> Thresholds:
    th = Thresholds()
    for name, stat in profile.metrics.items():
        spec = METRICS.get(name)
        if spec is None or spec.slop_side == "none":
            continue
        th.metrics[name] = band_for(
            stat.mean, stat.within_std, stat.std, spec.slop_side, K_STEPS[0]
        )
    for tid, t in profile.tells.items():
        th.tells[tid] = max(t.rate_per_1k * TELL_STEPS[0], TELL_FLOORS[0])
    return th


def _rates(docs: list[DocSamples], thresholds: Thresholds) -> float:
    if not docs:
        return 0.0
    return sum(passes(d.metrics, d.tells, thresholds) for d in docs) / len(docs)


def calibrate_thresholds(
    profile: MetricsProfile,
    llm_docs: list[DocSamples],
    *,
    banned: set[str] | None = None,
    seed: int = 0,
    heldout_target: float = HELDOUT_PASS_TARGET,
) -> Thresholds:
    """Tune bands and limits: start loose, then tighten one criterion at a time, choosing
    the step that fails the most LLM-reference documents per held-out document lost, until
    LLM_FAIL_TARGET is met or no step keeps `heldout_target` (HELDOUT_PASS_TARGET; a small
    corpus asks for 1.0, every document of its own). A banned tell starts at the tightest
    floor and is relaxed, floor by floor, until the held-out documents pass."""
    heldout, method = split_heldout(profile.per_document, seed=seed)
    th = initial_thresholds(profile)
    banned = banned or set()
    for tid in banned:
        if tid in th.tells:
            th.tells[tid] = TELL_FLOORS[-1]
    floors = sorted(TELL_FLOORS)  # tightest first
    for tid in sorted(banned):
        while tid in th.tells and _rates(heldout, th) < heldout_target:
            looser = [f for f in floors if f > th.tells[tid]]
            if not looser:
                break
            th.tells[tid] = looser[0]

    # Candidate settings per criterion, loosest first.
    metric_steps: dict[str, list[Band]] = {}
    for name in th.metrics:
        stat = profile.metrics[name]
        side = METRICS[name].slop_side
        metric_steps[name] = [
            band_for(stat.mean, stat.within_std, stat.std, side, k) for k in K_STEPS
        ]
    tell_steps: dict[str, list[float]] = {}
    for tid, t in profile.tells.items():
        limits = sorted(
            {max(t.rate_per_1k * m, f) for m in TELL_STEPS for f in TELL_FLOORS}, reverse=True
        )
        tell_steps[tid] = limits
    position: dict[str, int] = {k: 0 for k in list(th.metrics) + list(th.tells)}

    def rates(t: Thresholds) -> tuple[float, float]:
        return _rates(heldout, t), 1.0 - _rates(llm_docs, t)

    heldout_pass, llm_fail = rates(th)
    while llm_fail < LLM_FAIL_TARGET:
        best: tuple[float, str, Thresholds, float, float] | None = None
        for key in position:
            steps = metric_steps.get(key) or tell_steps.get(key)
            if steps is None or position[key] + 1 >= len(steps):
                continue
            trial = th.model_copy(deep=True)
            nxt = steps[position[key] + 1]
            if key in trial.metrics:
                trial.metrics[key] = nxt  # type: ignore[assignment]
            else:
                trial.tells[key] = nxt  # type: ignore[assignment]
            hp, lf = rates(trial)
            if hp < heldout_target:
                continue
            gain = lf - llm_fail
            cost = heldout_pass - hp
            score = gain - 2.0 * cost
            if gain > 0 and (best is None or score > best[0]):
                best = (score, key, trial, hp, lf)
        if best is None:
            break
        _, key, th, heldout_pass, llm_fail = best
        position[key] += 1

    # Second phase: keep tightening wherever it costs no held-out document, so the margin
    # over the target grows and a text cannot pass by avoiding one tell alone. Bands stop at
    # K_FREE_MIN and limits at TELL_FREE_MIN times the voice rate: with few held-out
    # documents a free step is weak evidence, and tighter settings would fit the sample.
    changed = method.startswith("20%")  # only with a real split is a free step evidence
    while changed:
        changed = False
        for key in list(position):
            steps = metric_steps.get(key) or tell_steps.get(key)
            if steps is None or position[key] + 1 >= len(steps):
                continue
            nxt = steps[position[key] + 1]
            if isinstance(nxt, Band) and nxt.k < K_FREE_MIN:
                continue
            if isinstance(nxt, float) and nxt < max(
                TELL_FREE_MIN * profile.tells[key].rate_per_1k, TELL_FLOORS[2]
            ):
                continue
            trial = th.model_copy(deep=True)
            if key in trial.metrics:
                trial.metrics[key] = nxt  # type: ignore[assignment]
            else:
                trial.tells[key] = nxt  # type: ignore[assignment]
            hp, lf = rates(trial)
            if hp >= heldout_pass and lf >= llm_fail:
                th, heldout_pass, llm_fail = trial, hp, lf
                position[key] += 1
                changed = True

    raw_h = sorted(
        raw_distance(d.metrics, d.tells, th, {n: s.mean for n, s in profile.metrics.items()})
        for d in heldout
    )
    raw_l = sorted(
        raw_distance(d.metrics, d.tells, th, {n: s.mean for n, s in profile.metrics.items()})
        for d in llm_docs
    )
    th.stats = CalibrationStats(
        method=method,
        heldout_docs=len(heldout),
        heldout_authors=sorted({d.author for d in heldout}),
        llm_docs=len(llm_docs),
        heldout_pass_rate=round(heldout_pass, 3),
        llm_fail_rate=round(llm_fail, 3),
        raw_heldout_p90=raw_h[int(0.9 * (len(raw_h) - 1))] if raw_h else 0.0,
        raw_heldout_median=raw_h[len(raw_h) // 2] if raw_h else 0.0,
        raw_llm_median=raw_l[len(raw_l) // 2] if raw_l else 0.0,
        raw_llm_p90=raw_l[int(0.9 * (len(raw_l) - 1))] if raw_l else 0.0,
    )
    return th
