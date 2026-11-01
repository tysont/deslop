# ABOUTME: Tests for the apply loop with scripted providers: candidate sampling in rotation,
# ABOUTME: locks, length and primary-text gates, the comparative judge, fallback, edit mode.
from __future__ import annotations

import json

from deslop.apply import DEFAULT_MODELS, JudgePlan, ProviderPool, apply, rewrite_units
from deslop.nlp import analyze
from deslop.providers.base import CachingProvider, Completion, Usage
from deslop.reference import PrimaryText
from deslop.score import score


class Scripted(CachingProvider):
    """Answers by the system prompt it sees. Rewrites prepend a marker so the candidate
    differs from the original while keeping every locked item; the comparative judge picks
    marked text as the writer's with `confidence`."""

    def __init__(
        self,
        name: str,
        *,
        confidence: float = 0.9,
        lexical_break: bool = False,
        pad_words: int = 0,
        add_quote: str = "",
    ):
        super().__init__(use_cache=False, cache_only=False)
        self.name = name
        self.default_model = f"{name}-model"
        self.confidence = confidence
        self.lexical_break = lexical_break
        self.pad_words = pad_words
        self.add_quote = add_quote
        self.calls: list[tuple[str, str]] = []

    def _raw_complete(self, system: str, user: str, model: str, temperature: float) -> Completion:
        self.calls.append((system[:30], user[:80]))
        usage = Usage(calls=1, input_tokens=10, output_tokens=5)
        if "list the propositions" in system:
            text = "1. The chapter withholds its answer.\n2. Abraham is told to take his son."
        elif "compare an original passage" in system:
            text = json.dumps(
                {"original": [{"n": 1, "kept": True}, {"n": 2, "kept": True}], "added": []}
            )
        elif "two paragraphs, A and B" in system:
            a = user.split("PARAGRAPH A:\n", 1)[1].split("\n\nPARAGRAPH B:", 1)[0]
            b = user.split("PARAGRAPH B:\n", 1)[1]
            marked = ("Plainly," in a[:12] or "Rewritten:" in a[:12], "Plainly," in b[:12])
            pick = (
                "A" if marked[0] and not marked[1] else "B" if marked[1] and not marked[0] else "A"
            )
            text = json.dumps(
                {
                    "more_like_writer": pick,
                    "more_like_model": "B" if pick == "A" else "A",
                    "confidence": self.confidence if marked[0] != marked[1] else 0.55,
                    "note": "marked text reads as the writer",
                }
            )
        elif "demanding editor" in system:
            para = user.split("PARAGRAPH:\n", 1)[-1]
            v = 5 if para.startswith(("Plainly,", "Rewritten:")) else 2
            text = json.dumps({"voice": v, "model_prose": 6 - v, "fidelity": 5, "note": "ok"})
        elif "write prose in one specific writer" in system:
            unit = user.split("to write afresh\n", 1)[-1].split("\n\n# Its propositions", 1)[0]
            if "to rewrite\n" in user:  # edit mode template
                unit = user.split("to rewrite\n", 1)[-1].split("\n\n# What is wrong", 1)[0]
                text = "Rewritten: " + unit.strip()
            else:
                text = "Plainly, " + unit.strip()
            if self.lexical_break:
                text = text.replace("Abraham", "Isaac")
            if self.pad_words:
                text += " " + " ".join(["more"] * self.pad_words)
            if self.add_quote:
                text += f' As the text says, "{self.add_quote}" (Gen 22:2).'
        else:
            text = "?"
        return Completion(text=text, model=model, provider=self.name, usage=usage)


def factory_for(providers: dict[str, Scripted]):
    aliases = {"claude": "anthropic", "gpt": "openai"}

    def factory(name: str) -> CachingProvider:
        return providers[aliases.get(name, name)]

    return factory


def test_rewrite_mode_samples_in_rotation_and_judges_with_the_other_provider(
    fixture_voice, slop_text
):
    a = Scripted("anthropic")
    b = Scripted("openai")
    result = apply(
        slop_text,
        fixture_voice,
        mode="rewrite",
        n=3,
        models=DEFAULT_MODELS,
        provider_factory=factory_for({"anthropic": a, "openai": b}),
    )
    assert result.mode == "rewrite" and result.units
    first = next(u for u in result.units if not u.kept_original)
    writers = [c.provider for c in first.candidates if c.text]
    assert "anthropic" in writers and "openai" in writers, "candidates rotate across providers"
    chosen = first.candidates[first.chosen]
    assert chosen.accepted and chosen.beats_original and chosen.win_rate is not None
    assert all(p.provider != chosen.provider for p in chosen.pairs), "judge never the writer"
    assert any(p.against == "original" for p in chosen.pairs)
    assert any(p.against.startswith("candidate") for c in first.candidates for p in c.pairs), (
        "round-robin between candidates a judge is clear of"
    )
    assert chosen.judge is not None and chosen.judge.provider != chosen.provider
    assert chosen.semantic is not None and chosen.semantic.ok
    assert result.text.count("Plainly,") >= 1
    assert "judge gate on" in result.judge_gate
    assert result.growth_pct > 0 and not result.unverified_additions
    assert "✓ The chapter withholds its answer." in result.meaning_diff
    assert result.exemplars_used and result.cost.calls > 0
    assert result.report_after.distance <= result.report_before.distance or result.text != slop_text


def test_lexical_lock_rejects_and_original_is_kept(fixture_voice, slop_text):
    a = Scripted("anthropic", lexical_break=True)
    b = Scripted("openai", lexical_break=True)
    result = apply(
        slop_text,
        fixture_voice,
        n=2,
        provider_factory=factory_for({"anthropic": a, "openai": b}),
    )
    unit = next(u for u in result.units if "Abraham" in u.original)
    assert unit.kept_original and "no candidate passed" in unit.reason
    assert any("content lock" in c.reason for c in unit.candidates)
    assert len(unit.candidates) == 4, "n candidates, then one retry round"
    assert unit.original in result.text


def test_judge_gate_keeps_original_when_confidence_is_low(fixture_voice, slop_text):
    a = Scripted("anthropic", confidence=0.6)
    b = Scripted("openai", confidence=0.6)
    result = apply(
        slop_text,
        fixture_voice,
        n=1,
        min_confidence=0.7,
        provider_factory=factory_for({"anthropic": a, "openai": b}),
    )
    assert all(u.kept_original for u in result.units)
    assert all(
        "beat the original only at 0.60" in c.reason
        for u in result.units
        for c in u.candidates
        if c.text
    )


def test_growth_limit_rejects_long_candidates(fixture_voice, slop_text):
    a = Scripted("anthropic", pad_words=40)
    b = Scripted("openai", pad_words=40)
    result = apply(
        slop_text,
        fixture_voice,
        n=1,
        max_growth=15.0,
        provider_factory=factory_for({"anthropic": a, "openai": b}),
    )
    assert all(u.kept_original for u in result.units)
    reasons = [c.reason for u in result.units for c in u.candidates if c.text]
    assert reasons and all(r.startswith("length:") for r in reasons)
    assert all(
        c.growth_pct and c.growth_pct > 15 for u in result.units for c in u.candidates if c.text
    )


def test_primary_text_rejects_fabricated_quotation_and_lists_unverified(
    fixture_voice, slop_text, tmp_path
):
    marked = tmp_path / "gen.txt"
    marked.write_text(
        "# title: Genesis\n# book: Genesis\n\n{{Genesis 22:2}} Take now thy son, thine only "
        "son Isaac, whom thou lovest.\n",
        encoding="utf-8",
    )
    a = Scripted("anthropic", add_quote="Take now thy son, thine only son Isaac")
    b = Scripted("openai", add_quote="Take now thy daughter")
    providers = factory_for({"anthropic": a, "openai": b})
    result = apply(
        slop_text, fixture_voice, n=2, primary_texts=[str(marked)], provider_factory=providers
    )
    rewritten = [u for u in result.units if not u.kept_original]
    assert rewritten and result.primary_texts == ["Genesis"]
    for u in rewritten:
        assert u.candidates[u.chosen].provider == "anthropic", "the fabricated quote lost"
        rejected = [c for c in u.candidates if c.reason.startswith("primary text:")]
        assert rejected and "Take now thy daughter" in rejected[0].reason
    assert not result.unverified_additions
    # Without primary texts the same additions pass and are listed.
    loose = apply(slop_text, fixture_voice, n=1, provider_factory=providers)
    assert loose.unverified_additions and "Gen 22:2" in "".join(loose.unverified_additions)
    assert "unverified additions" in loose.meaning_diff
    assert PrimaryText.load(marked).resolve("Gen 22:2") == "Genesis 22:2"


def test_judge_plan_drops_uncalibrated_judges_and_turns_the_gate_off():
    a, b = Scripted("anthropic"), Scripted("openai")
    pool = ProviderPool(DEFAULT_MODELS, factory_for({"anthropic": a, "openai": b}))
    good = {
        "provider": "x",
        "model": "m",
        "pairs": 10,
        "accuracy": 0.9,
        "model_prose_accuracy": 0.9,
        "mean_confidence": 0.8,
    }
    bad = dict(good, accuracy=0.6)
    plan = JudgePlan(pool, {"anthropic": bad, "openai": good})
    assert plan.enabled and [j.name for j in plan.judges] == ["openai"]
    assert plan.writers == ["claude"], "only the provider the judge is clear of may write"
    assert plan.judges_for("anthropic")[0].name == "openai" and not plan.judges_for("openai")
    off = JudgePlan(pool, {"anthropic": bad, "openai": bad})
    assert not off.enabled and off.writers == ["claude", "gpt"] and "OFF" in off.describe()
    trusted = JudgePlan(pool, {})
    assert trusted.enabled and len(trusted.judges) == 2 and "uncalibrated" in trusted.describe()


def test_gate_off_accepts_lock_passing_candidates(fixture_voice, slop_text):
    a, b = Scripted("anthropic"), Scripted("openai")
    bad = {
        "provider": "x",
        "model": "m",
        "pairs": 10,
        "accuracy": 0.5,
        "model_prose_accuracy": 0.5,
        "mean_confidence": 0.6,
    }
    fixture_voice.meta.judge_calibration = {"anthropic": bad, "openai": bad}
    try:
        result = apply(
            slop_text,
            fixture_voice,
            n=1,
            provider_factory=factory_for({"anthropic": a, "openai": b}),
        )
    finally:
        fixture_voice.meta.judge_calibration = {}
    assert "OFF" in result.judge_gate
    assert any(not u.kept_original for u in result.units)
    assert all(c.pairs == [] for u in result.units for c in u.candidates)


def test_rewrite_units_merge_paragraphs_that_need_room(fixture_voice, slop_text):
    report = score(slop_text, fixture_voice)
    units = rewrite_units(slop_text, analyze(slop_text), report)
    assert units and all(u.kind in ("paragraph", "section") for u in units)
    assert sum(u.text.count("\n\n") + 1 for u in units) == len(analyze(slop_text).paragraphs)


def test_edit_mode_uses_locks_and_iterates(fixture_voice, slop_text):
    a = Scripted("anthropic")
    b = Scripted("openai")
    result = apply(
        slop_text,
        fixture_voice,
        mode="edit",
        max_iter=1,
        provider_factory=factory_for({"anthropic": a, "openai": b}),
    )
    assert result.mode == "edit" and result.iterations and result.iterations[0].units > 0
    assert any(c.accepted for u in result.units for c in u.candidates)
    # The result is the best iteration by distance, so a rewrite only shows up when it helped.
    assert result.text == slop_text or "Rewritten:" in result.text


def test_brief_is_drafted_then_rewritten(fixture_voice, fixtures_dir):
    from deslop.draft import load_brief

    class Drafting(Scripted):
        def _raw_complete(self, system, user, model, temperature):
            if "from a brief" in system:
                return Completion(
                    text="Abraham climbs the mountain in Gen 22:1-3. The chapter withholds its "
                    "answer and the reader waits with him. Nothing is explained.",
                    model=model,
                    provider=self.name,
                    usage=Usage(calls=1),
                )
            return super()._raw_complete(system, user, model, temperature)

    a, b = Drafting("anthropic"), Drafting("openai")
    brief = load_brief(fixtures_dir / "brief_sample.yaml")
    result = apply(
        brief, fixture_voice, n=1, provider_factory=factory_for({"anthropic": a, "openai": b})
    )
    assert result.draft is not None and len(result.draft) == len(brief.sections)
    assert result.draft_text and "Abraham" in result.draft_text
    assert result.mode == "rewrite"
