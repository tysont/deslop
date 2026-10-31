# ABOUTME: Tests for the apply loop with scripted providers: candidate sampling in rotation,
# ABOUTME: lock and judge gating, fallback to the original, edit mode, and the meaning diff.
from __future__ import annotations

import json

from deslop.apply import DEFAULT_MODELS, apply, rewrite_units
from deslop.nlp import analyze
from deslop.providers.base import CachingProvider, Completion, Usage
from deslop.score import score


class Scripted(CachingProvider):
    """Answers by the system prompt it sees. Rewrites prepend a marker so the candidate
    differs from the original while keeping every locked item; the judge likes marked text."""

    def __init__(self, name: str, *, judge_high: float = 5, lexical_break: bool = False):
        super().__init__(use_cache=False, cache_only=False)
        self.name = name
        self.default_model = f"{name}-model"
        self.judge_high = judge_high
        self.lexical_break = lexical_break
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
        elif "demanding editor" in system:
            para = user.split("PARAGRAPH:\n", 1)[-1]
            high = para.startswith("Plainly,") or para.startswith("Rewritten:")
            v = self.judge_high if high else 2
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
    assert chosen.accepted and chosen.judge is not None and chosen.judge.provider != chosen.provider
    assert chosen.semantic is not None and chosen.semantic.ok
    assert result.text.count("Plainly,") >= 1
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


def test_judge_gate_keeps_original_when_scores_are_low(fixture_voice, slop_text):
    a = Scripted("anthropic", judge_high=3)
    b = Scripted("openai", judge_high=3)
    result = apply(
        slop_text,
        fixture_voice,
        n=1,
        judge_min=4.0,
        provider_factory=factory_for({"anthropic": a, "openai": b}),
    )
    assert all(u.kept_original for u in result.units)
    assert all("judge" in c.reason for u in result.units for c in u.candidates if c.text)


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
