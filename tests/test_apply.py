# ABOUTME: Tests for the document-level apply loop with scripted providers: candidate sampling
# ABOUTME: in rotation, the locks, growth, primary texts, coherence, the four-item judge, fallback.
from __future__ import annotations

import json

from deslop.apply import DEFAULT_MODELS, JudgePlan, ProviderPool, apply, sections_of
from deslop.nlp import analyze
from deslop.providers.base import CachingProvider, Completion, Usage
from deslop.reference import PrimaryText


class Scripted(CachingProvider):
    """Answers by the prompt it sees. Rewrites prefix every paragraph with a marker so the
    candidate differs from the original while keeping every locked item; the document judge
    picks the marked text on every item (or the unmarked one when `loses`)."""

    def __init__(
        self,
        name: str,
        *,
        confidence: float = 0.9,
        loses: bool = False,
        lexical_break: bool = False,
        break_on: str = "",
        pad_words: int = 0,
        add_quote: str = "",
        drop_paragraph: bool = False,
    ):
        super().__init__(use_cache=False, cache_only=False)
        self.name = name
        self.default_model = f"{name}-model"
        self.confidence = confidence
        self.loses = loses
        self.lexical_break = lexical_break
        self.break_on = break_on
        self.pad_words = pad_words
        self.add_quote = add_quote
        self.drop_paragraph = drop_paragraph
        self.calls: list[tuple[str, str]] = []

    def _raw_complete(self, system: str, user: str, model: str, temperature: float) -> Completion:
        self.calls.append((system, user))
        usage = Usage(calls=1, input_tokens=10, output_tokens=5, cost_usd=0.01)
        if "list the propositions" in system:
            text = "1. The chapter withholds its answer.\n2. Abraham is told to take his son."
        elif "compare an original passage" in system:
            text = json.dumps(
                {"original": [{"n": 1, "kept": True}, {"n": 2, "kept": True}], "added": []}
            )
        elif "Answer four questions" in system:
            a = user.split("TEXT A:\n", 1)[1].split("\n\nTEXT B:", 1)[0]
            b = user.split("TEXT B:\n", 1)[1]
            marked = ("Plainly," in a, "Plainly," in b)
            if marked[0] == marked[1]:
                pick = "tie"
            else:
                pick = "A" if marked[0] else "B"
                if self.loses:
                    pick = "B" if pick == "A" else "A"
            text = json.dumps(
                {
                    "voice": pick,
                    "coherent": pick,
                    "tense_person": "tie",
                    "order": pick,
                    "confidence": self.confidence,
                    "note": "the marked text reads as the writer",
                }
            )
        elif "demanding editor" in system:
            para = user.split("PARAGRAPH:\n", 1)[-1]
            v = 5 if para.startswith(("Plainly,", "Rewritten:")) else 2
            text = json.dumps({"voice": v, "model_prose": 6 - v, "fidelity": 5, "note": "ok"})
        elif "write prose in one specific writer" in system:
            if "to rewrite\n" in user:  # edit mode template
                unit = user.split("to rewrite\n", 1)[-1].split("\n\n# What is wrong", 1)[0]
                text = "Rewritten: " + unit.strip()
            else:
                if "# The section to write now" in user:
                    source = user.split("# The section to write now", 1)[1]
                    source = source.split("\n", 1)[1].split("\n\n# Already written", 1)[0]
                else:
                    source = user.split("# The original, in full\n", 1)[1]
                    source = source.split("\n\n# What it asserts", 1)[0]
                paragraphs = [p.strip() for p in source.split("\n\n") if p.strip()]
                if self.drop_paragraph:
                    paragraphs = paragraphs[:1]
                text = "\n\n".join(p if p.startswith("#") else "Plainly, " + p for p in paragraphs)
                if self.lexical_break or (self.break_on and self.break_on in source):
                    text = text.replace("Abraham", "Isaac").replace("Genesis 22", "Exodus 3")
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


def pair(**kw) -> tuple[Scripted, Scripted, object]:
    a, b = Scripted("anthropic", **kw), Scripted("openai", **kw)
    return a, b, factory_for({"anthropic": a, "openai": b})


def test_rewrites_the_document_as_a_whole_and_judges_with_the_other_provider(
    fixture_voice, slop_text
):
    a, b, factory = pair()
    result = apply(slop_text, fixture_voice, n=3, models=DEFAULT_MODELS, provider_factory=factory)
    assert result.mode == "rewrite" and len(result.units) == 1
    unit = result.units[0]
    assert unit.kind == "document" and not unit.kept_original
    assert len(unit.propositions_by_paragraph) == len(analyze(slop_text).paragraphs)
    writers = [c.provider for c in unit.candidates if c.text]
    assert "anthropic" in writers and "openai" in writers, "candidates rotate across providers"
    chosen = unit.candidates[unit.chosen]
    assert chosen.accepted and chosen.judge_ok and chosen.win_rate is not None
    assert all(v.provider != chosen.provider for v in chosen.verdicts), "judge never the writer"
    assert chosen.vs_original is not None and chosen.vs_original.no_loss
    assert any(v.against.startswith("candidate") for c in unit.candidates for v in c.verdicts), (
        "round-robin between candidates a judge is clear of"
    )
    assert chosen.semantic is not None and chosen.semantic.ok
    assert chosen.semantic.checks[0].proposition.startswith("¶1 ")
    assert chosen.coherence is not None and chosen.coherence.ok
    assert result.coherence is not None and set(result.judge_items) == {
        "voice",
        "coherent",
        "tense_person",
        "order",
    }
    assert result.metric_table and any(r.band_high is not None for r in result.metric_table)
    assert result.text.count("Plainly,") == len(analyze(slop_text).paragraphs)
    assert "judge gate on" in result.judge_gate and result.stopped == "done"
    assert result.growth_pct > 0 and not result.unverified_additions
    assert "✓ ¶1 The chapter withholds its answer." in result.meaning_diff
    assert result.exemplars_used and result.cost.calls > 0
    # The writer's prompt holds the profile, the rules, the exemplars, the whole original and
    # the propositions by paragraph, and says nothing about metrics or sentence length.
    system, user = next(c for c in a.calls if "write prose" in c[0])
    assert "# The original, in full" in user and "Paragraph 1:" in user and "Exemplars" in user
    assert fixture_voice.profile.strip()[:40] in system
    instruction = user.split("# Instruction", 1)[1].lower()
    for banned in ("metric", "sentence length", "opener", "citation", "words"):
        assert banned not in instruction, banned


def test_long_documents_are_rewritten_in_sections_with_the_previous_rewrite_in_context(
    fixture_voice, slop_text
):
    sections = sections_of(analyze(slop_text), 150)
    assert len(sections) > 1 and sections[0].kind == f"section 1/{len(sections)}"
    assert sum(s.last - s.first + 1 for s in sections) == len(analyze(slop_text).paragraphs)
    a, b, factory = pair()
    result = apply(slop_text, fixture_voice, n=1, chunk_words=150, provider_factory=factory)
    assert [u.kind for u in result.units] == [s.kind for s in sections]
    assert all(not u.kept_original for u in result.units)
    assert result.text.count("Plainly,") == len(analyze(slop_text).paragraphs)
    later = [u for _, u in a.calls + b.calls if "# The section to write now" in u]
    assert later and any("Plainly," in u.split("# Already written", 1)[1] for u in later)
    assert any("What follows the section" in u for u in later)


def test_lexical_lock_keeps_the_original_and_the_retry_names_what_a_reader_noticed(
    fixture_voice, slop_text
):
    a, b, factory = pair(lexical_break=True)
    result = apply(slop_text, fixture_voice, n=2, provider_factory=factory)
    unit = result.units[0]
    assert unit.kept_original and "no candidate passed" in unit.reason
    assert any("missing" in c.reason for c in unit.candidates)
    assert len(unit.candidates) == 4, "n candidates, then one retry round"
    assert result.text == slop_text and result.stopped == "no_candidate_passed"
    assert result.coherence is None and result.judge_items == {}
    retry = [u for _, u in a.calls + b.calls if "What a reader noticed" in u]
    assert retry and "from the original" in retry[0]


def test_judge_loss_on_any_item_keeps_the_original(fixture_voice, slop_text):
    a, b, factory = pair(loses=True)
    result = apply(slop_text, fixture_voice, n=1, provider_factory=factory)
    unit = result.units[0]
    assert unit.kept_original and result.text == slop_text
    assert all("judge preferred the original" in c.reason for c in unit.candidates if c.text)
    assert all(c.judge_ok is False for c in unit.candidates if c.text)


def test_growth_limit_keeps_the_original(fixture_voice, slop_text):
    a, b, factory = pair(pad_words=120)
    result = apply(slop_text, fixture_voice, n=1, max_growth=15.0, provider_factory=factory)
    unit = result.units[0]
    assert unit.kept_original
    assert all("words against the original" in c.reason for c in unit.candidates if c.text)
    assert all(c.growth_pct and c.growth_pct > 15 for c in unit.candidates if c.text)


def test_coherence_failure_keeps_the_original(fixture_voice, slop_text):
    a, b, factory = pair(drop_paragraph=True)
    result = apply(slop_text, fixture_voice, n=1, provider_factory=factory)
    unit = result.units[0]
    assert unit.kept_original
    cand = next(c for c in unit.candidates if c.text)
    assert cand.coherence is not None and not cand.coherence.ok
    assert "paragraphs where the original has" in cand.reason


def test_a_failing_later_section_reverts_the_whole_document(fixture_voice, slop_text):
    a, b, factory = pair(break_on="In the end")
    result = apply(slop_text, fixture_voice, n=1, chunk_words=150, provider_factory=factory)
    assert result.text == slop_text and result.stopped == "no_candidate_passed"
    assert all(u.kept_original for u in result.units)
    assert any("reverted" in u.reason for u in result.units)
    assert any("no candidate passed" in u.reason for u in result.units)


def test_budget_stops_before_the_first_candidate(fixture_voice, slop_text):
    a, b, factory = pair()
    result = apply(slop_text, fixture_voice, n=1, budget=0.0, provider_factory=factory)
    assert result.stopped == "budget" and result.text == slop_text


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
    unit = result.units[0]
    assert not unit.kept_original and result.primary_texts == ["Genesis"]
    assert unit.candidates[unit.chosen].provider == "anthropic", "the fabricated quote lost"
    rejected = [c for c in unit.candidates if "not in the primary text" in c.reason]
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


def test_without_a_qualified_judge_the_original_is_kept(fixture_voice, slop_text):
    a, b, factory = pair()
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
        result = apply(slop_text, fixture_voice, n=1, provider_factory=factory)
    finally:
        fixture_voice.meta.judge_calibration = {}
    assert "OFF" in result.judge_gate and result.text == slop_text
    assert result.stopped == "no_judge" and result.cost.calls == 0
    assert all(u.kept_original and not u.candidates for u in result.units)
    assert "no provider qualifies as judge" in result.units[0].reason


def test_judge_provider_failure_keeps_the_original_without_crashing(fixture_voice, slop_text):
    from deslop.providers.base import ProviderError

    class Broken(Scripted):
        def _raw_complete(self, system, user, model, temperature):
            if "Answer four questions" in system or "compare an original passage" in system:
                raise ProviderError("credit_balance_exhausted")
            return super()._raw_complete(system, user, model, temperature)

    a, b = Broken("anthropic"), Broken("openai")
    result = apply(
        slop_text, fixture_voice, n=1, provider_factory=factory_for({"anthropic": a, "openai": b})
    )
    assert result.text == slop_text and result.stopped == "no_candidate_passed"
    reasons = [c.reason for u in result.units for c in u.candidates if c.text]
    assert reasons and all("unavailable" in r for r in reasons)


def test_edit_mode_uses_locks_and_iterates(fixture_voice, slop_text):
    a, b, factory = pair()
    result = apply(slop_text, fixture_voice, mode="edit", max_iter=1, provider_factory=factory)
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
