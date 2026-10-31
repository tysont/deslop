# ABOUTME: Tests for the conversion loop using a scripted fake provider (no network).
# ABOUTME: Covers unit selection, validation, retry with the other provider, splicing, pairs log, budget.
import json

import pytest

from deslop.convert import (
    ProviderPool,
    Unit,
    clean_response,
    convert,
    length_bounds,
    select_units,
    validate,
)
from deslop.nlp import analyze
from deslop.patterns import load_catalog
from deslop.providers.base import CachingProvider, Completion, ProviderError, Usage
from deslop.score import score


class FakeProvider(CachingProvider):
    """Returns scripted rewrites keyed by the unit text found in the prompt."""

    def __init__(self, name, script, cost=0.01, fail=False):
        super().__init__(use_cache=False, max_retries=1)
        self.name = name
        self.default_model = f"{name}-fake"
        self.script = script
        self.cost = cost
        self.fail = fail
        self.prompts = []

    def _raw_complete(self, system, user, model, temperature):
        if self.fail:
            raise RuntimeError("down")
        self.prompts.append((system, user))
        unit = user.split(" to rewrite\n", 1)[1].split("\n\n# What is wrong", 1)[0].strip()
        text = self.script.get(unit, unit)
        return Completion(
            text=text,
            model=model,
            provider=self.name,
            usage=Usage(input_tokens=100, output_tokens=20, cost_usd=self.cost, calls=1),
        )


@pytest.fixture
def catalog():
    return load_catalog()


def unit_for(text, kind="sentence"):
    return Unit(
        index=0,
        kind=kind,
        start=0,
        end=len(text),
        text=text,
        para_index=0,
        hits=[],
        metric_notes=[],
    )


def test_clean_response():
    assert clean_response('"Quoted rewrite."') == "Quoted rewrite."
    assert clean_response("```\nFenced.\n```") == "Fenced."
    assert clean_response('She said "hi" and "bye".') == 'She said "hi" and "bye".'


def test_length_bounds():
    u = unit_for("one two three four five six seven eight nine ten")
    assert length_bounds(u, 0.15) == (4, 17)
    p = unit_for(" ".join(["w"] * 100), kind="paragraph")
    assert length_bounds(p, 0.15) == (85, 116)


def test_validate_rejections(catalog):
    u = unit_for("The chapter asks what Abraham owes in Gen 22.")
    kw = dict(
        context_before="",
        context_after="",
        lock_classes=("citations", "numbers", "proper_nouns"),
        preserve_length=0.15,
        catalog=catalog,
    )
    assert validate(u, u.text, **kw) == "empty or unchanged"
    assert validate(u, "Short.", **kw).startswith("length")
    assert validate(u, "The text asks what Isaac owes in Gen 22.", **kw).startswith("content lock")
    assert validate(u, "The narrator refuses to say what Abraham owes in Gen 22.", **kw) is None
    assert (
        validate(u, "What does Abraham owe in Gen 22?\nNobody says.", **kw)
        == "contains line breaks"
    )
    assert validate(
        u, "The chapter asks, and the text insists, what Abraham owes in Gen 22.", **kw
    ).startswith("new tells")


def test_validate_rejects_repeated_context(catalog):
    ctx = "This is the preceding sentence which must not be repeated by the model at all."
    u = unit_for("The chapter asks what Abraham owes.")
    reason = validate(
        u,
        ctx + " Abraham owes a great deal.",
        context_before=ctx,
        context_after="",
        lock_classes=("proper_nouns",),
        preserve_length=5.0,
        catalog=catalog,
    )
    assert reason == "repeated the surrounding context"


def test_select_units_edit_and_rewrite(fixture_voice, slop_text):
    report = score(slop_text, fixture_voice)
    a = analyze(slop_text)
    edit = select_units(slop_text, a, report, "edit")
    rewrite = select_units(slop_text, a, report, "rewrite")
    assert all(u.kind == "sentence" for u in edit) and len(edit) >= 8
    assert all(u.kind == "paragraph" for u in rewrite) and len(rewrite) == report.paragraphs_count
    assert all(slop_text[u.start : u.end] == u.text for u in edit + rewrite)
    assert any("T1" in u.tell_ids for u in edit)
    assert all(u.metric_notes for u in edit)  # slop sample has flagged metrics


def test_select_units_metric_only_falls_back_to_similar_run(fixture_voice, heldout_texts):
    report = score(heldout_texts[1], fixture_voice, sigma=0.01, tell_mult=1000, tell_floor=1000)
    assert report.flagged_metrics() and not report.flagged_tells()
    a = analyze(heldout_texts[1])
    units = select_units(heldout_texts[1], a, report, "edit")
    assert units and all(u.metric_notes for u in units) and all(not u.hits for u in units)


def test_provider_pool_alternates_and_skips_unavailable():
    good = FakeProvider("claude", {})

    def factory(name):
        if name == "gpt":
            raise ProviderError("no key")
        return good

    pool = ProviderPool(("claude", "gpt"), factory)
    assert pool.for_iteration(0) is good
    assert pool.for_iteration(1) is good  # gpt unavailable, falls back
    assert pool.other_than(good) is good
    assert "gpt" in pool.unavailable


def test_convert_pass_returns_unchanged(fixture_voice, heldout_texts, tmp_path):
    calls = FakeProvider("claude", {})
    res = convert(
        heldout_texts[0],
        fixture_voice,
        provider_factory=lambda n: calls,
        pairs_path=tmp_path / "p.jsonl",
    )
    assert res.stopped == "pass" and res.text == heldout_texts[0] and res.iterations == []
    assert res.cost.calls == 0 and not calls.prompts


def test_convert_loop_with_scripted_rewrites(fixture_voice, tmp_path):
    text = (
        "Abraham climbs the mountain with Isaac in Genesis 22. The chapter asks what he owes. "
        "The narrator refuses to look inside him. He builds the altar and binds his son.\n\n"
        "Sarah is absent from the whole account. The text insists on her silence. "
        "Isaac carries the wood up the slope while Abraham carries the knife and the fire."
    )
    script = {
        "The chapter asks what he owes.": "What he owes is never said.",
        "The narrator refuses to look inside him.": "We are never shown what he feels.",
        "The text insists on her silence.": "Her silence is the fact the account leaves us with.",
    }
    claude = FakeProvider("claude", script, cost=0.02)
    gpt = FakeProvider("gpt", script, cost=0.03)
    providers = {"claude": claude, "gpt": gpt}
    pairs = tmp_path / "pairs.jsonl"
    res = convert(
        text, fixture_voice, provider_factory=lambda n: providers[n], pairs_path=pairs, max_iter=2
    )
    assert res.report_before.verdict == "FAIL"
    assert "chapter asks" not in res.text
    assert (
        "Abraham climbs the mountain with Isaac in Genesis 22." in res.text
    )  # unflagged, byte-identical
    assert res.text.count("\n\n") == 1
    accepted = [u for u in res.units if u.accepted]
    assert accepted and all(u.before in script for u in accepted)
    assert res.cost.calls == len(res.units) and res.cost.usd > 0
    assert res.iterations[0].provider == "claude"
    assert "--- before" in res.diff and "+++ after" in res.diff
    records = [json.loads(line) for line in pairs.read_text().splitlines()]
    assert len(records) == len(res.units) and records[0]["voice"] == "fixture"
    assert res.report_after.distance <= res.report_before.distance


def test_convert_retries_with_other_provider_then_keeps_original(fixture_voice, tmp_path):
    text = "Abraham climbs the hill in Genesis 22. The chapter asks what he owes. He builds the altar there."
    bad = {
        "The chapter asks what he owes.": "The chapter asks what Isaac owes."
    }  # adds a name: lock violation
    claude = FakeProvider("claude", bad)
    gpt = FakeProvider("gpt", bad)
    providers = {"claude": claude, "gpt": gpt}
    res = convert(
        text,
        fixture_voice,
        provider_factory=lambda n: providers[n],
        pairs_path=tmp_path / "p.jsonl",
        max_iter=1,
    )
    assert res.text == text
    reasons = [u.reason for u in res.units]
    assert len(reasons) == 2 and all(r.startswith("content lock") for r in reasons)
    assert [u.provider for u in res.units] == ["claude", "gpt"]
    assert res.stopped == "no_progress"


def test_convert_provider_failure_falls_through(fixture_voice, tmp_path):
    text = "Abraham climbs the hill in Genesis 22. The chapter asks what he owes. He builds the altar there."
    script = {"The chapter asks what he owes.": "What he owes, no one explains."}
    down = FakeProvider("claude", script, fail=True)
    gpt = FakeProvider("gpt", script)
    providers = {"claude": down, "gpt": gpt}
    res = convert(
        text,
        fixture_voice,
        provider_factory=lambda n: providers[n],
        pairs_path=tmp_path / "p.jsonl",
        max_iter=1,
    )
    assert any(u.reason.startswith("provider error") for u in res.units)
    assert any(u.accepted and u.provider == "gpt" for u in res.units)


def test_convert_budget_stops_early(fixture_voice, slop_text, tmp_path):
    claude = FakeProvider("claude", {}, cost=1.0)
    res = convert(
        slop_text,
        fixture_voice,
        provider_factory=lambda n: claude,
        pairs_path=tmp_path / "p.jsonl",
        budget=0.5,
        max_iter=3,
    )
    assert res.stopped == "budget"
    assert res.cost.calls <= 2
    assert res.text == slop_text


def test_convert_rewrite_mode_units_are_paragraphs(fixture_voice, tmp_path):
    text = (
        "Abraham climbs the mountain with Isaac in Genesis 22. The chapter asks what he owes.\n\n"
        "Sarah waits at home in Beersheba. She says nothing at all about it."
    )
    para = "Abraham climbs the mountain with Isaac in Genesis 22. The chapter asks what he owes."
    script = {
        para: "In Genesis 22 Abraham climbs the mountain with Isaac, and what he owes is never said."
    }
    claude = FakeProvider("claude", script)
    res = convert(
        text,
        fixture_voice,
        mode="rewrite",
        provider_factory=lambda n: claude,
        pairs_path=tmp_path / "p.jsonl",
        max_iter=1,
    )
    assert res.units[0].kind == "paragraph"
    assert res.text.endswith("Sarah waits at home in Beersheba. She says nothing at all about it.")
    assert res.text.startswith("In Genesis 22 Abraham climbs")
