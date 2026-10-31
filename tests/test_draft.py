# ABOUTME: Tests for draft(brief, voice): section generation, verbatim refs and quotes,
# ABOUTME: stripping of unlisted citations and quotations, assembly, and the convert hand-off.
from pathlib import Path

import pytest

from deslop.draft import Brief, DraftResult, draft, load_brief, strip_unlisted
from deslop.providers.base import CachingProvider, Completion, Usage

FIXTURES = Path(__file__).parent / "fixtures"


class SectionProvider(CachingProvider):
    """Answers drafting prompts with scripted section text keyed by heading."""

    def __init__(self, script):
        super().__init__(use_cache=False, max_retries=1)
        self.name = "claude"
        self.default_model = "fake"
        self.script = script
        self.prompts = []

    def _raw_complete(self, system, user, model, temperature):
        self.prompts.append(user)
        heading = user.split("# Section: ", 1)[1].splitlines()[0].strip()
        attempt = sum(1 for p in self.prompts if f"# Section: {heading}" in p)
        text = self.script[heading]
        if isinstance(text, list):
            text = text[min(attempt, len(text)) - 1]
        return Completion(
            text=text,
            model=model,
            provider=self.name,
            usage=Usage(input_tokens=50, output_tokens=50, cost_usd=0.01, calls=1),
        )


@pytest.fixture
def brief() -> Brief:
    return load_brief(FIXTURES / "brief_sample.yaml")


def test_load_brief_parses_sections(brief):
    assert brief.title.startswith("What Johnson")
    assert [s.heading for s in brief.sections] == [
        "General nature",
        "No heroes",
        "The stream of time",
    ]
    assert brief.sections[0].refs == ["(Johnson 1765)"]
    assert brief.sections[0].quotes[0].startswith("Nothing can please many")
    assert brief.words_per_section == 180


def test_load_brief_rejects_missing_claim(tmp_path):
    bad = tmp_path / "b.yaml"
    bad.write_text("title: x\nsections:\n  - heading: h\n")
    with pytest.raises(ValueError, match="claim"):
        load_brief(bad)


def test_strip_unlisted_removes_foreign_quotes_and_citations():
    text = (
        'Johnson wrote that "just representations of general nature" please long (Johnson 1765). '
        'Pope had said "true wit is nature to advantage dressed" (Pope 1711). '
        "Nobody disputes the first claim."
    )
    out, stripped = strip_unlisted(
        text, refs=["(Johnson 1765)"], quotes=["just representations of general nature"]
    )
    assert "Pope" not in out and "(Pope 1711)" not in out
    assert "(Johnson 1765)" in out and "Nobody disputes the first claim." in out
    assert sorted(stripped) == ["(Pope 1711)", "true wit is nature to advantage dressed"]


def test_strip_unlisted_drops_bare_citation_but_keeps_sentence():
    text = "The point is made at Genesis 22:1 and again later. Then nothing."
    out, stripped = strip_unlisted(text, refs=[], quotes=[])
    assert stripped == ["Genesis 22:1"]
    assert out == "The point is made at and again later. Then nothing."


def test_draft_assembles_sections_strips_and_requires_verbatim(fixture_voice, brief, tmp_path):
    q1, q2, q3 = (s.quotes[0] for s in brief.sections)
    script = {
        "General nature": (
            f'Johnson begins with a plain test. "{q1}" That is the whole claim (Johnson 1765). '
            'He adds, "as Dryden also said" (Dryden 1668), that nothing else lasts.'
        ),
        "No heroes": [
            "A first attempt that forgets the quotation and the reference entirely.",
            f'Johnson is blunt: "{q2}" (Johnson 1765). They talk as we would.',
        ],
        "The stream of time": f'"{q3}" So Johnson writes (Johnson 1765), and he means it.',
    }
    provider = SectionProvider(script)
    res = draft(
        brief,
        fixture_voice,
        provider_factory=lambda n: provider,
        convert_max_iter=0,
        pairs_path=tmp_path / "pairs.jsonl",
    )
    assert isinstance(res, DraftResult)
    assert res.text.startswith("# What Johnson Asked of Shakespeare\n\n## General nature\n\n")
    assert "## No heroes" in res.text and "## The stream of time" in res.text
    assert "Dryden" not in res.text and "(Dryden 1668)" not in res.text
    assert "as Dryden also said" not in res.text
    assert res.sections[0].stripped == ["(Dryden 1668)", "as Dryden also said"]
    assert res.sections[1].attempts == 2 and q2 in res.text
    assert all(s.complete for s in res.sections)
    assert res.cost.calls == 4
    assert res.convert.iterations == [] and res.report.verdict in ("PASS", "FAIL")
    # every section prompt carries the claim, refs and quotes, and the voice profile
    assert all("(Johnson 1765)" in p and "# Section: " in p for p in provider.prompts)


def test_draft_reports_incomplete_section_after_two_attempts(fixture_voice, brief, tmp_path):
    script = {s.heading: "Never uses what it was given." for s in brief.sections}
    res = draft(
        brief,
        fixture_voice,
        provider_factory=lambda n: SectionProvider(script),
        convert_max_iter=0,
        pairs_path=tmp_path / "pairs.jsonl",
    )
    assert all(not s.complete and s.attempts == 2 for s in res.sections)
    assert res.sections[0].missing
