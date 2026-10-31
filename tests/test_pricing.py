# ABOUTME: Tests for the pricing table: packaged pricing.yaml, user override file, prefix matching,
# ABOUTME: and the unpriced flag for models missing from the table.
import pytest

from deslop import pricing


@pytest.fixture(autouse=True)
def fresh_table(monkeypatch, tmp_path):
    monkeypatch.setenv("DESLOP_HOME", str(tmp_path))
    pricing.load_pricing.cache_clear()
    yield
    pricing.load_pricing.cache_clear()


def test_packaged_table_prices_every_default_model():
    table = pricing.load_pricing()
    for model in ("claude-opus-5-5", "claude-sonnet-5-5", "gpt-5.5", "gpt-4.1"):
        assert pricing.is_priced(model), model
    assert table.models["gpt-5.5"] == (5.0, 30.0)
    assert table.models["claude-opus-5-5"] == (4.0, 20.0)
    assert table.sources and table.fetched


def test_price_uses_longest_matching_prefix():
    assert pricing.price("gpt-5.5-2026-01-01", 1_000_000, 0) == 5.0
    assert pricing.price("gpt-5-mini-2025-08-07", 1_000_000, 1_000_000) == 0.25 + 2.0
    assert pricing.price("claude-sonnet-5-5-20260101", 0, 1_000_000) == 10.0


def test_unknown_model_is_unpriced_and_costs_zero():
    assert not pricing.is_priced("mystery-9")
    assert pricing.price("mystery-9", 1000, 1000) == 0.0


def test_user_override_file_adds_and_replaces(tmp_path):
    (tmp_path / "pricing.yaml").write_text(
        "models:\n  mystery-9: {input: 1, output: 2}\n  gpt-5.5: {input: 100, output: 200}\n"
    )
    pricing.load_pricing.cache_clear()
    assert pricing.is_priced("mystery-9")
    assert pricing.price("mystery-9", 1_000_000, 1_000_000) == 3.0
    assert pricing.price("gpt-5.5", 1_000_000, 0) == 100.0
    assert pricing.is_priced("claude-opus-5-5")  # packaged entries survive


def test_training_prices_available_for_fine_tuning():
    table = pricing.load_pricing()
    assert table.training["gpt-4.1"] == 25.0
    assert pricing.training_price("gpt-4.1-mini", 1_000_000) == 5.0
    assert pricing.training_price("gpt-5.5", 1_000_000) is None
