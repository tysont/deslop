# ABOUTME: Tests for the lab helpers: paragraph pools for judge calibration and the blind-set
# ABOUTME: export (shuffled pairs plus a separate answer key).
from __future__ import annotations

import json

from deslop import lab
from deslop.blind import Pair, Pairing


def test_llm_paragraphs_reads_cached_json_and_markdown(tmp_path):
    long = " ".join(["word"] * 50)
    (tmp_path / "a.json").write_text(json.dumps({"text": f"{long}\n\nshort"}), encoding="utf-8")
    (tmp_path / "b.md").write_text(f"{long} two\n\n{long} three", encoding="utf-8")
    (tmp_path / "c.bin").write_bytes(b"\x00")
    paras = lab.llm_paragraphs(tmp_path)
    assert len(paras) == 3 and all(p.startswith("word") for p in paras)


def test_heldout_paragraphs_uses_the_voice_split(fixture_voice, fixtures_dir):
    paras, method = lab.heldout_paragraphs(fixture_voice, [fixtures_dir / "corpus_small"])
    assert paras and "corpus too small" in method, "four authors: every document is used"
    from deslop.nlp import count_words

    assert all(count_words(p) >= lab.MIN_WORDS for p in paras)


def test_export_blind_writes_pairs_without_answers_and_a_key(tmp_path):
    pairing = Pairing(
        pairs=[
            Pair(
                candidate="Machine paragraph one.",
                candidate_source="001.v2.md",
                control="Human paragraph one.",
                control_source="essay.txt",
                overlap=0.2,
                candidate_position="B",
            )
        ],
        skipped=1,
    )
    pairs_path, key_path = lab.export_blind(pairing, tmp_path / "blind", "tutor")
    text = pairs_path.read_text(encoding="utf-8")
    assert "## Pair 1" in text and "**A**" in text and "Human paragraph one." in text
    assert "001.v2.md" not in text and "candidate" not in text.lower().split("## pair 1")[1]
    key = json.loads(key_path.read_text(encoding="utf-8"))
    assert key == [
        {
            "pair": 1,
            "candidate": "B",
            "candidate_source": "001.v2.md",
            "control_source": "essay.txt",
            "overlap": 0.2,
        }
    ]
