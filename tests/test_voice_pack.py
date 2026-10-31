# ABOUTME: Tests for voice pack/unpack: zip round trip, checksum verification, and loading a
# ABOUTME: packed voice read-only through Voice.load and the CLI.
from __future__ import annotations

import os
import zipfile

import pytest
from typer.testing import CliRunner

from deslop.cli.app import app
from deslop.voice import Voice
from deslop.voice.pack import PackError, pack_voice, unpack_voice

runner = CliRunner()


def test_pack_unpack_round_trip_and_checksum(fixture_voice, tmp_path):
    zip_path = pack_voice(fixture_voice.path, tmp_path / "fixture.voice.zip")
    assert zipfile.is_zipfile(zip_path)
    out = unpack_voice(zip_path, tmp_path / "unpacked.voice")
    again = Voice.load(out)
    assert again.meta.checksum and again.profile == fixture_voice.profile
    assert again.metrics.metrics == fixture_voice.metrics.metrics


def test_tampered_zip_is_rejected(fixture_voice, tmp_path):
    zip_path = pack_voice(fixture_voice.path, tmp_path / "fixture.voice.zip")
    bad = tmp_path / "bad.voice.zip"
    with zipfile.ZipFile(zip_path) as src, zipfile.ZipFile(bad, "w") as dst:
        for item in src.infolist():
            data = src.read(item.filename)
            if item.filename.endswith("rules.md"):
                data += b"\n- tampered\n"
            dst.writestr(item, data)
    with pytest.raises(PackError):
        unpack_voice(bad, tmp_path / "x")
    with pytest.raises(PackError):
        Voice.load(bad)


def test_voice_load_accepts_zip_read_only(fixture_voice, tmp_path):
    zip_path = pack_voice(fixture_voice.path, tmp_path / "fixture.voice.zip")
    v = Voice.load(zip_path)
    assert v.name == "fixture" and v.path is not None and v.path.is_dir()
    assert not os.access(v.path / "rules.md", os.W_OK)
    with pytest.raises(PackError):
        v.save(v.path)


def test_cli_pack_unpack_and_score_with_zip(fixture_voice, tmp_path, fixtures_dir):
    zip_path = tmp_path / "fixture.voice.zip"
    r = runner.invoke(app, ["voice", "pack", str(fixture_voice.path), "--out", str(zip_path)])
    assert r.exit_code == 0, r.output
    r = runner.invoke(app, ["voice", "unpack", str(zip_path), "--out", str(tmp_path / "u.voice")])
    assert r.exit_code == 0 and (tmp_path / "u.voice" / "voice.yaml").exists()
    r = runner.invoke(
        app, ["score", str(fixtures_dir / "heldout_bradley.md"), "--voice", str(zip_path), "--json"]
    )
    assert r.exit_code in (0, 2), r.output
    assert '"voice": "fixture"' in r.stdout or '"voice":"fixture"' in r.stdout
    r = runner.invoke(app, ["voice", "show", str(zip_path), "--clusters"])
    assert r.exit_code == 0 and "register" in r.stdout.lower()
