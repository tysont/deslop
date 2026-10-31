# ABOUTME: Tests for voice pack/unpack plumbing: directory to zip and back, checksum
# ABOUTME: verification, and that commands accept the zip directly.
from __future__ import annotations

import os
import zipfile

import pytest

from deslop.voice import Voice
from deslop.voice.pack import PackError, pack_voice, unpack_voice


def test_pack_unpack_round_trip_and_checksum(fixture_voice, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    fixture_voice.save(work)
    zip_path = pack_voice(work, tmp_path / "fixture.voice")
    assert zipfile.is_zipfile(zip_path)
    out = unpack_voice(zip_path, tmp_path / "unpacked")
    again = Voice.load(out)
    assert again.meta.checksum and again.profile == fixture_voice.profile
    assert again.metrics.metrics == fixture_voice.metrics.metrics


def test_tampered_zip_is_rejected(fixture_voice, tmp_path):
    zip_path = fixture_voice.save(tmp_path / "fixture.voice")
    bad = tmp_path / "bad.voice"
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


def test_voice_load_from_zip_is_read_only(fixture_voice, tmp_path):
    zip_path = fixture_voice.save(tmp_path / "fixture.voice")
    v = Voice.load(zip_path)
    assert v.name == "fixture" and v.read_only and v.path == zip_path
    with pytest.raises(PackError):
        v.save(zip_path)
    other = v.save(tmp_path / "copy.voice")
    assert zipfile.is_zipfile(other) and not os.path.samefile(other, zip_path)
