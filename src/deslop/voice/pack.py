# ABOUTME: Packs a voice directory into a zip with a content checksum in voice.yaml, and
# ABOUTME: unpacks or temporarily extracts it (read-only) with the checksum verified.
from __future__ import annotations

import atexit
import hashlib
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import yaml

VOICE_FILES = ("metrics.json", "lexicon.json", "rules.md", "profile.md", "exemplars.md")


class PackError(Exception):
    pass


def checksum(root: Path) -> str:
    """sha256 over the five content files of a voice directory, in a fixed order."""
    h = hashlib.sha256()
    for name in VOICE_FILES:
        h.update(name.encode())
        h.update((root / name).read_bytes())
    return h.hexdigest()


def pack_voice(directory: Path | str, out: Path | str) -> Path:
    """Zip a voice directory to `out`; voice.yaml inside the zip carries the checksum."""
    root = Path(directory)
    if not (root / "voice.yaml").exists():
        raise PackError(f"not a voice directory: {root}")
    meta = yaml.safe_load((root / "voice.yaml").read_text(encoding="utf-8"))
    meta["checksum"] = checksum(root)
    out_path = Path(out)
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("voice.yaml", yaml.safe_dump(meta, sort_keys=False, allow_unicode=True))
        for name in VOICE_FILES:
            zf.write(root / name, name)
    return out_path


def unpack_voice(archive: Path | str, out: Path | str) -> Path:
    """Extract a packed voice into `out` and verify its checksum."""
    out_path = Path(out)
    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
        missing = [n for n in ("voice.yaml", *VOICE_FILES) if n not in names]
        if missing:
            raise PackError(f"{archive} is missing {', '.join(missing)}")
        out_path.mkdir(parents=True, exist_ok=True)
        for name in ("voice.yaml", *VOICE_FILES):
            (out_path / name).write_bytes(zf.read(name))
    meta = yaml.safe_load((out_path / "voice.yaml").read_text(encoding="utf-8"))
    expected = meta.get("checksum")
    actual = checksum(out_path)
    if expected != actual:
        shutil.rmtree(out_path, ignore_errors=True)
        raise PackError(
            f"checksum mismatch in {archive}: voice.yaml says {expected}, files hash to {actual}"
        )
    return out_path


def extract_read_only(archive: Path | str) -> Path:
    """Unpack into a temporary directory whose files are read-only; removed at exit."""
    tmp = Path(tempfile.mkdtemp(prefix="deslop-voice-"))
    atexit.register(shutil.rmtree, tmp, True)
    unpack_voice(archive, tmp)
    for name in ("voice.yaml", *VOICE_FILES):
        os.chmod(tmp / name, 0o444)
    return tmp


def is_packed(path: Path | str) -> bool:
    p = Path(path)
    return p.is_file() and zipfile.is_zipfile(p)
