# ABOUTME: Turns files and directories into an internal corpus for calibration, resolving each
# ABOUTME: document's author from an author map, YAML frontmatter, a manifest, or the filename.
from __future__ import annotations

import fnmatch
import re
import tempfile
from collections.abc import Callable
from pathlib import Path

import yaml

from deslop.corpus import Corpus, DuplicateError
from deslop.corpus.ingest import IngestError

TEXT_SUFFIXES = {".txt", ".text", ".md", ".markdown", ".html", ".htm", ".epub", ".pdf", ".docx"}
FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.S)


class CollectReport:
    def __init__(self) -> None:
        self.added: list[str] = []
        self.duplicates: list[str] = []
        self.failed: list[tuple[str, str]] = []


def load_author_map(path: Path | str | None) -> dict[str, str]:
    """A yaml mapping of filename glob (or exact name) to author; the key `default` names
    the author for files nothing else matches."""
    if path is None:
        return {}
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: author map must be a mapping of pattern to author")
    return {str(k): str(v) for k, v in data.items()}


def frontmatter(path: Path) -> dict[str, str]:
    if path.suffix.lower() not in (".md", ".markdown", ".txt", ".text"):
        return {}
    try:
        head = path.read_text(encoding="utf-8", errors="ignore")[:4000]
    except OSError:
        return {}
    m = FRONTMATTER_RE.match(head)
    if not m:
        return {}
    try:
        data = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        return {}
    return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}


def author_for(path: Path, author_map: dict[str, str]) -> str:
    """Resolution order: author map (glob on the path or its name), frontmatter `author`,
    then the filename's first underscore-separated part, title-cased."""
    for pattern, author in author_map.items():
        if pattern == "default":
            continue
        if fnmatch.fnmatch(path.name, pattern) or fnmatch.fnmatch(str(path), pattern):
            return author
    fm = frontmatter(path)
    if fm.get("author"):
        return fm["author"]
    if "default" in author_map:
        return author_map["default"]
    stem = path.stem.split("_")[0].replace("-", " ").strip()
    return stem.title() if stem else "Unknown"


def title_for(path: Path) -> str:
    fm = frontmatter(path)
    if fm.get("title"):
        return fm["title"]
    parts = path.stem.split("_", 1)
    raw = parts[1] if len(parts) == 2 else parts[0]
    return raw.replace("_", " ").replace("-", " ").strip() or path.stem


def gather_files(paths: list[Path]) -> list[Path]:
    """Files from the given paths; directories are walked recursively. A directory holding
    `manifest.jsonl` is returned as that directory so its manifest is used instead."""
    out: list[Path] = []

    def walk(directory: Path) -> None:
        if (directory / "manifest.jsonl").is_file():
            out.append(directory)
            return
        for child in sorted(directory.iterdir()):
            if child.is_dir():
                walk(child)
            elif child.suffix.lower() in TEXT_SUFFIXES and child.name != "corpus.yaml":
                out.append(child)

    for p in paths:
        if p.is_dir():
            walk(p)
        elif p.is_file():
            out.append(p)
        else:
            raise IngestError(f"no such file or directory: {p}")
    return out


def collect(
    paths: list[Path],
    *,
    name: str,
    author_map: dict[str, str] | None = None,
    root: Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[Corpus, CollectReport]:
    """Build a corpus under `root` (a fresh temporary directory by default) from files,
    directories and manifest directories. Near-duplicates are skipped, not fatal."""
    root = root or Path(tempfile.mkdtemp(prefix="deslop-corpus-"))
    corpus = Corpus.init(root, name=name)
    report = CollectReport()
    author_map = author_map or {}
    for item in gather_files(paths):
        if item.is_dir():
            manifest_report = corpus.add_manifest(item)
            report.added += [e.title for e in manifest_report.added]
            report.duplicates += manifest_report.duplicates
            report.failed += manifest_report.failed
            if progress:
                progress(f"{item}: {len(manifest_report.added)} documents from manifest")
            continue
        author = author_for(item, author_map)
        title = title_for(item)
        fm = frontmatter(item)
        try:
            corpus.add(
                str(item),
                author=author,
                title=title,
                year=_year(fm.get("year") or fm.get("date")),
                license=fm.get("license", ""),
                origin=fm.get("url") or fm.get("source") or "",
            )
        except DuplicateError:
            report.duplicates.append(title)
            continue
        except IngestError as exc:
            report.failed.append((title, str(exc)))
            continue
        report.added.append(title)
        if progress:
            progress(f"added {author}: {title}")
    return corpus, report


def _year(value: str | None) -> int | None:
    if not value:
        return None
    m = re.match(r"\s*(\d{4})", str(value))
    return int(m.group(1)) if m else None
