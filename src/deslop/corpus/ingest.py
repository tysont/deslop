# ABOUTME: Converts files and URLs (txt, md, html, epub, pdf) into normalized plain text.
# ABOUTME: Uses trafilatura for HTML, pandoc for epub/docx, pypdf for PDF.
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import httpx
import trafilatura

from deslop.corpus.normalize import normalize


class IngestError(Exception):
    pass


def ingest(source: str) -> tuple[str, str]:
    """Return (normalized_text, suggested_title) for a path or URL."""
    if source.startswith(("http://", "https://")):
        return ingest_url(source)
    path = Path(source)
    if not path.exists():
        raise IngestError(f"no such file: {source}")
    suffix = path.suffix.lower()
    if suffix in (".txt", ".text", ""):
        return normalize(strip_frontmatter(read_text(path))), path.stem
    if suffix in (".md", ".markdown"):
        return normalize(strip_markdown(read_text(path)), gutenberg=False), path.stem
    if suffix in (".html", ".htm", ".xhtml"):
        return normalize(extract_html(read_text(path)), gutenberg=False), path.stem
    if suffix in (".epub", ".docx", ".odt", ".rtf"):
        return normalize(pandoc_to_text(path), gutenberg=False), path.stem
    if suffix == ".pdf":
        return normalize(pdf_to_text(path), gutenberg=False), path.stem
    raise IngestError(f"unsupported file type: {suffix}")


def read_text(path: Path) -> str:
    data = path.read_bytes()
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    raise IngestError(f"cannot decode {path}")


def ingest_url(url: str) -> tuple[str, str]:
    resp = httpx.get(url, follow_redirects=True, timeout=60.0, headers={"User-Agent": "deslop/0.1"})
    resp.raise_for_status()
    ctype = resp.headers.get("content-type", "")
    if "text/plain" in ctype:
        return normalize(resp.text), url.rsplit("/", 1)[-1] or url
    text = extract_html(resp.text)
    title = trafilatura.extract_metadata(resp.text)
    name = title.title if title and title.title else url
    return normalize(text, gutenberg=False), name


def extract_html(html: str) -> str:
    text = trafilatura.extract(
        html, include_comments=False, include_tables=False, favor_precision=True
    )
    if not text:
        raise IngestError("could not extract article text from HTML")
    return text


def strip_frontmatter(text: str) -> str:
    import re

    return re.sub(r"\A---\s*\n.*?\n---\s*\n", "", text, count=1, flags=re.S)


def strip_markdown(text: str) -> str:
    import re

    text = strip_frontmatter(text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"^(?:[-*+]|\d+\.)\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^>\s?", "", text, flags=re.MULTILINE)
    text = re.sub(r"`{1,3}([^`]*)`{1,3}", r"\1", text)
    text = re.sub(r"(\*\*|__)(.+?)\1", r"\2", text)
    return text


def pandoc_to_text(path: Path) -> str:
    if path.suffix.lower() == ".epub" and _epub_is_drm(path):
        raise IngestError(f"{path.name} is DRM-protected; deslop does not circumvent DRM")
    if shutil.which("pandoc") is None:
        raise IngestError("pandoc is required to ingest this file type; install it first")
    proc = subprocess.run(
        ["pandoc", str(path), "-t", "plain", "--wrap=none"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise IngestError(f"pandoc failed: {proc.stderr.strip()[:300]}")
    return proc.stdout


def _epub_is_drm(path: Path) -> bool:
    import zipfile

    try:
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
    except zipfile.BadZipFile as exc:
        raise IngestError(f"{path.name} is not a valid epub") from exc
    return "META-INF/encryption.xml" in names or "META-INF/rights.xml" in names


def pdf_to_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise IngestError("pypdf is required for PDF ingest") from exc
    reader = PdfReader(str(path))
    if reader.is_encrypted:
        raise IngestError(f"{path.name} is encrypted; deslop does not circumvent DRM")
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)
