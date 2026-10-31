# ABOUTME: Ingest tests for HTML, epub (via pandoc), URL plain text, and DRM detection.
# ABOUTME: Skips epub when pandoc is unavailable.
import shutil
import subprocess
import zipfile

import pytest

from deslop.corpus.ingest import IngestError, ingest

ARTICLE = (
    "<html><head><title>T</title></head><body><nav>Home | About</nav><main><article>"
    + "".join(
        f"<p>Paragraph {i} of the article contains enough prose to be extracted as real content by the parser.</p>"
        for i in range(8)
    )
    + "</article></main><footer>Copyright</footer></body></html>"
)


def test_html_boilerplate_stripped(tmp_path):
    f = tmp_path / "a.html"
    f.write_text(ARTICLE, encoding="utf-8")
    text, title = ingest(str(f))
    assert "Paragraph 1 of the article" in text and "Paragraph 7" in text
    assert "Home | About" not in text and "Copyright" not in text
    assert title == "a"


@pytest.mark.skipif(shutil.which("pandoc") is None, reason="pandoc not installed")
def test_epub_round_trip(tmp_path):
    md = tmp_path / "book.md"
    md.write_text(
        "---\ntitle: Book\n---\n\n# One\n\n"
        + "A sentence of prose in the first chapter. " * 20
        + "\n\n# Two\n\n"
        + "Another chapter with its own prose here. " * 20
    )
    epub = tmp_path / "book.epub"
    subprocess.run(["pandoc", str(md), "-o", str(epub)], check=True)
    text, title = ingest(str(epub))
    assert "A sentence of prose in the first chapter." in text
    assert "Another chapter with its own prose here." in text
    assert title == "book"


def test_drm_epub_rejected(tmp_path):
    epub = tmp_path / "locked.epub"
    with zipfile.ZipFile(epub, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip")
        zf.writestr("META-INF/encryption.xml", "<encryption/>")
    with pytest.raises(IngestError, match="DRM"):
        ingest(str(epub))


def test_url_plain_text(monkeypatch):
    import httpx

    def fake_get(url, **kwargs):
        return httpx.Response(
            200,
            text="Line one\nwraps here.\n\nPara two.",
            headers={"content-type": "text/plain"},
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr(httpx, "get", fake_get)
    text, title = ingest("https://example.org/files/doc.txt")
    assert text == "Line one wraps here.\n\nPara two.\n"
    assert title == "doc.txt"
