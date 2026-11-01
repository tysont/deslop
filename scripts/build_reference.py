# ABOUTME: Builds the bundled primary texts under src/deslop/reference from Project Gutenberg
# ABOUTME: plain text: KJV Genesis (eBook #10) and Jowett's Republic with Stephanus pages (#55201).
"""Usage: python scripts/build_reference.py DOWNLOAD_DIR

DOWNLOAD_DIR holds pg10.txt (The King James Version of the Bible) and pg55201.txt (The
Republic of Plato, Jowett, 1888 text with the complete Stephanus numbering), fetched from
https://www.gutenberg.org/cache/epub/{10,55201}/pg{10,55201}.txt. Both are public domain.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "src" / "deslop" / "reference"


def kjv_genesis(raw: str) -> str:
    start = raw.rindex("The First Book of Moses: Called Genesis")
    end = raw.rindex("The Second Book of Moses: Called Exodus")
    body = raw[start:end]
    verses: list[tuple[int, int, str]] = []
    current: list[str] = []
    ref: tuple[int, int] | None = None
    for line in body.splitlines():
        m = re.match(r"^(\d{1,3}):(\d{1,3})\s+(.*)$", line)
        if m:
            if ref:
                verses.append((ref[0], ref[1], " ".join(current)))
            ref = (int(m.group(1)), int(m.group(2)))
            current = [m.group(3).strip()]
        elif line.strip() and ref:
            current.append(line.strip())
    if ref:
        verses.append((ref[0], ref[1], " ".join(current)))
    lines = [
        "# title: Genesis (King James Version)",
        "# book: Genesis",
        "# aliases: Gen, Ge, Gn",
        "# source: Project Gutenberg eBook #10, The King James Version of the Bible",
        "# license: public domain",
        "",
    ]
    chapter = 0
    for c, v, text in verses:
        if c != chapter:
            if chapter:
                lines.append("")
            chapter = c
        lines.append(f"{{{{Genesis {c}:{v}}}}} {text}")
    return "\n".join(lines) + "\n"


def jowett_republic(raw: str) -> str:
    start = raw.index("\nBOOK I\n") + 1
    end = raw.index("\nINDEX.", start)
    body = raw[start:end]
    body = re.sub(r"\[Sidenote:.*?\]", "", body, flags=re.S)
    body = re.sub(r"\[Footnote \d+:.*?\]", "", body, flags=re.S)
    body = re.sub(r"\[\d+\]", "", body)
    body = re.sub(r"\{\d+\}", "", body)
    body = re.sub(r"\*Ed\. Steph\. (\d{3})\*", r"{{\1a}}", body)
    body = re.sub(
        r"\*(\d{3})([A-E])\*", lambda m: "{{" + m.group(1) + m.group(2).lower() + "}}", body
    )
    paragraphs = []
    for para in re.split(r"\n\s*\n", body):
        text = " ".join(line.strip() for line in para.splitlines() if line.strip())
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            paragraphs.append(text)
    lines = [
        "# title: The Republic (Jowett)",
        "# book: Republic",
        "# aliases: Rep",
        "# source: Project Gutenberg eBook #55201, The Republic of Plato, translated by "
        "Benjamin Jowett (third edition, 1888, with Stephanus pages)",
        "# license: public domain",
        "",
    ]
    return "\n".join(lines) + "\n\n".join(paragraphs) + "\n"


def main() -> None:
    src = Path(sys.argv[1])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "kjv_genesis.txt").write_text(
        kjv_genesis((src / "pg10.txt").read_text(encoding="utf-8")), encoding="utf-8"
    )
    (OUT / "jowett_republic.txt").write_text(
        jowett_republic((src / "pg55201.txt").read_text(encoding="utf-8")), encoding="utf-8"
    )
    for name in ("kjv_genesis.txt", "jowett_republic.txt"):
        text = (OUT / name).read_text(encoding="utf-8")
        print(name, len(text.split()), "words,", text.count("{{"), "markers")


if __name__ == "__main__":
    main()
